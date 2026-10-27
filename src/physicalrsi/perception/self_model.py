"""Own-robot projection from calibrated geometry and measured actuator positions.

This trusted provider has no simulator handle or scene-object interface. A visual
convex hull is conservative: foreground objects may overlap it, and calibration
errors or shadows can invalidate self-motion exclusion.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

from physicalrsi.runtime.sensors import decode_rgb, validate_observation


class RobotSelfModel:
    def __init__(self, urdf, *, mounts, intrinsic_cv, extrinsic_cv, image_size_px,
                 camera, embodiment_revision, calibration_revision):
        import cv2
        import numpy as np
        import scipy
        from scipy.spatial.transform import Rotation
        import trimesh
        if not all(isinstance(x, str) and x for x in (camera, embodiment_revision, calibration_revision)):
            raise ValueError('Declare camera, embodiment and calibration identities')
        if (len(image_size_px) != 2 or any(type(x) is not int or x <= 0 for x in image_size_px)):
            raise ValueError('Declare calibrated image dimensions')
        def matrix(value, shape):
            result = np.array(value, dtype=float, copy=True)
            if result.shape != shape or not np.isfinite(result).all():
                raise ValueError('Invalid calibration matrix')
            return result
        def rigid(value):
            result = matrix(value, (4, 4))
            if (not np.allclose(result[3], [0, 0, 0, 1])
                    or not np.allclose(result[:3,:3].T @ result[:3,:3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(result[:3,:3]), 1, atol=1e-5)):
                raise ValueError('Expected rigid calibration transform')
            return result
        self._K = matrix(intrinsic_cv, (3, 3))
        if self._K[0,0] <= 0 or self._K[1,1] <= 0 or not np.allclose(self._K[2], [0,0,1]):
            raise ValueError('Invalid pinhole intrinsics')
        extrinsic = np.asarray(extrinsic_cv, dtype=float)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        self._E = rigid(extrinsic)
        self._camera, self._size = camera, tuple(image_size_px)
        self._embodiment, self._calibration = embodiment_revision, calibration_revision
        path = Path(urdf)
        raw = path.read_bytes()
        tree = ET.fromstring(raw)
        def vector(node, name, default):
            result = np.array([float(x) for x in node.get(name, default).split()])
            if result.shape != (3,) or not np.isfinite(result).all():
                raise ValueError('Invalid URDF vector')
            return result
        def origin(node):
            result = np.eye(4)
            if node is not None:
                result[:3,3] = vector(node, 'xyz', '0 0 0')
                result[:3,:3] = Rotation.from_euler('xyz', vector(node, 'rpy', '0 0 0')).as_matrix()
            return result
        joints = []
        for node in tree.findall('joint'):
            kind = node.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError('Unsupported joint type')
            axis = vector(node.find('axis'), 'xyz', '1 0 0') if node.find('axis') is not None else np.array([1.,0,0])
            if not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError('Joint axis must be normalized')
            joints.append(dict(name=node.get('name'), kind=kind, parent=node.find('parent').get('link'),
                child=node.find('child').get('link'), origin=origin(node.find('origin')), axis=axis))
        links = {node.get('name') for node in tree.findall('link')}
        roots = links - {j['child'] for j in joints}
        if len(roots) != 1:
            raise ValueError('Expected one robot root')
        self._root = next(iter(roots))
        known = {self._root}; ordered = []; pending = list(joints)
        while pending:
            ready = [j for j in pending if j['parent'] in known]
            if not ready:
                raise ValueError('Disconnected or cyclic robot model')
            for j in ready:
                if j['child'] in known or j['child'] not in links:
                    raise ValueError('Invalid robot tree')
                ordered.append(j); known.add(j['child']); pending.remove(j)
        self._joints = ordered
        dynamic = {j['name'] for j in joints if j['kind'] != 'fixed'}
        if not isinstance(mounts, dict) or not mounts:
            raise ValueError('Declare robot mounts and actuator bindings')
        self._mounts = deepcopy(mounts)
        for mount in self._mounts.values():
            if set(mount) != {'base_to_world', 'bindings'} or set(mount['bindings']) != dynamic:
                raise ValueError('Bindings must cover every moving joint')
            rigid(mount['base_to_world'])
            for binding in mount['bindings'].values():
                if (set(binding) != {'index', 'scale', 'offset'}
                        or type(binding['index']) is not int or binding['index'] < 0
                        or not all(type(binding[k]) in (int,float) and np.isfinite(binding[k]) for k in ('scale','offset'))):
                    raise ValueError('Invalid actuator binding')
        self._actuator_count = 1 + max(b['index'] for m in self._mounts.values() for b in m['bindings'].values())
        self._geometry = []; assets = {'urdf': hashlib.sha256(raw).hexdigest()}
        for link in tree.findall('link'):
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('This projector requires explicit visual meshes')
                asset = path.parent / mesh.get('filename')
                asset_bytes = asset.read_bytes()
                loaded = trimesh.load(str(asset), force='mesh', process=False)
                if hashlib.sha256(asset.read_bytes()).digest() != hashlib.sha256(asset_bytes).digest():
                    raise ValueError('Robot asset changed while loading')
                vertices = np.asarray(loaded.convex_hull.vertices) * vector(mesh, 'scale', '1 1 1')
                vertices = (origin(visual.find('origin')) @ np.c_[vertices, np.ones(len(vertices))].T).T
                if not np.isfinite(vertices).all():
                    raise ValueError('Invalid robot mesh')
                self._geometry.append((link.get('name'), vertices))
                assets[mesh.get('filename')] = hashlib.sha256(asset_bytes).hexdigest()
        self._identity = dict(adapter='physicalrsi.robot-self-model/v1', assets=assets, mounts=self._mounts,
            intrinsic_cv=self._K.tolist(), extrinsic_cv=self._E.tolist(), image_size_px=list(self._size),
            camera=camera, embodiment_revision=embodiment_revision, calibration_revision=calibration_revision,
            numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__, trimesh=trimesh.__version__)

    def identity(self):
        return deepcopy(self._identity)

    def forward_kinematics(self, observation):
        import numpy as np
        from scipy.spatial.transform import Rotation
        sensors = validate_observation(observation)
        if (sensors['embodiment_revision'] != self._embodiment
                or sensors['calibration_revision'] != self._calibration):
            raise ValueError('Robot model does not match sensor configuration')
        values = sensors['actuator_position']; output = {}
        if len(values) != self._actuator_count:
            raise ValueError('Actuator layout does not match robot model')
        for name, mount in self._mounts.items():
            q = {}
            for joint, binding in mount['bindings'].items():
                if binding['index'] >= len(values):
                    raise ValueError('Actuator layout does not match robot model')
                q[joint] = values[binding['index']]*binding['scale'] + binding['offset']
            transforms = {self._root: np.array(mount['base_to_world'], dtype=float)}
            for joint in self._joints:
                delta = np.eye(4)
                if joint['kind'] in ('revolute', 'continuous'):
                    delta[:3,:3] = Rotation.from_rotvec(joint['axis']*q[joint['name']]).as_matrix()
                elif joint['kind'] == 'prismatic':
                    delta[:3,3] = joint['axis']*q[joint['name']]
                transforms[joint['child']] = transforms[joint['parent']] @ joint['origin'] @ delta
            output[name] = transforms
        return output

    def mask(self, observation, *, camera):
        import cv2
        import numpy as np
        if camera != self._camera:
            raise ValueError('Camera does not match robot projection calibration')
        poses = self.forward_kinematics(observation)
        image = decode_rgb(observation['cameras'][camera]['rgb_png_b64'])
        height, width = image.shape[:2]
        if (width, height) != self._size:
            raise ValueError('Image dimensions do not match calibration')
        mask = np.zeros((height, width), np.uint8)
        for transforms in poses.values():
            for name, vertices in self._geometry:
                xyz = (self._E @ transforms[name] @ vertices.T).T[:,:3]
                if np.any(xyz[:,2] <= .001):
                    raise ValueError('Robot mesh crosses camera near plane')
                uv = (self._K @ xyz.T).T
                uv = uv[:,:2] / uv[:,2:3]
                hull = cv2.convexHull(np.rint(np.clip(uv, -10000, 10000)).astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 255)
        return mask.astype(bool)
