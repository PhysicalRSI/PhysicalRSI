"""Bounded public-depth client for the pinned ASPIRE Contact-GraspNet service."""
import base64
import http.client
import io
import math
import json
import socket
import time
from PhysicalRSI_core.infra.rpc.authenticated_http import SocketDeadline


def _encode(array):
    import numpy as np
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode()


def _decode(value):
    import numpy as np
    if not isinstance(value, str) or len(value) > 4*1024*1024:
        raise ValueError('Grasp array exceeds byte allowance')
    raw = base64.b64decode(value, validate=True)
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError('Unsupported grasp array encoding')
    if (dtype.kind not in 'fi' or len(shape) > 3 or any(x < 0 for x in shape)
            or math.prod(shape) > 16384):
        raise ValueError('Invalid grasp array shape or type')
    if len(raw)-stream.tell() != math.prod(shape)*dtype.itemsize:
        raise ValueError('Grasp array length differs from shape')
    stream.seek(0)
    result = np.load(stream, allow_pickle=False)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite grasp array')
    return result


class ContactGraspClient:
    def __init__(self, *, port=8115, max_grasps=32, depth_range_m=(.2, 2.)):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid local service port')
        if type(max_grasps) is not int or not 1 <= max_grasps <= 64:
            raise ValueError('Invalid grasp limit')
        if (not isinstance(depth_range_m, (tuple, list)) or len(depth_range_m) != 2 or
                not all(isinstance(x, (int, float)) and math.isfinite(x) for x in depth_range_m) or
                not .015 <= depth_range_m[0] < depth_range_m[1] <= 20.):
            raise ValueError('Invalid metric grasp depth range')
        self.port, self.max_grasps = port, max_grasps
        self.depth_range_m = tuple(float(x) for x in depth_range_m)

    def __call__(self, depth, intrinsics, mask, *, deadline):
        import numpy as np
        depth = np.asarray(depth, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        mask = np.asarray(mask)
        if (depth.ndim != 2 or min(depth.shape) < 1 or max(depth.shape) > 512
                or mask.shape != depth.shape or not np.isin(mask, [0, 1]).all()
                or not np.isfinite(depth).all() or np.any(depth <= 0)
                or intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
                or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
                or not np.allclose(intrinsics[2], [0, 0, 1])):
            raise ValueError('Invalid calibrated depth or segmentation mask')
        if np.count_nonzero(mask) < 20:
            raise ValueError('Insufficient segmented pixels for grasp planning')
        remaining = deadline-time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError('Grasp planning deadline reached')
        body = json.dumps({'depth_base64': _encode(depth), 'cam_K_base64': _encode(intrinsics),
            'segmap_base64': _encode(mask.astype(np.uint8)), 'segmap_id': 1,
            'local_regions': True, 'filter_grasps': True, 'skip_border_objects': False,
            'z_range': list(self.depth_range_m), 'forward_passes': 1, 'max_retries': 1}).encode()
        return self._request(body, '/plan', 'camera-optical', deadline)

    def from_point_clouds(self, scene_points, target_points, *, deadline):
        """Infer in the caller's shared point-cloud frame from public geometry."""
        import numpy as np
        arrays = []
        for value, maximum in ((scene_points, 100000), (target_points, 50000)):
            points = np.asarray(value, dtype=np.float32)
            if (points.ndim != 2 or points.shape[1] != 3 or not 20 <= len(points) <= maximum
                    or not np.isfinite(points).all() or np.any(np.abs(points) > 20.)):
                raise ValueError('Invalid bounded public point cloud')
            arrays.append(points)
        body = json.dumps({'pc_full_base64': _encode(arrays[0]),
            'pc_segment_base64': _encode(arrays[1]), 'segmap_id': 1,
            'local_regions': True, 'filter_grasps': True,
            'forward_passes': 1, 'max_retries': 1}).encode()
        return self._request(body, '/plan_point_clouds', 'input-point-cloud', deadline)

    def _request(self, body, path, frame, deadline):
        import numpy as np
        remaining = deadline-time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError('Grasp planning deadline reached')
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=remaining)
        watchdog = None
        try:
            connection.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.sock.settimeout(remaining)
            watchdog = SocketDeadline(connection.sock, remaining)
            connection.sock.connect(('127.0.0.1', self.port))
            connection.request('POST', path, body=body, headers={'Content-Type':'application/json'})
            with connection.getresponse() as response:
                data = response.read(8*1024*1024+1)
                if response.status != 200:
                    raise RuntimeError('Contact-GraspNet service rejected the request')
            if len(data) > 8*1024*1024:
                raise ValueError('Grasp response exceeds byte allowance')
            if watchdog.expired or time.monotonic() >= deadline:
                raise TimeoutError('Grasp planning deadline reached')
        finally:
            if watchdog is not None:
                watchdog.cancel()
            connection.close()
        response = json.loads(data)
        grasps, scores = _decode(response['grasps_base64']), _decode(response['scores_base64'])
        if grasps.size == 0 and scores.size == 0:
            return {'frame':frame, 'grasps':[]}
        if (grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or scores.shape != (len(grasps),)
                or len(grasps) > 1024 or np.any(scores < 0) or np.any(scores > 1)
                or not np.allclose(grasps[:, 3], [0, 0, 0, 1], atol=1e-5)):
            raise ValueError('Invalid grasp poses or scores')
        rotations = grasps[:, :3, :3]
        if (not np.allclose(rotations.transpose(0, 2, 1)@rotations, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)):
            raise ValueError('Invalid grasp rotation')
        contacts = None
        if 'contact_pts_base64' in response:
            contacts = _decode(response['contact_pts_base64'])
            if contacts.shape != (len(grasps), 3):
                raise ValueError('Contact predictions must align with grasp poses')
        order = np.argsort(-scores)[:self.max_grasps]
        return {'frame':frame, 'grasps':[
            {'matrix':grasps[i].tolist(), 'score':float(scores[i]),
             **({'contact_point':contacts[i].tolist()} if contacts is not None else {})}
            for i in order]}
