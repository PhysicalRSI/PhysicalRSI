"""Public RoboTwin RGB-D surface location and its read-only CAP callback."""
import math
import time

import numpy as np

from .public_geometry import masked_surface_geometry
from .robotwin_primitives import NativeRoboTwinPrimitives


def public_world_camera(observation, camera='head_camera'):
    if camera not in ('head_camera', 'front_camera', 'left_camera', 'right_camera'):
        raise ValueError('Unsupported public RoboTwin camera')
    source = observation['cameras'][camera]
    extrinsic = np.asarray(source['extrinsic_cv'], dtype=float)
    gl = np.asarray(source['cam2world_gl'], dtype=float)
    if extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all():
        raise ValueError('Invalid public world-to-camera extrinsic')
    matrix = np.eye(4)
    matrix[:3] = extrinsic
    transform = np.linalg.inv(matrix)
    if (gl.shape != (4, 4) or not np.isfinite(gl).all()
            or not np.allclose(transform, gl @ np.diag([1, -1, -1, 1]), atol=1e-5)):
        raise ValueError('Public optical and OpenGL camera calibrations disagree')
    # Pinned RoboTwin camera.get_depth exports millimetres; no image flip.
    return {'images': {'rgb': source['rgb'],
                       'depth': np.asarray(source['depth'], dtype=float) / 1000.},
            'intrinsics': source['intrinsic_cv'], 'pose_mat': transform}


class PublicRoboTwinLocator:
    def __init__(self, *, segmenter):
        self.segmenter = segmenter

    def __call__(self, observation, prompt, *, camera='head_camera', deadline):
        source = public_world_camera(observation, camera)
        result = self.segmenter(source['images']['rgb'], prompt, deadline=deadline)
        if not result['detections']:
            return {'status': 'not_detected', 'frame': 'world',
                    'semantic_identity_verified': False}
        detection = max(result['detections'], key=lambda row: row['score'])
        geometry = masked_surface_geometry(source, detection['mask'])
        # The shared geometry operation applies pose_mat. Here that transform
        # maps into world coordinates rather than the LIBERO robot base.
        return {**geometry, 'frame': 'world',
                'detection_score': float(detection['score']),
                'semantic_identity_verified': False}


class PerceivedRoboTwinPrimitives(NativeRoboTwinPrimitives):
    def __init__(self, *args, object_locator, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_locator = object_locator

    @property
    def handlers(self):
        handlers = super().handlers

        def locate(args, kwargs, *, deadline):
            if not math.isfinite(deadline):
                raise ValueError('A finite perception deadline is required')
            if time.monotonic() >= deadline:
                raise TimeoutError('RoboTwin location deadline reached')
            value = self._object_locator(self.get_observation(), *args, **kwargs,
                                         deadline=deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError('RoboTwin location deadline reached')
            return value

        handlers['locate_object'] = locate
        return handlers
