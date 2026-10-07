"""Read-only CAP location callbacks over host-whitelisted RoboEvolve RGB-D.

The host supplies an atomic camera reader returning public_native_camera packets.
Neither a simulator environment nor task evaluator is passed to the segmenter.
Native calls still require an outer supervisor to bound blocking work.
"""
import math
import time

from .public_geometry import masked_surface_geometry


def public_location_handlers(*, camera_reader, segmenter, cameras):
    cameras = frozenset(cameras)
    if not cameras or cameras - {'head_camera', 'front_camera', 'left_camera', 'right_camera'}:
        raise ValueError('Declare supported public cameras')

    def check(deadline):
        if not math.isfinite(deadline):
            raise ValueError('A finite perception deadline is required')
        if time.monotonic() >= deadline:
            raise TimeoutError('RoboEvolve perception deadline reached')

    def locate_many(args, kwargs, *, deadline):
        check(deadline)
        if (len(args) != 1 or not isinstance(args[0], str)
                or not args[0].strip() or len(args[0]) > 256
                or set(kwargs) != {'camera'} or kwargs['camera'] not in cameras):
            raise ValueError('Expected one description and a configured camera')
        packet = camera_reader(kwargs['camera'])
        check(deadline)
        source = packet['camera']
        protocol = packet['sensor_protocol']
        if (protocol.get('depth_source') != 'simulator-rendered'
                or protocol.get('output_frame') != 'world'
                or protocol.get('output_depth') != 'optical-axis-metres'):
            raise ValueError('Expected declared public RoboEvolve RGB-D protocol')
        detections = segmenter(source['images']['rgb'], args[0], deadline=deadline)['detections']
        check(deadline)
        objects = []
        for detection in detections:
            score = float(detection['score'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid detection confidence')
            geometry = masked_surface_geometry(source, detection['mask'])
            if geometry['status'] != 'estimated_surface':
                continue
            objects.append({**geometry, 'frame': 'world', 'detection_score': score,
                            'semantic_identity_verified': False})
        check(deadline)
        return {'status': 'estimated_surfaces' if objects else 'not_located',
                'objects': objects, 'frame': 'world',
                'semantic_identity_verified': False,
                'depth_source': 'simulator-rendered'}

    def locate_one(args, kwargs, *, deadline):
        reply = locate_many(args, kwargs, deadline=deadline)
        if len(reply['objects']) == 1:
            return reply['objects'][0]
        return {'status': 'ambiguous' if reply['objects'] else 'not_located',
                'candidate_count': len(reply['objects']), 'frame': 'world',
                'semantic_identity_verified': False}

    return {'locate_object': locate_one, 'locate_objects': locate_many}
