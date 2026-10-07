import time

import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_location import public_location_handlers
from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_sensor_boundary import public_native_camera


def setup(detections):
    calls = []
    observation = {'rgb': np.zeros((10, 10, 3), dtype=np.uint8),
                   'intrinsic': np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]]),
                   'camera_to_world': np.eye(4), 'private_pose': object()}
    packet = public_native_camera(observation, {'distance_to_image_plane': np.ones((10, 10))})

    def read(camera):
        calls.append(camera)
        return packet

    def segment(rgb, prompt, *, deadline):
        assert rgb.shape == (10, 10, 3)
        assert prompt == 'red block'
        return {'detections': detections}

    return public_location_handlers(camera_reader=read, segmenter=segment,
                                    cameras=['head_camera']), calls


def invoke(handler, **kwargs):
    return handler(['red block'], {'camera': 'head_camera'},
                   deadline=kwargs.get('deadline', time.monotonic()+5))


def test_world_geometry_and_no_confidence_winner_for_multiple_objects():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    handlers, calls = setup([{'score': .9, 'mask': mask}, {'score': .5, 'mask': mask}])
    result = invoke(handlers['locate_objects'])
    assert len(result['objects']) == 2
    assert result['objects'][0]['surface_quantiles']['median'][2] == -1
    assert result['objects'][0]['frame'] == 'world'
    assert not result['objects'][0]['semantic_identity_verified']
    assert invoke(handlers['locate_object'])['status'] == 'ambiguous'
    assert calls == ['head_camera', 'head_camera']


def test_no_detection_or_insufficient_depth_does_not_supply_position():
    for detections in [[], [{'score': .9, 'mask': np.zeros((10, 10), dtype=bool)}]]:
        handlers, _ = setup(detections)
        assert invoke(handlers['locate_object'])['status'] == 'not_located'


def test_expired_deadline_and_invalid_camera_do_not_read_sensor():
    handlers, calls = setup([])
    with pytest.raises(TimeoutError):
        invoke(handlers['locate_objects'], deadline=time.monotonic()-1)
    with pytest.raises(ValueError):
        handlers['locate_objects'](['red block'], {'camera': 'object_gt'},
                                   deadline=time.monotonic()+5)
    assert not calls
