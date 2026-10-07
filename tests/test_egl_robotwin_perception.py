import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.robotwin_perception import (
    PublicRoboTwinLocator, public_world_camera,
)


def test_robo_depth_units_and_world_frame_are_not_libero_base_coordinates():
    transform = np.eye(4)
    transform[:3, 3] = [2, 3, 4]
    source = {'rgb': np.zeros((8, 8, 3), dtype=np.uint8),
              'depth': np.full((8, 8), 1000.),
              'intrinsic_cv': np.array([[4, 0, 3.5], [0, 4, 3.5], [0, 0, 1]]),
              'extrinsic_cv': np.linalg.inv(transform)[:3],
              'cam2world_gl': transform @ np.diag([1, -1, -1, 1])}
    observation = {'cameras': {'head_camera': source}}
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True
    locator = PublicRoboTwinLocator(segmenter=lambda *a, **kw: {
        'detections': [{'score': .9, 'mask': mask}]})
    result = locator(observation, 'red block', deadline=100.)
    assert result['surface_quantiles']['median'] == [2., 3., 5.]
    assert result['frame'] == 'world'
    assert not result['semantic_identity_verified']
    assert np.all(source['depth'] == 1000.)
    source['cam2world_gl'][0, 3] += .01
    with pytest.raises(ValueError, match='calibrations disagree'):
        public_world_camera(observation)


def test_policy_observation_filters_native_object_truth_and_segmentation():
    from types import SimpleNamespace
    from PhysicalRSI_baselines.embodied_goodharts_law.robotwin_primitives import NativeRoboTwinPrimitives

    public_camera = {name: name for name in (
        'rgb', 'depth', 'intrinsic_cv', 'extrinsic_cv', 'cam2world_gl')}
    native = {
        'observation': {'head_camera': {
            **public_camera, 'actor_segmentation': 'private actor mask',
            'mesh_segmentation': 'private mesh mask', 'object_poses': 'private poses',
        }},
        'object_poses': 'private scene poses', 'official_success': True,
    }
    api = object.__new__(NativeRoboTwinPrimitives)
    api._task = SimpleNamespace(get_obs=lambda: native)
    api.get_proprioception = lambda: {'endpose': {'left': [0.] * 7}}
    result = api.get_observation()
    assert result == {'cameras': {'head_camera': public_camera},
                      'endpose': {'left': [0.] * 7}}
    assert native['observation']['head_camera']['object_poses'] == 'private poses'
