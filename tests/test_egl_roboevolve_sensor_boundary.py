import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_sensor_boundary import public_native_camera, named_native_depth_frame


def observation():
    return {'rgb': np.full((3, 3, 3), 42, dtype=np.uint8),
            'intrinsic': np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]),
            'camera_to_world': np.eye(4), 'depth': np.full((3, 3), 999.),
            'prim_path': '/private', 'pointcloud': 'private',
            'object_pose': 'private'}


def test_flat_native_observation_prefers_explicit_plane_depth():
    native = observation()
    result = public_native_camera(native, {'depth': native['depth'],
        'distance_to_camera': np.full((3, 3), 5.),
        'distance_to_image_plane': np.ones((3, 3)), 'segmentation': 'private'})
    assert set(result) == {'camera', 'sensor_protocol'}
    assert set(result['camera']) == {'images', 'intrinsics', 'pose_mat'}
    np.testing.assert_array_equal(result['camera']['images']['rgb'], native['rgb'])
    np.testing.assert_allclose(result['camera']['images']['depth'], 1.)
    assert result['sensor_protocol']['native_depth_channel'] == 'distance_to_image_plane'
    native['rgb'][:] = 0
    assert result['camera']['images']['rgb'].min() == 42


def test_explicit_ray_channel_is_converted_and_declared():
    result = public_native_camera(observation(), {'distance_to_camera': np.ones((3, 3))})
    assert result['camera']['images']['depth'][0, 0] == pytest.approx(1/np.sqrt(3))
    assert result['sensor_protocol']['native_depth_channel'] == 'distance_to_camera'


def test_ambiguous_or_invalid_named_depth_is_not_silently_replaced():
    with pytest.raises(ValueError, match='named'):
        public_native_camera(observation(), {'depth': np.ones((3, 3))})
    with pytest.raises(ValueError, match='Invalid'):
        public_native_camera(observation(), {'distance_to_image_plane': np.ones((2, 2)),
                                            'distance_to_camera': np.ones((3, 3))})


def test_named_annotator_supports_native_frame_without_depth_and_copies_data():
    from types import SimpleNamespace
    depth = np.ones((3, 3))
    sensor = SimpleNamespace(get_current_frame=lambda: {'depth': depth * 999},
        _custom_annotators={'distance_to_image_plane': SimpleNamespace(get_data=lambda: depth)})
    frame = named_native_depth_frame(sensor)
    depth[:] = 4
    result = public_native_camera(observation(), frame)
    np.testing.assert_allclose(result['camera']['images']['depth'], 1.)


def test_generic_depth_and_segmentation_are_never_read_as_named_depth():
    from types import SimpleNamespace
    def forbidden():
        pytest.fail('Generic depth or segmentation was consulted')
    sensor = SimpleNamespace(get_current_frame=lambda: {'depth': np.ones((3, 3))},
        get_depth=forbidden, _custom_annotators={'segmentation': SimpleNamespace(get_data=forbidden)})
    with pytest.raises(ValueError, match='named'):
        named_native_depth_frame(sensor)


def test_named_annotator_failure_is_not_hidden_by_alternative_channel():
    from types import SimpleNamespace
    def failed():
        raise RuntimeError('Sensor unavailable')
    sensor = SimpleNamespace(get_current_frame=lambda: {'distance_to_camera': np.ones((3, 3))},
        _custom_annotators={'distance_to_image_plane': SimpleNamespace(get_data=failed)})
    with pytest.raises(RuntimeError, match='Sensor unavailable'):
        named_native_depth_frame(sensor)
