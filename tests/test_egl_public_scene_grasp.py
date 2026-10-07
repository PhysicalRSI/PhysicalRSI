import time
from types import SimpleNamespace

import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.public_scene_grasp import PublicSceneGraspPlanner


def fixture():
    calls = []
    def world(*args, **kwargs):
        calls.append(('world', args, kwargs))
        return SimpleNamespace(mesh=[SimpleNamespace(name=n) for n in ('observed_target', 'observed_scene')])
    def plan(*args, **kwargs):
        calls.append(('plan', args, kwargs))
        return True, np.zeros((3, 7)), 0
    api = SimpleNamespace(create_curobo_world_from_depth_with_object=world, plan_to_grasp_poses=plan)
    inputs = dict(depth=np.ones((4, 4)), object_mask=np.eye(4, dtype=bool),
                  intrinsics=np.diag([100., 100., 1.]), camera_to_base=np.eye(4),
                  robot_joints=np.zeros(7), grasp_poses=[[.5, 0, .2, 0, 1, 0, 0]],
                  pose_reference='hand', deadline=time.monotonic() + 30)
    return api, inputs, calls


@pytest.mark.parametrize('reference', ['hand', 'fingertip'])
def test_scene_collision_and_explicit_tool_reference(reference):
    api, inputs, calls = fixture()
    result = PublicSceneGraspPlanner(api)(**{**inputs, 'pose_reference': reference})
    assert result['status'] == 'planned' and not result['execution_verified']
    assert calls[0][2]['object_pose_override'] is None
    assert calls[1][2]['use_world_collision'] is True
    assert calls[1][2]['ignore_obstacle_names'] == ['observed_target']
    assert calls[1][2]['grasp_pose_is_fingertip'] == (reference == 'fingertip')
    assert calls[1][2]['position_threshold'] == .001
    assert calls[1][2]['rotation_threshold'] == .005
    assert calls[1][2]['position_threshold_z'] is None


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_tolerance_rejected(value):
    api, _, calls = fixture()
    with pytest.raises(ValueError):
        PublicSceneGraspPlanner(api, position_tolerance_m=value)
    assert not calls


def test_missing_scene_does_not_fall_back_to_robot_only_planning():
    api, inputs, calls = fixture()
    api.create_curobo_world_from_depth_with_object = lambda *a, **k: SimpleNamespace(mesh=[])
    assert PublicSceneGraspPlanner(api)(**inputs)['status'] == 'incomplete_observed_world'
    assert not calls


@pytest.mark.parametrize('enabled', [False, True])
def test_optional_approach_constraint_preserves_strict_thresholds(enabled):
    api, inputs, calls = fixture()
    result = PublicSceneGraspPlanner(api, use_grasp_approach=enabled)(**inputs)
    assert calls[1][2]['use_grasp_approach'] is enabled
    assert calls[1][2]['position_threshold'] == .001
    assert calls[1][2]['rotation_threshold'] == .005
    assert result['use_grasp_approach'] is enabled
    assert not result['execution_verified']


def test_fine_target_uses_public_transformed_points_and_keeps_scene():
    api, inputs, calls = fixture()
    rebuilt = []
    def mesh(points, **kwargs):
        rebuilt.append((points.copy(), kwargs))
        return SimpleNamespace(name=kwargs['name'])
    api.Mesh = SimpleNamespace(from_pointcloud=mesh)
    api.WorldConfig = SimpleNamespace
    inputs['camera_to_base'][:3, 3] = [2., 3., 4.]
    result = PublicSceneGraspPlanner(api, allow_target_contact=False,
        scene_mesh_pitch_m=.01, target_mesh_pitch_m=.003)(**inputs)
    assert np.allclose(rebuilt[0][0], [[2+i/100, 3+i/100, 5] for i in range(4)])
    assert rebuilt[0][1]['pitch'] == .003
    assert calls[0][2]['marching_cubes_pitch'] == .01
    assert [m.name for m in calls[1][1][0].mesh] == ['observed_scene', 'observed_target']
    assert calls[1][2]['ignore_obstacle_names'] == []
    assert result['allow_target_contact'] is False
    assert not result['execution_verified']


@pytest.mark.parametrize('pitch', [0., -.01, .2, float('nan'), float('inf')])
def test_invalid_mesh_resolution_rejected(pitch):
    api, _, calls = fixture()
    with pytest.raises(ValueError, match='mesh pitch'):
        PublicSceneGraspPlanner(api, target_mesh_pitch_m=pitch)
    assert not calls


@pytest.mark.parametrize('update', [
    {'camera_to_base': np.diag([1., 1., -1., 1.])},
    {'object_mask': np.ones((4, 4), dtype=bool)},
    {'grasp_poses': [[0, 0, 0, 2, 0, 0, 0]]},
    {'pose_reference': 'unspecified'},
])
def test_invalid_geometry_rejected_before_backend(update):
    api, inputs, calls = fixture()
    with pytest.raises(ValueError):
        PublicSceneGraspPlanner(api)(**{**inputs, **update})
    assert not calls


def test_false_planner_success_cannot_admit_nonfinite_path():
    api, inputs, _ = fixture()
    api.plan_to_grasp_poses = lambda *a, **k: (True, np.full((2, 7), np.nan), 0)
    with pytest.raises(ValueError):
        PublicSceneGraspPlanner(api)(**inputs)
