import time
from types import SimpleNamespace
import pytest
import numpy as np
from test_egl_roboevolve_primitives import Robot
from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_pose_primitives import RoboEvolvePosePrimitives, bounded_joint_waypoint_indices


def make(plan, *, applier=None, budget=3):
    robot = Robot()
    robot._trans_from_gripper_to_endlink = lambda pose, arm: pose
    planner = SimpleNamespace(world_config=None, plan_path=lambda *a, **k: plan)
    return RoboEvolvePosePrimitives(SimpleNamespace(robot=robot), planners={'left': planner},
        joint_limits={'left': [[-1, 1], [-2, 2]]}, max_control_actions=budget,
        settle_attempts=1, action_applier=applier or (lambda *a, **k: True))


def move(api, pose=None):
    return api.handlers['move_to_pose'](['left', pose or [1, 2, 3, 1, 0, 0, 0]], {}, deadline=time.monotonic()+10)


def test_plan_does_not_prove_tracking_success():
    api = make({'status': 'Success', 'position': [[.1, .2]]})
    result = move(api, [1.1, 2, 3, 1, 0, 0, 0])
    assert result['status'] == 'tracking_failed' and not result['reached']
    assert result['control_attempts'] == 2 and not result['task_success_claimed']


def test_measured_endpoint_is_required_even_after_valid_plan():
    result = move(make({'status': 'Success', 'position': [[.1, .2]]}))
    assert result['reached'] and not result['scene_collision_checked']
    assert not result['task_success_claimed']


@pytest.mark.parametrize('positions', [[], [[9, 0]], [[float('nan'), 0]], [[.015*(i % 2), 0] for i in range(141)]])
def test_bad_plan_never_moves(positions):
    api = make({'status': 'Success', 'position': positions}, applier=lambda *a, **k: pytest.fail('unexpected motion'))
    with pytest.raises(ValueError):move(api)
    assert not api.trace


def test_control_budget_checked_before_first_motion():
    api = make({'status': 'Success', 'position': [[.1, .2]]}, budget=1)
    with pytest.raises(RuntimeError, match='budget'):move(api)
    assert not api.trace


@pytest.mark.parametrize('positions,reason', [([[.015*(i % 2), 0] for i in range(141)], 'waypoint_budget'),
                                            ([[9, 0]], 'joint_limits')])
def test_public_planning_diagnostics_distinguish_budget_and_joint_limits(positions, reason):
    api = make({'status': 'Success', 'position': positions})
    with pytest.raises(ValueError):move(api)
    trace = api.planning_trace
    assert trace[0]['rejection'] == reason
    assert trace[0]['control_attempts_before'] == 0
    trace[0]['rejection'] = 'modified'
    assert api.planning_trace[0]['rejection'] == reason


def test_dense_path_sampling_preserves_endpoints_and_bounds_accumulated_motion():
    values = np.column_stack([np.sin(np.linspace(0, np.pi*2, 488))*.2, np.zeros(488)])
    indices = bounded_joint_waypoint_indices(values)
    assert indices[0] == 0 and indices[-1] == len(values)-1
    assert 20 < len(indices) < 140  # Returning to the start must not erase the loop.
    for a, b in zip(indices, indices[1:]):
        assert np.max(np.abs(np.diff(values[a:b+1], axis=0)), axis=1).sum() <= .02+1e-12
    with pytest.raises(ValueError, match='increment'):
        bounded_joint_waypoint_indices([[0, 0], [.1, 0]])


def test_dense_valid_plan_fits_existing_execution_budget_after_sampling():
    api = make({'status': 'Success', 'position': np.column_stack([np.linspace(0, .2, 488), np.zeros(488)])}, budget=30)
    result = move(api)
    assert result['reached']
    assert api.planning_trace[0]['original_waypoints'] == 488
    assert len(api.trace) <= 30


def test_planning_failure_and_map_change_never_move():
    api = make({'status': 'Failure'})
    assert move(api)['status'] == 'planning_failed' and not api.trace
    api._planners['left'].world_config = {'objects': []}
    with pytest.raises(ValueError, match='scene'):move(api)
    assert not api.trace
