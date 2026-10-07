import time
import numpy as np
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.libero_primitives import NativeLiberoPrimitives


def fixture_api(solver):
    api = NativeLiberoPrimitives.__new__(NativeLiberoPrimitives)
    api._pose_solver = solver
    api._segmenter = None
    api._pose_trace = []
    api._deadline = None
    api._remaining = 10
    api._steps = 0
    api.home_joint_position = np.zeros(7)
    api.get_observation = lambda: {'robot_joint_pos': np.r_[np.zeros(7), .5],
        'robot_cartesian_pos': np.array([0., 0., 0., 1., 0., 0., 0., .5])}
    return api


def test_pose_capability_is_explicit_and_invalid_target_never_solves():
    calls = []
    api = fixture_api(lambda **kw: calls.append(kw))
    assert 'move_to_pose' not in fixture_api(None).handlers
    for target in [[0]*6, [0, 0, float('nan'), 1, 0, 0, 0], [0]*7]:
        with pytest.raises(ValueError):
            api.handlers['move_to_pose']([target], {}, deadline=time.monotonic()+10)
    assert calls == [] and api._deadline is None


def test_solver_overrun_rejected_before_motion():
    def solve(**kw):
        api._deadline = time.monotonic()-1
        return {'joints': [0]*7}
    api = fixture_api(solve)
    motions = []
    api._track_joint_target = lambda *a, **kw: motions.append(a)
    with pytest.raises(TimeoutError):
        api.handlers['move_to_pose']([[0, 0, 0, 1, 0, 0, 0]], {}, deadline=time.monotonic()+10)
    assert motions == [] and api._deadline is None


def test_joint_success_cannot_mask_cartesian_failure():
    api = fixture_api(lambda **kw: {'joints': [0]*7})
    api._track_joint_target = lambda *a, **kw: {'converged': True, 'joint_converged': True}
    with pytest.raises(RuntimeError, match='Cartesian tolerance'):
        api.handlers['move_to_pose']([[0, 0, .01, 1, 0, 0, 0]], {}, deadline=time.monotonic()+10)
    assert api.pose_trace[0]['state'] == 'failed'
    assert api.pose_trace[0]['position_error_m'] == .01
    trace = api.pose_trace
    trace[0]['target'][0] = 99
    assert api.pose_trace[0]['target'][0] == 0


def test_solver_preserves_virtual_environment_python_path(tmp_path, monkeypatch):
    import hashlib
    import sys
    from PhysicalRSI_baselines.embodied_goodharts_law import aspire_ik
    python = tmp_path / 'venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    model = tmp_path / 'fixture.urdf'
    model.write_bytes(b'test-only model')
    monkeypatch.setattr(aspire_ik, 'PANDA_SHA256', hashlib.sha256(model.read_bytes()).hexdigest())
    solver = aspire_ik.AspireHandIK(interpreter=python, urdf=model,
        native_hand_translation=[0, 0, .1065], native_hand_quaternion_wxyz=[1, 0, 0, 0])
    assert solver.interpreter == str(python)
    assert solver.interpreter != str(python.resolve())


def test_exhausted_joint_criterion_requires_measured_cartesian_success():
    api = fixture_api(lambda **kw: {'joints': [0]*7})
    def exhausted(*args, **kwargs):
        raise RuntimeError('Joint primitive did not converge within its step allowance')
    api._track_joint_target = exhausted
    result = api.handlers['move_to_pose']([[0, 0, 0, 1, 0, 0, 0]], {}, deadline=time.monotonic()+10)
    assert result['converged'] and not result['joint_converged']
    assert api.pose_trace[-1]['state'] == 'completed'
    with pytest.raises(RuntimeError, match='Cartesian tolerance'):
        api.handlers['move_to_pose']([[0, 0, .01, 1, 0, 0, 0]], {}, deadline=time.monotonic()+10)
    assert api.pose_trace[-1]['state'] == 'failed'


def test_try_pose_exposes_only_recoverable_reach_failure():
    api = fixture_api(lambda **kw: {'joints': [0]*7})
    api._track_joint_target = lambda *a, **kw: {'converged': True, 'joint_converged': True}
    result = api.handlers['try_move_to_pose']([[0, 0, .01, 1, 0, 0, 0]], {}, deadline=time.monotonic()+10)
    assert not result['reached'] and api.pose_trace[-1]['state'] == 'failed'
    def exhausted(*args, **kwargs):
        raise RuntimeError('Primitive physics-step budget exhausted')
    api._track_joint_target = exhausted
    with pytest.raises(RuntimeError, match='budget exhausted'):
        api.handlers['try_move_to_pose']([[0, 0, 0, 1, 0, 0, 0]], {}, deadline=time.monotonic()+10)


def test_ik_seed_projects_finite_measurements_without_an_arbitrary_excursion_cutoff():
    from PhysicalRSI_baselines.embodied_goodharts_law.aspire_ik import gripper_seed_fraction
    assert gripper_seed_fraction(1.0000660947096738) == 1.
    assert gripper_seed_fraction(1.0038978707491508) == 1.
    assert gripper_seed_fraction(1.00113888024653) == 1.
    assert gripper_seed_fraction(1.0102256484363124) == 1.
    assert gripper_seed_fraction(2.) == 1.
    assert gripper_seed_fraction(-.5) == 0.
    assert gripper_seed_fraction(-.0001) == 0.
    assert gripper_seed_fraction(.8) == .8
    for invalid in [float('nan'), float('inf'), -float('inf')]:
        with pytest.raises(ValueError, match='measured gripper'):
            gripper_seed_fraction(invalid)


def test_cartesian_tracking_requires_consecutive_goal_samples_and_stops_early():
    api = fixture_api(None)
    api._joint_limits = np.array([[-2., 2.]] * 7)
    api._observation = {'robot0_joint_pos': np.zeros(7)}
    api._gripper_fraction = 1.
    def step(action):
        assert len(action) == 8
        api._steps += 1
    api._step = step
    samples = iter([True, False, True, True])
    result = api._track_joint_target(np.ones(7), goal=lambda: next(samples))
    assert result['physics_steps'] == 4
    assert result['converged'] and not result['joint_converged']


def test_joint_convergence_does_not_end_unreached_cartesian_motion():
    api = fixture_api(None)
    api._joint_limits = np.array([[-2., 2.]] * 7)
    api._observation = {'robot0_joint_pos': np.zeros(7)}
    api._gripper_fraction = 1.
    api._step = lambda action: None
    with pytest.raises(RuntimeError, match='step allowance'):
        api._track_joint_target(np.zeros(7), max_steps=3, goal=lambda: False)
