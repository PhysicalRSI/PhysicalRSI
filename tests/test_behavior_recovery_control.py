"""Controller-contract tests only; no simulated task improvement claim."""
from copy import deepcopy

import numpy as np

from PhysicalRSI_baselines.behavior_baseline import recovery_control as control


def observation(tick, *, batch=1, moving_view=False, base_speed=0):
    arms = np.full((batch, 7), min(tick, 40) * .03, dtype=np.float32)
    left, right = np.zeros_like(arms), arms.copy()
    if batch == 2:
        left[1], right[1] = arms[1], 0
    rgb = np.full((batch, 16, 16, 3), 90, dtype=np.uint8)
    if moving_view:
        rgb[-1] = (tick * 3) % 255
    return dict(proprio=dict(left_arm_qpos=left, right_arm_qpos=right,
                            base_qvel=np.full((batch, 3), base_speed, dtype=np.float32)),
                rgb=dict(left_wrist=rgb.copy(), right_wrist=rgb.copy()))


def primitive(obs):
    action = np.zeros((len(obs['proprio']['base_qvel']), 23), dtype=np.float32)
    action[:, 3:7] = [.1, -.2, .3, -.4]
    action[:, 7:14] = obs['proprio']['left_arm_qpos']
    action[:, 15:22] = obs['proprio']['right_arm_qpos']
    action[:, [14, 22]] = [.6, -.7]
    return action


def test_recovery_is_observation_triggered_bounded_and_preserves_other_controls():
    state = control.initialize('public task')
    affected = 0
    calls = 0
    for tick in range(1200):
        obs = observation(tick, batch=2, moving_view=True)
        original = deepcopy(obs)
        baseline = primitive(obs)
        saved_baseline = baseline.copy()
        def infer(public):
            nonlocal calls
            calls += 1
            return baseline
        actual = control.step(obs, state, infer)
        assert actual.shape == (2, 23) and np.isfinite(actual).all()
        assert np.max(np.abs(actual - baseline)) <= .030001
        np.testing.assert_array_equal(actual[:, :7], baseline[:, :7])
        np.testing.assert_array_equal(actual[:, [14, 22]], baseline[:, [14, 22]])
        np.testing.assert_array_equal(actual[0, 7:14], baseline[0, 7:14])
        # The second member's wrist image keeps changing: no recovery there.
        np.testing.assert_array_equal(actual[1], baseline[1])
        np.testing.assert_array_equal(baseline, saved_baseline)
        for group in original:
            for name in original[group]:
                np.testing.assert_array_equal(obs[group][name], original[group][name])
        if not np.array_equal(actual[0], baseline[0]):
            affected += 1
            assert tick >= 191
    assert 0 < affected <= 3 * 96
    assert calls == 1200
    assert list(state['attempts']) == [3, 0]
    assert np.all(state['recovery'] == -1)
    fresh = control.initialize('public task')
    obs = observation(0, batch=2)
    np.testing.assert_array_equal(control.step(obs, fresh, primitive), primitive(obs))
    assert not fresh['attempts'].any()


def test_no_exploration_without_prior_arm_motion_or_while_base_moves():
    for stationary_arm in (True, False):
        state = control.initialize('task')
        for tick in range(400):
            obs = observation(0 if stationary_arm else tick,
                              base_speed=0 if stationary_arm else .1)
            np.testing.assert_array_equal(control.step(obs, state, primitive), primitive(obs))
        assert not state['attempts'].any()


def test_base_motion_aborts_an_active_recovery():
    state = control.initialize('task')
    for tick in range(200):
        obs = observation(tick)
        action = control.step(obs, state, primitive)
    assert state['recovery'][0] >= 0
    assert not np.array_equal(action, primitive(obs))
    obs = observation(200, base_speed=.1)
    np.testing.assert_array_equal(control.step(obs, state, primitive), primitive(obs))
    assert state['recovery'][0] == -1
    assert state['cooldown'][0] == 96
