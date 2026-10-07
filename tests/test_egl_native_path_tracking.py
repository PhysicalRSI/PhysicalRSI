import time
import numpy as np
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.native_path_tracking import track_native_path


class Api:
    _deadline = None
    _gripper_fraction = 1.
    _joint_limits = np.array([[-2., 2.]]*7)
    def __init__(self, *, budget=100, cartesian_error=0.):
        self.q = np.zeros(7)
        self.physics_steps = 0
        self.budget = budget
        self.cartesian_error = cartesian_error
    def get_robot_state(self):
        return {'robot_joint_pos': self.q.tolist(),
                'robot_cartesian_pos': [self.q[0]+self.cartesian_error, 0, 0, 1, 0, 0, 0]}
    def _step(self, action):
        if self.physics_steps >= self.budget:
            raise RuntimeError('Primitive physics-step budget exhausted')
        self.q += np.asarray(action[:7])*.02
        self.physics_steps += 1
    def _track_joint_target(self, q, *, tolerance, max_steps, goal):
        consecutive = 0
        for _ in range(max_steps):
            self._step(np.r_[np.clip((q-self.q)/.05, -1, 1), -1])
            consecutive = consecutive+1 if goal() else 0
            if consecutive >= 2:
                return
        raise RuntimeError('Joint primitive did not converge within its step allowance')


def run(api, **kwargs):
    return track_native_path(api, {'status': 'planned', 'trajectory': [[x, 0, 0, 0, 0, 0, 0] for x in [0, .03, .06, .1]]},
                             [.1, 0, 0, 1, 0, 0, 0], deadline=time.monotonic()+10, **kwargs)


def test_intermediate_proximity_keeps_strict_final_acceptance():
    api = Api()
    result = run(api)
    assert result['execution_verified'] and result['position_error_m'] <= .001
    assert result['tracking_steps'] == api.physics_steps and api._deadline is None


def test_joint_arrival_does_not_hide_cartesian_failure():
    api = Api(cartesian_error=.01)
    result = run(api, max_steps=25)
    assert not result['execution_verified'] and api.physics_steps <= 25


def test_native_budget_failure_propagates_and_restores_deadline():
    api = Api(budget=2)
    with pytest.raises(RuntimeError, match='physics-step budget exhausted'):
        run(api)
    assert api.physics_steps == 2 and api._deadline is None


def test_stale_start_is_rejected_without_motion():
    api = Api()
    api.q[0] = .1
    assert run(api)['status'] == 'stale_plan_start'
    assert api.physics_steps == 0


def test_out_of_limits_path_is_rejected_before_motion():
    api = Api()
    api._joint_limits = np.array([[-.05, .05]]*7)
    with pytest.raises(ValueError, match='joint limits'):
        run(api)
    assert api.physics_steps == 0
