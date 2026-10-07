import time
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.native_endpoint_settle import settle_native_endpoint


class Api:
    _deadline = None
    physics_steps = 0
    def __init__(self, position):
        self.position = position
    def get_robot_state(self):
        return {'robot_cartesian_pos': [self.position, 0, .2, 0, 1, 0, 0]}
    def _track_joint_target(self, joints, *, tolerance, max_steps, goal):
        assert max_steps == 120 and self._deadline is not None
        self.physics_steps += 2
        if not goal():
            raise RuntimeError('Joint primitive did not converge within its step allowance')


@pytest.mark.parametrize('position,passed', [(.5, True), (.502, False)])
def test_settling_uses_measured_cartesian_acceptance_and_restores_deadline(position, passed):
    api = Api(position)
    result = settle_native_endpoint(api, [0]*7, [.5, 0, .2, 0, 1, 0, 0], deadline=time.monotonic()+5)
    assert result['execution_verified'] is passed
    assert result['settling_steps'] == 2 and api._deadline is None


def test_expired_settling_never_moves():
    api = Api(.5)
    with pytest.raises(TimeoutError):
        settle_native_endpoint(api, [0]*7, [.5, 0, .2, 0, 1, 0, 0], deadline=time.monotonic()-1)
    assert api.physics_steps == 0
