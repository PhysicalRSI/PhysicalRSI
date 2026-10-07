import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.bounded_joint_return import BoundedJointReturnMixin

class Tracker:
    _steps = 120
    def move_to_joints_blocking(self, joints, *, tolerance, max_steps):
        assert tolerance == .01 and max_steps == 120
        raise self.error
    def get_robot_state(self):return {'robot_joint_pos':[.02]*7+[1.]}

class Robot(BoundedJointReturnMixin, Tracker):pass

def test_native_step_exhaustion_returns_failure_with_measured_state():
    robot=Robot();robot.error=RuntimeError('Joint primitive did not converge within its step allowance')
    reply=robot.move_to_joints_blocking([0.]*7)
    assert not reply['converged'] and not reply['joint_converged']
    assert reply['joint_error']==pytest.approx(.02*7**.5)
    assert reply['physics_steps']==120

@pytest.mark.parametrize('error',[RuntimeError('physics crashed'),ValueError('Invalid primitive convergence limits')])
def test_unexpected_errors_still_propagate(error):
    robot=Robot();robot.error=error
    with pytest.raises(type(error),match=str(error)):
        robot.move_to_joints_blocking([0.]*7)
