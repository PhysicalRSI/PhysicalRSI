"""Represent bounded joint tracking failure without aborting a CAP campaign."""


class BoundedJointReturnMixin:
    def move_to_joints_blocking(self, joints, *, tolerance=0.01, max_steps=120):
        try:
            return super().move_to_joints_blocking(
                joints, tolerance=tolerance, max_steps=max_steps)
        except RuntimeError as exc:
            if str(exc) != 'Joint primitive did not converge within its step allowance':
                raise
            state = self.get_robot_state()
            measured = state['robot_joint_pos'][:7]
            error = sum((measured[i]-joints[i])**2 for i in range(7))**.5
            return {'converged': False, 'joint_converged': False,
                    'joint_error': error, 'reason': str(exc),
                    'robot_state': state, 'physics_steps': self._steps}
