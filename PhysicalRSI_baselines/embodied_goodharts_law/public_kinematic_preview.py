"""Bounded robot-only IK preview, without advancing simulator state."""
import math
import time

import numpy as np

from .persistent_aspire_ik import IKNonConvergence


class PublicKinematicPreviewMixin:
    @property
    def handlers(self):
        handlers = super().handlers

        def preview(args, kwargs, *, deadline):
            if kwargs or len(args) != 1:
                raise ValueError("Expected one pose sequence")
            if not math.isfinite(deadline):
                raise ValueError("A finite IK preview deadline is required")
            poses = np.asarray(args[0], dtype=float)
            if (poses.ndim != 2 or poses.shape[1] != 7 or not 1 <= len(poses) <= 8
                    or not np.isfinite(poses).all()
                    or np.any(np.abs(np.linalg.norm(poses[:, 3:], axis=1)-1) > 1e-5)):
                raise ValueError("Expected one to eight finite xyz+wxyz unit-quaternion poses")
            if time.monotonic() >= deadline:
                raise TimeoutError("IK preview deadline reached")
            state = self.get_robot_state()
            joints = np.asarray(state['robot_joint_pos'][:7], dtype=float)
            gripper = state['robot_joint_pos'][7]
            solved = 0
            for pose in poses:
                if time.monotonic() >= deadline:
                    raise TimeoutError("IK preview deadline reached")
                try:
                    solution = self._pose_solver(pose=pose, joints=joints,
                        gripper_fraction=gripper, deadline=deadline)
                except IKNonConvergence:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("IK preview deadline reached")
                    return self._preview_result(False, solved, 'solver_nonconvergence')
                if time.monotonic() >= deadline:
                    raise TimeoutError("IK preview deadline reached")
                joints = np.asarray(solution['joints'], dtype=float)
                if joints.shape != (7,) or not np.isfinite(joints).all():
                    raise ValueError("Invalid IK preview solution")
                if np.any(joints < self._joint_limits[:, 0]) or np.any(joints > self._joint_limits[:, 1]):
                    return self._preview_result(False, solved, 'joint_limits')
                solved += 1
            return self._preview_result(True, solved, 'all_waypoints_solved')

        handlers['preview_poses'] = preview
        return handlers

    @staticmethod
    def _preview_result(solved, count, reason):
        return {'kinematically_solved': solved, 'solved_waypoints': count,
                'reason': reason, 'native_effects': False,
                'collision_checked': False, 'execution_verified': False,
                'scope': 'robot-model-and-proprioception'}
