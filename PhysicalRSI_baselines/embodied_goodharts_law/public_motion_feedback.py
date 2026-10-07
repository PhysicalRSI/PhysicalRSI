"""Robot command and proprioception feedback, excluding scene/evaluator state."""
import numpy as np


def motion_feedback(trace_row, robot_state):
    """Whitelist the robot quantities needed to diagnose native tracking misses."""
    commanded = np.asarray(trace_row["solver_result"]["joints"], dtype=float)
    measured = np.asarray(robot_state["robot_joint_pos"][:7], dtype=float)
    target = np.asarray(trace_row["target"], dtype=float)
    actual = np.asarray(robot_state["robot_cartesian_pos"][:7], dtype=float)
    if any(x.shape != (7,) or not np.isfinite(x).all()
           for x in (commanded, measured, target, actual)):
        raise ValueError("Invalid robot motion feedback")
    return {"commanded_joint_positions": commanded.tolist(),
            "measured_joint_positions": measured.tolist(),
            "joint_error_rad": (commanded-measured).tolist(),
            "target_hand_pose": target.tolist(),
            "measured_hand_pose": actual.tolist(),
            "scope": "robot-commands-and-proprioception"}


class PublicMotionFeedbackMixin:
    def try_move_to_pose(self, pose):
        before = len(self._pose_trace)
        result = super().try_move_to_pose(pose)
        if len(self._pose_trace) > before:
            result = {**result, "motion_feedback": motion_feedback(
                self._pose_trace[-1], self.get_robot_state())}
        return result
