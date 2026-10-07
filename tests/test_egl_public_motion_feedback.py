import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.public_motion_feedback import (
    PublicMotionFeedbackMixin, motion_feedback)


def values():
    return ({"solver_result": {"joints": [1.]*7, "private": "withheld"},
             "target": [0, 0, 1, 1, 0, 0, 0], "object_poses": "withheld"},
            {"robot_joint_pos": [.5]*7+[1.],
             "robot_cartesian_pos": [0, 0, .8, 1, 0, 0, 0, 1],
             "actor_segmentation": "withheld"})


def test_only_robot_command_and_proprioception_fields_are_exported():
    result = motion_feedback(*values())
    assert result["joint_error_rad"] == [.5]*7
    assert set(result) == {"commanded_joint_positions", "measured_joint_positions",
                           "joint_error_rad", "target_hand_pose", "measured_hand_pose", "scope"}


def test_nonfinite_command_rejected():
    row, state = values()
    row["solver_result"]["joints"][0] = float("nan")
    with pytest.raises(ValueError):
        motion_feedback(row, state)


def test_solver_miss_cannot_reuse_previous_motion_trace():
    class Base:
        def try_move_to_pose(self, pose):
            return {"reached": False, "native_effects": False}
        def get_robot_state(self):
            raise AssertionError("No new native motion; no stale feedback")
    class API(PublicMotionFeedbackMixin, Base):
        _pose_trace = [values()[0]]
    assert API().try_move_to_pose([0]*7) == {"reached": False, "native_effects": False}


def test_native_tracking_miss_preserves_failure_with_feedback():
    class Base:
        def try_move_to_pose(self, pose):
            self._pose_trace.append(values()[0])
            return {"reached": False, "reason": "tracking miss"}
        def get_robot_state(self):
            return values()[1]
    class API(PublicMotionFeedbackMixin, Base):
        def __init__(self): self._pose_trace = []
    result = API().try_move_to_pose([0]*7)
    assert not result["reached"]
    assert result["reason"] == "tracking miss"
    assert result["motion_feedback"]["joint_error_rad"] == [.5]*7
