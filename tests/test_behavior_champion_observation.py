"""Prevent silent state corruption when transferring the 2025 parent."""
import numpy as np
import pytest

from PhysicalRSI_baselines.behavior_baseline.observations import CAMERAS, PROPRIO, project
from PhysicalRSI_baselines.behavior_baseline.champion_observation import champion_observation


def observation():
    raw = {"robot::proprio": np.arange(256, dtype=np.float32)}
    raw.update({prefix + "::rgb": np.full((8, 8, 3), i, dtype=np.uint8)
                for i, prefix in enumerate(CAMERAS.values())})
    return raw


def test_transfers_joint_semantics_and_excludes_privileged_coordinates():
    public = project(observation(), task="turning_on_radio")
    legacy = champion_observation(public)
    state = legacy["robot_r1::proprio"]
    # Expected offsets independently transcribed from the pinned 2025 evaluator.
    expected = [(253, 256, "base_qvel"), (236, 240, "trunk_qpos"),
                (158, 165, "left_arm_qpos"), (193, 195, "left_gripper_qpos"),
                (197, 204, "right_arm_qpos"), (232, 234, "right_gripper_qpos")]
    consumed = np.zeros(256, dtype=bool)
    for start, stop, name in expected:
        np.testing.assert_array_equal(state[start:stop], np.array(PROPRIO[name]))
        consumed[start:stop] = True
    assert not state[~consumed].any()
    assert set(legacy) == {"robot_r1::proprio", *(
        p.replace("robot::robot:", "robot_r1::robot_r1:") + "::rgb" for p in CAMERAS.values())}
    state[253] = -999
    assert public["proprio"]["base_qvel"][0, 0] == 0


def test_rejects_batch_instead_of_mixing_episode_state():
    public = project(observation(), task="turning_on_radio")
    public["proprio"]["base_qvel"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="one finite"):
        champion_observation(public)
