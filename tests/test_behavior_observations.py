import numpy as np
import pytest

from PhysicalRSI_baselines.behavior_baseline.observations import CAMERAS, PROPRIO, project, validate_action


def native_observation():
    raw = {"robot::proprio": np.arange(256, dtype=np.float32),
           "object_states": {"target_position": [100, 200, 300]}, "task_id": 123}
    for prefix in CAMERAS.values():
        raw[prefix + "::rgb"] = np.zeros((8, 8, 4), dtype=np.uint8)
        raw[prefix + "::depth_linear"] = np.ones((8, 8), dtype=np.float32)
        raw[prefix + "::seg_instance"] = np.full((8, 8), 99)
    return raw


def test_privileged_fields_cannot_cross_public_projection():
    raw = native_observation()
    result = project(raw, task="turn on the radio")
    assert set(result) == {"task", "proprio", "rgb"}
    assert set(result["proprio"]) == set(PROPRIO)
    assert sum(v.shape[-1] for v in result["proprio"].values()) == 25
    assert all(v.max() <= 56 for v in result["proprio"].values())
    assert all(v.shape == (1, 8, 8, 3) for v in result["rgb"].values())
    raw["robot::proprio"][:] = -1
    assert result["proprio"]["base_qvel"][0, 0] == 0


def test_changing_native_privileged_state_does_not_change_policy_inputs():
    a, b = native_observation(), native_observation()
    b["robot::proprio"][140:256] = -999
    b["object_states"] = {"success": True}
    b["task_id"] = 777
    x, y = project(a, task="task"), project(b, task="task")
    assert x["task"] == y["task"]
    for field in ("rgb", "proprio"):
        for key in x[field]:
            np.testing.assert_array_equal(x[field][key], y[field][key])


def test_depth_is_explicit_and_invalid_inputs_fail_closed():
    raw = native_observation()
    assert "depth" in project(raw, task="task", depth=True)
    raw["robot::proprio"][0] = np.nan
    with pytest.raises(ValueError, match="proprioception"):
        project(raw, task="task")
    with pytest.raises(ValueError):
        validate_action(np.full((1, 23), np.nan), batch_size=1)
    with pytest.raises(ValueError):
        validate_action(np.zeros((1, 24)), batch_size=1)
    assert validate_action(np.zeros(23), batch_size=1).shape == (1, 23)


def test_pinned_evaluator_robot_r1_names_map_without_changing_values():
    legacy = native_observation()
    # Exact default r1pro.yaml name and sensor namespace from the pinned evaluator.
    renamed = {key.replace("robot::robot:", "robot_r1::robot_r1:").replace(
        "robot::proprio", "robot_r1::proprio"): value for key, value in legacy.items()}
    expected = project(legacy, task="task", depth=True)
    actual = project(renamed, task="task", depth=True)
    assert set(actual) == {"task", "proprio", "rgb", "depth"}
    for group in ("proprio", "rgb", "depth"):
        for name in expected[group]:
            np.testing.assert_array_equal(actual[group][name], expected[group][name])
    renamed["robot::proprio"] = legacy["robot::proprio"]
    with pytest.raises(ValueError, match="exactly one"):
        project(renamed, task="task")
    del renamed["robot::proprio"]
    del renamed["robot_r1::robot_r1:zed_link:Camera:0::rgb"]
    renamed["robot::robot:zed_link:Camera:0::rgb"] = legacy["robot::robot:zed_link:Camera:0::rgb"]
    with pytest.raises(KeyError):
        project(renamed, task="task")
