"""Low-level execution contract; these tests do not claim native success."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "native_execution", Path(__file__).parents[1] / "PhysicalRSI_baselines/robodyna/execution.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
InterceptionExecutor = module.InterceptionExecutor


class Actor:
    def __init__(self, position):
        self.position = list(position)

    def get_pose(self):
        return SimpleNamespace(p=self.position, q=[1, 0, 0, 0])


class Environment:
    plan_success = True
    table_top = 0.74
    CUP_CENTER_Z = 0.043
    _push_active = True
    _box_half_xy = [0.06, 0.06]
    _box_height = 0.06
    PUSH_BEHIND_STANDOFF = 0.055
    PUSH_FINGER_HEIGHT_FRAC = 0.4
    PUSH_CONTACT_GAP = 0.012

    def __init__(self):
        self.cup = Actor([0.1, -0.2, 0.783])
        self.bowl = Actor([0.15, -0.2, 0.77])
        self.actions = []
        self.robot = SimpleNamespace(
            get_left_tcp_pose=lambda: [-0.2, -0.3, 1.0, 1, 0, 0, 0],
            get_right_tcp_pose=lambda: [0.2, -0.3, 1.0, 1, 0, 0, 0])

    def move(self, action):
        self.actions.append(action)

    def grasp_actor(self, actor, **kwargs):
        return ("grasp", kwargs)

    def move_by_displacement(self, **kwargs):
        return ("displace", kwargs)

    def open_gripper(self, arm):
        return ("open", arm)

    def close_gripper(self, arm):
        return ("close", arm)

    def move_to_pose(self, arm, pose):
        return ("pose", pose)

    def _finger_drop(self, arm):
        return 0.043


def test_ramp_single_placement_does_not_claim_grasp_success():
    env = Environment()
    env.cup.position[0] = -0.01
    executor = InterceptionExecutor(env, "catch_ramp_ball")
    target = [0.2, -0.05, 0.82]
    result = executor.execute_target(target)
    assert result["phase"] == "lift"
    assert result["arm"] == "right"  # Supplied target, not cup spawn side.
    assert result["plan_success"]
    assert result["grasp_verified"] is None
    assert result["motion_before"]["catcher_position_m"] == [-0.01, -0.2, 0.783]
    assert result["motion_after"]["tcp_pose_world"] == [0.2, -0.3, 1., 1., 0., 0., 0.]
    assert result["catcher_displacement_m"] == [0., 0., 0.]
    assert executor.execute_target(target)["phase"] == "place"
    assert executor.execute_target(target)["phase"] == "release"
    delta = env.actions[-1][1]
    assert delta["x"] == pytest.approx(0.21)
    assert delta["y"] == pytest.approx(0.15)
    # Fixture cup did not follow the gripper. No repeated correction or false
    # attachment claim is made even though every motion plan was successful.
    assert executor.execute_target(target)["phase"] == "retreat"
    assert executor.execute_target(target)["phase"] == "done"
    action_count = len(env.actions)
    executor.execute_target([-0.2, -0.1, 0.82])
    assert len(env.actions) == action_count
    assert executor.arm == "right"


def test_failed_plan_is_terminal_without_resetting_native_latch():
    env = Environment()
    executor = InterceptionExecutor(env, "catch_ramp_ball")
    env.plan_success = False
    assert executor.execute_target([0, 0, 1])["phase"] == "failed"
    assert not env.actions
    assert not env.plan_success


def test_valley_requires_official_dynamic_setup():
    env = Environment()
    env._push_active = False
    with pytest.raises(RuntimeError, match="native interactive"):
        InterceptionExecutor(env, "catch_valley_ball")


def test_valley_one_contact_phase_each_call(monkeypatch):
    monkeypatch.setitem(sys.modules, "envs._GLOBAL_CONFIGS", SimpleNamespace(
        GRASP_DIRECTION_DIC={"top_down": [-0.5, 0.5, -0.5, -0.5]}))
    env = Environment()
    executor = InterceptionExecutor(env, "catch_valley_ball")
    target = [0.3, -0.1, 0.9]
    for phase in ("approach", "lower", "push", "push"):
        before = len(env.actions)
        assert executor.execute_target(target)["phase"] == phase
        assert len(env.actions) == before + 1
    assert env.bowl.position == [0.15, -0.2, 0.77]
    assert env._push_active


@pytest.mark.parametrize("target", ([float("nan"), 0, 1], [0, 1], [0, 1, float("inf")]))
def test_invalid_targets_fail_before_motion(target):
    env = Environment()
    executor = InterceptionExecutor(env, "catch_ramp_ball")
    with pytest.raises(ValueError):
        executor.execute_target(target)
    assert not env.actions
