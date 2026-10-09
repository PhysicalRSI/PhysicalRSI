"""Exercise the real CodexPolicy/Core bridge with a software-only environment."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment
from PhysicalRSI_core.experiments import ExperimentRuntime, Budget
from hybrid_rollout.robodojo.skill.run import CodexPolicy

from PhysicalRSI_demos.dexterous_manipulation.common import file_hash, save
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_sim import DexJoCoProfile, validate_response
from PhysicalRSI_demos.dexterous_manipulation.evidence import verify_media
from PhysicalRSI_demos.dexterous_manipulation.policy import GalbotPolicy


class FixtureEnvironment:
    """Synthetic protocol observations; no robot or native simulator is used."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.moves = []
        self.stopped = False
        self.latest = None

    def identity(self):
        return dict(name="agent-bridge-software-fixture", revision="1")

    def describe(self):
        contract = Contract("agent-bridge-fixture/v1", unit="software", embodiment="software")
        return Embodiment("software-fixture", "1", "software", contract, contract)

    def observe(self):
        tick = len(self.moves)
        folder = self.root / "observations" / str(tick)
        folder.mkdir(parents=True)
        images = []
        for camera in ("base", "wrist_left", "wrist_right"):
            path = folder / (camera + ".png")
            Image.new("RGB", (32, 32), (tick, 0, 0)).save(path)
            images.append(dict(camera=camera, path=str(path), sha256=file_hash(path)))
        state = []
        for side in ("right", "left"):
            state.extend(self.target[side]["position"] + self.target[side]["quaternion_wxyz"])
        for side in ("right", "left"):
            state.extend(self.target[side]["hand_joints_rad"])
        self.latest = dict(request_id=str(tick), observation_path=str(folder / "observation.json"),
            images=images, state=state, current_eef=copy.deepcopy(self.target),
            hand_limits_rad={side: [[-1., 1.]] * 16 for side in self.target},
            limits=dict(max_steps=30, max_translation_m=.02, max_rotation_rad=.2),
            rollout_finished=tick >= 2)
        save(self.latest["observation_path"], self.latest)
        return self.latest

    def reset(self, case, context):
        self.target = {side: dict(position=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.],
                                 hand_joints_rad=[0.] * 16) for side in ("right", "left")}
        return dict(ready=True, observation=self.observe())

    def step(self, action, context):
        validated = validate_response(action, self.latest, self.latest["limits"])
        self.moves.append(validated)
        self.target = copy.deepcopy(validated["target"])
        observation = self.observe()
        return dict(observation=observation, terminated=observation["rollout_finished"])

    def stop_only(self):
        self.stopped = True
        return dict(quiescent=True, software_fixture=True)


class FixtureVerifier:
    def identity(self):
        return dict(name="software-protocol-verifier", revision="1")

    def verify(self, case, trace):
        return dict(outcome="uncertain", reason="Synthetic protocol observations cannot establish task success",
                    measurements=verify_media(trace))


def action_for(observation):
    target = copy.deepcopy(observation["current_eef"])
    target["right"]["position"][0] += .005
    return dict(mode="eef", steps=1, target=target, request_id=observation["request_id"],
                reason="Exercise bounded action feedback in a software fixture")


def run_core(root, env, policy):
    return ExperimentRuntime(root).run("episode", task="Move the test object", case={},
        scope="protocol-test", environment=env, policy=policy, verifier=FixtureVerifier(),
        budget=Budget(max_steps=4, seconds=60))


def test_core_writes_valid_receipt_and_preserves_media(tmp_path):
    env = FixtureEnvironment(tmp_path / "evidence")
    def worker(tools):
        tools.start(task="Move the test object", output_dir="unused")
        while tools.phase != "done":
            tools.act(observation_path=tools.observation_path, response=action_for(tools.request), output_dir="unused")
    policy = GalbotPolicy(env, DexJoCoProfile(), tmp_path / "galbot", worker_factory=worker)
    receipt = run_core(tmp_path / "core", env, policy)
    assert receipt["state"] == "completed" and receipt["outcome"] == "uncertain"
    assert receipt["steps"] == 2
    assert ExperimentRuntime(tmp_path / "core").read("episode") == receipt
    trace = json.loads((tmp_path / "core/episode/trajectory.json").read_text())
    assert verify_media(trace)["verified_images"] == 9
    assert env.stopped and not policy.thread.is_alive()
    Path(trace[0]["observation"]["images"][0]["path"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="RGB evidence"):
        verify_media(trace)


class FakeAppServer:
    """The upstream agent consumes app-server events and actual PNG tool replies."""
    def __init__(self, argv, workspace):
        self.process = SimpleNamespace(pid=12345)
        self.calls, self.replies = [], []
        self.index, self.last = 0, None
        self.closed = False

    def request(self, method, params, timeout):
        self.calls.append((method, params))
        if method == "initialize": return {}
        if method == "thread/start":
            assert {s["name"] for s in params["dynamicTools"]} == {"dexjoco_start", "dexjoco_act"}
            assert "Allegro" in params["developerInstructions"]
            return dict(thread=dict(id="thread"), model="gpt-6-astra", reasoningEffort="xhigh")
        if method == "turn/start": return dict(turn=dict(id="turn"))
        raise AssertionError(method)

    def notify(self, method, params): pass
    def close(self): self.closed = True

    def reply(self, identifier, result):
        self.replies.append(result)
        packet = json.loads(result["contentItems"][0]["text"])
        if result["success"]:
            self.last = packet
            assert len([x for x in result["contentItems"] if x["type"] == "inputImage"]) == 3
            assert len(packet["state"]) == 46
        else:
            assert packet["no_execution"] is True

    def next_message(self, timeout):
        i = self.index
        self.index += 1
        if i == 0:
            name, arguments = "dexjoco_start", dict(task="Move the test object", output_dir="unused")
        elif i in {1, 2, 3}:
            response = action_for(self.last)
            if i == 1: response["target"] = [0.] * 14
            name, arguments = "dexjoco_act", dict(observation_path=self.last["observation_path"],
                                                  response=response, output_dir="unused")
        else:
            return dict(method="turn/completed", params=dict(threadId="thread", turn=dict(id="turn", status="completed")))
        return dict(id=i+1, method="item/tool/call", params=dict(threadId="thread", turnId="turn", tool=name,
                    arguments=arguments, callId=str(i)))


def test_actual_galbot_loop_with_core_and_new_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("hybrid_rollout.robodojo.skill.run.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(stdout="codex-test-double"))
    env = FixtureEnvironment(tmp_path / "evidence")
    profile = DexJoCoProfile()
    controllers = []
    def worker(tools):
        controller = CodexPolicy(tmp_path / "evidence/galbot", "unused", runtime=profile,
                                 method="gpt_only", transport_factory=FakeAppServer)
        controllers.append(controller)
        try:
            controller.run(tools)
        finally:
            controller.close()
    policy = GalbotPolicy(env, profile, tmp_path / "evidence/galbot", worker_factory=worker)
    receipt = run_core(tmp_path / "core", env, policy)
    assert receipt["outcome"] == "uncertain"
    assert len(env.moves) == 2  # The invalid request caused no extra environment step.
    transport = controllers[0].transport
    assert [r["success"] for r in transport.replies] == [True, False, True, True]
    assert sum(method == "thread/start" for method, _ in transport.calls) == 1
    assert env.stopped and not policy.thread.is_alive() and transport.closed


def test_policy_error_stops_and_retains_reconciliation(tmp_path):
    env = FixtureEnvironment(tmp_path / "evidence")
    def broken(tools):
        tools.start(task="Move the test object", output_dir="unused")
        tools.act(observation_path=tools.observation_path, response=action_for(tools.request), output_dir="unused")
        raise RuntimeError("Model disconnected")
    policy = GalbotPolicy(env, DexJoCoProfile(), tmp_path / "galbot", worker_factory=broken)
    with pytest.raises(RuntimeError, match="policy failed"):
        run_core(tmp_path / "core", env, policy)
    receipt = json.loads((tmp_path / "core/episode/receipt.json").read_text())
    assert receipt["state"] == "needs_reconciliation"
    assert env.stopped and not policy.thread.is_alive()
    assert len(env.moves) == 1


def test_image_attachment_setting_is_part_of_frozen_policy_identity(monkeypatch):
    profile = DexJoCoProfile()
    monkeypatch.setenv("CODEX_IMAGE_MAX_EDGE", "480")
    original = profile.identity()
    monkeypatch.setenv("CODEX_IMAGE_MAX_EDGE", "640")
    assert profile.identity() != original
    assert profile.identity()["image_max_edge"] == 640
