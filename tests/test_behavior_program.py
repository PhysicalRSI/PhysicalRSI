import numpy as np
import pytest

from PhysicalRSI_baselines.behavior_baseline.program import ProgramPolicy
from PhysicalRSI_baselines.behavior_baseline.observations import CAMERAS
from PhysicalRSI_core.contracts import Context
from PhysicalRSI_core.embodiment import Embodiment
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_baselines.behavior_baseline.program import OBSERVATION, ACTION
from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI_core.self_harness.artifacts import CLOSURE
from PhysicalRSI_baselines.behavior_baseline.engine import manifest as engine_manifest


class Backend:
    def identity(self):
        return {"kind": "unit-test-backend", "revision": "1"}

    def reset(self, task):
        self.task = task

    def act(self, observation, context):
        assert set(observation) == {"task", "rgb", "proprio"}
        assert observation["task"] == self.task
        return np.zeros((1, 23), dtype=np.float32)


def make_policy(tmp_path, control_source=None):
    # Complete synthetic closure exercises core binding, not benchmark admission.
    atomic_json(tmp_path / "test-dependencies.json", {"scope": "unit test only"})
    control = tmp_path / "control.py"
    control.write_text(control_source or 'def initialize(task):\n    return {}\ndef step(obs, state, primitive):\n    return primitive(obs)\n')
    candidate = dict(id="test-policy", root=str(tmp_path), components={
        kind: {"test-dependencies.json": file_digest(tmp_path / "test-dependencies.json")} for kind in CLOSURE})
    candidate["components"]["control"] = {"control.py": file_digest(control)}
    atomic_json(tmp_path / "backend.json", Backend().identity())
    candidate["components"]["foundation"] = {"backend.json": file_digest(tmp_path / "backend.json")}
    atomic_json(tmp_path / "engine.json", engine_manifest())
    candidate["components"]["dependencies"]["engine.json"] = file_digest(tmp_path / "engine.json")
    return ProgramPolicy(candidate, Backend())


def test_development_trace_preserves_actions_and_pre_control_observation(tmp_path):
    import json
    from PhysicalRSI_baselines.behavior_baseline.diagnostics import DevelopmentTrace

    source = ('def initialize(task):\n    return {}\n'
              'def step(obs, state, primitive):\n'
              '    obs["proprio"]["base_qvel"][:] = 7\n'
              '    action = primitive(obs)\n'
              '    action[:] += 0.125\n'
              '    return action\n')
    base = make_policy(tmp_path, source)
    path = tmp_path / "development.jsonl"
    recorder = DevelopmentTrace(path, split="development", candidate_sha256=base.revision,
                                task="radio", case_id="development/radio/1")
    traced = ProgramPolicy(base.candidate, Backend(), diagnostics=recorder)
    raw = {"robot::proprio": np.zeros(256), "object_ground_truth": "excluded"}
    raw.update({prefix + "::rgb": np.zeros((8, 8, 3), dtype=np.uint8) for prefix in CAMERAS.values()})
    ctx = Context("test", harness_revision=base.revision)
    for policy in (base, traced):
        policy.begin_episode("radio", {}, raw, ctx)
    np.testing.assert_array_equal(base.act(raw, ctx), traced.act(raw, ctx))
    receipt = traced.end_episode(ctx)["diagnostics"]
    assert receipt["recorded_steps"] == 1
    row = json.loads(path.read_text().splitlines()[1])
    np.testing.assert_array_equal(row["proprio"]["base_qvel"], np.zeros((1, 3)))
    np.testing.assert_array_equal(row["primitive_actions"][0], np.zeros((1, 23)))
    np.testing.assert_array_equal(row["issued_action"], np.full((1, 23), 0.125))
    assert traced.identity()["diagnostics"]["case_id"] == "development/radio/1"
    with pytest.raises(ValueError, match="exactly one"):
        traced.begin_episode("radio", {}, raw, ctx)


def test_diagnostics_reject_wrong_candidate_and_task(tmp_path):
    from PhysicalRSI_baselines.behavior_baseline.diagnostics import DevelopmentTrace
    base = make_policy(tmp_path)
    with DevelopmentTrace(tmp_path / "wrong.jsonl", split="development",
                          candidate_sha256="a" * 64, task="radio", case_id="case") as recorder:
        with pytest.raises(ValueError, match="attribution"):
            ProgramPolicy(base.candidate, Backend(), diagnostics=recorder)
    with DevelopmentTrace(tmp_path / "wrong-task.jsonl", split="development",
                          candidate_sha256=base.revision, task="other", case_id="case") as recorder:
        traced = ProgramPolicy(base.candidate, Backend(), diagnostics=recorder)
        with pytest.raises(ValueError, match="exactly one"):
            traced.begin_episode("radio", {}, {}, Context("test", harness_revision=base.revision))


def test_fixed_program_receives_onboard_data_and_forwards_real_action_port(tmp_path):
    policy = make_policy(tmp_path)
    context = Context("test", harness_revision=policy.revision)
    raw = {"robot::proprio": np.zeros(256), "object_ground_truth": "must not pass"}
    raw.update({prefix + "::rgb": np.zeros((8, 8, 3), dtype=np.uint8) for prefix in CAMERAS.values()})
    policy.begin_episode("radio", {}, raw, context)
    assert np.asarray(policy.act(raw, context)).shape == (1, 23)
    assert policy.end_episode(context)["stopped"]
    with pytest.raises(ValueError, match="Initialize"):
        policy.act(raw, context)


def test_program_edits_are_rejected_until_new_candidate_is_materialized(tmp_path):
    policy = make_policy(tmp_path)
    context = Context("test", harness_revision=policy.revision)
    policy.begin_episode("radio", {}, {}, context)
    (tmp_path / "control.py").write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="Changed or missing harness input"):
        policy.act({}, context)


def test_shared_engine_change_is_rejected_before_action(tmp_path, monkeypatch):
    from PhysicalRSI_baselines.behavior_baseline import engine
    policy = make_policy(tmp_path)
    changed = engine.manifest()
    changed["files"]["PhysicalRSI_baselines/behavior_baseline/observations.py"] = "0" * 64
    monkeypatch.setattr(engine, "manifest", lambda: changed)
    with pytest.raises(ValueError, match="Shared System 1 implementation"):
        policy.act({}, Context("fixture", harness_revision=policy.revision))


def test_program_runs_through_core_json_lifecycle(tmp_path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    policy = make_policy(candidate_root)
    raw = {"robot::proprio": [0.] * 256}
    raw.update({prefix + "::rgb": np.zeros((8, 8, 3), dtype=np.uint8).tolist() for prefix in CAMERAS.values()})

    class Environment:
        def identity(self):
            return {"kind": "software-interface-fixture"}

        def describe(self):
            return Embodiment("R1Pro", "fixture", "software", OBSERVATION, ACTION)

        def reset(self, case, context):
            return {"ready": True, "observation": raw}

        def step(self, action, context):
            assert np.asarray(action).shape == (1, 23)
            return {"observation": raw, "terminated": True}

    class Verifier:
        def identity(self):
            return {"kind": "fixture-action-contract"}

        def verify(self, case, trace):
            return {"outcome": "success", "reason": "Software interface fixture only", "measurements": {}}

    receipt = ExperimentRuntime(tmp_path / "runs").run(
        "fixture", task="radio", case={"seed": 0}, scope="software interface test only",
        environment=Environment(), policy=policy, verifier=Verifier(), budget=Budget(1, 30))
    assert receipt["outcome"] == "success"
    assert receipt["qualification"] is None
