import hashlib
import json

import numpy as np
import pytest

from PhysicalRSI_baselines.behavior_baseline.diagnostics import DevelopmentTrace
from PhysicalRSI_baselines.behavior_baseline.observations import PROPRIO


def trace(path, **overrides):
    return DevelopmentTrace(path, **dict(split="development", candidate_sha256="a" * 64,
                                        task="radio", case_id="development/radio/1", **overrides))


def public():
    return dict(task="radio", proprio={name: np.zeros((1, len(indices))) for name, indices in PROPRIO.items()},
                rgb={"head": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
                evaluator_score=999, object_states={"privileged": True})


def test_trace_keeps_only_public_commands_and_proprioception(tmp_path):
    path = tmp_path / "trace.jsonl"
    action = np.arange(23, dtype=np.float32)[None, :]
    original = action.copy()
    with trace(path) as writer:
        writer.append(public(), primitive_actions=[action], issued_action=action)
        action[:] = -1
    record = writer.close()
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["recorded_steps"] == 1 and record["qualification"] is None
    header, row = [json.loads(line) for line in path.read_text().splitlines()]
    assert header["split"] == "development"
    assert set(row) == {"step", "proprio", "primitive_actions", "issued_action"}
    assert sum(len(v[0]) for v in row["proprio"].values()) == 25
    np.testing.assert_array_equal(row["issued_action"], original)
    np.testing.assert_array_equal(row["primitive_actions"][0], original)
    with pytest.raises(ValueError, match="closed"):
        writer.append(public(), primitive_actions=[], issued_action=original)
    with pytest.raises(FileExistsError):
        trace(path)


def test_validation_and_test_cannot_open_diagnostics(tmp_path):
    for split in ("validation", "public_test", "test"):
        with pytest.raises(ValueError, match="development"):
            DevelopmentTrace(tmp_path / split, split=split, candidate_sha256="a" * 64,
                             task="radio", case_id="case")
        assert not (tmp_path / split).exists()


def test_invalid_actions_do_not_write_partial_rows_or_mutate_inputs(tmp_path):
    observation = public()
    action = np.zeros((1, 23), dtype=np.float32)
    with trace(tmp_path / "trace.jsonl") as writer:
        before = writer.path.read_bytes()
        with pytest.raises(ValueError):
            writer.append(observation, primitive_actions=[np.full((1, 23), np.nan)], issued_action=action)
        assert writer.path.read_bytes() == before
        writer.append(observation, primitive_actions=[], issued_action=action)
    assert writer.close()["recorded_steps"] == 1
    np.testing.assert_array_equal(action, np.zeros((1, 23)))
    assert all(not value.any() for value in observation["proprio"].values())


def test_binding_requires_frozen_development_membership(tmp_path):
    from PhysicalRSI_baselines.behavior_baseline.diagnostics import validate_binding
    from PhysicalRSI_core.self_harness.protocol import CaseProtocol
    from PhysicalRSI_core.infra.storage import digest

    def case(index):
        return dict(task="radio", mode="train", instance_id=index, instance_index=index, rollout_id=0)
    protocol = CaseProtocol(tmp_path / "cases", identity_keys=("instance_id",),
                            splits={split: {"radio": [case(index)]} for split, index in
                                    [("development", 1), ("validation", 200), ("test", 301)]})
    binding = dict(schema="physicalrsi.behavior-development-binding/v1", split="development",
                   candidate_sha256="a" * 64, task="radio", case=case(1),
                   protocol_path=str(protocol.root / "protocol.json"), protocol_sha256=protocol.revision)
    assert validate_binding(binding, candidate_sha256="a" * 64, task="radio") == digest(["radio", {"instance_id": 1}])
    for changes in ({"case": case(200)}, {"split": "validation"}, {"protocol_sha256": "0" * 64},
                    {"candidate_sha256": "b" * 64}):
        with pytest.raises(ValueError):
            validate_binding(dict(binding, **changes), candidate_sha256="a" * 64, task="radio")


def test_sealed_trace_checks_counts_fields_and_preserves_snapshot(tmp_path):
    from PhysicalRSI_baselines.behavior_baseline.diagnostics import seal_trace, verify_trace
    source, target = tmp_path / "live.jsonl", tmp_path / "sealed.jsonl"
    with trace(source) as writer:
        writer.append(public(), primitive_actions=[], issued_action=np.zeros((1, 23)))
    identity = writer.identity()
    with pytest.raises(ValueError, match="count"):
        seal_trace(source, target, expected_identity=identity, expected_steps=2)
    assert not target.exists()
    sealed = seal_trace(source, target, expected_identity=identity, expected_steps=1)
    source.write_text("later external corruption")
    assert verify_trace(target, expected_identity=identity, expected_steps=1) == sealed
    with pytest.raises(FileExistsError):
        seal_trace(source, target, expected_identity=identity, expected_steps=1)
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    rows[1]["privileged_object_state"] = "forbidden"
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="fields"):
        verify_trace(target, expected_identity=identity, expected_steps=1)
