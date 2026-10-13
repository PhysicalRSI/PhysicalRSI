from copy import deepcopy
from pathlib import Path

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState, StateConflict
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.retention import RetentionSelector
from PhysicalRSI_demos.retention_observations import replay_known_cases, setup


def test_retained_comparison_preserves_floor_and_archived_evidence(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path)
    result = campaign.run(initial)
    assert [r["outcome"] for r in result["rounds"]] == ["retain_parent", "retain_parent"]
    first, second = [campaign.state.read(r["result_revision"]) for r in result["rounds"]]
    anchor = campaign.state.read(result["rounds"][0]["parent_revision"])
    assert first["action"] == second["action"] == "assessment"
    assert first["previous"] == anchor["revision"] and second["previous"] == first["revision"]
    assert anchor["harness"] == first["harness"] == second["harness"]
    assert first["decision"]["metrics"]["initial"]["per_task"]["axis"] == 1.
    decision = second["decision"]
    assert decision["metrics"]["initial"]["per_task"]["axis"] == .5
    assert decision["metrics"]["candidate-1"]["normalized_score"] > decision["metrics"]["initial"]["normalized_score"]
    assert decision["retention"]["high_water"]["axis"] == 1.
    assert decision["rejected"]["candidate-1"]["reason"] == "historical_capability_regression"
    archive = CandidateArchive(tmp_path / "archive")
    entries = [archive.read(p.parent.name) for p in (archive.root / "entries").glob("*/record.json")]
    assert len(entries) == 2 and all(e["outcome"] == "retain_parent" for e in entries)
    assert all(e["result_revision"] != e["parent_revision"] for e in entries)
    assert len(suite.resets) == quota.status()["reserved_trials"] == 20
    reopened, seed, observed, _ = setup(tmp_path)
    assert reopened.run(seed) == result and observed.resets == []
    with pytest.raises(StateConflict):
        campaign.state.commit(anchor["revision"], anchor["harness"], evidence=first["evidence"],
                              decision=first["decision"])
    # No selected generation exists; assessments alone do not invent one.
    with pytest.raises(ValueError, match="No previous harness"):
        campaign.state.rollback(second["revision"])
    # Old cases are explicitly diagnostic replays, never fresh validation.
    replay = replay_known_cases(campaign, result, suite, quota)
    assert replay["candidates"]["initial"]["successes"] == 2
    assert replay["candidates"]["candidate-1"]["successes"] == 0
    assert len(suite.resets) == quota.status()["reserved_trials"] == 24
    assert replay_known_cases(campaign, result, suite, quota) == replay
    assert len(suite.resets) == 24 and campaign.state.resolve() == second


@pytest.mark.parametrize("boundary", ["history", "pointer", "commit", "archive"])
def test_retained_assessment_crash_recovery_does_not_repeat_trials(tmp_path, monkeypatch, boundary):
    from PhysicalRSI_core import lineage
    from PhysicalRSI_core import self_harness
    from PhysicalRSI_core.self_harness import evaluation
    campaign, initial, suite, quota = setup(tmp_path)
    write, outer, freeze = lineage.atomic_json, self_harness.atomic_json, evaluation.freeze_json
    stopped = False

    def hit():
        nonlocal stopped
        stopped = True
        raise RuntimeError("retained assessment publication interrupted")

    def publish(path, value):
        write(path, value)
        if stopped:
            return
        if boundary == "history" and Path(path).parent == campaign.state.root / "history" and value.get("action") == "assessment":
            hit()
        if boundary == "pointer" and Path(path) == campaign.state.root / "current.json":
            if campaign.state._metadata(value["revision"])["action"] == "assessment":
                hit()

    def receipt(path, value):
        if boundary == "commit" and Path(path).name == "commit.json" and not stopped:
            hit()
        return outer(path, value)

    def archive_receipt(path, value):
        result = freeze(path, value)
        if boundary == "archive" and Path(path).name == "archive.json" and not stopped:
            hit()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(lineage, "atomic_json", publish)
        patch.setattr(self_harness, "atomic_json", receipt)
        patch.setattr(evaluation, "freeze_json", archive_receipt)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            campaign.run(initial)
    assert stopped and len(suite.resets) == quota.status()["reserved_trials"] == 10
    result = campaign.run(initial)
    assert [r["outcome"] for r in result["rounds"]] == ["retain_parent", "retain_parent"]
    assert len(suite.resets) == quota.status()["reserved_trials"] == 20
    assert len(list((campaign.state.root / "history").glob("*.json"))) == 3
    assert campaign.run(initial) == result and len(suite.resets) == 20


def test_retained_historical_evidence_damage_blocks_new_floor(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path)
    result = campaign.run(initial)
    first = campaign.state.read(result["rounds"][0]["result_revision"])
    trajectory = next(Path(p) for p in first["evidence"] if p.endswith("trajectory.json"))
    trajectory.write_text("corrupted retained observation")
    with pytest.raises(ValueError, match="Committed evidence changed"):
        RetentionSelector(state=campaign.state, parent_revision=result["current"]["revision"],
                          maximum_regression={"axis": 0., "vector": 0.})
    assert len(suite.resets) == 20


def test_default_rollback_skips_assessments_and_restores_previous_artifact(tmp_path):
    from PhysicalRSI_demos.recursive_research import setup as research_setup
    campaign, initial, _, _ = research_setup(tmp_path)
    result = campaign.run(initial)
    assert result["current"]["action"] == "assessment"
    assert result["current"]["harness"]["id"] != initial["id"]
    restored = campaign.state.rollback(result["current"]["revision"], reason="restore previous generation")
    assert restored["harness"]["id"] == initial["id"]
    assert restored["rollback_to"] == result["rounds"][0]["result_revision"]
    assert campaign.state.rollback(result["current"]["revision"], reason="restore previous generation") == restored
    with pytest.raises(ValueError, match="No previous harness"):
        campaign.state.rollback(restored["revision"])


def test_improvement_campaign_labels_unchanged_artifact_as_retained(tmp_path, monkeypatch):
    from test_improvement_campaign import Proposal, setup as improvement_setup
    propose = Proposal.propose
    def ineffective(self, parent, feedback, output):
        children = propose(self, parent, feedback, output)
        for child in children:
            path = Path(child["root"]) / "control.json"
            atomic_json(path, dict(increment=4))
            child["components"]["control"]["control.json"] = file_digest(path)
        return children
    monkeypatch.setattr(Proposal, "propose", ineffective)
    campaign, initial, suite, quota = improvement_setup(tmp_path, max_rounds=1)
    result = campaign.run(initial)
    assert result["rounds"][0]["outcome"] == "retained"
    assert result["current"]["action"] == "assessment" and result["current"]["harness"]["id"] == initial["id"]
    assert result["rounds"][0]["parent_revision"] != result["rounds"][0]["result_revision"]
    assert campaign.run(initial) == result and len(suite.resets) == quota.status()["reserved_trials"] == 5


@pytest.mark.parametrize("prior_action", ["bootstrap", "selection"])
def test_robot_adapter_preserves_original_promotion_and_qualification(tmp_path, monkeypatch, prior_action):
    # Synthetic adapter boundary fixture; no robot or physical qualification is asserted.
    from PhysicalRSI_core import lineage
    from PhysicalRSI_demos.real_robot import learning
    manifest = dict(id="policy", root=str(tmp_path / "policy"))
    atomic_json(Path(manifest["root"]) / "configuration.json", dict(task={"task": "fixture"}))
    evidence = tmp_path / "selected/result.json"
    atomic_json(evidence, dict(schema="tianji.dual-system.result/v1", policy_id="policy",
        qualification="offline", split="validation", status="completed", l2_calls=0, l3_calls=0))
    selected = dict(action=prior_action, harness=manifest, freeze_sha256="a"*64,
                    evidence={str(evidence): file_digest(evidence)})
    assessed = dict(action="assessment", previous="selected", harness=manifest,
                    freeze_sha256="a"*64, evidence={})
    class State:
        def __init__(self, root):
            pass
        def resolve(self):
            return deepcopy(assessed)
        def read(self, revision):
            assert revision == "selected"
            return deepcopy(selected)
    class Task:
        def public(self):
            return {"task": "fixture"}
    monkeypatch.setattr(lineage, "HarnessState", State)
    monkeypatch.setattr(learning.PolicyLibrary, "load_for_execution", lambda manifest: None)
    if prior_action == "bootstrap":
        with pytest.raises(ValueError, match="no independently selected"):
            learning.selected_policy(tmp_path, task=Task(), qualification="offline")
    else:
        assert learning.selected_policy(tmp_path, task=Task(), qualification="offline") == manifest
        with pytest.raises(ValueError, match="matching independent L1-only qualification"):
            learning.selected_policy(tmp_path, task=Task(), qualification="physical")
