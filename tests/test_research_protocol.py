from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.shadow import ReplayCase, ShadowGate, TemporalTrigger, shadow_replay
from test_improvement_campaign import setup, manifest


def splits():
    return {"development": {"counter": [{"initial": 0, "target": 3}, {"initial": 0, "target": 2}]},
            "validation": {"counter": [{"initial": 0, "target": 5}, {"initial": 0, "target": 7}]},
            "test": {"counter": [{"initial": 0, "target": 11}]}}


def protocol(tmp_path):
    return CaseProtocol(tmp_path / "protocol", splits=splits(), identity_keys=("initial", "target"))


def test_protocol_disjoint_immutable_and_single_final_claim(tmp_path):
    plan = protocol(tmp_path)
    cases = plan.cases("development")
    cases["counter"][0]["target"] = 99
    assert plan.cases("development") == splits()["development"]
    assert protocol(tmp_path).identity() == plan.identity()
    with pytest.raises(ValueError, match="final-candidate"):
        plan.cases("test")
    assert plan.claim_test(candidate_sha256="1" * 64, selection_sha256="2" * 64) == splits()["test"]
    assert plan.claim_test(candidate_sha256="1" * 64, selection_sha256="2" * 64) == splits()["test"]
    with pytest.raises(ValueError, match="already bound"):
        plan.claim_test(candidate_sha256="3" * 64, selection_sha256="2" * 64)
    atomic_json(plan.root / "protocol.json", {})
    with pytest.raises(ValueError, match="changed"):
        plan.cases("development")


def test_case_metadata_cannot_disguise_split_leakage(tmp_path):
    cases = splits()
    cases["test"]["counter"] = [dict(cases["development"]["counter"][0], note="test")]
    with pytest.raises(ValueError, match="reused"):
        CaseProtocol(tmp_path / "invalid", splits=cases, identity_keys=("initial", "target"))
    assert not (tmp_path / "invalid").exists()


def test_validation_cannot_be_reused_for_another_adaptive_round(tmp_path):
    from PhysicalRSI_core.self_harness import now
    plan = protocol(tmp_path)
    comparison = dict(schema_version=1, round_id="first", parent_id="parent", frozen_at=now(),
        candidates={"parent": "1" * 64, "child": "2" * 64},
        profile=dict(tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1],
            maximum_regression=0)}, minimum_gain=0, tie_tolerance=1e-10))
    assert plan.claim_validation(comparison) == splits()["validation"]
    assert plan.claim_validation(deepcopy(comparison)) == splits()["validation"]
    comparison["round_id"] = "second"
    with pytest.raises(ValueError, match="fresh round"):
        plan.claim_validation(comparison)


def prepare(tmp_path):
    campaign, parent, suite, quota = setup(tmp_path, max_trials=8, max_rounds=1)
    loop = campaign.build_loop(tmp_path / "unused")
    plan = protocol(tmp_path)
    loop.evaluator.suite = PreregisteredSuite(suite, plan)
    # Build a fresh evaluator identity, as required for any protocol change.
    from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
    evaluator = ExperimentEvaluator(suite=loop.evaluator.suite, budgets=loop.evaluator.budgets,
        scope=loop.evaluator.scope, system2=loop.evaluator.system2, quota=quota)
    evaluator.development(parent, tmp_path / "development")
    receipts = sorted((tmp_path / "development").glob("experiments/development/*/trials/*/receipt.json"))
    assert len(receipts) == 2 and suite.resets == [3, 2]
    root = tmp_path / "child"
    shutil.copytree(parent["root"], root)
    atomic_json(root / "control.json", {"increment": 1})
    child = manifest(root, "child")
    cases = [ReplayCase(p, 2 if read_json(p)["outcome"] == "failure" else None) for p in receipts]
    return parent, child, suite, plan, cases


def replay(parent, child, cases, **kwargs):
    return shadow_replay(parent_sha256=verify_harness(parent), candidate_sha256=verify_harness(child),
                         trigger=TemporalTrigger(("position",), 3, consecutive=1), cases=cases, **kwargs)


def test_replay_and_admission_use_actual_experiment_evidence_without_effects(tmp_path):
    parent, child, suite, plan, cases = prepare(tmp_path)
    report = replay(parent, child, cases)
    assert report["passed"] and report["timely_detections"] == 1
    assert report["false_positives"] == 0 and report["qualification"] is None
    assert suite.resets == [3, 2]  # Replay did not reset/actuate anything.
    path = tmp_path / "shadow.json"
    atomic_json(path, report)
    gate = ShadowGate(parent_sha256=verify_harness(parent), reports={path: file_digest(path)})
    wrapped = PreregisteredSuite(suite, plan, shadow_gate=gate)
    assert wrapped.admit(parent)["accepted"] and wrapped.admit(child)["accepted"]
    assert "shadow" in wrapped.admit(child)["evidence"]
    trajectory = cases[0].receipt.with_name("trajectory.json")
    trajectory.write_text("[]")
    with pytest.raises(ValueError, match="evidence changed"):
        wrapped.admit(child)


@pytest.mark.parametrize("kind", ["missing", "late", "false_positive", "no_controls", "unknown_divergence"])
def test_inconclusive_or_inaccurate_detector_cannot_pass(tmp_path, kind):
    parent, child, suite, plan, cases = prepare(tmp_path)
    trigger = TemporalTrigger(("position",), 3, consecutive=1)
    if kind == "missing":
        trigger = TemporalTrigger(("unlogged",), 3)
    elif kind == "late":
        cases = [ReplayCase(c.receipt, 1 if c.divergence_index is not None else None) for c in cases]
    elif kind == "false_positive":
        trigger = TemporalTrigger(("position",), 0, consecutive=1)
    elif kind == "no_controls":
        cases = [c for c in cases if c.divergence_index is not None]
    else:
        cases = [ReplayCase(c.receipt) for c in cases]
    report = shadow_replay(parent_sha256=verify_harness(parent), candidate_sha256=verify_harness(child),
                           trigger=trigger, cases=cases)
    assert not report["passed"]
    path = tmp_path / "shadow.json"
    atomic_json(path, report)
    gate = ShadowGate(parent_sha256=verify_harness(parent), reports={path: file_digest(path)})
    assert not gate.check(child)["accepted"]


def test_wrong_parent_duplicate_budget_and_tampering_rejected(tmp_path):
    parent, child, _, _, cases = prepare(tmp_path)
    with pytest.raises(ValueError, match="frozen parent"):
        shadow_replay(parent_sha256="f" * 64, candidate_sha256=verify_harness(child),
                      trigger=TemporalTrigger(("position",), 3), cases=cases)
    with pytest.raises(ValueError, match="Repeated"):
        replay(parent, child, cases + cases)
    with pytest.raises(ValueError, match="budget"):
        replay(parent, child, cases, max_total_frames=1)
    cases[0].receipt.with_name("trajectory.json").write_text("[]")
    with pytest.raises(ValueError):
        replay(parent, child, cases)


def test_temporal_history_resets_per_episode_and_missing_data_is_not_negative_evidence():
    trigger = TemporalTrigger(("progress",), 1, consecutive=2)
    assert trigger.replay([{"progress": 1}])["first_trigger_index"] is None
    assert trigger.replay([{"progress": 1}])["first_trigger_index"] is None
    result = trigger.replay([{"progress": 1}, {}, {"progress": 1}, {"progress": 1}])
    assert result == {"first_trigger_index": 3, "unavailable_indices": [1]}


@pytest.mark.parametrize("split", ["validation", "test"])
def test_evaluation_receipts_cannot_enter_development_replay(tmp_path, split):
    from PhysicalRSI_core.embodiment import Revision, System2Trial
    from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
    parent, child, suite, _, _ = prepare(tmp_path)
    runtime = ExperimentRuntime(tmp_path / "reserved-evaluation")
    runtime.run("held", task="counter", case={"initial": 0, "target": 9}, scope="software",
        environment=suite.environment("counter"), policy=suite.policy(parent, "counter"),
        verifier=suite.verifier("counter"), budget=Budget(10, 30),
        system2=System2Trial(Revision("reviewed", "1"), parent["id"], verify_harness(parent),
                            split, "1" * 64, "2" * 64))
    with pytest.raises(ValueError, match="development trials"):
        replay(parent, child, [ReplayCase(runtime.root / "held/receipt.json", 2)])


def test_failed_shadow_blocks_provider_admission(tmp_path):
    parent, child, suite, plan, _ = prepare(tmp_path)
    gate = ShadowGate(parent_sha256=verify_harness(parent), reports={})
    suite.admit = lambda candidate: pytest.fail("Rejected trigger must not reach provider admission")
    wrapped = PreregisteredSuite(suite, plan, shadow_gate=gate)
    assert wrapped.admit(child)["accepted"] is False


def test_preregistered_suite_runs_existing_self_harness_end_to_end(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path, max_trials=7, max_rounds=1)
    from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
    template = campaign.build_loop(tmp_path / "template")
    plan = protocol(tmp_path)

    def build(directory):
        from PhysicalRSI_core.self_harness import SelfHarness
        from test_improvement_campaign import Proposal, Selector
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, plan), budgets=template.evaluator.budgets,
            scope=template.evaluator.scope, system2=template.evaluator.system2, quota=quota)
        return SelfHarness(directory, state=campaign.state, proposer=Proposal(evaluator),
            evaluator=evaluator, selector=Selector(), scope="software", max_candidates=1,
            profile=template.config["profile"],
            protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"))

    campaign.build_loop = build
    result = campaign.run(initial)
    assert result["status"] == "completed" and result["current"]["harness"]["id"] == "child"
    assert suite.resets == [3, 2, 5, 7, 5, 7]
    assert 11 not in suite.resets
