import time
from pathlib import Path

import pytest

from PhysicalRSI_core.contracts import Cancelled, InfrastructureFault
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.selection import select_survivor

from core_fixture import CounterEnvironment, CounterPolicy, CounterVerifier
from test_improvement_campaign import manifest, setup


class LostLinkEnvironment(CounterEnvironment):
    def reset(self, case, context):
        raise InfrastructureFault("robot link lost")


class SlowPolicy(CounterPolicy):
    def act(self, observation, context):
        time.sleep(0.03)
        return super().act(observation, context)


class CancellingPolicy(CounterPolicy):
    def act(self, observation, context):
        context.cancelled.set()
        return super().act(observation, context)


def run(runtime, run_id, *, environment=None, policy=None, budget=None):
    return runtime.run(run_id, task="counter", case={"initial": 0, "target": 1},
                       scope="software", environment=environment or CounterEnvironment(),
                       policy=policy or CounterPolicy(), verifier=CounterVerifier(),
                       budget=budget or Budget(1, 30))


def test_explicit_fault_closes_attempt_and_is_auditable(tmp_path):
    runtime = ExperimentRuntime(tmp_path / "trials")
    receipt = run(runtime, "lost", environment=LostLinkEnvironment())
    assert receipt["outcome"] == "infrastructure-fault"
    audited = runtime.read_evidence("lost")
    assert audited["fault"]["reason"] == "robot link lost"
    assert all(not journal["trailing_partial"] for journal in audited.get("journals", {}).values())


def test_policy_deadline_is_failure_and_cancellation_still_raises(tmp_path):
    runtime = ExperimentRuntime(tmp_path / "trials")
    receipt = run(runtime, "deadline", policy=SlowPolicy(), budget=Budget(1, 0.005))
    assert receipt["outcome"] == "deadline"
    assert receipt["timing"]["call_seconds"]["policy"] > 0.005
    with pytest.raises(Cancelled):
        run(runtime, "cancelled", policy=CancellingPolicy())


def test_host_stall_outside_policy_budget_is_bookkeeping(tmp_path, monkeypatch):
    import PhysicalRSI_core.experiments as experiments

    original = experiments.atomic_json

    def stalled(path, value):
        if Path(path).name == "verdict.json":
            time.sleep(0.03)
        return original(path, value)

    monkeypatch.setattr(experiments, "atomic_json", stalled)
    receipt = run(ExperimentRuntime(tmp_path / "trials"), "stall", budget=Budget(1, 0.005))
    assert receipt["outcome"] == "success"
    assert receipt["timing"]["call_seconds"]["policy"] < 0.005
    assert receipt["timing"]["phase_seconds"]["bookkeeping"] > 0.02


def test_safety_ceiling_turns_an_unbounded_bookkeeping_stall_into_fault(tmp_path, monkeypatch):
    import PhysicalRSI_core.experiments as experiments

    original = experiments.atomic_json

    def stalled(path, value):
        if Path(path).name == "verdict.json":
            time.sleep(0.06)
        return original(path, value)

    monkeypatch.setattr(experiments, "atomic_json", stalled)
    runtime = ExperimentRuntime(tmp_path / "trials", safety_ceiling_multiple=2)
    receipt = run(runtime, "ceiling", budget=Budget(1, 0.02))
    assert receipt["outcome"] == "infrastructure-fault"
    assert receipt["timing"]["phase_seconds"]["bookkeeping"] > 0.05
    assert runtime.read_evidence("ceiling")["fault"]["reason"] == "Trial safety ceiling exceeded"


def test_bookkeeping_stall_preserves_paired_selection_and_statistics(tmp_path, monkeypatch):
    import PhysicalRSI_core.experiments as experiments

    def campaign_result(root, *, slow):
        campaign, parent, suite, quota = setup(root, max_trials=20, max_rounds=1,
                                                budget_seconds=0.005)
        if slow:
            original = experiments.atomic_json
            stalled = {"value": False}

            def write(path, value):
                if (not stalled["value"] and Path(path).name == "verdict.json"
                        and "validation" in Path(path).parts):
                    stalled["value"] = True
                    time.sleep(0.03)
                return original(path, value)

            with monkeypatch.context() as patch:
                patch.setattr(experiments, "atomic_json", write)
                result = campaign.run(parent)
            assert stalled["value"]
        else:
            result = campaign.run(parent)
        selection = read_json(campaign.root / "rounds/round-0001/selection.json")["output"]
        return result, selection

    _, baseline = campaign_result(tmp_path / "baseline", slow=False)
    _, slowed = campaign_result(tmp_path / "slowed", slow=True)
    fields = ("cohort_cases", "cohort_power", "decision", "evaluated_cases", "metrics",
              "rejected", "survivor_id")
    assert {field: slowed[field] for field in fields} == {field: baseline[field] for field in fields}
    trial_root = tmp_path / "slowed/campaign/rounds/round-0001/experiments/validation/parent/trials"
    receipts = list(trial_root.glob("*/receipt.json"))
    assert len(receipts) == 2
    assert any(read_json(path)["timing"]["phase_seconds"]["bookkeeping"] > 0.02
               for path in receipts)


def test_exhausted_pair_is_dropped_from_paired_statistics(tmp_path):
    source = tmp_path / "evidence.json"
    atomic_json(source, {})
    comparison = dict(schema_version=1, round_id="fault-round", frozen_at=now(),
                      parent_id="parent", candidates={"parent": "1" * 64, "child": "2" * 64},
                      profile=dict(tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1],
                                                           maximum_regression=0)},
                                   minimum_gain=0, tie_tolerance=0))
    layouts = [digest(["counter", index]) for index in range(2)]
    cohort = dict(comparison_sha256=digest(comparison), generated_at=now(),
                  layouts={"counter": layouts})
    results = []
    for candidate, scores in (("parent", [0.5]), ("child", [0.75])):
        results.append(dict(candidate_id=candidate, kind="policy_evaluation", native_exit_code=0,
                            comparison_sha256=digest(comparison), cohort_sha256=digest(cohort),
                            freeze_sha256=comparison["candidates"][candidate], evaluator_revision="r",
                            episodes=[dict(task="counter", layout_sha256=layouts[1], state="completed",
                                           score=scores[0], success=True, measurements={},
                                           evidence_sha256={"evidence.json": file_digest(source)})]))
    selection = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    assert selection["cohort_power"]["n_cases"] == 1


def test_evaluator_retries_same_case_after_infrastructure_fault(tmp_path):
    campaign, parent, suite, quota = setup(tmp_path, max_trials=20, max_rounds=1)
    evaluator = campaign.build_loop(tmp_path / "loop").evaluator
    evaluator.development(parent, tmp_path / "evaluation")
    child_root = tmp_path / "child"
    import shutil
    shutil.copytree(parent["root"], child_root)
    atomic_json(child_root / "control.json", {"increment": 1})
    child = manifest(child_root, "child")
    comparison = dict(schema_version=1, round_id="retry-round", frozen_at=now(),
                      parent_id="parent", candidates={"parent": verify_harness(parent),
                      "child": verify_harness(child)}, scope="software",
                      profile=dict(tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1],
                                                           maximum_regression=0)},
                                   minimum_gain=0, tie_tolerance=0, infrastructure_retries=2))
    original = suite.environment
    failed = {"value": False}

    def environment(task):
        adapter = original(task)
        reset = adapter.reset

        def flaky(case, context):
            if case["target"] == 5 and not failed["value"]:
                failed["value"] = True
                raise InfrastructureFault("transient sensor timeout")
            return reset(case, context)

        adapter.reset = flaky
        return adapter

    suite.environment = environment
    cohort = evaluator.validation(comparison, tmp_path / "evaluation")
    result = evaluator.evaluate(parent, comparison, cohort, tmp_path / "evaluation")
    assert len(result["episodes"]) == 2
    metadata = read_json(tmp_path / "evaluation/experiments/validation/parent/infrastructure.json")
    retry = next(iter(metadata["trials"].values()))
    assert len(retry["attempts"]) == 2 and retry["selected"] == retry["attempts"][-1]


def test_evaluator_exhaustion_marks_the_pair_unavailable(tmp_path):
    campaign, parent, suite, quota = setup(tmp_path, max_trials=40, max_rounds=1)
    evaluator = campaign.build_loop(tmp_path / "loop").evaluator
    evaluator.development(parent, tmp_path / "evaluation")
    import shutil
    child_root = tmp_path / "child"
    shutil.copytree(parent["root"], child_root)
    atomic_json(child_root / "control.json", {"increment": 1})
    child = manifest(child_root, "child")
    comparison = dict(schema_version=1, round_id="exhausted-round", frozen_at=now(),
                      parent_id="parent", candidates={"parent": verify_harness(parent),
                      "child": verify_harness(child)}, scope="software",
                      profile=dict(tasks={"counter": dict(weight=1, episodes=2,
                                                           score_range=[0, 1], maximum_regression=0)},
                                   minimum_gain=0, tie_tolerance=0, infrastructure_retries=2))
    original = suite.environment

    def environment(task):
        adapter = original(task)
        reset = adapter.reset

        def always_fault(case, context):
            if case["target"] == 5:
                raise InfrastructureFault("persistent sensor timeout")
            return reset(case, context)

        adapter.reset = always_fault
        return adapter

    suite.environment = environment
    cohort = evaluator.validation(comparison, tmp_path / "evaluation")
    results = [evaluator.evaluate(candidate, comparison, cohort, tmp_path / "evaluation")
               for candidate in (parent, child)]
    assert all(len(result["episodes"]) == 1 for result in results)
    selection = select_survivor(comparison, cohort, results, evidence_root=tmp_path / "evaluation")
    assert selection["cohort_power"]["n_cases"] == 1
