from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState, StateConflict
from PhysicalRSI_core.self_harness import SelfHarness, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


def manifest(root, name, **metadata):
    return dict(id=name, root=str(root), components={
        kind: {kind + ".json": file_digest(root / (kind + ".json"))} for kind in CLOSURE}, **metadata)


class Suite:
    def __init__(self):
        self.resets = []
        self.fail_after_effect = False

    def identity(self):
        return dict(kind="test-counter-suite", implementation=file_digest(Path(__file__)))

    def admit(self, candidate):
        return dict(accepted=True, evidence=dict(closure=verify_harness(candidate)))

    def development_cases(self, parent):
        return {"counter": [dict(initial=0, target=3)]}

    def validation_cases(self, comparison):
        return {"counter": [dict(initial=0, target=target) for target in (5, 7)]}

    def environment(self, task):
        suite = self

        class Environment(CounterEnvironment):
            def reset(self, case, context):
                suite.resets.append(case["target"])
                return super().reset(case, context)

            def step(self, action, context):
                result = super().step(action, context)
                if self.target == 5 and suite.fail_after_effect:
                    raise RuntimeError("Connection lost after an action")
                return result

        return Environment()

    def policy(self, candidate, task):
        sha = verify_harness(candidate)

        class Policy(CounterPolicy):
            def describe(self):
                return replace(super().describe(), revision=sha)

        return Policy(read_json(Path(candidate["root"]) / "control.json")["increment"])

    def verifier(self, task):
        return CounterVerifier()


class Proposal:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def identity(self):
        return dict(kind="reviewed-counter-repair", implementation=file_digest(Path(__file__)))

    def develop(self, parent, output):
        return self.evaluator.development(parent, output)

    def resume_development(self, parent, output):
        return self.evaluator.development(parent, output)

    def propose(self, parent, feedback, output):
        if all(row["outcome"] == "success" for row in feedback["episodes"]):
            return []
        root = output / "candidate"
        shutil.copytree(parent["root"], root)
        atomic_json(root / "control.json", dict(increment=1))
        return [manifest(root, "child", parent_sha256=verify_harness(parent),
                         method="scripted", changes=dict(increment=1), environment="software-counter",
                         evidence=feedback["evidence"], costs=dict(proposals=1))]


class Selector:
    def identity(self):
        return dict(source=file_digest(Path(selection.__file__)))

    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


def setup(tmp_path, *, max_trials=6, max_rounds=2, score_measurement=None):
    seed = tmp_path / "seed"
    for kind in CLOSURE:
        atomic_json(seed / (kind + ".json"), dict(increment=2) if kind == "control" else dict(kind=kind))
    initial = manifest(seed, "parent")
    state = HarnessState(tmp_path / "state")
    suite = Suite()
    quota = TrialQuota(tmp_path / "quota", max_trials=max_trials)
    evaluator = ExperimentEvaluator(suite=suite, budgets={"counter": Budget(10, 30)},
                                    scope="software", system2=Revision("test-repair", "1"), quota=quota,
                                    score_measurement=score_measurement)

    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=Proposal(evaluator), evaluator=evaluator,
                           selector=Selector(), scope="software", max_candidates=1,
                           profile=dict(tasks={"counter": dict(weight=1, episodes=2,
                               score_range=[0, 1], maximum_regression=0)}, minimum_gain=0, tie_tolerance=1e-10),
                           protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"))

    campaign = ImprovementCampaign(tmp_path / "campaign", state=state, build_loop=build_loop, max_rounds=max_rounds)
    return campaign, initial, suite, quota


@pytest.mark.parametrize("changed_file", ["develop.json", ".hidden-evidence.json"])
def test_campaign_inherits_then_retains_and_resumes_without_new_effects(tmp_path, changed_file):
    campaign, initial, suite, quota = setup(tmp_path)
    result = campaign.run(initial)
    assert result["status"] == "completed" and result["qualification"] is None
    assert [r["outcome"] for r in result["rounds"]] == ["inherited", "retained"]
    assert result["current"]["harness"]["id"] == "child"
    assert len(suite.resets) == quota.status()["reserved_trials"] == 6
    parent = campaign.state.read(result["rounds"][0]["parent_revision"])
    assert parent["harness"]["id"] == "parent"
    assert campaign.run(initial) == result
    assert len(suite.resets) == 6
    # Retained rounds have no new lineage node, but remain audited evidence.
    path = campaign.root / "rounds/round-0002" / changed_file
    path.write_text("{}")
    with pytest.raises(ValueError, match="Completed campaign round changed"):
        campaign.run(initial)


@pytest.mark.parametrize("split", ["development", "validation"])
def test_completed_trials_resume_after_outer_evaluation_interruption(tmp_path, monkeypatch, split):
    campaign, initial, suite, quota = setup(tmp_path)
    run = ExperimentRuntime.run
    interrupted = False

    def stop_after_receipt(self, *args, **kwargs):
        nonlocal interrupted
        result = run(self, *args, **kwargs)
        if kwargs["system2"].split == split and not interrupted:
            interrupted = True
            raise RuntimeError("Process stopped after a durable trial")
        return result

    monkeypatch.setattr(ExperimentRuntime, "run", stop_after_receipt)
    with pytest.raises(RuntimeError, match="durable trial"):
        campaign.run(initial)
    assert len(suite.resets) == (2 if split == "validation" else 1)
    monkeypatch.setattr(ExperimentRuntime, "run", run)
    result = campaign.run(initial)
    assert result["current"]["harness"]["id"] == "child"
    assert len(suite.resets) == quota.status()["reserved_trials"] == 6
    record = read_json(campaign.root / "rounds/round-0001" /
                       ("evaluate_parent.json" if split == "validation" else "develop.json"))
    assert record["recovery_attempts"] == 1


def test_validation_checkpoint_recovery_keeps_sampled_cases_and_reservations(tmp_path, monkeypatch):
    import PhysicalRSI_core.self_harness as module
    campaign, initial, suite, quota = setup(tmp_path)
    write = module.atomic_json

    def crash(path, value):
        if path.name == "validation.json" and value["state"] == "completed":
            raise RuntimeError("Stopped after freezing and reserving cases")
        return write(path, value)

    monkeypatch.setattr(module, "atomic_json", crash)
    with pytest.raises(RuntimeError, match="freezing and reserving"):
        campaign.run(initial)
    assert quota.status()["reserved_trials"] == 5 and suite.resets == [3]
    cohort = read_json(campaign.root / "rounds/round-0001/validation-cohort.json")
    suite.validation_cases = lambda comparison: pytest.fail("Do not resample frozen cases")
    monkeypatch.setattr(module, "atomic_json", write)
    assert campaign.run(initial)["completed_rounds"] == 2
    assert read_json(campaign.root / "rounds/round-0001/validation-cohort.json") == cohort
    assert len(suite.resets) == quota.status()["reserved_trials"] == 6


def test_unknown_physical_effect_is_never_retried_or_scored(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path)
    suite.fail_after_effect = True
    with pytest.raises(RuntimeError, match="Connection lost"):
        campaign.run(initial)
    suite.fail_after_effect = False
    with pytest.raises(ReconciliationRequired):
        campaign.run(initial)
    assert len(suite.resets) == 2
    assert quota.status()["reserved_trials"] == 5
    assert campaign.state.resolve()["harness"]["id"] == "parent"


@pytest.mark.parametrize("outcome", ["uncertain", "invalid"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_unscorable_validation_stops_before_more_effects_even_on_resume(tmp_path, monkeypatch, outcome, interrupted):
    campaign, initial, suite, quota = setup(tmp_path)
    environment_factory = suite.environment

    def environment(task):
        instance = environment_factory(task)
        reset = instance.reset

        def guarded_reset(case, context):
            result = reset(case, context)
            if case["target"] == 5 and outcome == "invalid":
                result["ready"] = False
            return result

        instance.reset = guarded_reset
        return instance

    class Verifier(CounterVerifier):
        def verify(self, case, trace):
            if case["target"] == 5 and outcome == "uncertain":
                return dict(outcome="uncertain", reason="Measurement unavailable", measurements={})
            return super().verify(case, trace)

    suite.environment = environment
    suite.verifier = lambda task: Verifier()
    run = ExperimentRuntime.run

    def crash_after_receipt(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        if kwargs["system2"].split == "validation":
            raise RuntimeError("Interrupted after unscorable receipt")
        return result

    if interrupted:
        monkeypatch.setattr(ExperimentRuntime, "run", crash_after_receipt)
        with pytest.raises(RuntimeError, match="unscorable receipt"):
            campaign.run(initial)
        monkeypatch.setattr(ExperimentRuntime, "run", run)
    for _ in range(2):
        with pytest.raises(ReconciliationRequired, match="definite.*outcome"):
            campaign.run(initial)
        assert suite.resets == [3, 5]
    receipts = list((campaign.root / "rounds/round-0001/experiments/validation").rglob("receipt.json"))
    assert len(receipts) == 1
    receipt = read_json(receipts[0])
    assert receipt["state"] == "completed" and receipt["outcome"] == outcome
    assert quota.status()["reserved_trials"] == 5
    assert campaign.state.resolve()["harness"]["id"] == "parent"


@pytest.mark.parametrize("outcome", ["uncertain", "invalid"])
def test_development_preserves_unscorable_feedback(tmp_path, outcome):
    campaign, initial, suite, quota = setup(tmp_path)
    environment_factory = suite.environment

    def environment(task):
        instance = environment_factory(task)
        reset = instance.reset

        def guarded_reset(case, context):
            result = reset(case, context)
            if outcome == "invalid":
                result["ready"] = False
            return result

        instance.reset = guarded_reset
        return instance

    class Verifier(CounterVerifier):
        def verify(self, case, trace):
            return dict(outcome="uncertain", reason="Measurement unavailable", measurements={})

    suite.environment = environment
    suite.verifier = lambda task: Verifier()
    loop = campaign.build_loop(campaign.root / "development-check")
    feedback = loop.evaluator.development(initial, loop.root)
    assert [row["outcome"] for row in feedback["episodes"]] == [outcome]
    assert loop.evaluator.development(initial, loop.root) == feedback
    assert suite.resets == [3] and quota.status()["reserved_trials"] == 1


@pytest.mark.parametrize("measurement", [None, True, "1", 2])
@pytest.mark.parametrize("interrupted", [False, True])
def test_invalid_measurement_blocks_more_validation_and_resume(tmp_path, monkeypatch, measurement, interrupted):
    campaign, initial, suite, quota = setup(tmp_path, score_measurement="quality")

    class Verifier(CounterVerifier):
        def verify(self, case, trace):
            result = super().verify(case, trace)
            if case["target"] == 5:
                if measurement is not None:
                    result["measurements"]["quality"] = measurement
            else:
                result["measurements"]["quality"] = int(result["outcome"] == "success")
            return result

    suite.verifier = lambda task: Verifier()
    run = ExperimentRuntime.run

    def stop_after_receipt(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        if kwargs["system2"].split == "validation":
            raise RuntimeError("Interrupted after measured receipt")
        return result

    if interrupted:
        monkeypatch.setattr(ExperimentRuntime, "run", stop_after_receipt)
        with pytest.raises(RuntimeError, match="measured receipt"):
            campaign.run(initial)
        monkeypatch.setattr(ExperimentRuntime, "run", run)
    for _ in range(2):
        with pytest.raises(ReconciliationRequired, match="invalid score measurement"):
            campaign.run(initial)
        assert suite.resets == [3, 5]
    assert quota.status()["reserved_trials"] == 5
    assert campaign.state.resolve()["harness"]["id"] == "parent"


def test_declared_valid_measurements_drive_inheritance_and_reuse(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path, score_measurement="quality")

    class Verifier(CounterVerifier):
        def verify(self, case, trace):
            result = super().verify(case, trace)
            result["measurements"]["quality"] = int(result["outcome"] == "success")
            return result

    suite.verifier = lambda task: Verifier()
    result = campaign.run(initial)
    assert [row["outcome"] for row in result["rounds"]] == ["inherited", "retained"]
    assert campaign.run(initial) == result and len(suite.resets) == 6


def test_budget_reserves_whole_paired_pool_before_any_validation(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path, max_trials=4)
    result = campaign.run(initial)
    assert result["status"] == "budget_exhausted" and result["completed_rounds"] == 0
    assert suite.resets == [3]
    assert quota.status()["reserved_trials"] == 1
    assert campaign.run(initial) == result
    assert suite.resets == [3]


@pytest.mark.parametrize("boundary", ["commit", "campaign"])
def test_lineage_cas_crash_recovers_without_repeating_comparison(tmp_path, monkeypatch, boundary):
    import PhysicalRSI_core.self_harness as loop_module
    import PhysicalRSI_core.self_harness.campaign as campaign_module
    campaign, initial, suite, quota = setup(tmp_path)
    module = loop_module if boundary == "commit" else campaign_module
    write = module.atomic_json

    def crash(path, value):
        if (path.name == "commit.json" if boundary == "commit"
                else path.name == "campaign.json" and len(value["rounds"]) == 1):
            raise RuntimeError("Lost writer before checkpoint")
        return write(path, value)

    monkeypatch.setattr(module, "atomic_json", crash)
    with pytest.raises(RuntimeError, match="Lost writer"):
        campaign.run(initial)
    assert campaign.state.resolve()["harness"]["id"] == "child"
    assert len(suite.resets) == 5
    monkeypatch.setattr(module, "atomic_json", write)
    assert campaign.run(initial)["completed_rounds"] == 2
    assert len(suite.resets) == quota.status()["reserved_trials"] == 6


def test_stage_without_explicit_recovery_keeps_blocking(tmp_path):
    campaign, _, _, _ = setup(tmp_path)
    loop = campaign.build_loop(campaign.root / "unrecoverable")

    def fail():
        raise RuntimeError("An external proposal may have started")

    with pytest.raises(RuntimeError):
        loop._step("external", {"input": 1}, fail)
    with pytest.raises(ReconciliationRequired):
        loop._step("external", {"input": 1}, lambda: pytest.fail("Must not replay"))


def test_campaign_rejects_external_lineage_change(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path)
    result = campaign.run(initial)
    campaign.state.rollback(result["current"]["revision"])
    with pytest.raises(StateConflict, match="outside|after"):
        campaign.run(initial)
    assert len(suite.resets) == 6


def test_development_cases_cannot_be_reused_for_selection(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path)
    suite.validation_cases = lambda comparison: {"counter": [dict(initial=0, target=t) for t in (3, 5)]}
    with pytest.raises(ValueError, match="overlap development"):
        campaign.run(initial)
    assert suite.resets == [3]


def test_mutable_infrastructure_cannot_become_round_evidence(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path)
    loop = campaign.build_loop(campaign.root / "rounds/round-0001")
    # All stores in this fixture are descendants of tmp_path. Such a broad
    # comparison directory would accidentally commit mutable infrastructure.
    with pytest.raises(ValueError, match="outside round evidence"):
        loop.evaluator.development(initial, tmp_path)
    assert suite.resets == []
