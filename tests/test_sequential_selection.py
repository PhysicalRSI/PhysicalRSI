import random
import shutil
from pathlib import Path

import pytest

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, now
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator, SequentialEvaluator
from PhysicalRSI_core.self_harness.selection import (
    betting_evidence,
    select_survivor,
    validate_profile,
)
from test_improvement_campaign import Proposal, Selector, Suite, manifest


def test_betting_stops_on_a_clear_prefix_and_is_undecided_on_null():
    evidence = betting_evidence([0.8] * 100, 0.0, 0.05, min_cases=8)
    assert evidence["decision"] == "accept"
    assert evidence["stopping_index"] < evidence["n_used"]
    assert betting_evidence([0.0] * 100, 0.0, 0.05)["decision"] == "undecided"


def test_betting_optional_stopping_null_acceptance_is_controlled():
    rng = random.Random(20261101)
    accepted = sum(
        betting_evidence([rng.choice((-1.0, 1.0)) for _ in range(160)], 0.0, 0.05)["decision"]
        == "accept"
        for _ in range(2000)
    )
    assert accepted / 2000 <= 0.065


def _paired_prefix(tmp_path, *, mismatch=False):
    source = tmp_path / "evidence.json"
    source.write_text("{}")
    episodes = 40
    profile = dict(
        tasks={"task": dict(weight=1, episodes=episodes, score_range=[0, 1], maximum_regression=0)},
        minimum_gain=0.1,
        tie_tolerance=0,
        statistics=dict(method="paired-betting", alpha=0.05, max_comparisons=1, min_cases=8),
    )
    comparison = dict(schema_version=1, round_id="sequential", frozen_at=now(),
                      parent_id="parent", candidates={"parent": "1" * 64, "child": "2" * 64},
                      profile=profile)
    cohort = dict(comparison_sha256=digest(comparison), generated_at=now(),
                  layouts={"task": [digest(["task", i]) for i in range(episodes)]})
    rows = []
    for candidate, score in (("parent", 0.1), ("child", 0.9)):
        prefix = cohort["layouts"]["task"][:24]
        if mismatch and candidate == "child":
            prefix = cohort["layouts"]["task"][1:25]
        rows.append(dict(candidate_id=candidate, kind="policy_evaluation", native_exit_code=0,
                         comparison_sha256=digest(comparison), cohort_sha256=digest(cohort),
                         freeze_sha256=comparison["candidates"][candidate], evaluator_revision="frozen",
                         episodes=[dict(task="task", layout_sha256=case, state="completed", score=score,
                                        success=True, measurements={},
                                        evidence_sha256={"evidence.json": file_digest(source)}) for case in prefix]))
    return comparison, cohort, rows


def test_selector_accepts_a_consistent_prefix_and_rejects_a_mismatch(tmp_path):
    comparison, cohort, results = _paired_prefix(tmp_path)
    decision = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    assert decision["survivor_id"] == "child"
    assert decision["evaluated_cases"] == 24 and decision["cohort_cases"] == 40
    comparison, cohort, results = _paired_prefix(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="frozen prefix"):
        select_survivor(comparison, cohort, results, evidence_root=tmp_path)


def test_profile_accepts_paired_betting():
    validate_profile(dict(
        tasks={"task": dict(weight=1, episodes=8, score_range=[0, 1], maximum_regression=0)},
        minimum_gain=0, tie_tolerance=0,
        statistics=dict(method="paired-betting", alpha=0.05, max_comparisons=2,
                        min_cases=8, bet_clip=0.5),
    ))


class SequentialSuite(Suite):
    def identity(self):
        return dict(kind="sequential-counter-suite", implementation=file_digest(Path(__file__)))

    def validation_cases(self, comparison):
        # Every gap is odd: increment 1 succeeds, increments 2 and 4 overshoot.
        # Distinct offsets give 32 unique layouts, disjoint from development.
        return {"counter": [dict(initial=2 * i, target=2 * i + 5) for i in range(32)]}


class ScriptedProposal(Proposal):
    def __init__(self, evaluator, increment):
        super().__init__(evaluator)
        self.increment = increment

    def identity(self):
        return dict(kind="sequential-scripted-child", increment=self.increment,
                    implementation=file_digest(Path(__file__)))

    def propose(self, parent, feedback, output):
        root = output / "candidate"
        shutil.copytree(parent["root"], root)
        atomic_json(root / "control.json", dict(increment=self.increment))
        return [manifest(root, "child", parent_sha256=verify_harness(parent),
                         method="scripted", changes=dict(increment=self.increment),
                         environment="software-counter", evidence=feedback["evidence"],
                         costs=dict(proposals=1))]


def _sequential_campaign(tmp_path, *, parent_increment=2, child_increment=1):
    seed = tmp_path / "seed"
    for kind in CLOSURE:
        atomic_json(seed / (kind + ".json"),
                    dict(increment=parent_increment) if kind == "control" else dict(kind=kind))
    initial = manifest(seed, "parent")
    state = HarnessState(tmp_path / "state")
    suite = SequentialSuite()
    quota = TrialQuota(tmp_path / "quota", max_trials=1 + 32 * 2)
    profile = dict(
        # Allow regression here so rejection is decided by the betting rule.
        tasks={"counter": dict(weight=1, episodes=32, score_range=[0, 1], maximum_regression=1)},
        minimum_gain=0, tie_tolerance=0,
        statistics=dict(method="paired-betting", alpha=0.05, max_comparisons=3, min_cases=8),
    )
    evaluator = SequentialEvaluator(
        ExperimentEvaluator(suite=suite, budgets={"counter": Budget(10, 30)}, scope="software",
                            system2=Revision("sequential-counter", "1"), quota=quota),
        block_cases=4, selector_profile=profile,
    )

    def build_loop(directory):
        return SelfHarness(directory, state=state,
                           proposer=ScriptedProposal(evaluator, child_increment), evaluator=evaluator,
                           selector=Selector(), scope="software", max_candidates=1, profile=profile,
                           protocol=dict(identity=evaluator.revision,
                                         evaluation_kind="local_contract_evaluation"))

    campaign = ImprovementCampaign(tmp_path / "campaign", state=state,
                                   build_loop=build_loop, max_rounds=1)
    return campaign, initial, suite, quota


def _sequential_records(campaign):
    directory = campaign.root / "rounds/round-0001"
    selection = read_json(directory / "selection.json")["output"]
    sequential = read_json(directory / "sequential.json")
    cohort = read_json(directory / "validation-cohort.json")
    results = {name: read_json(directory / f"evaluate_{name}.json")["output"]
               for name in ("parent", "child")}
    for result in results.values():
        assert [row["layout_sha256"] for row in result["episodes"]] == (
            cohort["layouts"]["counter"][:selection["evaluated_cases"]])
    assert sequential["evaluated_cases"] == selection["evaluated_cases"]
    assert len(sequential["blocks"]) == sequential["blocks_run"]
    for index, block in enumerate(sequential["blocks"]):
        assert block["block"] == index
        assert block["cases_evaluated"] == (index + 1) * 4
        assert set(block["states"]) == {"child"}
        betting = block["states"]["child"]
        assert betting["n_used"] == block["cases_evaluated"]
        assert block["decisions"] == {"child": betting["decision"]}
    return selection, sequential, results


def _assert_trial_count(campaign, suite, quota, selection):
    directory = campaign.root / "rounds/round-0001"
    development = read_json(directory / "develop.json")["output"]
    pool = read_json(directory / "freeze.json")["output"]["candidates"]
    expected = len(development["episodes"]) + selection["evaluated_cases"] * len(pool)
    trial_directories = [path for path in directory.glob("experiments/**/trials/trial-*")
                         if path.is_dir()]
    assert len(trial_directories) == quota.status()["reserved_trials"] == len(suite.resets) == expected
    assert all((path / "receipt.json").is_file() for path in trial_directories)


def test_sequential_campaign_stops_early_and_inherits(tmp_path):
    campaign, initial, suite, quota = _sequential_campaign(tmp_path)
    result = campaign.run(initial)
    assert result["status"] == "completed"
    assert result["rounds"][0]["outcome"] == "inherited"
    assert result["current"] == campaign.state.resolve()
    assert result["current"]["harness"]["id"] == "child"
    selection, sequential, results = _sequential_records(campaign)
    assert selection["decision"] == "inherit_child"
    assert 8 <= selection["evaluated_cases"] < selection["cohort_cases"] == 32
    assert all(row["score"] == 0 for row in results["parent"]["episodes"])
    assert all(row["score"] == 1 for row in results["child"]["episodes"])
    assert all(block["decisions"]["child"] == "undecided" for block in sequential["blocks"][:-1])
    stopping = sequential["blocks"][-1]
    assert stopping["decisions"]["child"] == "accept"
    assert stopping["cases_evaluated"] - 4 < stopping["states"]["child"]["stopping_index"] <= (
        stopping["cases_evaluated"])
    assert result["current"]["decision"] == selection
    _assert_trial_count(campaign, suite, quota, selection)


def test_sequential_campaign_resume_reproduces_prefix(tmp_path, monkeypatch):
    baseline, initial, _, _ = _sequential_campaign(tmp_path / "baseline")
    expected = baseline.run(initial)
    expected_selection, expected_log, expected_results = _sequential_records(baseline)
    campaign, initial, suite, quota = _sequential_campaign(tmp_path / "resumed")
    run = ExperimentRuntime.run
    executions = []
    interrupted = False

    def interrupt_after_trial(self, run_id, **kwargs):
        nonlocal interrupted
        fresh = not (self.root / run_id / "receipt.json").exists()
        receipt = run(self, run_id, **kwargs)
        if fresh:
            executions.append((str(self.root), run_id))
        # Include the development trial: nine completed trials means the
        # first validation block (four cases for each candidate) is durable.
        if not interrupted and fresh and len(executions) == 9:
            interrupted = True
            raise RuntimeError("Interrupted sequential campaign")
        return receipt

    monkeypatch.setattr(ExperimentRuntime, "run", interrupt_after_trial)
    with pytest.raises(RuntimeError, match="Interrupted sequential campaign"):
        campaign.run(initial)
    assert interrupted
    assert campaign.state.resolve()["harness"]["id"] == "parent"
    receipts = {path: file_digest(path) for path in campaign.root.rglob("receipt.json")}
    result = campaign.run(initial)
    selection, sequential, results = _sequential_records(campaign)
    assert selection["evaluated_cases"] == expected_selection["evaluated_cases"]
    assert sequential["blocks"] == expected_log["blocks"]
    for name in results:
        def prefix(value):
            return [(row["task"], row["layout_sha256"], row["score"], row["success"])
                    for row in value["episodes"]]
        assert prefix(results[name]) == prefix(expected_results[name])
    assert result["current"]["harness"]["id"] == expected["current"]["harness"]["id"] == "child"
    assert result["current"]["freeze_sha256"] == expected["current"]["freeze_sha256"]
    assert result["rounds"][0]["outcome"] == "inherited"
    assert all(file_digest(path) == sha for path, sha in receipts.items())
    assert len(executions) == len(set(executions)) == len(suite.resets)
    _assert_trial_count(campaign, suite, quota, selection)
    assert campaign.run(initial) == result
    _assert_trial_count(campaign, suite, quota, selection)


def test_sequential_null_child_retains_parent_without_exhausting_cohort_when_rejected(tmp_path):
    campaign, initial, suite, quota = _sequential_campaign(tmp_path, parent_increment=1, child_increment=2)
    result = campaign.run(initial)
    selection, sequential, results = _sequential_records(campaign)
    assert result["status"] == "completed"
    assert result["rounds"][0]["outcome"] == "retained"
    assert result["current"]["harness"]["id"] == "parent"
    assert selection["decision"] == "retain_parent"
    assert selection["evaluated_cases"] < selection["cohort_cases"] == 32
    assert all(row["score"] == 1 for row in results["parent"]["episodes"])
    assert all(row["score"] == 0 for row in results["child"]["episodes"])
    rejected = selection["rejected"]["child"]
    assert rejected["reason"] == "insufficient_evidence"
    assert rejected["betting_state"]["decision"] == "reject"
    assert 8 <= rejected["betting_state"]["stopping_index"] < selection["cohort_cases"]
    stopping = sequential["blocks"][-1]
    assert stopping["decisions"] == {"child": "reject"}
    for key, value in stopping["states"]["child"].items():
        assert rejected["betting_state"][key] == value
    _assert_trial_count(campaign, suite, quota, selection)


def test_sequential_undecided_runs_full_cohort(tmp_path):
    campaign, initial, suite, quota = _sequential_campaign(tmp_path, child_increment=4)
    result = campaign.run(initial)
    selection, sequential, results = _sequential_records(campaign)
    assert result["status"] == "completed"
    assert result["rounds"][0]["outcome"] == "retained"
    assert result["current"]["harness"]["id"] == "parent"
    assert selection["decision"] == "retain_parent"
    assert selection["evaluated_cases"] == selection["cohort_cases"] == 32
    assert [row["score"] for row in results["parent"]["episodes"]] == (
        [row["score"] for row in results["child"]["episodes"]])
    assert len(sequential["blocks"]) == 8
    assert all(block["decisions"] == {"child": "undecided"} for block in sequential["blocks"])
    betting = selection["rejected"]["child"]["betting_state"]
    assert betting["decision"] == "undecided" and betting["stopping_index"] is None
    _assert_trial_count(campaign, suite, quota, selection)
