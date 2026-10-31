import json
from pathlib import Path

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.language_proposer import (
    LanguageMutationEvolver,
    ScriptedModelClient,
)
from PhysicalRSI_core.self_harness.rollout import RolloutProposer
from PhysicalRSI_core.self_harness.search_space import ParameterSpace, SearchField

from core_fixture import CounterEnvironment, CounterPolicy, CounterVerifier
from test_improvement_campaign import Selector, Suite, manifest


def _parent(tmp_path, increment=2):
    root = tmp_path / "seed"
    for kind in CLOSURE:
        atomic_json(root / (kind + ".json"),
                    {"increment": increment} if kind == "control" else {"kind": kind})
    return manifest(root, "parent")


def _space():
    return ParameterSpace([SearchField("control", "control.json", ("increment",), 0, 3, 0)])


def _evaluations():
    return [{"trial": {"task": "counter", "case": {"initial": 0, "target": 3},
                         "scope": "software"},
             "outcome": "failure", "verdict": {"measurements": {"position": 2}},
             "evidence": {}}]


def _evolver(response, tmp_path, *, max_candidates=2):
    return LanguageMutationEvolver(client=ScriptedModelClient([response]), space=_space(),
                                   max_candidates=max_candidates)


def test_valid_script_materializes_bounded_candidates_with_provenance(tmp_path):
    response = json.dumps({"candidates": [
        {"changes": {"increment": 1}, "hypothesis": "one step reaches the target",
         "expected_effect": "development succeeds"},
        {"changes": {"increment": 3}, "hypothesis": "three steps overshoot less",
         "expected_effect": "development succeeds"},
    ]})
    evolver = _evolver(response, tmp_path)
    children = evolver.propose(_parent(tmp_path), _evaluations(), None, tmp_path / "round")
    assert len(children) == 2
    assert all(child["method"] == "language-mutation" for child in children)
    assert all(child["parent_sha256"] for child in children)
    assert all(child["changes"]["control"][0]["value"] in (1, 3) for child in children)
    assert all(child["evidence"] for child in children)
    assert children[0]["costs"]["attempts"] == 1


def test_invalid_edits_are_dropped_and_recorded(tmp_path):
    response = json.dumps({"candidates": [
        {"changes": {"increment": 99}, "hypothesis": "bad", "expected_effect": "bad"},
        {"changes": {"missing": 1}, "hypothesis": "bad", "expected_effect": "bad"},
    ]})
    evolver = LanguageMutationEvolver(client=ScriptedModelClient([response]), space=_space(),
                                      max_candidates=1, max_attempts=1)
    assert evolver.propose(_parent(tmp_path), _evaluations(), None, tmp_path / "round") == []
    validation = read_json(tmp_path / "round/language-proposal/attempt-1/validation.json")
    assert len(validation["errors"]) >= 2
    assert any("outside bounds" in row["reason"] for row in validation["errors"])
    assert any("unknown field" in row["reason"] for row in validation["errors"])


def test_malformed_first_response_retries_and_records_attempt_cost(tmp_path):
    client = ScriptedModelClient([
        "not json",
        '{"candidates":[{"changes":{"increment":1},"hypothesis":"repair","expected_effect":"success"}]}',
    ])
    evolver = LanguageMutationEvolver(client=client, space=_space(), max_candidates=1, max_attempts=2)
    children = evolver.propose(_parent(tmp_path), _evaluations(), None, tmp_path / "round")
    assert len(children) == 1 and children[0]["costs"]["attempts"] == 2
    assert (tmp_path / "round/language-proposal/attempt-2/validation.json").exists()


def test_identical_candidate_is_dropped_and_resume_is_frozen(tmp_path):
    client = ScriptedModelClient([
        '{"candidates":[{"changes":{"increment":2},"hypothesis":"same","expected_effect":"same"}]}',
    ])
    evolver = LanguageMutationEvolver(client=client, space=_space(), max_candidates=1, max_attempts=1)
    parent = _parent(tmp_path)
    output = tmp_path / "round"
    assert evolver.propose(parent, _evaluations(), None, output) == []
    assert evolver.resume_proposal(parent, _evaluations(), None, output) == []
    assert client._index == 1
    validation = read_json(output / "language-proposal/attempt-1/validation.json")
    assert "identical to the parent" in validation["errors"][0]["reason"]


def test_two_round_counter_campaign_uses_scripted_language_improvement(tmp_path):
    initial = _parent(tmp_path)
    state = HarnessState(tmp_path / "state")
    suite = Suite()
    quota = TrialQuota(tmp_path / "quota", max_trials=6)
    evaluator = ExperimentEvaluator(suite=suite, budgets={"counter": Budget(10, 30)},
                                    scope="software", system2=Revision("test-repair", "1"),
                                    quota=quota)
    evolver = LanguageMutationEvolver(
        client=ScriptedModelClient([
            '{"candidates":[{"changes":{"increment":1},"hypothesis":"reduce to one","expected_effect":"hit target"}]}'
        ]), space=_space(), max_candidates=1)

    def build_loop(directory):
        return SelfHarness(
            directory, state=state, proposer=RolloutProposer(evaluator, evolver), evaluator=evaluator,
            selector=Selector(), scope="software", max_candidates=1,
            profile=dict(tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1], maximum_regression=0)},
                         minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"),
        )

    result = ImprovementCampaign(tmp_path / "campaign", state=state, build_loop=build_loop,
                                 max_rounds=2).run(initial)
    assert result["status"] == "completed"
    assert result["current"]["harness"]["id"].startswith("language-")
    proposal = read_json(tmp_path / "campaign/rounds/round-0001/language-proposal/proposal.json")
    assert proposal["candidates"][0]["hypothesis"] == "reduce to one"
