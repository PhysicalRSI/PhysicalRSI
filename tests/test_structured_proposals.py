from copy import deepcopy
from pathlib import Path

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import atomic_json, canonical, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.proposals import (
    EnumeratedEdits, JsonEditContract, ProposalLimits, StructuredProposer,
)
from test_improvement_campaign import Selector, Suite, manifest


def contract():
    return JsonEditContract({"control.json": dict(component="control", schema=dict(
        type="object", properties=dict(increment=dict(type="integer", minimum=1, maximum=3)),
        required=["increment"], additionalProperties=False))})


class Strategy(EnumeratedEdits):
    def __init__(self):
        super().__init__([{"control.json": dict(increment=1)}])
        self.calls = []

    def generate(self, request, limits):
        self.calls.append(deepcopy(request))
        return super().generate(request, limits)


def setup(tmp_path, strategy=None, **limit_values):
    seed = tmp_path / "seed"
    for kind in CLOSURE:
        atomic_json(seed / (kind + ".json"), dict(increment=2) if kind == "control" else dict(kind=kind))
    # This file must never be copied or exposed to the strategy.
    (seed / "undeclared.txt").write_text("private workspace content")
    parent = manifest(seed, "parent")
    quota = TrialQuota(tmp_path / "quota", max_trials=10)
    suite = Suite()
    evaluator = ExperimentEvaluator(suite=suite, budgets={"counter": Budget(10, 30)},
                                    scope="software", system2=Revision("structured-search", "1"), quota=quota)
    strategy = strategy or Strategy()
    proposer = StructuredProposer(evaluator=evaluator, strategy=strategy, contract=contract(),
                                  objective="Reach the counter target using bounded positive increments",
                                  limits=ProposalLimits(max_candidates=2, **limit_values))
    state = HarnessState(tmp_path / "state")

    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=proposer, evaluator=evaluator,
                           selector=Selector(), scope="software", max_candidates=2,
                           profile=dict(tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1],
                               maximum_regression=0)}, minimum_gain=0, tie_tolerance=1e-10),
                           protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"))

    campaign = ImprovementCampaign(tmp_path / "campaign", state=state, build_loop=build_loop, max_rounds=2)
    return parent, proposer, suite, campaign


def test_generic_search_drives_real_experiments_and_lineage_without_leaking_workspace(tmp_path):
    parent, proposer, suite, campaign = setup(tmp_path)
    original = verify_harness(parent)
    result = campaign.run(parent)
    assert [row["outcome"] for row in result["rounds"]] == ["inherited", "retained"]
    child = result["current"]["harness"]
    assert read_json(Path(child["root"]) / "control.json") == dict(increment=1)
    assert verify_harness(parent) == original
    assert not (Path(child["root"]) / "undeclared.txt").exists()
    assert len(proposer.strategy.calls) == 2
    first = proposer.strategy.calls[0]
    assert str(tmp_path) not in canonical(first).decode()
    assert "private workspace content" not in canonical(first).decode()
    assert first["development"]["episodes"][0]["case"]["target"] == 3
    assert "validation" not in first
    assert len(suite.resets) == 6
    assert campaign.run(parent) == result
    assert len(proposer.strategy.calls) == 2 and len(suite.resets) == 6


@pytest.mark.parametrize("stage", ["reply", "candidate", "outer"])
def test_durable_response_resumes_materialization_without_another_strategy_call(tmp_path, monkeypatch, stage):
    import PhysicalRSI_core.self_harness as harness_module
    import PhysicalRSI_core.self_harness.proposals as module
    parent, proposer, suite, campaign = setup(tmp_path)
    if stage == "reply":
        original = proposer._materialize

        def crash(*args):
            raise RuntimeError("after response")

        monkeypatch.setattr(proposer, "_materialize", crash)
    else:
        target = harness_module if stage == "outer" else module
        original = target.atomic_json

        def crash(path, value):
            if ((stage == "outer" and path.name == "propose.json" and value["state"] == "completed") or
                    (stage == "candidate" and path.name == "materialization.json")):
                raise RuntimeError("after response")
            return original(path, value)

        monkeypatch.setattr(target, "atomic_json", crash)
    with pytest.raises(RuntimeError, match="after response"):
        campaign.run(parent)
    assert len(proposer.strategy.calls) == 1 and suite.resets == [3]
    if stage == "reply":
        monkeypatch.setattr(proposer, "_materialize", original)
    else:
        monkeypatch.setattr(target, "atomic_json", original)
    assert campaign.run(parent)["completed_rounds"] == 2
    assert len(proposer.strategy.calls) == 2 and len(suite.resets) == 6


def test_unknown_call_is_not_repeated_or_scored_and_does_not_log_exception_secrets(tmp_path):
    parent, proposer, suite, campaign = setup(tmp_path)

    def lose_reply(request, limits):
        proposer.strategy.calls.append(request)
        raise TimeoutError("fake-api-key-that-must-not-be-logged")

    proposer.strategy.generate = lose_reply
    for _ in range(2):
        with pytest.raises(ReconciliationRequired):
            campaign.run(parent)
    assert len(proposer.strategy.calls) == 1 and suite.resets == [3]
    files = list((tmp_path / "campaign").rglob("*.json"))
    assert all("fake-api-key-that-must-not-be-logged" not in p.read_text() for p in files)
    assert campaign.state.resolve()["harness"] == parent


@pytest.mark.parametrize("bad", ["foundation", "traversal", "absolute", "alias", "stale", "schema", "extra", "duplicate"])
def test_untrusted_edits_are_rejected_before_materialization(tmp_path, bad):
    parent, proposer, _, _ = setup(tmp_path)
    output = tmp_path / "round"
    feedback = proposer.develop(parent, output)
    generate = proposer.strategy.generate

    def invalid(request, limits):
        reply = generate(request, limits)
        from PhysicalRSI_core.self_harness.proposals import strict_json
        batch = strict_json(reply["text"])
        candidate = batch["candidates"][0]
        edit = candidate["edits"][0]
        if bad in {"foundation", "traversal", "absolute", "alias"}:
            edit["path"] = {"foundation": "foundation.json", "traversal": "../control.json",
                            "absolute": "/tmp/control.json", "alias": "./control.json"}[bad]
        elif bad == "stale":
            edit["before_sha256"] = "0" * 64
        elif bad == "schema":
            edit["value"] = dict(increment=100)
        elif bad == "extra":
            candidate["success_rate"] = 1
        else:
            candidate["edits"].append(deepcopy(edit))
        return dict(reply, text=canonical(batch).decode())

    proposer.strategy.generate = invalid
    before = verify_harness(parent)
    assert proposer.propose(parent, feedback, output) == []
    assert verify_harness(parent) == before
    assert not (output / "proposal/candidates").exists()
    report = read_json(output / "proposal/materialization.json")["result"]
    assert report["rejected"] == [dict(index=0, reason="edit_contract_rejected")]
    assert report["qualification"] is None


@pytest.mark.parametrize("reply", ["```json\n{}\n```", '{"candidates":[],"candidates":[]}', '{"x":NaN}', '{"x":1e999}'])
def test_invalid_responses_are_durable_rejections_with_no_retry(tmp_path, reply):
    parent, proposer, _, _ = setup(tmp_path)
    output = tmp_path / "round"
    feedback = proposer.develop(parent, output)
    proposer.strategy.generate = lambda request, limits: dict(text=reply)
    assert proposer.propose(parent, feedback, output) == []
    proposer.strategy.generate = lambda *_: pytest.fail("Known invalid response must not be retried")
    assert proposer.resume_proposal(parent, feedback, output) == []


def test_inputs_and_saved_response_are_verified_on_resume(tmp_path):
    parent, proposer, _, _ = setup(tmp_path)
    output = tmp_path / "round"
    feedback = proposer.develop(parent, output)
    proposer.propose(parent, feedback, output)
    path = output / "proposal/call.json"
    call = read_json(path)
    call["reply"]["text"] = "{}"
    atomic_json(path, call)
    with pytest.raises(ValueError, match="response changed"):
        proposer.propose(parent, feedback, output)


def test_changed_development_or_exposed_validation_cannot_start_a_new_call(tmp_path):
    parent, proposer, _, _ = setup(tmp_path)
    output = tmp_path / "round"
    feedback = proposer.develop(parent, output)
    file = output / next(iter(feedback["evidence"]))
    original = file.read_bytes()
    file.write_text("{}")
    with pytest.raises(ValueError, match="Development evidence changed"):
        proposer.propose(parent, feedback, output)
    file.write_bytes(original)
    atomic_json(output / "freeze.json", {})
    with pytest.raises(ValueError, match="after validation"):
        proposer.propose(parent, feedback, output)
    assert not proposer.strategy.calls


def test_contract_disallows_foundation_aliases_and_external_schema_references(tmp_path):
    with pytest.raises(ValueError, match="non-foundation"):
        JsonEditContract({"foundation.json": dict(component="foundation", schema={})})
    with pytest.raises(ValueError, match="without references"):
        JsonEditContract({"control.json": dict(component="control", schema={"$ref": "https://example.invalid/schema"})})
    parent, proposer, _, _ = setup(tmp_path)
    parent["components"]["foundation"]["control.json"] = parent["components"]["control"]["control.json"]
    with pytest.raises(ValueError, match="exactly its declared component"):
        proposer.contract.inputs(parent, proposer.limits)


@pytest.mark.parametrize("limit", ["request_bytes", "closure_bytes"])
def test_byte_limits_block_before_the_strategy_is_called(tmp_path, limit):
    parent, proposer, _, _ = setup(tmp_path, **{limit: 10})
    output = tmp_path / "round"
    feedback = proposer.develop(parent, output)
    with pytest.raises(ValueError, match="budget|allowance"):
        proposer.propose(parent, feedback, output)
    assert not proposer.strategy.calls


@pytest.mark.parametrize("receipt_saved", [True, False])
def test_resume_rejects_extra_files_in_a_previously_materialized_candidate(tmp_path, receipt_saved):
    parent, proposer, _, _ = setup(tmp_path)
    output = tmp_path / "round"
    feedback = proposer.develop(parent, output)
    child = proposer.propose(parent, feedback, output)[0]
    # Both completed materialization and a crash before its checkpoint must
    # reject an undeclared import/config inserted into an existing child.
    if not receipt_saved:
        (output / "proposal/materialization.json").unlink()
    (Path(child["root"]) / "undeclared.py").write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="undeclared files"):
        proposer.resume_proposal(parent, feedback, output)
    assert len(proposer.strategy.calls) == 1


def test_formatting_only_edits_and_duplicates_do_not_create_trials(tmp_path):
    parent, proposer, _, _ = setup(tmp_path)
    control = Path(parent["root"]) / "control.json"
    control.write_text('{ "increment": 2 }')
    parent["components"]["control"]["control.json"] = file_digest(control)
    output = tmp_path / "round"
    feedback = proposer.develop(parent, output)

    def generate(request, limits):
        edit = dict(path="control.json", before_sha256=request["editable"]["control.json"]["sha256"],
                    value=dict(increment=2))
        return dict(text=canonical(dict(parent_sha256=request["parent_sha256"], candidates=[
            dict(rationale="Same value in different formatting", edits=[edit])])).decode())

    proposer.strategy.generate = generate
    assert proposer.propose(parent, feedback, output) == []
    assert read_json(output / "proposal/materialization.json")["result"]["rejected"] == [
        dict(index=0, reason="unchanged_or_duplicate")]
    assert control.read_text() == '{ "increment": 2 }'


def test_multi_candidate_search_selects_from_independent_outcomes(tmp_path):
    alternatives = EnumeratedEdits([{"control.json": dict(increment=3)}, {"control.json": dict(increment=1)}])
    parent, _, suite, campaign = setup(tmp_path, alternatives)
    result = campaign.run(parent)
    assert result["status"] == "budget_exhausted"  # The second paired pool cannot fit.
    assert result["completed_rounds"] == 1
    child = result["current"]["harness"]
    assert read_json(Path(child["root"]) / "control.json") == dict(increment=1)
    assert len(suite.resets) == 8  # Seven first-round trials and next development.
    first = read_json(tmp_path / "campaign/rounds/round-0001/proposal/materialization.json")
    assert len(first["result"]["candidates"]) == 2
    assert result["current"]["decision"]["metrics"][child["id"]]["success_rate"] == {"counter": 1}
