from copy import deepcopy
from pathlib import Path

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.storage import atomic_json, digest, read_json
from PhysicalRSI_core.self_harness.bindings import validate_admission
from PhysicalRSI_demos.recursive_research import setup


def test_research_rejects_reflector_identity_type_drift_before_trials(tmp_path, monkeypatch):
    from PhysicalRSI_demos.recursive_research import Reflector
    settings = {'version': 1}
    identity = Reflector.identity
    monkeypatch.setattr(Reflector, 'identity', lambda self: dict(identity(self), settings=settings))
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=1)
    settings['version'] = True
    assert campaign.reflector.identity() == campaign.reflector_identity
    with pytest.raises(ValueError, match='Frozen reflector identity changed'):
        campaign.run(initial)
    assert suite.resets == [] and campaign.reflector.calls == 0
    assert quota.status()['reserved_trials'] == 0
    settings['version'] = 1
    result = campaign.run(initial)
    count = len(suite.resets)
    assert count > 0 and campaign.reflector.calls == 1
    assert campaign.run(initial) == result
    assert len(suite.resets) == count and campaign.reflector.calls == 1


def test_three_rounds_learn_across_distinct_task_contracts_and_resume(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path)
    result = campaign.run(initial)
    assert result["completed_rounds"] == 3 and result["qualification"] is None
    assert [row["outcome"] for row in result["rounds"]] == ["retain_parent", "inherit_child", "retain_parent"]
    first = result["rounds"][0]
    assert first["parent_revision"] != first["result_revision"]
    assessed = campaign.state.read(first["result_revision"])
    assert assessed["action"] == "assessment"
    assert assessed["harness"] == campaign.state.read(first["parent_revision"])["harness"]
    assert result["rounds"][0]["parent_memory"] != result["rounds"][0]["result_memory"]
    assert quota.status()["reserved_trials"] == len(suite.resets) == 30
    for index in range(3):
        path = campaign.root / "rounds" / f"round-{index+1:04d}" / "execution"
        proposal = read_json(path / "propose.json")["output"][0]
        assert proposal["used_lessons"] == (["unit-step-and-copy"] if index else [])
        comparison = read_json(path / "freeze.json")["output"]
        assert comparison["round_index"] == index
    assert campaign.run(initial) == result
    assert len(suite.resets) == 30 and campaign.reflector.calls == 3
    restarted, initial_again, fresh_suite, _ = setup(tmp_path)
    assert restarted.run(initial_again) == result
    assert fresh_suite.resets == [] and restarted.reflector.calls == 0
    assert not any(target in (21, "final") for _, target in suite.resets)


def test_final_test_is_report_only_bound_to_selection_and_resumes(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=2, max_trials=22)
    result = campaign.run(initial)
    before = (result["revision"], result["memory_revision"], campaign.reflector.calls)
    report = campaign.report_test(initial)
    assert report["selection_sha256"] == result["revision"]
    assert report["qualification"] is None
    assert all(row["success_rate"] == 1 for row in report["metrics"].values())
    assert quota.status()["reserved_trials"] == len(suite.resets) == 22
    assert campaign.report_test(initial) == report
    resumed = campaign.run(initial)
    assert (resumed["revision"], resumed["memory_revision"], campaign.reflector.calls) == before
    assert len(suite.resets) == 22


def test_frozen_memory_control_repeats_failures_under_same_trial_budget(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path, update_memory=False)
    result = campaign.run(initial)
    assert [row["outcome"] for row in result["rounds"]] == ["retain_parent"] * 3
    assert result["memory_revision"] == campaign.initial_memory
    assert quota.status()["reserved_trials"] == len(suite.resets) == 30
    for index in range(3):
        path = campaign.root / "rounds" / f"round-{index+1:04d}" / "execution" / "develop.json"
        assert all(row["outcome"] == "failure" for row in read_json(path)["output"]["episodes"])


@pytest.mark.parametrize("boundary", ["reflection_receipt", "cursor", "before_cursor"])
def test_restart_after_selection_or_memory_update_never_repeats_trials(tmp_path, monkeypatch, boundary):
    from PhysicalRSI_core.self_harness import research_campaign as module
    campaign, initial, suite, quota = setup(tmp_path)
    original = module.atomic_json
    stopped = False

    def crash(path, value):
        nonlocal stopped
        if (boundary == "before_cursor" and Path(path) == campaign.root / "current.json"
                and len(suite.resets) == 20 and not stopped):
            stopped = True
            raise RuntimeError("simulated process interruption")
        original(path, value)
        hit = ((boundary == "reflection_receipt" and Path(path).name == "reflection.json"
                and value.get("state") == "completed" and len(suite.resets) == 20) or
               (boundary == "cursor" and Path(path) == campaign.root / "current.json"
                and len(suite.resets) == 20))
        if hit and not stopped:
            stopped = True
            raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(module, "atomic_json", crash)
    with pytest.raises(RuntimeError, match="interruption"):
        campaign.run(initial)
    assert len(suite.resets) == 20
    result = campaign.run(initial)
    assert result["completed_rounds"] == 3 and len(suite.resets) == 30
    assert campaign.reflector.calls == 3


def test_uncertain_reflector_requires_explicit_recovery(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path, max_rounds=1)
    original = campaign.reflector.reflect

    def interrupted(*args):
        raise RuntimeError("provider response lost")

    campaign.reflector.reflect = interrupted
    with pytest.raises(RuntimeError, match="response lost"):
        campaign.run(initial)
    assert len(suite.resets) == 10
    campaign.reflector.reflect = original
    with pytest.raises(ReconciliationRequired, match="reflection"):
        campaign.run(initial)
    campaign.reflector.resume_reflection = original
    assert campaign.run(initial)["completed_rounds"] == 1
    assert len(suite.resets) == 10


def test_memory_delta_cannot_import_validation_evidence_or_overwrite_lessons(tmp_path):
    campaign, initial, _, _ = setup(tmp_path, max_rounds=1)
    result = campaign.run(initial)
    directory = campaign.root / "rounds/round-0001"
    delta = read_json(directory / "reflection.json")["output"]
    evidence = delta["operations"][0]["lesson"]["evidence"]
    forbidden = directory / "execution/validation-cases.json"
    from PhysicalRSI_core.infra.storage import file_digest
    changed = deepcopy(delta)
    changed["operations"][0]["lesson"]["evidence"] = {str(forbidden): file_digest(forbidden)}
    with pytest.raises(ValueError, match="development evidence"):
        campaign.memory.apply_delta(campaign.initial_memory, changed,
                                    allowed_evidence=evidence, context=campaign.context)
    with pytest.raises(ValueError, match="overwrite"):
        campaign.memory.apply_delta(result["memory_revision"], delta,
                                    allowed_evidence=evidence, context=campaign.context)


def test_completed_retained_round_is_audited(tmp_path):
    campaign, initial, _, _ = setup(tmp_path, max_rounds=1)
    campaign.run(initial)
    path = campaign.root / "rounds/round-0001/receipt.json"
    atomic_json(path, {})
    with pytest.raises(ValueError, match="round changed"):
        campaign.run(initial)


def test_cross_round_reset_identity_cannot_move_between_splits(tmp_path):
    from PhysicalRSI_core.lineage import HarnessState
    state = HarnessState(tmp_path)
    original = {k: [digest(k)] for k in ("development", "validation", "test")}
    state.reserve_splits("first", original)
    changed = deepcopy(original)
    changed["validation"] = [digest("new")]
    state.reserve_splits("second", changed)
    changed["development"] = original["validation"]
    with pytest.raises(ValueError, match="reused"):
        state.reserve_splits("third", changed)


def test_memory_and_shadow_rejections_satisfy_common_admission_contract(tmp_path):
    from test_research_memory import fixture, gate_for
    from PhysicalRSI_core.self_harness.artifacts import verify_harness
    from PhysicalRSI_core.self_harness.shadow import ShadowGate
    data = fixture(tmp_path)
    report = deepcopy(data[5])
    report["lessons"]["recovery"]["activation"]["passed"] = False
    freeze = verify_harness(data[1])
    memory = gate_for(tmp_path, data, report).check(data[1])
    shadow = ShadowGate(parent_sha256=verify_harness(data[0]), reports={}).check(data[1])
    for result in (memory, shadow):
        assert not result["accepted"]
        validate_admission(dict(result, freeze_sha256=freeze), freeze)


@pytest.mark.parametrize("mode", ["no_candidate", "all_rejected"])
def test_nonselected_rounds_still_advance_research_memory(tmp_path, mode):
    from PhysicalRSI_core.self_harness.artifacts import verify_harness
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=1)
    if mode == "all_rejected":
        suite.admit = lambda candidate: dict(accepted=candidate["id"] == "parent",
            reason="Review rejected this hypothesis", evidence=dict(closure=verify_harness(candidate)))
    else:
        build = campaign.build_loop

        def no_candidates(directory, context):
            loop = build(directory, context)
            loop.proposer.propose_with_context = lambda *args: []
            return loop

        campaign.build_loop = no_candidates
    result = campaign.run(initial)
    assert result["rounds"][0]["outcome"] == ("no_changed_candidate" if mode == "no_candidate"
                                               else "all_candidates_rejected")
    assert result["memory_revision"] != campaign.initial_memory
    assert len(suite.resets) == quota.status()["reserved_trials"] == 2
    assert campaign.run(initial) == result


def test_budget_stop_never_repeats_completed_development(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=1, max_trials=3)
    result = campaign.run(initial)
    assert result["status"] == "budget_exhausted" and result["completed_rounds"] == 0
    assert result["memory_revision"] == campaign.initial_memory
    assert campaign.run(initial) == result
    assert len(suite.resets) == quota.status()["reserved_trials"] == 2


def test_delta_preserves_counterexamples_and_restricts_refinement(tmp_path):
    campaign, _, _, _ = setup(tmp_path, max_rounds=1)
    revision = campaign.initial_memory
    evidence = campaign.memory.read(revision)["lessons"]["baseline"]["evidence"]
    def apply(operation, parent):
        return campaign.memory.apply_delta(parent, dict(schema="physicalrsi.memory-delta/v1", operations=[operation]),
                                           allowed_evidence=evidence, context=campaign.context)
    suspended = apply(dict(kind="counterexample", lesson_id="baseline", check="negative-control",
                           explanation="Observed contradiction", evidence=evidence), revision)
    assert campaign.memory.retrieve(suspended, campaign.context) == {}
    operation = dict(kind="refine", lesson_id="baseline", conditions=campaign.context,
                     hypothesis="Narrowly revised hypothesis", evidence=evidence)
    refined = apply(operation, suspended)
    assert campaign.memory.read(refined)["lessons"]["baseline"]["counterexample_checks"] == ["negative-control"]
    operation["conditions"] = {}
    with pytest.raises(ValueError, match="broaden"):
        apply(operation, refined)


def test_completed_only_report_rejects_fresh_and_partial_search_without_execution(tmp_path, monkeypatch):
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=2, max_trials=22)
    with pytest.raises(ValueError, match='already completed'):
        campaign.report_test(initial, require_completed=True)
    assert suite.resets == [] and quota.status()['reserved_trials'] == 0
    assert campaign.reflector.calls == 0
    publish = campaign._publish

    def interrupted(body):
        revision = publish(body)
        if body['index'] == 1:
            raise RuntimeError('stop after first committed round')
        return revision

    monkeypatch.setattr(campaign, '_publish', interrupted)
    with pytest.raises(RuntimeError, match='first committed round'):
        campaign.run(initial)
    assert len(suite.resets) == 10 and campaign.reflector.calls == 1
    with pytest.raises(ValueError, match='already completed'):
        campaign.report_test(initial, require_completed=True)
    assert len(suite.resets) == quota.status()['reserved_trials'] == 10
    assert campaign.reflector.calls == 1
    monkeypatch.setattr(campaign, '_publish', publish)
    result = campaign.run(initial)
    assert len(suite.resets) == 20 and campaign.reflector.calls == 2
    report = campaign.report_test(initial, require_completed=True)
    assert report['selection_sha256'] == result['revision']
    assert len(suite.resets) == quota.status()['reserved_trials'] == 22
    assert campaign.report_test(initial, require_completed=True) == report
    assert len(suite.resets) == 22 and campaign.reflector.calls == 2


def test_completed_only_report_still_audits_completed_evidence(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=1, max_trials=12)
    campaign.run(initial)
    path = campaign.root / 'rounds/round-0001/receipt.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='Completed research round changed'):
        campaign.report_test(initial, require_completed=True)
    assert len(suite.resets) == quota.status()['reserved_trials'] == 10


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_completed_only_report_requires_explicit_boolean(tmp_path, value):
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=1)
    with pytest.raises(ValueError, match='boolean'):
        campaign.report_test(initial, require_completed=value)
    assert suite.resets == [] and quota.status()['reserved_trials'] == 0
