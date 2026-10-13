from copy import deepcopy
from pathlib import Path

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_core.self_harness.rollout import development_evaluations
from PhysicalRSI_demos.rollout_evolution import setup


@pytest.mark.parametrize("gains", [{"axis": 2.0, "vector": 0.5}, {"axis": 0.25, "vector": 3.0}])
def test_rollout_fitted_control_improves_two_tasks_on_unseen_targets(tmp_path, gains):
    campaign, initial, suite, quota = setup(tmp_path, gains=gains)
    result = campaign.run(initial)
    assert [row["outcome"] for row in result["rounds"]] == ["inherit_child", "no_changed_candidate"]
    control = read_json(Path(result["current"]["harness"]["root"]) / "control.json")
    assert control == pytest.approx({task: 1 / gain for task, gain in gains.items()})
    first = campaign.root / "rounds/round-0001/execution"
    feedback = read_json(first / "develop.json")["output"]
    assert [row["outcome"] for row in feedback["evaluations"]] == ["failure", "failure"]
    for row in feedback["evaluations"]:
        assert len(row["trajectory"]) == 2
        assert "action" in row["trajectory"][1]
    assert all(case["target"][0] != 23 for _, case in suite.resets)
    final = campaign.report_test(initial)
    assert all(row["success_rate"] == 1 for row in final["metrics"].values())
    assert result["qualification"] is final["qualification"] is None
    assert len(suite.resets) == quota.status()["reserved_trials"] == 14
    assert campaign.run(initial) == result
    restarted, seed, new_suite, _ = setup(tmp_path, gains=gains)
    assert restarted.run(seed) == result and restarted.report_test(seed) == final
    assert new_suite.resets == []


@pytest.mark.parametrize("fault", ["trace", "outcome", "omit", "duplicate", "validation"])
def test_reflection_requires_complete_unmodified_development_rollouts(tmp_path, fault):
    campaign, initial, _, _ = setup(tmp_path)
    context = campaign._context(0, "0" * 64, campaign.initial_memory, None)
    loop = campaign.build_loop(tmp_path / "probe", context)
    feedback = loop.proposer.develop(initial, loop.root)
    changed = deepcopy(feedback)
    if fault == "trace":
        changed["evaluations"][0]["trajectory"][1]["action"] = 42
    elif fault == "outcome":
        changed["episodes"][0]["outcome"] = "success"
    elif fault == "omit":
        changed["episodes"].pop()
        changed["evaluations"].pop()
    elif fault == "duplicate":
        changed["evaluations"].append(changed["evaluations"][0])
    else:
        changed["evaluations"][0]["trial"]["system2"]["split"] = "validation"
    with pytest.raises(ValueError):
        development_evaluations(initial, changed, loop.root)
    assert len(development_evaluations(initial, feedback, loop.root)) == 2


def test_unknown_evolver_call_requires_provider_reconciliation(tmp_path):
    campaign, initial, _, _ = setup(tmp_path)
    context = campaign._context(0, "0" * 64, campaign.initial_memory, None)
    loop = campaign.build_loop(tmp_path / "probe", context)
    feedback = loop.proposer.develop(initial, loop.root)
    with pytest.raises(ReconciliationRequired, match="reconcile"):
        loop.proposer.resume_proposal_with_context(initial, feedback, context, loop.root)


def test_unrepresentable_score_span_is_rejected_before_rollout(tmp_path, monkeypatch):
    from PhysicalRSI_demos import rollout_evolution as demo
    initialize = demo.SelfHarness

    def overflowing_profile(*args, **kwargs):
        kwargs['profile'] = deepcopy(kwargs['profile'])
        for spec in kwargs['profile']['tasks'].values():
            spec['score_range'] = [-1e308, 1e308]
        return initialize(*args, **kwargs)

    monkeypatch.setattr(demo, 'SelfHarness', overflowing_profile)
    campaign, initial, suite, quota = setup(tmp_path)
    with pytest.raises(ValueError, match='native score range'):
        campaign.run(initial)
    assert suite.resets == []
    assert quota.status()['reserved_trials'] == 0
    assert not (campaign.state.root / 'current.json').exists()
