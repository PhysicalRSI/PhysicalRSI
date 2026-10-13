from copy import deepcopy
from pathlib import Path
from threading import Event, Lock

import pytest

from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import atomic_json, read_json
from PhysicalRSI_core.lineage import StateConflict
from PhysicalRSI_core.self_harness.scheduled import ScheduledEvaluator
from PhysicalRSI_demos.rollout_evolution import setup


def test_out_of_order_candidates_are_paired_and_restart_does_not_rerun(tmp_path, monkeypatch):
    campaign, initial, suite, quota = setup(tmp_path, scheduled_workers=2)
    execute = ScheduledEvaluator.execute
    entered, completed = [], []
    lock, both, child_done = Lock(), Event(), Event()

    def reordered(self, payload):
        name = payload["candidate"]["id"]
        with lock:
            entered.append(name)
            if len(entered) == 2:
                assert self.scheduler.status()["used"] == {"cpu": 2}
                both.set()
        assert both.wait(10)
        if name == initial["id"]:
            assert child_done.wait(10)
        value = execute(self, payload)
        completed.append(name)
        if name != initial["id"]:
            child_done.set()
        return value

    monkeypatch.setattr(ScheduledEvaluator, "execute", reordered)
    result = campaign.run(initial)
    assert completed == ["calibrated-0", "uncalibrated"]
    assert [r["outcome"] for r in result["rounds"]] == ["inherit_child", "no_changed_candidate"]
    assert all(m["success_rate"] == 1 for m in campaign.report_test(initial)["metrics"].values())
    scheduler = JobScheduler(tmp_path / "scheduler", capacities={"cpu": 2})
    assert scheduler.status()["used"] == {"cpu": 0}
    assert len(scheduler.status()["jobs"]) == 2
    assert len(suite.resets) == quota.status()["reserved_trials"] == 14
    restarted, seed, new_suite, _ = setup(tmp_path, scheduled_workers=2)
    assert restarted.run(seed) == result
    restarted.report_test(seed)
    assert new_suite.resets == []


def test_completed_rollouts_recover_after_queue_publication_failure(tmp_path, monkeypatch):
    campaign, initial, suite, quota = setup(tmp_path, scheduled_workers=2)
    complete = JobScheduler._complete
    stopped = None

    def interrupt(self, key, result, **kwargs):
        nonlocal stopped
        if stopped is None:
            stopped = key
        # Keep publication unavailable for this job until the fault is removed.
        # A concurrent read-only collector must not repair a one-shot fault
        # before the test can inspect the retained reservation.
        if key == stopped:
            raise RuntimeError("completion publication interrupted")
        return complete(self, key, result, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(JobScheduler, "_complete", interrupt)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            campaign.run(initial)
    assert len(suite.resets) == 10
    assert JobScheduler(tmp_path / "scheduler", capacities={"cpu": 2}).status()["used"] == {"cpu": 1}
    result = campaign.run(initial)
    assert result["completed_rounds"] == 2
    assert len(suite.resets) == quota.status()["reserved_trials"] == 12
    assert JobScheduler(tmp_path / "scheduler", capacities={"cpu": 2}).status()["used"] == {"cpu": 0}


def test_concurrent_head_change_blocks_async_promotion(tmp_path, monkeypatch):
    campaign, initial, suite, _ = setup(tmp_path, scheduled_workers=2)
    evaluate_pool = ScheduledEvaluator.evaluate_pool
    new_head = None

    def changed(self, *args):
        nonlocal new_head
        values = evaluate_pool(self, *args)
        current = campaign.state.resolve()
        # Publish a separate lineage event during the comparison. The content
        # is unchanged; the stale parent revision must still fail CAS.
        body = deepcopy(current)
        body.pop("revision")
        body.update(previous=current["revision"], action="concurrent-maintenance")
        new_head = campaign.state._publish(body)
        return values

    monkeypatch.setattr(ScheduledEvaluator, "evaluate_pool", changed)
    with pytest.raises(StateConflict):
        campaign.run(initial)
    assert campaign.state.resolve() == new_head
    assert campaign.state.resolve()["harness"]["id"] == initial["id"]
    assert len(suite.resets) == 10
    assert not list((tmp_path / "archive/entries").glob("*/record.json"))


def test_read_only_evaluation_cannot_fill_missing_rollouts(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path, scheduled_workers=2)
    campaign.run(initial)
    directory = campaign.root / "rounds/round-0001"
    loop = campaign.build_loop(directory / "execution", read_json(directory / "context.json"))
    comparison = read_json(loop.root / "freeze.json")["output"]
    cohort = read_json(loop.root / "validation.json")["output"]
    child = read_json(loop.root / "propose.json")["output"][0]
    next((loop.root / "experiments/validation" / child["id"]).glob("trials/*/receipt.json")).unlink()
    before = len(suite.resets)
    with pytest.raises(FileNotFoundError):
        loop.evaluator.read_evaluation(child, comparison, cohort, loop.root)
    assert len(suite.resets) == before == 12
