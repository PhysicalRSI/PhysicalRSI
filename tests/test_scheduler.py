import multiprocessing
import os
from pathlib import Path
from time import monotonic, sleep

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.scheduler import JobBusy, JobScheduler
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json


class FileHandler:
    def __init__(self, root):
        self.root = Path(root)

    def identity(self):
        return dict(kind="scheduler-test-handler/v1", root=str(self.root))

    def execute(self, payload):
        folder = self.root / payload["name"]
        with locked(folder / ".lock"):
            path = folder / "calls.json"
            atomic_json(path, read_json(path) + 1 if path.exists() else 1)
        if payload.get("mode") == "hold":
            atomic_json(folder / "ready.json", True)
            until = monotonic() + 10
            while not (folder / "release.json").exists():
                if monotonic() > until:
                    raise TimeoutError("test worker release missing")
                sleep(.01)
        if payload.get("mode") == "die-partial":
            os._exit(27)
        value = dict(input_sha256=digest(payload), value=payload["name"])
        atomic_json(folder / "result.json", value)
        if payload.get("mode") == "die-complete":
            os._exit(28)
        return value

    def collect(self, payload):
        value = read_json(self.root / payload["name"] / "result.json")
        if value != dict(input_sha256=digest(payload), value=payload["name"]):
            raise ValueError("Native evidence changed")
        return value


def worker(root, capacity, evidence, ids):
    JobScheduler(root, capacities={"cpu": capacity}).work_once(FileHandler(evidence), ids=ids)


def start(root, scheduler, handler, ids):
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=worker, args=(scheduler.root, scheduler.capacities["cpu"], handler.root, ids))
    process.start()
    return process


def test_dead_worker_keeps_capacity_and_never_replays_unknown_effect(tmp_path):
    scheduler, handler = JobScheduler(tmp_path / "queue", capacities={"cpu": 1}), FileHandler(tmp_path / "effects")
    ids = scheduler.submit([dict(name="first", mode="die-partial")], handler=handler, resources={"cpu": 1})
    process = start(tmp_path, scheduler, handler, ids)
    process.join(10)
    assert process.exitcode == 27
    status = scheduler.status()
    assert status["used"] == {"cpu": 1} and not status["jobs"][ids[0]]["owner_active"]
    with pytest.raises(ReconciliationRequired):
        scheduler.run_until(ids, handler)
    later = scheduler.submit([dict(name="later")], handler=handler, resources={"cpu": 1})
    assert scheduler.work_once(handler) is None
    proof = tmp_path / "quiescence.json"
    atomic_json(proof, dict(exitcode=process.exitcode, alive=process.is_alive()))
    scheduler.abandon(ids[0], operator="test", reason="Local child exit independently observed",
                      evidence={str(proof): file_digest(proof)})
    assert scheduler.run_until(later, handler)[0]["value"] == "later"
    assert read_json(handler.root / "first/calls.json") == 1
    with pytest.raises(ReconciliationRequired, match="abandoned"):
        scheduler.run_until(ids, handler)


def test_complete_native_evidence_recovers_after_worker_exit_without_reexecution(tmp_path):
    scheduler, handler = JobScheduler(tmp_path / "queue", capacities={"cpu": 1}), FileHandler(tmp_path / "effects")
    payloads = [dict(name="finished", mode="die-complete")]
    ids = scheduler.submit(payloads, handler=handler, resources={"cpu": 1})
    process = start(tmp_path, scheduler, handler, ids)
    process.join(10)
    assert process.exitcode == 28
    reopened = JobScheduler(scheduler.root, capacities={"cpu": 1})
    assert reopened.submit(payloads, handler=handler, resources={"cpu": 1}) == ids
    assert reopened.run_until(ids, handler)[0]["value"] == "finished"
    assert reopened.status()["used"] == {"cpu": 0}
    assert reopened.status()["jobs"][ids[0]]["recovered"]
    assert read_json(handler.root / "finished/calls.json") == 1
    atomic_json(handler.root / "finished/result.json", {"changed": True})
    with pytest.raises(ValueError, match="evidence changed"):
        reopened.run_until(ids, handler)
    pending = reopened.submit([dict(name="not-started")], handler=handler, resources={"cpu": 1})
    with pytest.raises(ValueError, match="evidence changed"):
        reopened.run_until([*pending, *ids], handler)
    assert not (handler.root / "not-started/calls.json").exists()


def test_live_owner_cannot_be_replaced_and_fitting_job_bypasses_blocked_request(tmp_path):
    scheduler, handler = JobScheduler(tmp_path / "queue", capacities={"cpu": 2}), FileHandler(tmp_path / "effects")
    first = scheduler.submit([dict(name="held", mode="hold")], handler=handler, resources={"cpu": 1})
    process = start(tmp_path, scheduler, handler, first)
    try:
        until = monotonic() + 8
        while not (handler.root / "held/ready.json").exists():
            assert process.is_alive() and monotonic() < until
            sleep(.01)
        assert scheduler.status()["jobs"][first[0]]["owner_active"]
        with pytest.raises(JobBusy):
            scheduler.recover_completed(first[0], handler)
        proof = handler.root / "held/ready.json"
        with pytest.raises(JobBusy):
            scheduler.abandon(first[0], operator="test", reason="not a stop proof", evidence={str(proof): file_digest(proof)})
        with pytest.raises(TimeoutError, match="existing jobs"):
            scheduler.run_until(first, handler, wait_seconds=.04)
        blocked = scheduler.submit([dict(name="large")], handler=handler, resources={"cpu": 2}, priority=10)
        fitting = scheduler.submit([dict(name="small")], handler=handler, resources={"cpu": 1})
        assert scheduler.work_once(handler) == fitting[0]
        assert scheduler.status()["jobs"][blocked[0]]["state"] == "queued"
        assert scheduler.status()["used"] == {"cpu": 1}
        atomic_json(handler.root / "held/release.json", True)
        process.join(10)
        assert process.exitcode == 0
        assert scheduler.run_until(blocked, handler)[0]["value"] == "large"
        assert scheduler.status()["used"] == {"cpu": 0}
        assert read_json(handler.root / "held/calls.json") == 1
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_impossible_resources_and_configuration_changes_precede_dispatch(tmp_path):
    scheduler, handler = JobScheduler(tmp_path / "queue", capacities={"cpu": 1}), FileHandler(tmp_path / "effects")
    with pytest.raises(ValueError, match="capacity"):
        scheduler.submit([dict(name="never")], handler=handler, resources={"cpu": 2})
    with pytest.raises(ValueError, match="configuration"):
        JobScheduler(scheduler.root, capacities={"cpu": 2})
    assert not handler.root.exists() and scheduler.status()["jobs"] == {}
