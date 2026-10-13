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


def test_local_worker_failure_between_polls_preserves_original_error(tmp_path, monkeypatch):
    import PhysicalRSI_core.infra.scheduler as module

    class FailingHandler(FileHandler):
        def execute(self, payload):
            raise RuntimeError('specific startup failure')

    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    handler = FailingHandler(tmp_path / 'effects')
    ids = scheduler.submit([dict(name='failed')], handler=handler, resources={'cpu': 1})
    real_wait = module.wait
    delayed = False

    def finish_after_observation(futures, **options):
        nonlocal delayed
        if not delayed:
            delayed = True
            # Model a worker finishing just after a timed wait reported no
            # completion, before the next durable-state scan.
            real_wait(futures)
            return set(), set(futures)
        return real_wait(futures, **options)

    monkeypatch.setattr(module, 'wait', finish_after_observation)
    with pytest.raises(RuntimeError, match='specific startup failure'):
        scheduler.run_until(ids, handler)
    job = scheduler.status()['jobs'][ids[0]]
    assert job['state'] == 'needs_reconciliation'
    assert not job['owner_active']


@pytest.mark.parametrize('value', [True, 1.0])
@pytest.mark.parametrize('entry', ['reopen', 'execute'])
def test_scheduler_rejects_stored_capacity_type_drift(tmp_path, value, entry):
    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    handler = FileHandler(tmp_path / 'effects')
    key, = scheduler.submit([dict(name='target')], handler=handler, resources={'cpu': 1})
    path = scheduler.root / 'queue.json'
    original = path.read_bytes()
    changed = read_json(path)
    changed['identity']['capacities']['cpu'] = value
    atomic_json(path, changed)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Frozen scheduler configuration changed'):
        JobScheduler(scheduler.root, capacities={'cpu': 1}) if entry == 'reopen' else scheduler.work_once(handler)
    assert path.read_bytes() == before and not handler.root.exists()
    path.write_bytes(original)
    assert scheduler.work_once(handler) == key
    assert scheduler.work_once(handler) is None
    assert read_json(handler.root / 'target/calls.json') == 1
    assert scheduler.status()['used'] == {'cpu': 0}


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


@pytest.mark.parametrize('expiry', ['reservation', 'owner'])
def test_dispatch_deadline_stops_new_effects_and_allows_explicit_resume(tmp_path, monkeypatch, expiry):
    from PhysicalRSI_core.infra import scheduler as module
    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 2})
    handler = FileHandler(tmp_path / 'effects')
    ids = scheduler.submit([dict(name='first'), dict(name='second')],
                           handler=handler, resources={'cpu': 1})
    clock = [0.]
    in_claim = [False]
    claim, save, owner = scheduler._claim, scheduler._save, scheduler._owner

    def claiming(*args, **kwargs):
        in_claim[0] = True
        try:
            return claim(*args, **kwargs)
        finally:
            in_claim[0] = False

    def slow_save(ledger):
        save(ledger)
        if in_claim[0] and expiry == 'reservation':
            clock[0] = 2.

    def slow_owner(key):
        result = owner(key)
        if in_claim[0] and expiry == 'owner':
            clock[0] = 2.
        return result

    with monkeypatch.context() as patch:
        patch.setattr(module, 'monotonic', lambda: clock[0])
        patch.setattr(scheduler, '_claim', claiming)
        patch.setattr(scheduler, '_save', slow_save)
        patch.setattr(scheduler, '_owner', slow_owner)
        with pytest.raises(TimeoutError, match='Observation deadline'):
            scheduler.run_until(ids, handler, workers=2, wait_seconds=1.)
    status = scheduler.status()
    expected = ['completed', 'queued'] if expiry == 'reservation' else ['queued', 'queued']
    assert [status['jobs'][key]['state'] for key in ids] == expected
    assert status['used'] == {'cpu': 0}
    assert not any(job['owner_active'] for job in status['jobs'].values())
    assert not (handler.root / 'second/calls.json').exists()
    if expiry == 'owner':
        assert not handler.root.exists()
    assert [row['value'] for row in scheduler.run_until(ids, handler, workers=2)] == ['first', 'second']
    assert all(read_json(handler.root / name / 'calls.json') == 1 for name in ('first', 'second'))


@pytest.mark.parametrize('accepted', [True, False])
def test_submit_error_preserves_ownership_until_executor_is_quiescent(tmp_path, monkeypatch, accepted):
    from threading import Event
    from PhysicalRSI_core.infra import scheduler as module
    started, release = Event(), Event()

    class HeldHandler(FileHandler):
        def execute(self, payload):
            started.set()
            assert release.wait(5), 'executor did not release test worker'
            return super().execute(payload)

    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    handler = HeldHandler(tmp_path / 'effects')
    ids = scheduler.submit([dict(name='job')], handler=handler, resources={'cpu': 1})
    observations = []

    class AmbiguousExecutor(module.ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            if accepted:
                super().submit(*args, **kwargs)
                assert started.wait(5), 'test worker did not start'
            raise RuntimeError('injected submission failure')

        def shutdown(self, *args, **kwargs):
            if not observations:
                observations.append(scheduler.status()['jobs'][ids[0]]['owner_active'])
            release.set()
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(module, 'ThreadPoolExecutor', AmbiguousExecutor)
    with pytest.raises(RuntimeError, match='injected submission failure'):
        scheduler.run_until(ids, handler)
    assert observations == [True]
    status = scheduler.status()
    assert not status['jobs'][ids[0]]['owner_active']
    if accepted:
        assert status['jobs'][ids[0]]['state'] == 'completed'
        assert status['used'] == {'cpu': 0}
        assert scheduler.recover_completed(ids[0], handler)['value'] == 'job'
        assert read_json(handler.root / 'job/calls.json') == 1
    else:
        assert status['jobs'][ids[0]]['state'] == 'running'
        assert status['used'] == {'cpu': 1}
        assert not handler.root.exists()
        with pytest.raises(ReconciliationRequired):
            scheduler.recover_completed(ids[0], handler)
        assert scheduler.work_once(handler) is None


@pytest.mark.parametrize('change_after', ['first', 'last'])
def test_submit_freezes_one_handler_for_the_entire_batch(tmp_path, change_after):
    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    handler = FileHandler(tmp_path / 'effects')
    existing = scheduler.submit([dict(name='existing')], handler=handler, resources={'cpu': 1})
    before = (scheduler.root / 'queue.json').read_bytes()
    original_root = handler.root

    def payloads():
        yield dict(name='first')
        if change_after == 'first':
            handler.root = tmp_path / 'different-effects'
        yield dict(name='second')
        if change_after == 'last':
            handler.root = tmp_path / 'different-effects'

    with pytest.raises(ValueError, match='handler identity changed'):
        scheduler.submit(payloads(), handler=handler, resources={'cpu': 1})
    assert (scheduler.root / 'queue.json').read_bytes() == before
    assert not original_root.exists() and not handler.root.exists()
    assert scheduler.status()['used'] == {'cpu': 0}
    handler.root = original_root
    stable = [dict(name='first'), dict(name='second')]
    ids = scheduler.submit(iter(stable), handler=handler, resources={'cpu': 1})
    assert scheduler.submit(stable, handler=handler, resources={'cpu': 1}) == ids
    assert [r['value'] for r in scheduler.run_until(ids, handler)] == ['first', 'second']
    assert all(read_json(handler.root / name / 'calls.json') == 1 for name in ('first', 'second'))
    assert scheduler.status(existing)['jobs'][existing[0]]['state'] == 'queued'
    assert scheduler.status()['used'] == {'cpu': 0}


def test_queue_scan_observes_handler_once_without_caching_across_claims(tmp_path, monkeypatch):
    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    other = FileHandler(tmp_path / 'other-effects')
    handler = FileHandler(tmp_path / 'effects')
    queued = scheduler.submit([dict(name=str(i)) for i in range(64)],
                              handler=other, resources={'cpu': 1})
    target, = scheduler.submit([dict(name='target')], handler=handler, resources={'cpu': 1})
    identity = handler.identity
    observations = []

    def observe():
        observations.append(str(handler.root))
        return identity()

    monkeypatch.setattr(handler, 'identity', observe)
    assert scheduler.work_once(handler) == target
    # One observation for scanning, plus before/after execute and collect.
    assert observations == [str(handler.root)] * 5
    assert read_json(handler.root / 'target/calls.json') == 1
    assert all(job['state'] == 'queued' for job in scheduler.status(queued)['jobs'].values())
    observations.clear()
    handler.root = other.root
    assert scheduler.work_once(handler) == queued[0]
    assert observations == [str(other.root)] * 5
    assert read_json(other.root / '0/calls.json') == 1
    assert scheduler.status()['used'] == {'cpu': 0}


def test_handler_change_after_claim_stops_execution_and_keeps_reservation(tmp_path):
    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    handler = FileHandler(tmp_path / 'effects')
    key, = scheduler.submit([dict(name='target')], handler=handler, resources={'cpu': 1})
    claim = scheduler._claim(handler)
    original = handler.root
    handler.root = tmp_path / 'changed-effects'
    with pytest.raises(ValueError, match='handler identity changed'):
        scheduler._execute(claim, handler)
    assert not original.exists() and not handler.root.exists()
    status = scheduler.status()
    assert status['jobs'][key]['state'] == 'needs_reconciliation'
    assert not status['jobs'][key]['owner_active']
    assert status['used'] == {'cpu': 1}


def test_observation_scan_checks_handler_once_for_all_waiting_jobs(tmp_path, monkeypatch):
    import PhysicalRSI_core.infra.scheduler as module
    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    handler = FileHandler(tmp_path / 'effects')
    ids = scheduler.submit([dict(name=str(i)) for i in range(64)],
                           handler=handler, resources={'cpu': 1})
    identity = handler.identity
    observations = []

    def observe():
        observations.append(True)
        return identity()

    monkeypatch.setattr(handler, 'identity', observe)
    times = iter([0., 2.])
    monkeypatch.setattr(module, 'monotonic', lambda: next(times))
    with pytest.raises(TimeoutError, match='Observation deadline'):
        scheduler.run_until(ids, handler, wait_seconds=1.)
    assert len(observations) == 1
    assert not handler.root.exists()
    assert all(row['state'] == 'queued' for row in scheduler.status(ids)['jobs'].values())


def test_observation_scan_rechecks_handler_on_next_poll(tmp_path, monkeypatch):
    scheduler = JobScheduler(tmp_path / 'queue', capacities={'cpu': 1})
    handler = FileHandler(tmp_path / 'effects')
    holder, target = scheduler.submit([dict(name='holder'), dict(name='target')],
                                      handler=handler, resources={'cpu': 1})
    claim = scheduler._claim(handler, [holder])
    status = scheduler.status
    scans = []
    original = handler.root

    def observe(ids=None):
        view = status(ids)
        scans.append(True)
        if len(scans) == 2:
            handler.root = tmp_path / 'changed-effects'
        return view

    monkeypatch.setattr(scheduler, 'status', observe)
    try:
        with pytest.raises(ValueError, match='handler identity changed'):
            scheduler.run_until([target], handler)
        assert len(scans) == 2
        assert not original.exists() and not handler.root.exists()
        assert status([target])['jobs'][target]['state'] == 'queued'
    finally:
        claim[2].close()
