"""Durable bounded jobs for cooperating processes on one POSIX host.

Capacity is configured, never inferred. An owner lock proves local liveness;
losing that lock does not release a durable reservation or authorize a retry.
Handlers must provide a read-only collect() that verifies completed evidence.
This is a controller queue, not process isolation or a distributed GPU service.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
from time import monotonic, sleep

from ..contracts import ReconciliationRequired
from .storage import atomic_json, canonical, digest, file_digest, locked, read_json


def _json(value):
    """Own one JSON snapshot, independent of mutable provider return values."""
    return json.loads(canonical(value))


class JobBusy(RuntimeError):
    pass


class JobScheduler:
    def __init__(self, root, *, capacities):
        if (not isinstance(capacities, dict) or not capacities
                or any(not isinstance(k, str) or not k or type(v) is not int or v < 1
                       for k, v in capacities.items())):
            raise ValueError("Declare positive integer resource capacities")
        self.root, self.capacities = Path(root).resolve(), deepcopy(capacities)
        with locked(self.root / ".lock"):
            path = self.root / "queue.json"
            if not path.exists():
                atomic_json(path, dict(identity=self.identity(), jobs={}))
            self._read()

    def identity(self):
        return dict(schema="physicalrsi.job-scheduler/v1", root=str(self.root),
                    capacities=self.capacities, implementation=file_digest(Path(__file__)))

    def _read(self):
        value = read_json(self.root / "queue.json")
        if digest(value["identity"]) != digest(self.identity()):
            raise ValueError("Frozen scheduler configuration changed")
        for key, job in value["jobs"].items():
            if key != digest(job["request"]) or job["state"] not in {
                    "queued", "running", "needs_reconciliation", "completed", "abandoned"}:
                raise ValueError("Scheduler job identity or state changed")
            if job["state"] == "completed" and digest(job["result"]) != job["result_sha256"]:
                raise ValueError("Scheduled result changed")
            for row in job.get('resumes', []):
                if row.get('revision') != digest({k: v for k, v in row.items() if k != 'revision'}):
                    raise ValueError('Scheduled replacement history changed')
        return value

    def _save(self, ledger):
        atomic_json(self.root / "queue.json", ledger)

    def _owner(self, job_id):
        folder = self.root / "owners"
        folder.mkdir(exist_ok=True)
        stream = (folder / (job_id + ".lock")).open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return None
        return stream

    def _used(self, ledger):
        used = dict.fromkeys(self.capacities, 0)
        for job in ledger["jobs"].values():
            if job["state"] in {"running", "needs_reconciliation"}:
                for name, amount in job["request"]["resources"].items():
                    used[name] += amount
        return used

    def submit(self, payloads, *, handler, resources, priority=0):
        """Idempotently enqueue an entire input set; no execution occurs here."""
        resources = deepcopy(resources)
        if (not isinstance(resources, dict) or not resources or type(priority) is not int
                or any(k not in self.capacities or type(v) is not int or not 1 <= v <= self.capacities[k]
                       for k, v in resources.items())):
            raise ValueError("Job resource request exceeds configured capacity")
        handler_identity = _json(handler.identity())
        requests = [_json(dict(payload=p, handler=handler_identity,
                         resources=resources, priority=priority)) for p in payloads]
        ids = [digest(request) for request in requests]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Submit a nonempty set of distinct jobs")
        with locked(self.root / ".lock"):
            ledger = self._read()
            self._handler(dict(handler=handler_identity), handler)
            for key, request in zip(ids, requests):
                if key not in ledger["jobs"]:
                    ledger["jobs"][key] = dict(request=request, state="queued", submitted=len(ledger["jobs"]))
            self._save(ledger)
        return ids

    def status(self, ids=None):
        with locked(self.root / ".lock"):
            ledger = self._read()
            selected = ledger["jobs"] if ids is None else {key: ledger["jobs"][key] for key in ids}
            jobs = deepcopy(selected)
            for key, job in jobs.items():
                owner = self._owner(key)
                job["owner_active"] = owner is None
                if owner is not None:
                    owner.close()
            return dict(jobs=jobs, used=self._used(ledger), capacities=deepcopy(self.capacities))

    def _claim(self, handler, ids=None, *, deadline=None):
        with locked(self.root / ".lock"):
            if deadline is not None and monotonic() >= deadline:
                return None
            ledger = self._read()
            allowed = set(ledger["jobs"] if ids is None else ids)
            used = self._used(ledger)
            jobs = sorted(ledger["jobs"].items(), key=lambda item: (
                -item[1]["request"]["priority"], item[1]["submitted"]))
            handler_sha = None
            for key, job in jobs:
                request = job["request"]
                if key not in allowed or job["state"] != "queued":
                    continue
                if handler_sha is None:
                    handler_sha = digest(_json(handler.identity()))
                if digest(request["handler"]) != handler_sha:
                    continue
                if any(used[k] + v > self.capacities[k] for k, v in request["resources"].items()):
                    continue
                owner = self._owner(key)
                if owner is None:
                    continue
                try:
                    # Lock acquisition and queue inspection can consume the
                    # remaining observation budget. Do not reserve new work.
                    if deadline is not None and monotonic() >= deadline:
                        owner.close()
                        return None
                    job.update(state="running")
                    self._save(ledger)
                    return key, deepcopy(request), owner
                except BaseException:
                    owner.close()
                    raise
        return None

    @staticmethod
    def _handler(request, handler):
        if digest(request["handler"]) != digest(_json(handler.identity())):
            raise ValueError("Scheduled handler identity changed")

    def _invoke(self, request, handler, method):
        self._handler(request, handler)
        result = _json(getattr(handler, method)(_json(request["payload"])))
        self._handler(request, handler)
        return result

    def _complete(self, key, result, *, recovered=False):
        result = _json(result)
        with locked(self.root / ".lock"):
            ledger = self._read()
            job = ledger["jobs"][key]
            if job["state"] not in {"running", "needs_reconciliation"}:
                raise ValueError("Job is not awaiting completion")
            job.update(state="completed", result=result, result_sha256=digest(result),
                       recovered=recovered)
            self._save(ledger)

    def _execute(self, claim, handler):
        key, request, owner = claim
        try:
            result = self._invoke(request, handler, "execute")
            verified = self._invoke(request, handler, "collect")
            if digest(result) != digest(verified):
                raise ValueError("Worker output differs from verified evidence")
            self._complete(key, verified)
            return key
        except BaseException as error:
            with locked(self.root / ".lock"):
                ledger = self._read()
                if ledger["jobs"][key]["state"] != "completed":
                    ledger["jobs"][key].update(state="needs_reconciliation", error=type(error).__name__ + ": " + str(error))
                    self._save(ledger)
            raise
        finally:
            owner.close()

    def work_once(self, handler, *, ids=None):
        """Execute one fitting job; usable by independent local worker processes."""
        claim = self._claim(handler, ids)
        return None if claim is None else self._execute(claim, handler)

    def recover_completed(self, key, handler):
        """Collect existing evidence from an inactive job, without executing it."""
        with locked(self.root / ".lock"):
            job = self._read()["jobs"][key]
            self._handler(job["request"], handler)
            owner = self._owner(key)
            if owner is None:
                raise JobBusy("Cannot reconcile a live job owner")
        try:
            if job["state"] == "completed":
                result = self._invoke(job["request"], handler, "collect")
                if digest(result) != job["result_sha256"]:
                    raise ValueError("Completed scheduled evidence changed")
                return result
            if job["state"] not in {"running", "needs_reconciliation"}:
                raise ValueError("Only uncertain jobs require recovery")
            try:
                result = self._invoke(job["request"], handler, "collect")
            except Exception as error:
                raise ReconciliationRequired("Job has no verified complete result; inspect " + key) from error
            self._complete(key, result, recovered=True)
            return result
        finally:
            owner.close()

    def abandon(self, key, *, operator, reason, evidence):
        """Record independently established quiescence; never retry the input.

        Evidence maps physical file paths to hashes. DeviceRegistry claims, if
        any, require their own reconciliation; this releases queue capacity only.
        """
        if not operator or not reason or not isinstance(evidence, dict) or not evidence:
            raise ValueError("Abandonment requires operator, reason and quiescence evidence")
        for path, sha in evidence.items():
            if file_digest(Path(path)) != sha:
                raise ValueError("Quiescence evidence changed")
        with locked(self.root / ".lock"):
            ledger = self._read()
            job = ledger["jobs"][key]
            owner = self._owner(key)
            if owner is None:
                raise JobBusy("Cannot abandon a live job owner")
            try:
                if job["state"] not in {"running", "needs_reconciliation"}:
                    raise ValueError("Only uncertain jobs may be abandoned")
                job.update(state="abandoned", reconciliation=dict(operator=operator, reason=reason, evidence=evidence))
                self._save(ledger)
            finally:
                owner.close()

    def resume_replacements(self, key, handler, *, operator, reason, evidence):
        """Queue explicitly authorized new Trial identities, never replay unknown work.

        Only handlers exposing a read-only replacement_plan() can participate.
        The old dispatch error and its stopping evidence remain in the job's
        resume history. Completed and abandoned jobs cannot be reexecuted.
        """
        if (not isinstance(operator, str) or not operator.strip() or not isinstance(reason, str)
                or not reason.strip() or not isinstance(evidence, dict) or not evidence
                or not callable(getattr(handler, 'replacement_plan', None))):
            raise ValueError('Replacement resume requires a verifier, operator, reason and evidence')
        for path, sha in evidence.items():
            if file_digest(Path(path)) != sha:
                raise ValueError('Replacement resume evidence changed')
        with locked(self.root / '.lock'):
            ledger = self._read()
            job = ledger['jobs'][key]
            self._handler(job['request'], handler)
            owner = self._owner(key)
            if owner is None:
                raise JobBusy('Cannot resume a live job owner')
            try:
                history = job.get('resumes', [])
                if job['state'] in {'queued', 'completed'} and history:
                    previous = history[-1]
                    if all(previous[k] == v for k, v in dict(operator=operator, reason=reason, evidence=evidence).items()):
                        return deepcopy(previous)
                if job['state'] != 'needs_reconciliation':
                    raise ValueError('Only an interrupted dispatch may resume replacements')
                proof = self._invoke(job['request'], handler, 'replacement_plan')
                if (proof.get('schema') != 'physicalrsi.scheduled-replacements/v1'
                        or not isinstance(proof.get('authorizations'), list)
                        or not proof['authorizations'] or not proof.get('pending')):
                    raise ReconciliationRequired('No authorized replacement work; collect completed evidence instead')
                used = {sha for row in history for sha in row['proof']['authorizations']}
                if not set(proof['authorizations']) - used:
                    raise ReconciliationRequired('No new replacement authorization since the previous dispatch')
                row = dict(previous_state=job['state'], error=job.get('error'), operator=operator,
                           reason=reason, evidence=_json(evidence), proof=proof)
                job['resumes'] = [*history, dict(row, revision=digest(row))]
                job.update(state='queued')
                job.pop('error', None)
                self._save(ledger)
                return deepcopy(job['resumes'][-1])
            finally:
                owner.close()

    def run_until(self, ids, handler, *, workers=1, wait_seconds=60):
        """Run independent jobs concurrently and collect in requested order.

        The observation deadline stops new dispatch, but owned workers are joined
        before returning. Adapter execution budgets must bound their effects.
        External live owners are observed, never replaced on timeout.
        """
        ids = list(ids)
        if (not ids or len(set(ids)) != len(ids) or type(workers) is not int or workers < 1
                or not math.isfinite(wait_seconds) or wait_seconds <= 0):
            raise ValueError("Declare jobs, worker count and a finite observation budget")
        deadline, futures, checked = monotonic() + wait_seconds, {}, set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                view = self.status(ids)["jobs"]
                handler_sha = digest(_json(handler.identity()))
                for key, job in view.items():
                    if digest(job["request"]["handler"]) != handler_sha:
                        raise ValueError("Scheduled handler identity changed")
                    if job["state"] == "abandoned":
                        raise ReconciliationRequired("Scheduled job was abandoned: " + key)
                    if job["state"] == "completed" and key not in checked:
                        if digest(self._invoke(job["request"], handler, "collect")) != job["result_sha256"]:
                            raise ValueError("Completed scheduled evidence changed")
                        checked.add(key)
                    # A local worker can release its owner lock between polls
                    # before its Future is consumed. Preserve its exception;
                    # recovery is for jobs without an outstanding local Future.
                    if (job["state"] in {"running", "needs_reconciliation"}
                            and not job["owner_active"] and key not in futures.values()):
                        try:
                            self.recover_completed(key, handler)
                        except JobBusy:
                            pass  # Another collector proved ownership first.
                if not futures and all(job["state"] == "completed" for job in view.values()):
                    results = []
                    for key in ids:
                        job = view[key]
                        verified = self._invoke(job["request"], handler, "collect")
                        if digest(verified) != job["result_sha256"]:
                            raise ValueError("Completed scheduled evidence changed")
                        results.append(verified)
                    return results
                if monotonic() >= deadline:
                    raise TimeoutError("Observation deadline reached; inspect the existing jobs")
                while len(futures) < workers:
                    claim = self._claim(handler, ids, deadline=deadline)
                    if claim is None:
                        break
                    try:
                        futures[pool.submit(self._execute, claim, handler)] = claim[0]
                    except BaseException:
                        # submit() may enqueue work before failing to start a
                        # thread. Keep ownership until all possible workers stop.
                        pool.shutdown(wait=True)
                        claim[2].close()
                        raise
                if futures:
                    done, _ = wait(futures, timeout=.02, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        future.result()
                else:
                    sleep(.02)
