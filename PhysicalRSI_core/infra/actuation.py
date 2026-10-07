"""Device-side actuation fencing and an independently polled command watchdog.

Run this boundary with the actuator backend, separately from a policy/controller
process. Every path to the actuator must pass through it. This Python reference
implementation requires bounded backend calls; hardware still needs a watchdog
that survives loss of this process/host and independently qualified stop logic.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import fcntl
import os
from pathlib import Path
from threading import RLock
from time import monotonic
import uuid

from ..contracts import ReconciliationRequired
from ..timing import SensorSample, finite_seconds
from .control_authority import ControlAuthority, ControlClaimInvalid, claim_reference
from .rpc.journal import RequestJournal
from .storage import atomic_json, digest, file_digest, identifier, read_json


def local_clock_domain():
    """Identify a Linux monotonic clock without comparing host wall clocks."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    namespace = os.readlink("/proc/self/ns/time") if Path("/proc/self/ns/time").exists() else "initial"
    return "linux-boot:" + boot + ":" + namespace


@dataclass(frozen=True)
class ActuationLimits:
    lease_seconds: float = 3
    max_command_seconds: float = 1
    max_poll_gap_seconds: float = .2
    max_actions: int = 64

    def __post_init__(self):
        for value in (self.lease_seconds, self.max_command_seconds, self.max_poll_gap_seconds):
            finite_seconds(value)
        if self.max_command_seconds >= self.lease_seconds:
            raise ValueError("Command horizon must fit within an actuation lease")
        if type(self.max_actions) is not int or not 1 <= self.max_actions <= 4096:
            raise ValueError("Declare a bounded actuator command size")


class ActuationGuard:
    """One durable authority immediately before actuator writes.

    Backend methods: identity(), validate(actions), reset(case), observe(),
    apply(action), stop(), quiescence(), and tick(elapsed_seconds). All must be
    bounded and serialize here. stop() inhibits actuation; quiescence() measures
    whether motion has settled. Returning from stop is never proof of settling.
    observe() returns a newly captured SensorSample using this host's clock.
    tick() advances a simulator or services device I/O; it never renews ownership.
    """

    def __init__(self, root, *, backend, resources, limits=None, clock=monotonic, clock_domain=None):
        if not resources or len(set(resources)) != len(resources):
            raise ValueError("Declare unique actuator resources")
        if clock_domain is not None and (not isinstance(clock_domain, str) or not clock_domain):
            raise ValueError("An explicit actuator clock domain must be a nonempty string")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend, self.limits, self.clock = backend, limits or ActuationLimits(), clock
        self._lock, self._closed = RLock(), False
        self._owner = (self.root / ".owner.lock").open("a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._owner.close()
            raise RuntimeError("Another actuation backend owns this directory") from None
        try:
            self.authority = ControlAuthority(self.root / "authority.json", resources=resources,
                lease_seconds=self.limits.lease_seconds, boot_id=uuid.uuid4().hex, clock=clock)
            self.journal = RequestJournal(self.root / "requests")
            self._identity = dict(schema="physicalrsi.actuation-backend/v1", backend=backend.identity(),
                authority_id=self.authority.inspect()["authority_id"], resources=list(resources),
                limits=asdict(self.limits), clock_domain=clock_domain or local_clock_domain(), sources={
                    name: file_digest(Path(__file__).parent / name) for name in
                    ("actuation.py", "actuation_rpc.py", "clock_sync.py", "control_authority.py", "operator.py", "rpc/journal.py",
                     "rpc/rpc_facade.py", "rpc/authenticated_http.py", "rpc/tls.py", "rpc/http_rpc.py", "storage.py")})
            self._active, self._claim = None, None
            self._last_poll = finite_seconds(clock(), positive=False)
            self._fault = None
            # Startup inhibits output before accepting anything, including when
            # an earlier process died between actuator I/O and its receipt.
            backend.stop()
            self._state = dict(phase="inhibited", active_command=None, stop_reason="backend_startup")
            self._save()
        except BaseException:
            self._owner.close()
            raise

    def identity(self):
        return deepcopy(self._identity)

    def _save(self):
        atomic_json(self.root / "state.json", self._state)

    def _stable(self):
        if self._closed:
            raise RuntimeError("Actuation backend is closed")
        if self.backend.identity() != self._identity["backend"]:
            raise ReconciliationRequired("Actuator backend identity changed")
        if self._fault:
            raise ReconciliationRequired("Actuator polling/stop failed; local recovery is required")

    def _status_path(self, request_id):
        return self.root / "commands" / (identifier(request_id) + ".json")

    def status(self, request_id):
        with self._lock:
            path = self._status_path(request_id)
            return read_json(path) if path.exists() else self.journal.status(request_id)

    def inspect(self):
        with self._lock:
            return dict(identity=self.identity(), state=deepcopy(self._state),
                        ownership=self.authority.inspect(), quiescence=deepcopy(self.backend.quiescence()),
                        poll_fault=self._fault, last_poll_at=self._last_poll)

    def _stop(self, reason, *, completed=False):
        # Stopping output precedes evidence I/O. The durable active command and
        # ownership already exist; a crash here remains uncertain on restart.
        self.backend.stop()
        active, self._active = self._active, None
        self._state = dict(phase="inhibited", active_command=None, stop_reason=reason)
        if active:
            if not completed:
                self.authority.revoke(digest(dict(interrupted=active["request_id"])), self._claim, reason)
            receipt = dict(state="completed" if completed else "interrupted", request_id=active["request_id"],
                generation=active["generation"], applied_steps=active["index"] + 1,
                expected_steps=len(active["actions"]), reason=reason, stopped_at=self.clock(),
                quiescent=self.backend.quiescence().get("quiescent") is True, deadline=active["deadline"])
            if active.get("timing_evidence") is not None:
                receipt["timing_evidence"] = active["timing_evidence"]
            atomic_json(self._status_path(active["request_id"]), receipt)
        self._save()

    def poll(self):
        """Called by the device process even with no clients or RPC traffic."""
        with self._lock:
            if self._closed:
                return
            now = finite_seconds(self.clock(), positive=False)
            elapsed = now - self._last_poll
            self._last_poll = now
            try:
                self._stable()
                if elapsed < 0:
                    raise ReconciliationRequired("Actuator clock moved backwards")
                owner = self.authority.inspect()
                if self._active:
                    command = self._active
                    if owner["phase"] != "held" or elapsed > self.limits.max_poll_gap_seconds:
                        self._stop("ownership_expired_or_polling_late")
                    elif now >= command["deadline"]:
                        self._stop("command_deadline")
                    elif now >= command["ends_at"]:
                        complete = command["index"] == len(command["actions"]) - 1
                        self._stop("bounded_command_finished" if complete else "missed_actuation_tick", completed=complete)
                    else:
                        index = int((now - command["started_at"]) / command["period_seconds"])
                        if index > command["index"] + 1:
                            self._stop("missed_actuation_tick")
                        elif index > command["index"]:
                            self.authority.check(self._claim)
                            self.backend.apply(deepcopy(command["actions"][index]))
                            command["index"] = index
                # Advance only after enforcing the watchdog. A late poll never
                # simulates additional time under a now-expired effort value.
                self.backend.tick(elapsed)
            except BaseException as error:
                self._fault = type(error).__name__
                try:
                    self._stop("backend_fault")
                finally:
                    raise

    def _request(self, request_id, operation, inputs, function, *, claim=None, revision="", request_deadline=None,
                 recovery=False):
        with self._lock:
            # A closed object must not write even an intent after its process
            # lock has been handed to a replacement backend.
            if recovery:
                if self._closed:
                    raise RuntimeError("Actuation backend is closed")
            else:
                self._stable()
            binding = dict(operation=operation, inputs=inputs, claim=claim_reference(claim),
                           harness_revision=revision, identity=self.identity())
            intent = self.root / "intents" / (identifier(request_id) + ".json")
            if intent.exists():
                if read_json(intent) != binding:
                    raise ValueError("Actuation request identity reused with different inputs")
            else:
                atomic_json(intent, binding)
            # The request journal also serializes duplicate effects on replay.
            def admitted():
                if request_deadline is not None and self.clock() >= finite_seconds(request_deadline):
                    raise TimeoutError("Actuator request expired before dispatch")
                result = function()
                if request_deadline is not None and self.clock() >= request_deadline:
                    raise TimeoutError("Actuator request completed after its deadline")
                return result
            try:
                return self.journal.execute(request_id, binding, admitted)
            except BaseException:
                # A stale/foreign request must never stop the current owner.
                try:
                    self.authority.check(claim, live=False)
                except (ControlClaimInvalid, ValueError):
                    pass
                else:
                    self._stop("actuator_request_failed")
                    self.authority.revoke(digest(dict(failed=request_id)), claim, "actuator_request_failed")
                raise

    def acquire(self, request_id, *, owner, credential, harness_revision, request_deadline=None):
        from .control_authority import credential_digest
        def perform():
            self.poll()
            return self.authority.acquire(request_id, owner=owner, credential=credential,
                harness_revision=harness_revision, controller_idle=self.backend.quiescence().get("quiescent") is True
                and self._active is None)
        return self._request(request_id, "acquire", dict(owner=owner, credential_sha256=credential_digest(credential)),
                             perform, revision=harness_revision, request_deadline=request_deadline)

    def renew(self, request_id, *, claim, request_deadline=None):
        return self._request(request_id, "renew", {}, lambda: self.authority.renew(request_id, claim),
                             claim=claim, request_deadline=request_deadline)

    def reset(self, request_id, *, case, claim, harness_revision, request_deadline=None):
        def perform():
            self.authority.check(claim, harness_revision=harness_revision)
            if self._active or self.backend.quiescence().get("quiescent") is not True:
                raise ReconciliationRequired("Actuator reset requires measured quiescence")
            result = self.backend.reset(deepcopy(case))
            self.authority.check(claim, harness_revision=harness_revision)
            return result
        return self._request(request_id, "reset", dict(case=case), perform, claim=claim,
                             revision=harness_revision, request_deadline=request_deadline)

    def observe(self, *, claim, harness_revision, request_deadline=None):
        with self._lock:
            self._stable()
            if request_deadline is not None and self.clock() >= finite_seconds(request_deadline):
                raise TimeoutError("Actuator observation request expired")
            self.authority.check(claim, harness_revision=harness_revision)
            started = self.clock()
            sample = self.backend.observe()
            finished = self.clock()
            if (not isinstance(sample, SensorSample) or not
                    started <= sample.captured_at <= sample.captured_at + sample.capture_uncertainty_seconds <= finished):
                raise ReconciliationRequired("Actuator sensor sample was not freshly captured during this observation")
            self.authority.check(claim, harness_revision=harness_revision)
            if request_deadline is not None and finished >= request_deadline:
                raise TimeoutError("Actuator observation completed after its deadline")
            return asdict(sample)

    def submit(self, request_id, *, actions, period_seconds, deadline, claim, harness_revision, request_deadline=None,
               timing_evidence=None):
        actions = deepcopy(actions)
        def perform():
            self.poll()
            owner = self.authority.check(claim, harness_revision=harness_revision)
            period = finite_seconds(period_seconds)
            until = finite_seconds(deadline)
            now = self.clock()
            if (not isinstance(actions, list) or not 1 <= len(actions) <= self.limits.max_actions
                    or len(actions) * period > self.limits.max_command_seconds
                    or until - now > self.limits.max_command_seconds
                    or now + len(actions) * period >= min(until, owner["expires_at"])):
                raise ValueError("Actuation command does not fit its live deadline/lease/horizon")
            if self._active:
                raise ReconciliationRequired("An actuator command is already in flight")
            self.backend.validate(actions)
            self._claim = deepcopy(claim)
            atomic_json(self._status_path(request_id), dict(state="started", request_id=request_id,
                generation=owner["generation"], expected_steps=len(actions), deadline=until))
            self._state = dict(phase="executing", active_command=request_id, generation=owner["generation"])
            self._save()
            # Schedule against the actual dispatch time, without extending the
            # caller's absolute deadline or the independently expiring lease.
            now = self.clock()
            if now + len(actions) * period >= min(until, owner["expires_at"], request_deadline or until):
                raise ReconciliationRequired("Actuation admission exceeded the available horizon")
            self._active = dict(request_id=request_id, generation=owner["generation"], actions=actions,
                index=-1, period_seconds=period, started_at=now, ends_at=now + len(actions) * period, deadline=until,
                timing_evidence=deepcopy(timing_evidence))
            self.authority.check(claim, harness_revision=harness_revision)
            self.backend.apply(deepcopy(actions[0]))
            self._active["index"] = 0
            return dict(state="accepted", request_id=request_id, generation=owner["generation"])
        timing = {} if timing_evidence is None else dict(timing_evidence=deepcopy(timing_evidence))
        return self._request(request_id, "submit", dict(actions=actions, period_seconds=period_seconds, deadline=deadline, **timing),
                             perform, claim=claim, revision=harness_revision, request_deadline=request_deadline)

    def stop(self, request_id, *, claim, request_deadline=None):
        def perform():
            self.authority.check(claim, live=False)
            self._stop("owner_stop")
            return dict(state="completed", output_inhibited=True, quiescence=self.backend.quiescence())
        return self._request(request_id, "stop", {}, perform, claim=claim, request_deadline=request_deadline)

    def _release_verified(self, request_id, generation, observed):
        # Keep the exact measurement whose digest authorizes release, including
        # when the response or the outer request receipt is subsequently lost.
        measurement = dict(deepcopy(observed), observed_at=self.clock())
        atomic_json(self.root / "quiescence" / (identifier(request_id) + ".json"), measurement)
        result = self.authority.release(request_id, generation=generation, evidence=measurement)
        return dict(result, measurement=measurement)

    def release(self, request_id, *, claim, request_deadline=None):
        def perform():
            owner = self.authority.check(claim, live=False)
            if self._active:
                raise ReconciliationRequired("Stop the active actuator command before release")
            evidence = self.backend.quiescence()
            if evidence.get("quiescent") is not True:
                raise ReconciliationRequired("Actuator motion has not independently settled")
            return self._release_verified(request_id, owner["generation"], evidence)
        return self._request(request_id, "release", {}, perform, claim=claim, request_deadline=request_deadline)

    def recover(self, request_id, *, generation, operator, reason, evidence, request_deadline=None):
        """Local operator only: fence the exact owner, stop, then verify settling."""
        inputs = dict(generation=generation, operator=operator, reason=reason, evidence=deepcopy(evidence))
        def perform():
            self.authority.quarantine(digest(dict(recovery=request_id)), generation=generation,
                operator=operator, reason=reason, evidence=evidence)
            if self.authority.inspect()["generation"] != generation:
                raise ControlClaimInvalid("Recovery refers to a different actuator generation")
            self._stop("operator_recovery")
            observed = self.backend.quiescence()
            if observed.get("quiescent") is not True:
                raise ReconciliationRequired("Recovery has inhibited output; motion has not settled")
            result = self._release_verified(request_id, generation, observed)
            self._fault = None
            return result
        return self._request(request_id, "recover", inputs, perform, recovery=True, request_deadline=request_deadline)

    def close(self):
        with self._lock:
            if not self._closed:
                self._stop("backend_shutdown")
                self._closed = True
                self._owner.close()
