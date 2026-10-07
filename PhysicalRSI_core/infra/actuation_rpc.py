"""Bounded transport from a controller driver to an independent actuator.

The actuation process owns the watchdog clock and continues polling when its
controller client stalls or dies. Hosts can require explicit mutual TLS and
method grants; credential isolation and a hardware watchdog remain deployment
responsibilities. Recovery is absent from ordinary client RPC.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import secrets
from threading import Event, Lock, Thread
from time import monotonic, sleep
import uuid

from ..contracts import ReconciliationRequired
from ..timing import SensorSample, finite_seconds
from .actuation import local_clock_domain
from .clock_sync import ClockMapping, ClockSyncPolicy, ClockUnsynchronized
from .controller import ControlRejected
from .rpc import ServiceHost
from .rpc.authenticated_http import ControlTransport
from .storage import digest, file_digest, identifier, read_json


class ActuationHost(ControlTransport, ServiceHost):
    def __init__(self, guard, *, poll_seconds=.005, operator_socket=None, tls=None):
        if not 0 < poll_seconds < guard.limits.max_poll_gap_seconds:
            raise ValueError("Poll interval must fit the declared watchdog gap")
        super().__init__(metadata=guard.identity())
        self.guard, self.poll_seconds = guard, poll_seconds
        self.configure_transport(tls, guard.root / "transport")
        self.operator_socket, self._operator = operator_socket, None
        self._poll_stop, self._poll_thread = Event(), None
        self._rpc.update({"actuation.describe": guard.identity, "actuation.inspect": guard.inspect,
                          "actuation.status": guard.status, "actuation.call": self.call,
                          "actuation.clock": self.clock_exchange, "actuation.intent": self.intent})
        self._readonly_methods.update({"actuation.describe", "actuation.inspect", "actuation.status",
                                       "actuation.clock", "actuation.intent"})

    def clock_exchange(self, nonce):
        received = self.guard.clock()
        identifier(nonce)
        return dict(schema="physicalrsi.clock-exchange/v1", nonce=nonce,
                    clock_domain=self.guard.identity()["clock_domain"],
                    authority_id=self.guard.identity()["authority_id"],
                    received_at=received, sent_at=self.guard.clock())

    def intent(self, request_id):
        # Intents contain public inputs and credential digests, never tokens.
        path = self.guard.root / "intents" / (identifier(request_id) + ".json")
        return read_json(path) if path.exists() else None

    def call(self, operation, inputs, deadline, clock_domain):
        if clock_domain != self.guard.identity()["clock_domain"]:
            raise ClockUnsynchronized("Actuator request names a different device clock domain")
        if operation not in {"acquire", "renew", "reset", "observe", "submit", "stop", "release"}:
            raise ValueError("Unknown actuation operation")
        return getattr(self.guard, operation)(**dict(inputs, request_deadline=deadline))

    def _poll(self):
        while not self._poll_stop.wait(self.poll_seconds):
            try:
                self.guard.poll()
            except Exception:
                # The guard inhibits output and retains the fault. Status and
                # local recovery remain available; no ownership is released.
                pass

    def serve(self, **kwargs):
        from .operator import OperatorServer, RecoveryTarget
        try:
            if self.operator_socket:
                self._operator = OperatorServer(self.operator_socket, RecoveryTarget(self.guard, kind="actuator"))
            self._poll_thread = Thread(target=self._poll, name="actuation-watchdog", daemon=True)
            self._poll_thread.start()
            return super().serve(**kwargs)
        finally:
            if self._operator:
                self._operator.close()

    def close(self):
        if self._operator:
            self._operator.close()
        self._poll_stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=2)
            if self._poll_thread.is_alive():
                raise ReconciliationRequired("Actuator poll is blocked; hardware watchdog/inspection required")
        self.guard.close()


class ActuationClient:
    def __init__(self, rpc, *, expected, clock_policy=None, clock=monotonic, clock_domain=None):
        actual = rpc.call("actuation.describe", timeout_s=5)
        self.clock, self.local_domain = clock, clock_domain or local_clock_domain()
        if clock_policy is not None and not isinstance(clock_policy, ClockSyncPolicy):
            raise ValueError("Expected an explicit ClockSyncPolicy")
        if actual != expected or (actual["clock_domain"] != self.local_domain and clock_policy is None):
            raise ValueError("Actuator identity or local monotonic clock domain mismatch")
        self.rpc, self._identity = rpc, deepcopy(actual)
        self.clock_policy, self._mapping, self._clock_fault = clock_policy, None, None
        self._local_clock_fault = None
        self._clock_lock, self._time_lock, self._last_time = Lock(), Lock(), None

    def identity(self):
        return deepcopy(self._identity)

    def timing_identity(self):
        return dict(mode="bounded-clock-mapping" if self.clock_policy else "shared-monotonic-clock",
                    local_domain=self.local_domain, remote_domain=self._identity["clock_domain"],
                    policy=asdict(self.clock_policy) if self.clock_policy else None,
                    source=file_digest(Path(__file__).with_name("clock_sync.py")))

    def _now(self):
        with self._time_lock:
            now = finite_seconds(self.clock(), positive=False)
            if self._last_time is not None and now < self._last_time:
                self._local_clock_fault = "Controller client clock moved backwards"
            self._last_time = now
            if self._local_clock_fault:
                raise ClockUnsynchronized(self._local_clock_fault)
            return now

    def _remaining(self, context):
        context.check()
        if context.deadline is None:
            raise ValueError("Actuator calls require a bounded context")
        return finite_seconds(context.deadline - self._now())

    def synchronize(self, context, *, force=False):
        if self.clock_policy is None:
            return None
        # A read-only probe never repeats an actuator operation. Queue time for
        # concurrent synchronizations still counts against the caller's budget.
        if not self._clock_lock.acquire(timeout=self._remaining(context)):
            raise TimeoutError("Clock synchronization queue deadline reached")
        try:
            if self._clock_fault:
                raise ClockUnsynchronized(self._clock_fault)
            now = self._now()
            if not force and self._mapping:
                try:
                    self._mapping.offset_bounds(now, now=now)
                    return self._mapping
                except ClockUnsynchronized:
                    pass
            nonce = secrets.token_hex(16)
            sent = self._now()
            reply = self.rpc.call("actuation.clock", kwargs=dict(nonce=nonce), timeout_s=self._remaining(context))
            received = self._now()
            if (set(reply) != {"schema", "nonce", "clock_domain", "authority_id", "received_at", "sent_at"}
                    or reply["schema"] != "physicalrsi.clock-exchange/v1" or reply["nonce"] != nonce
                    or reply["clock_domain"] != self._identity["clock_domain"]
                    or reply["authority_id"] != self._identity["authority_id"]):
                self._clock_fault = "Clock reply identity differs from the pinned actuator"
                raise ClockUnsynchronized(self._clock_fault)
            mapping = ClockMapping(self.clock_policy, self.local_domain, reply["clock_domain"], nonce,
                                   sent, reply["received_at"], reply["sent_at"], received)
            if self._mapping:
                try:
                    mapping.follows(self._mapping)
                except ClockUnsynchronized as error:
                    self._clock_fault = str(error)
                    raise
            self._mapping = mapping
            return mapping
        finally:
            self._clock_lock.release()

    def _mapped_submission(self, inputs, mapping, context):
        if "timing_evidence" in inputs:
            raise ValueError("Mapped timing evidence is generated by the actuator transport")
        signature = digest(dict(inputs=inputs, local_domain=self.local_domain))
        intent = self.rpc.call("actuation.intent", kwargs=dict(request_id=inputs["request_id"]),
                               timeout_s=self._remaining(context))
        if intent is not None:
            # Reuse the first materialized device deadline across calibration
            # refresh or client restart. Never retime the same command ID.
            evidence = intent["inputs"].get("timing_evidence", {})
            if (intent["identity"] != self._identity or intent["operation"] != "submit"
                    or evidence.get("logical_request_sha256") != signature):
                raise ValueError("Actuator request identity reused with different inputs or clock")
            return dict(inputs, deadline=intent["inputs"]["deadline"], timing_evidence=evidence)
        deadline = mapping.remote_deadline(inputs["deadline"], now=self._now())
        evidence = dict(schema="physicalrsi.mapped-deadline/v1", logical_request_sha256=signature,
                        local_deadline=inputs["deadline"], remote_deadline=deadline, mapping=mapping.evidence())
        return dict(inputs, deadline=deadline, timing_evidence=evidence)

    def call(self, operation, context, **inputs):
        self._remaining(context)
        mapping = self.synchronize(context)
        if mapping is not None and operation == "submit":
            inputs = self._mapped_submission(inputs, mapping, context)
        deadline = context.deadline if mapping is None else mapping.remote_deadline(context.deadline, now=self._now())
        sent = self._now()
        result = self.rpc.call("actuation.call", kwargs=dict(operation=operation, inputs=inputs, deadline=deadline,
                              clock_domain=self._identity["clock_domain"]), timeout_s=self._remaining(context))
        received = self._now()
        if operation == "observe" and mapping is not None:
            sample = SensorSample(**result)
            earliest, latest = mapping.local_capture(sample.captured_at,
                sample.captured_at + sample.capture_uncertainty_seconds, local_sent=sent, local_received=received)
            return asdict(SensorSample(sample.payload, sample.sequence, earliest, latest - earliest,
                dict(schema="physicalrsi.mapped-capture/v1", source={key: value for key, value in result.items() if key != "payload"},
                     local_sent=sent, local_received=received, earliest=earliest, latest=latest,
                     mapping=mapping.evidence())))
        return result

    def status(self, request_id, context):
        remaining = self._remaining(context)
        return self.rpc.call("actuation.status", kwargs=dict(request_id=request_id),
                             timeout_s=remaining)

    def inspect(self, context):
        return self.rpc.call("actuation.inspect", timeout_s=self._remaining(context))


class ActuationDriver:
    """ControllerDriver using the fenced backend rather than a raw SDK handle.

    A task supplies pure validation and any task termination logic. The backend
    validates again immediately before writing. There is no background renewal:
    a blocked controller cannot keep its actuator session alive indefinitely.
    """

    def __init__(self, client, *, validator, validation_identity):
        self.client, self.validator = client, validator
        self.validation_identity = deepcopy(validation_identity)
        self._claim = None

    def identity(self):
        security = self.client.rpc.security_identity() if hasattr(self.client.rpc, "security_identity") else None
        return dict(kind="fenced-actuation-driver/v1", backend=self.client.identity(),
                    validation=deepcopy(self.validation_identity), clock_transport=self.client.timing_identity(),
                    transport_security=security, implementation=file_digest(Path(__file__)))

    def reset(self, case, context):
        if self._claim is not None:
            raise ReconciliationRequired("Previous actuator session requires release")
        credential = secrets.token_hex(32)
        grant = self.client.call("acquire", context, request_id=uuid.uuid4().hex, owner=context.episode,
                                 credential=credential, harness_revision=context.harness_revision)
        if grant["state"] == "rejected":
            raise ControlRejected(grant)
        self._claim = dict(grant["claim"], credential=credential)
        return self.client.call("reset", context, request_id=uuid.uuid4().hex, case=case,
                                 claim=self._claim, harness_revision=context.harness_revision)

    def observe(self, context):
        return SensorSample(**self.client.call("observe", context, claim=self._claim,
                                               harness_revision=context.harness_revision))

    def validate(self, actions):
        self.validator(actions)

    def execute(self, actions, *, period_seconds, deadline, context):
        self.validate(actions)
        renewed = self.client.call("renew", context, request_id=uuid.uuid4().hex, claim=self._claim)
        if renewed["state"] == "rejected":
            raise ControlRejected(renewed)
        request_id = uuid.uuid4().hex
        self.client.call("submit", context, request_id=request_id, actions=actions, period_seconds=period_seconds,
                         deadline=deadline, claim=self._claim, harness_revision=context.harness_revision)
        while True:
            context.check()
            if monotonic() >= deadline:
                raise ReconciliationRequired("Actuator command has no observed completion before its deadline")
            result = self.client.status(request_id, context)
            if result["state"] == "completed":
                return dict(terminated=False, executed_steps=result["applied_steps"], actuation=result)
            if result["state"] != "started":
                raise ReconciliationRequired("Actuator command interrupted: " + result["state"])
            sleep(min(.01, period_seconds / 2))

    def quiesce(self, context):
        if self._claim is None:
            # Never report a competing owner's moving device as quiescent, and
            # never stop it to clean up an unsuccessful acquisition elsewhere.
            state = self.client.inspect(context)
            return dict(quiescent=state["quiescence"].get("quiescent") is True and state["ownership"]["phase"] == "free",
                        actuation=state, reason="No actuator session acquired by this driver")
        self.client.call("stop", context, request_id=uuid.uuid4().hex, claim=self._claim)
        while True:
            context.check()
            observed = self.client.inspect(context)
            if observed["quiescence"].get("quiescent") is True:
                release = self.client.call("release", context, request_id=uuid.uuid4().hex, claim=self._claim)
                self._claim = None
                return dict(quiescent=True, actuation_release=release, measured=observed["quiescence"])
            sleep(.01)
