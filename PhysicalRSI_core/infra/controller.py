"""A serialized controller gateway with durable requests and source-time checks.

Drivers execute action chunks in their own control loop. The gateway bounds and
attributes dispatch; it cannot interrupt a blocked driver or replace a hardware
watchdog. All local clients of a device must use one gateway and device registry.
"""

import fcntl
import json
import secrets
import uuid
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Protocol

from ..contracts import Context, ReconciliationRequired
from ..embodiment import Embodiment
from ..timing import SensorSample, finite_seconds, observation_reference
from .control_authority import ControlAuthority, ControlClaimInvalid, claim_reference, credential_digest
from .rpc.journal import RequestJournal
from .storage import atomic_json, canonical, digest, file_digest, identifier, read_json


def _json(value):
    return json.loads(canonical(value))


class ControllerDriver(Protocol):
    def identity(self) -> dict: ...
    def reset(self, case: dict, context: Context) -> dict: ...
    def observe(self, context: Context) -> SensorSample: ...
    def validate(self, actions: list) -> None:
        """Validate the entire chunk without effects, including physical limits."""
        ...
    def execute(self, actions: list, *, period_seconds: float, deadline: float,
                context: Context) -> dict:
        """Return terminated: bool, executed_steps: int; bound every device call.

        Deadline uses the gateway monotonic clock. The driver owns timing each
        control tick, cancellation/stop handling and any controller watchdog.
        """
        ...
    def quiesce(self, context: Context) -> dict: ...


class ControlRejected(ValueError):
    """A completed admission receipt confirms that no action was dispatched."""
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt["reason"])


class ControllerGateway:
    def __init__(self, root, *, driver, embodiment: Embodiment, clock=monotonic, lease_seconds=30):
        if not isinstance(embodiment, Embodiment) or embodiment.timing is None:
            raise ValueError("Controller gateway requires an embodiment with control timing")
        if not embodiment.resources:
            raise ValueError("Controller gateway requires named device resources")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.driver, self.embodiment, self.clock = driver, embodiment, clock
        self._identity = _json(dict(driver=driver.identity(), embodiment=asdict(embodiment),
                                   gateway=file_digest(Path(__file__)),
                                   ownership=dict(lease_seconds=finite_seconds(lease_seconds),
                                                  resources=embodiment.resources),
                                   contracts=file_digest(Path(__file__).parents[1] / "contracts.py"),
                                   timing=file_digest(Path(__file__).parents[1] / "timing.py"),
                                   infrastructure={name: file_digest(Path(__file__).parent / name)
                                                   for name in ("controller_rpc.py", "control_authority.py", "operator.py", "storage.py", "rpc/journal.py",
                                                                "rpc/main_thread_serve.py", "rpc/rpc_facade.py",
                                                                "rpc/deadline_http.py", "rpc/http_rpc.py",
                                                                "rpc/authenticated_http.py", "rpc/tls.py")}))
        self._boot = uuid.uuid4().hex
        self._last_time = None
        self._mutex = RLock()
        self._request_lock = RLock()
        self.journal = RequestJournal(self.root / "requests")
        self._owner = (self.root / ".owner.lock").open("a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._owner.close()
            raise RuntimeError("Another controller gateway owns this directory") from None
        try:
            self.authority = ControlAuthority(self.root / "authority.json", resources=embodiment.resources,
                                              lease_seconds=lease_seconds, boot_id=self._boot, clock=clock)
            self._identity["ownership"]["authority_id"] = self.authority.inspect()["authority_id"]
        except BaseException:
            self._owner.close()
            raise

    def identity(self):
        return deepcopy(self._identity)

    def state(self):
        path = self.root / "controller.json"
        return read_json(path) if path.exists() else dict(phase="idle", episode=None)

    def status(self, request_id):
        result = self.journal.status(request_id)
        return self.authority.status(request_id) if result["state"] == "missing" else result

    def ownership(self):
        return self.authority.inspect()

    def close(self):
        """Release the local process handle; durable unfinished work stays blocked.

        Call quiesce explicitly to stop/release a controller episode. Closing a
        Python object alone is not evidence that a physical device stopped.
        """
        with self._mutex, self._request_lock:
            self._owner.close()

    def _now(self):
        value = finite_seconds(self.clock(), positive=False)
        if self._last_time is not None and value < self._last_time:
            raise ReconciliationRequired("Controller monotonic clock moved backwards")
        self._last_time = value
        return value

    def _save(self, state):
        atomic_json(self.root / "controller.json", state)

    def _stable(self):
        if _json(self.driver.identity()) != self._identity["driver"]:
            raise ReconciliationRequired("Controller driver identity changed")

    def _register(self, request_id, method, inputs, context, claim=None):
        identifier(request_id)
        binding = dict(method=method, inputs=inputs, identity=self.identity(),
                       harness_revision=context.harness_revision, claim=claim_reference(claim))
        with self._request_lock:
            if self._owner.closed:
                raise RuntimeError("Controller gateway is closed")
            # Keep the bounded public command, not just its binding hash, so an
            # operator can inspect a dispatched request after losing the client.
            intent = self.root / "intents" / (request_id + ".json")
            if intent.exists():
                if read_json(intent) != binding:
                    raise ValueError("Request identity reused with different inputs/session")
            else:
                atomic_json(intent, binding)
        return binding

    def acquire(self, request_id, owner, credential, context):
        context.check()
        inputs = dict(owner=owner, credential_sha256=credential_digest(credential))
        with self._request_lock:
            self._register(request_id, "acquire", inputs, context)
            context.check()
            return self.authority.acquire(request_id, owner=owner, credential=credential,
                                          harness_revision=context.harness_revision,
                                          controller_idle=self.state()["phase"] == "idle")

    def renew(self, request_id, claim, context):
        with self._request_lock:
            self._register(request_id, "renew", {}, context, claim)
            context.check()
            return self.authority.renew(request_id, claim)

    def revoke(self, request_id, claim, reason, context):
        with self._request_lock:
            self._register(request_id, "revoke", dict(reason=reason), context, claim)
            context.check()
            return self.authority.revoke(request_id, claim, reason)

    def _controlled(self, context, claim):
        def validity():
            context.check()
            self.authority.check(claim, harness_revision=context.harness_revision)
        return replace(context, validity=validity)

    def _request(self, request_id, method, inputs, context, function, *, claim=None):
        with self._mutex:
            binding = self._register(request_id, method, inputs, context, claim)
            try:
                return self.journal.execute(request_id, binding, function)
            except BaseException as error:
                state = self.state()
                if (state.get("last_request") == request_id
                        and self.journal.status(request_id)["state"] in {"started", "uncertain"}):
                    state.update(phase="needs_reconciliation", error=type(error).__name__ + ": " + str(error))
                    self._save(state)
                raise

    @staticmethod
    def _reject(request_id, reason):
        return dict(state="rejected", request_id=request_id, reason=reason, dispatched=False)

    def _sample(self, episode, context, *, after, previous=None):
        sample = self.driver.observe(context)
        if not isinstance(sample, SensorSample):
            raise ValueError("Controller driver must return a stamped SensorSample")
        now = self._now()
        if not after <= sample.captured_at <= sample.captured_at + sample.capture_uncertainty_seconds <= now:
            raise ReconciliationRequired("Sensor sample predates the completed effect or is in the future")
        if now - sample.captured_at >= self.embodiment.timing.max_observation_age_seconds:
            raise ReconciliationRequired("Sensor sample is already stale")
        if previous is not None and sample.sequence <= previous["sequence"]:
            raise ReconciliationRequired("Sensor sequence did not advance after execution")
        timing = (dict(capture_uncertainty_seconds=sample.capture_uncertainty_seconds,
                       timing_evidence=sample.timing_evidence)
                  if sample.capture_uncertainty_seconds or sample.timing_evidence is not None else {})
        return _json(dict(schema="physicalrsi.observation/v1", episode=episode,
                          clock_domain=self._boot, sequence=sample.sequence,
                          captured_at=sample.captured_at, payload=sample.payload, **timing))

    def reset(self, request_id, episode, case, context, *, claim=None):
        identifier(episode)
        case = _json(case)
        claim = deepcopy(claim)
        context = self._controlled(context, claim)

        def perform():
            try:
                owner = self.authority.check(claim, harness_revision=context.harness_revision)
            except ControlClaimInvalid as error:
                return self._reject(request_id, str(error))
            if self.state()["phase"] != "idle":
                return self._reject(request_id, "Controller episode requires quiescence before reset")
            context.check()
            self._stable()
            state = dict(phase="executing", episode=episode, clock_domain=self._boot,
                         harness_revision=context.harness_revision, last_request=request_id,
                         authority=dict(authority_id=claim["authority_id"], generation=owner["generation"], owner=owner["owner"]))
            self._save(state)
            context.check()
            result = _json(self.driver.reset(deepcopy(case), context))
            finished = self._now()
            context.check()
            if type(result.get("ready")) is not bool or type(result.get("terminated", False)) is not bool:
                raise ValueError("Driver reset must declare readiness and boolean termination")
            packet = self._sample(episode, context, after=finished) if result["ready"] else None
            context.check()
            self._stable()
            state.update(phase="ready", observation=packet, terminated=result.get("terminated", False))
            self._save(state)
            return dict(state="completed", ready=result["ready"], observation=packet,
                        terminated=state["terminated"], control=dict(episode=episode, request_id=request_id,
                                                                   authority=state["authority"]))

        return self._request(request_id, "reset", dict(episode=episode, case=case), context, perform, claim=claim)

    def step(self, request_id, command, context, *, claim=None):
        command = _json(command)
        claim = deepcopy(claim)
        context = self._controlled(context, claim)

        def perform():
            try:
                owner = self.authority.check(claim, harness_revision=context.harness_revision)
            except ControlClaimInvalid as error:
                return self._reject(request_id, str(error))
            state = self.state()
            packet = state.get("observation")
            if (state["phase"] != "ready" or not packet or state.get("terminated")
                    or state.get("clock_domain") != self._boot):
                return self._reject(request_id, "Controller has no current actionable observation")
            if state["harness_revision"] != context.harness_revision:
                return self._reject(request_id, "Executing System 1 revision differs from the episode")
            if state.get("authority", {}).get("generation") != owner["generation"]:
                return self._reject(request_id, "Controller episode belongs to another ownership generation")
            timing = self.embodiment.timing
            try:
                if not isinstance(command, dict) or command.get("schema") != "physicalrsi.action-chunk/v1":
                    raise ValueError("Expected an observation-bound action chunk")
                if command.get("observation") != observation_reference(packet):
                    raise ValueError("Action refers to a different observation, episode or controller clock")
                if finite_seconds(command.get("period_seconds")) != timing.period_seconds:
                    raise ValueError("Action period differs from the controller contract")
                actions = command.get("actions")
                if not isinstance(actions, list) or not 1 <= len(actions) <= timing.max_chunk_steps:
                    raise ValueError("Action chunk exceeds the declared horizon or is empty")
                self.driver.validate(deepcopy(actions))
            except ValueError as error:
                return self._reject(request_id, str(error))
            context.check()
            self._stable()
            # Persist dispatch intent, then check freshness again at the driver
            # boundary so disk/queue delay cannot make an old action admissible.
            state.update(phase="executing", last_request=request_id)
            self._save(state)
            dispatched = self._now()
            age = dispatched - packet["captured_at"]
            if age >= timing.max_observation_age_seconds:
                state.update(phase="ready")
                self._save(state)
                return self._reject(request_id, "Observation expired before controller dispatch")
            deadline = dispatched + len(actions) * timing.period_seconds + timing.execution_slack_seconds
            context.check()
            if deadline >= self.authority.check(claim)["expires_at"]:
                state.update(phase="ready")
                self._save(state)
                return self._reject(request_id, "Renew controller ownership before dispatching this action horizon")
            result = _json(self.driver.execute(deepcopy(actions), period_seconds=timing.period_seconds,
                                               deadline=deadline, context=context))
            finished = self._now()
            if finished > deadline:
                raise ReconciliationRequired("Controller execution exceeded its action deadline")
            context.check()
            executed = result.get("executed_steps")
            if (type(result.get("terminated")) is not bool or type(executed) is not int
                    or not 1 <= executed <= len(actions)
                    or (not result["terminated"] and executed != len(actions))):
                raise ValueError("Driver must report completed control steps and termination")
            observed = self._sample(state["episode"], context, after=finished, previous=packet)
            context.check()
            self._stable()
            state.update(phase="ready", observation=observed, terminated=result["terminated"])
            self._save(state)
            return dict(state="completed", observation=observed, terminated=result["terminated"], driver=result,
                        control=dict(request_id=request_id, source=command["observation"],
                                     dispatched_at=dispatched, completed_at=finished,
                                     observation_age_seconds=age, deadline=deadline,
                                     period_seconds=timing.period_seconds, executed_steps=executed,
                                     authority=state["authority"]))

        return self._request(request_id, "step", command, context, perform, claim=claim)

    def quiesce(self, request_id, episode, context, *, claim=None):
        identifier(episode)
        claim = deepcopy(claim)

        def perform():
            try:
                owner = self.authority.check(claim, live=False)
            except ControlClaimInvalid as error:
                return self._reject(request_id, str(error))
            state = self.state()
            if state["phase"] != "idle" and state.get("episode") != episode:
                return self._reject(request_id, "Quiescence requested for a different controller episode")
            context.check()
            self._stable()
            state.update(phase="quiescing", last_request=request_id)
            self._save(state)
            result = _json(self.driver.quiesce(context))
            context.check()
            if result.get("quiescent") is not True:
                raise ReconciliationRequired("Controller did not confirm quiescence")
            self._stable()
            state.update(phase="idle", observation=None)
            self._save(state)
            release = self.authority.release(request_id, generation=owner["generation"], evidence=result)
            return dict(result, state="completed", episode=episode, request_id=request_id, ownership_release=release)

        return self._request(request_id, "quiesce", dict(episode=episode), context, perform, claim=claim)

    def prepare_recovery(self, request_id, *, generation, operator, reason, evidence, context):
        """Revoke through the metadata lock before a thread-affine stop is queued."""
        inputs = _json(dict(generation=generation, operator=operator, reason=reason, evidence=evidence))
        with self._request_lock:
            self._register(request_id, "recover", inputs, context)
            context.check()
            self.authority.quarantine(digest(dict(recovery=request_id)), **inputs)

    def recover(self, request_id, *, generation, operator, reason, evidence, context):
        """Locally authorized recovery; revoke first, then verify the driver stop.

        This method is intentionally absent from ControllerHost's RPC surface.
        Revocation can interrupt cooperative driver checks; it cannot stop a
        blocked native call or replace the robot's independent watchdog.
        """
        inputs = _json(dict(generation=generation, operator=operator, reason=reason, evidence=evidence))
        self.prepare_recovery(request_id, **inputs, context=context)

        def perform():
            if self.authority.inspect()["generation"] != generation:
                raise ControlClaimInvalid("Recovery refers to a different controller generation")
            context.check()
            self._stable()
            state = self.state()
            state.update(phase="quiescing", last_request=request_id)
            self._save(state)
            result = _json(self.driver.quiesce(context))
            context.check()
            if result.get("quiescent") is not True:
                raise ReconciliationRequired("Controller recovery did not confirm quiescence")
            self._stable()
            state.update(phase="idle", observation=None)
            self._save(state)
            release = self.authority.release(request_id, generation=generation, evidence=result)
            return dict(state="completed", request_id=request_id, quiescent=True,
                        recovery=inputs, driver=result, ownership_release=release)

        return self._request(request_id, "recover", inputs, context, perform)


class GatewayEnvironment:
    """Use a controller gateway as an ExperimentRuntime environment port."""
    def __init__(self, gateway):
        self.gateway = gateway
        self.episode = None
        self._claim = None
        self._renew_at = None

    def identity(self):
        identity = self.gateway.identity()
        rpc = getattr(self.gateway, "rpc", None)
        security = rpc.security_identity() if hasattr(rpc, "security_identity") else None
        return identity if security is None else dict(identity, transport_security=security)

    def describe(self):
        return self.gateway.embodiment

    @staticmethod
    def _accepted(receipt):
        if receipt["state"] == "rejected":
            raise ControlRejected(receipt)
        if receipt["state"] != "completed":
            raise ReconciliationRequired("Controller operation has no completed receipt")
        return receipt

    def reset(self, case, context):
        if self._claim is not None:
            raise ReconciliationRequired("Previous controller claim has not been released")
        self.episode = uuid.uuid4().hex
        credential = secrets.token_hex(32)
        started = monotonic()
        grant = self._accepted(self.gateway.acquire(uuid.uuid4().hex, context.episode, credential, context))
        self._claim = dict(grant["claim"], credential=credential)
        self._renew_at = started + grant["lease_seconds"] / 2
        return self._accepted(self.gateway.reset(uuid.uuid4().hex, self.episode, case, context, claim=self._claim))

    def step(self, action, context):
        if self._renew_at is not None and monotonic() >= self._renew_at:
            started = monotonic()
            renewed = self._accepted(self.gateway.renew(uuid.uuid4().hex, self._claim, context))
            self._renew_at = started + renewed["lease_seconds"] / 2
        return self._accepted(self.gateway.step(uuid.uuid4().hex, action, context, claim=self._claim))

    def quiesce(self, context):
        result = self._accepted(self.gateway.quiesce(uuid.uuid4().hex, self.episode, context, claim=self._claim))
        self._claim, self._renew_at = None, None
        return result
