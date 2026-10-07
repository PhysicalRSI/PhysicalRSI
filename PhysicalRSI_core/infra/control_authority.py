"""Persistent control claims at one authoritative device gateway.

The gateway owns the process lock and is the only writer. Expiry or revocation
rejects commands; it never proves quiescence or makes a resource available.
Claims are bearer capabilities for cooperating clients, not transport identity.
"""

from copy import deepcopy
import hmac
from pathlib import Path
from threading import RLock
import uuid

from ..contracts import ReconciliationRequired
from ..timing import finite_seconds
from .storage import atomic_json, digest, identifier, read_json


class ControlClaimInvalid(ReconciliationRequired):
    pass


def credential_digest(credential):
    if (not isinstance(credential, str) or len(credential) != 64
            or any(c not in "0123456789abcdef" for c in credential)):
        raise ValueError("A control credential must contain 32 random bytes encoded as lowercase hex")
    return digest(dict(credential=credential))


def claim_reference(claim):
    """A persistable request binding; never retain a plaintext credential."""
    if claim is None:
        return None
    if (not isinstance(claim, dict) or set(claim) != {"authority_id", "generation", "credential"}
            or not isinstance(claim["authority_id"], str) or type(claim["generation"]) is not int
            or claim["generation"] < 1):
        raise ValueError("Malformed controller claim")
    return dict(authority_id=claim["authority_id"], generation=claim["generation"],
                credential_sha256=credential_digest(claim["credential"]))


class ControlAuthority:
    def __init__(self, path, *, resources, lease_seconds, boot_id, clock):
        self.path, self.clock, self.boot_id = Path(path), clock, boot_id
        self.config = dict(resources=list(resources), lease_seconds=finite_seconds(lease_seconds))
        self._lock, self._last_time = RLock(), None
        if self.path.exists():
            self._data = read_json(self.path)
            if self._data["config"] != self.config:
                raise ValueError("Controller authority configuration changed")
        else:
            self._data = dict(schema="physicalrsi.control-authority/v1", config=self.config,
                              authority_id=uuid.uuid4().hex, generation=0, current=None, requests={})
            atomic_json(self.path, self._data)

    def _now(self):
        value = finite_seconds(self.clock(), positive=False)
        if self._last_time is not None and value < self._last_time:
            raise ControlClaimInvalid("Controller authority clock moved backwards")
        self._last_time = value
        return value

    def _match(self, claim):
        reference = claim_reference(claim)
        current = self._data["current"]
        if (not reference or not current or reference["authority_id"] != self._data["authority_id"]
                or reference["generation"] != current["generation"]
                or not hmac.compare_digest(reference["credential_sha256"], current["credential_sha256"])):
            raise ControlClaimInvalid("Claim does not own this controller generation")
        return current

    def _live(self, current):
        if current["boot_id"] != self.boot_id:
            raise ControlClaimInvalid("Claim belongs to an earlier gateway boot; recover the controller")
        if current.get("revoked"):
            raise ControlClaimInvalid("Controller claim was revoked; quiescence is required")
        if self._now() >= current["expires_at"]:
            raise ControlClaimInvalid("Controller claim expired; quiescence is required")

    def check(self, claim, *, harness_revision=None, live=True):
        with self._lock:
            current = self._match(claim)
            if harness_revision is not None and harness_revision != current["harness_revision"]:
                raise ControlClaimInvalid("Claim is bound to another System 1 revision")
            if live:
                self._live(current)
            return deepcopy({k: v for k, v in current.items() if k != "credential_sha256"})

    def inspect(self):
        with self._lock:
            current = self._data["current"]
            phase = "free"
            if current:
                try:
                    self._live(current)
                    phase = "held"
                except ControlClaimInvalid:
                    phase = "recovery_required"
            return dict(authority_id=self._data["authority_id"], generation=self._data["generation"],
                        phase=phase, resources=self.config["resources"], lease_seconds=self.config["lease_seconds"],
                        current=({k: deepcopy(v) for k, v in current.items() if k != "credential_sha256"}
                                 if current else None))

    def status(self, request_id):
        with self._lock:
            return deepcopy(self._data["requests"].get(identifier(request_id), dict(state="missing")))

    def _apply(self, request_id, binding, operation):
        """Commit metadata and its receipt together; a lost response is readable."""
        identifier(request_id)
        signature = digest(binding)
        with self._lock:
            previous = self._data["requests"].get(request_id)
            if previous:
                if previous["binding"] != signature:
                    raise ValueError("Authority request reused with different inputs")
                return deepcopy(previous["result"])
            updated = deepcopy(self._data)
            result = operation(updated)
            updated["requests"][request_id] = dict(state="completed", binding=signature, result=result)
            try:
                atomic_json(self.path, updated)
            finally:
                # If fsync fails after replace, use the authoritative on-disk
                # transaction, never an older cached owner on this live server.
                self._data = read_json(self.path)
            return deepcopy(result)

    @staticmethod
    def _reject(request_id, reason):
        return dict(state="rejected", request_id=request_id, reason=reason, dispatched=False)

    def acquire(self, request_id, *, owner, credential, harness_revision, controller_idle):
        identifier(owner)
        sha = credential_digest(credential)
        binding = dict(operation="acquire", owner=owner, credential_sha256=sha, harness_revision=harness_revision)

        def apply(data):
            if data["current"] is not None or not controller_idle:
                return self._reject(request_id, "Controller ownership requires verified release of its previous episode")
            data["generation"] += 1
            data["current"] = dict(generation=data["generation"], owner=owner, credential_sha256=sha,
                                   harness_revision=harness_revision, boot_id=self.boot_id,
                                   expires_at=self._now() + self.config["lease_seconds"])
            return dict(state="completed", request_id=request_id,
                        claim=dict(authority_id=data["authority_id"], generation=data["generation"]),
                        lease_seconds=self.config["lease_seconds"])

        return self._apply(request_id, binding, apply)

    def renew(self, request_id, claim):
        binding = dict(operation="renew", claim=claim_reference(claim))

        def apply(data):
            try:
                self.check(claim)
            except ControlClaimInvalid as error:
                return self._reject(request_id, str(error))
            data["current"]["expires_at"] = self._now() + self.config["lease_seconds"]
            return dict(state="completed", request_id=request_id, lease_seconds=self.config["lease_seconds"])

        return self._apply(request_id, binding, apply)

    def revoke(self, request_id, claim, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Declare a revocation reason")
        binding = dict(operation="revoke", claim=claim_reference(claim), reason=reason)

        def apply(data):
            try:
                self.check(claim, live=False)
            except ControlClaimInvalid as error:
                return self._reject(request_id, str(error))
            data["current"]["revoked"] = reason
            return dict(state="completed", request_id=request_id, phase="recovery_required",
                        generation=data["generation"], quiescent=False)

        return self._apply(request_id, binding, apply)

    def quarantine(self, request_id, *, generation, operator, reason, evidence):
        """Local operator path; not an unauthenticated takeover RPC."""
        if (type(generation) is not int or generation < 0 or not isinstance(operator, str) or not operator.strip()
                or not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, dict) or not evidence):
            raise ValueError("Recovery requires an exact generation, operator, reason and evidence")
        binding = dict(operation="quarantine", generation=generation, operator=operator, reason=reason,
                       evidence=evidence)

        def apply(data):
            if data["generation"] != generation:
                raise ControlClaimInvalid("Recovery refers to a different controller generation")
            if data["current"]:
                data["current"]["revoked"] = reason
            return dict(state="completed", request_id=request_id, generation=generation, quiescent=False)

        return self._apply(request_id, binding, apply)

    def release(self, request_id, *, generation, evidence):
        """Gateway-only commit after the driver confirmed quiescence."""
        binding = dict(operation="release", generation=generation, evidence=evidence)

        def apply(data):
            if data["generation"] != generation or evidence.get("quiescent") is not True:
                raise ControlClaimInvalid("Release requires current-generation quiescence evidence")
            data["current"] = None
            return dict(state="completed", request_id=request_id, generation=generation,
                        quiescent=True, evidence_sha256=digest(evidence))

        return self._apply(request_id, binding, apply)
