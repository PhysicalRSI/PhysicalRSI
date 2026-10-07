"""Local, OS-authenticated inspection and explicit device recovery.

This is an operator capability, never a policy/model tool. A deployment must run
untrusted System 1/System 2 code under a different OS identity. Same-UID processes
already share filesystem/process privileges; this interface is not their sandbox.
"""

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import socket
import socketserver
import stat
import struct
from threading import Event, Lock, Thread
from time import monotonic
import uuid

from ..contracts import Context, ReconciliationRequired
from ..timing import finite_seconds
from .actuation import local_clock_domain
from .storage import atomic_json, canonical, digest, identifier, locked, read_json

REQUEST_BYTES = 64 * 1024
RESPONSE_BYTES = 1024 * 1024
SCHEMA = "physicalrsi.operator-recovery/v1"


def _json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate operator JSON member")
            result[key] = value
        return result
    value = json.loads(payload, object_pairs_hook=pairs)
    canonical(value)  # Reject NaN/Infinity before any recovery effect.
    return value


def _read(sock, count, deadline):
    value = bytearray()
    while len(value) < count:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Operator transport total deadline exceeded")
        sock.settimeout(remaining)
        block = sock.recv(count - len(value))
        if not block:
            raise ConnectionError("Operator connection closed mid-frame")
        value.extend(block)
    return bytes(value)


def _receive(sock, deadline, limit):
    size, = struct.unpack("!I", _read(sock, 4, deadline))
    if not 0 < size <= limit:
        raise ValueError("Operator message exceeds its byte limit")
    return _json(_read(sock, size, deadline))


def _send(sock, value, deadline, limit):
    payload = canonical(value)
    if len(payload) > limit:
        raise ValueError("Operator response exceeds its byte limit")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("Operator transport total deadline exceeded")
    sock.settimeout(remaining)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _private_directory(directory):
    directory = Path(directory)
    if not directory.is_absolute() or directory.resolve() != directory:
        raise ValueError("Operator socket directory must be an absolute physical directory")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("Operator socket directory must be owned by this UID and private (0700)")
    return directory


def _peer(sock):
    pid, uid, gid = struct.unpack("iII", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("iII")))
    return dict(pid=pid, uid=uid, gid=gid)


class RecoveryTarget:
    """Adapt existing recovery primitives without adding an ordinary RPC method."""

    def __init__(self, target, *, kind, main_thread=None):
        if kind not in {"actuator", "controller"} or (kind == "controller" and main_thread is None):
            raise ValueError("Controller recovery requires its owning thread dispatcher")
        self.target, self.kind, self.main_thread = target, kind, main_thread

    def identity(self):
        ownership = self.target.authority.inspect()
        return dict(kind=self.kind, identity_sha256=digest(self.target.identity()),
                    authority_id=ownership["authority_id"], resources=ownership["resources"])

    def inspect(self):
        if self.kind == "actuator":
            snapshot = self.target.inspect()
            return {key: snapshot[key] for key in ("state", "ownership", "quiescence", "poll_fault", "last_poll_at")}
        state = self.target.state()
        return dict(ownership=self.target.ownership(), state={key: state[key] for key in
            ("phase", "episode", "last_request", "error", "authority") if key in state})

    def core_status(self, request_id):
        # Read the durable journal without waiting on a blocked device mutex.
        return self.target.journal.status(request_id)

    def recover(self, record, deadline):
        request, peer = record["request"], record["peer"]
        args = dict(generation=request["target"]["generation"], operator="unix-uid:" + str(peer["uid"]),
                    reason=request["reason"], evidence=dict(
                        inspection=request["inspection"], supplied=deepcopy(request["evidence"]),
                        operator_request_sha256=record["request_sha256"], authenticated_peer=peer))
        if self.kind == "actuator":
            # This local listener owns both clocks. Preserve the remaining
            # budget for an explicitly supplied backend clock (e.g. a fixture
            # with another origin), without extending it for conversion work.
            device_now = self.target.clock()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Operator recovery expired before device dispatch")
            return self.target.recover(record["core_request_id"], **args, request_deadline=device_now + remaining)
        context = Context(record["core_request_id"], deadline=deadline)
        # Revocation is metadata-only and remains live while the device thread
        # is occupied. Actual SDK stop calls stay on the original driver thread.
        self.target.prepare_recovery(record["core_request_id"], **args, context=context)
        return self.main_thread(lambda: self.target.recover(record["core_request_id"], **args, context=context),
                                deadline=deadline)


def recovery_request(snapshot, *, request_id, reason, evidence):
    identifier(request_id)
    if (not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, dict) or not evidence):
        raise ValueError("Recovery requires an explicit reason and inspection evidence")
    return dict(schema=SCHEMA, request_id=request_id,
                target=dict(snapshot["target"], generation=snapshot["observation"]["ownership"]["generation"]),
                inspection=dict(id=snapshot["id"], sha256=digest(snapshot)), reason=reason, evidence=deepcopy(evidence))


class OperatorEndpoint:
    def __init__(self, target):
        self.target = target
        self.root = target.target.root / "operator"
        self._run_lock = Lock()
        self.closed = Event()

    def _request_path(self, request_id):
        return self.root / "requests" / (identifier(request_id) + ".json")

    def _checkpoint(self, path, record):
        # Another listener (or a replacement host) may already have recovered
        # the core receipt while this listener was returning its response.
        # A late error must never overwrite that completed outer receipt.
        with locked(self.root / ".requests.lock"):
            current = read_json(path)
            if (current["request"] != record["request"] or current["peer"] != record["peer"]
                    or current["core_request_id"] != record["core_request_id"]):
                raise ValueError("Operator recovery binding changed during dispatch")
            if current["state"] != "completed":
                atomic_json(path, record)

    def inspect(self, peer):
        snapshot = dict(schema="physicalrsi.operator-inspection/v1", id=uuid.uuid4().hex,
                        target=self.target.identity(), observation=self.target.inspect(), inspected_by=peer)
        atomic_json(self.root / "inspections" / (snapshot["id"] + ".json"), snapshot)
        return snapshot

    def status(self, request_id):
        path = self._request_path(request_id)
        if not path.exists():
            return dict(state="missing", request_id=request_id)
        record = read_json(path)
        if digest(record["request"]) != record["request_sha256"]:
            raise ValueError("Operator request receipt changed")
        result = dict(request_id=request_id, state=record["state"], request=record["request"], peer=record["peer"])
        if record["state"] == "completed":
            if digest(record["result"]) != record["result_sha256"]:
                raise ValueError("Operator result receipt changed")
            result["result"] = record["result"]
        else:
            result["core_status"] = self.target.core_status(record["core_request_id"])
            result["error_type"] = record.get("error_type")
        return result

    def _validate(self, request):
        if (not isinstance(request, dict) or set(request) !=
                {"schema", "request_id", "target", "inspection", "reason", "evidence"} or request["schema"] != SCHEMA):
            raise ValueError("Expected an explicit versioned operator recovery request")
        identifier(request["request_id"])
        inspection = request["inspection"]
        if not isinstance(inspection, dict) or set(inspection) != {"id", "sha256"}:
            raise ValueError("Recovery must reference a saved inspection")
        path = self.root / "inspections" / (identifier(inspection["id"]) + ".json")
        snapshot = read_json(path)
        if digest(snapshot) != inspection["sha256"]:
            raise ValueError("Operator inspection digest differs")
        expected = recovery_request(snapshot, request_id=request["request_id"], reason=request["reason"], evidence=request["evidence"])
        if request != expected or snapshot["target"] != self.target.identity():
            raise ValueError("Recovery refers to another device identity or inspection")

    def recover(self, request, peer, deadline):
        self._validate(request)
        remaining = deadline - monotonic()
        if remaining <= 0 or not self._run_lock.acquire(timeout=remaining):
            raise TimeoutError("Operator recovery queue deadline reached")
        try:
            if self.closed.is_set() or monotonic() >= deadline:
                raise TimeoutError("Operator recovery expired or endpoint closed")
            path = self._request_path(request["request_id"])
            with locked(self.root / ".requests.lock"):
                use_path = self.root / "inspection-use" / (identifier(request["inspection"]["id"]) + ".json")
                use = dict(request_id=request["request_id"], request_sha256=digest(request), uid=peer["uid"])
                if use_path.exists() and read_json(use_path) != use:
                    raise ValueError("Inspection already authorizes a different request; inspect the device again")
                if not use_path.exists():
                    atomic_json(use_path, use)
                if path.exists():
                    record = read_json(path)
                    if record["request"] != request or record["peer"]["uid"] != peer["uid"]:
                        raise ValueError("Operator request identity reused with different inputs or UID")
                    if digest(request) != record["request_sha256"]:
                        raise ValueError("Operator request receipt changed")
                    if record["state"] == "completed":
                        return self.status(request["request_id"])
                else:
                    record = dict(state="started", request=deepcopy(request), request_sha256=digest(request), peer=peer,
                                  core_request_id="operator-" + digest(dict(request_id=request["request_id"], uid=peer["uid"])))
                    atomic_json(path, record)
            core = self.target.core_status(record["core_request_id"])
            if core["state"] not in {"missing", "completed"}:
                raise ReconciliationRequired("Operator recovery has an uncertain device effect; inspect the original request")
            if core["state"] == "missing":
                current = self.target.target.authority.inspect()
                if current["generation"] != request["target"]["generation"]:
                    raise ValueError("Recovery inspection belongs to an obsolete ownership generation")
            try:
                result = self.target.recover(record, deadline)
            except BaseException as error:
                self._checkpoint(path, dict(record, error_type=type(error).__name__))
                raise
            self._checkpoint(path, dict(record, state="completed", result=result, result_sha256=digest(result)))
            return self.status(request["request_id"])
        finally:
            self._run_lock.release()

    def dispatch(self, message, peer, *, received_at):
        if not isinstance(message, dict) or set(message) != {"operation", "inputs", "deadline", "clock_domain"}:
            raise ValueError("Malformed operator message")
        deadline = finite_seconds(message["deadline"])
        if message["clock_domain"] != local_clock_domain():
            raise ValueError("Operator client and server must share a Linux monotonic clock domain")
        if deadline > received_at + 60:
            raise ValueError("Operator operation deadline must be within 60 seconds")
        if self.closed.is_set() or monotonic() >= deadline:
            raise TimeoutError("Operator request expired before dispatch or endpoint closed")
        operation, inputs = message["operation"], message["inputs"]
        if operation == "inspect" and inputs == {}:
            return self.inspect(peer)
        if operation == "status" and isinstance(inputs, dict) and set(inputs) == {"request_id"}:
            return self.status(inputs["request_id"])
        if operation == "recover":
            return self.recover(inputs, peer, deadline)
        raise ValueError("Unknown operator operation")


class _OperatorHandler(socketserver.BaseRequestHandler):
    def handle(self):
        received_at = monotonic()
        try:
            peer = _peer(self.request)
            if peer["uid"] != self.server.uid:
                raise PermissionError("Operator UID is not authorized")
            message = _receive(self.request, received_at + 5, REQUEST_BYTES)
            result = self.server.endpoint.dispatch(message, peer, received_at=received_at)
            response = dict(ok=True, result=result)
        except Exception as error:
            response = dict(ok=False, error_type=type(error).__name__, error=str(error))
        try:
            _send(self.request, response, monotonic() + 2, RESPONSE_BYTES)
        except (OSError, ValueError, TimeoutError):
            pass  # Durable recovery/status survives a lost reply.


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class OperatorServer:
    def __init__(self, socket_path, target):
        if not hasattr(socket, "SO_PEERCRED"):
            raise RuntimeError("Operator endpoint requires Linux peer credentials")
        self.path = Path(socket_path)
        _private_directory(self.path.parent)
        if len(os.fsencode(self.path)) >= 108:
            raise ValueError("Operator Unix socket path is too long; use a private runtime directory")
        self._owner = self.path.with_name(self.path.name + ".lock").open("a")
        self.server = self.thread = None
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists() or self.path.is_symlink():
                info = self.path.lstat()
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
                    raise PermissionError("Refusing to replace an unrelated operator path")
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(.2)
                    try:
                        probe.connect(str(self.path))
                    except ConnectionRefusedError:
                        self.path.unlink()  # Dead listener, under our process lock.
                    else:
                        raise RuntimeError("Operator endpoint is already live")
            self.server = _Server(str(self.path), _OperatorHandler)
            os.chmod(self.path, 0o600)
            self.server.uid, self.server.endpoint = os.geteuid(), OperatorEndpoint(target)
            self._inode = self.path.stat().st_ino
            self.thread = Thread(target=lambda: self.server.serve_forever(poll_interval=.05), name="operator-socket", daemon=True)
            self.thread.start()
        except BaseException:
            if self.server:
                self.server.server_close()
            self._owner.close()
            raise

    def close(self):
        if self._owner.closed:
            return
        self.server.endpoint.closed.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.path.exists() and self.path.lstat().st_ino == self._inode:
            self.path.unlink()
        self._owner.close()


class OperatorClient:
    def __init__(self, socket_path):
        self.path = Path(socket_path)

    def call(self, operation, inputs=None, *, timeout_seconds=10):
        finite_seconds(timeout_seconds)
        if timeout_seconds > 60:
            raise ValueError("Operator operation timeout must not exceed 60 seconds")
        deadline = monotonic() + timeout_seconds
        if not self.path.is_absolute() or self.path.parent.resolve() != self.path.parent:
            raise ValueError("Operator socket must use an absolute physical path")
        info = self.path.lstat()
        parent = self.path.parent.stat()
        if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid() or
                stat.S_IMODE(info.st_mode) & 0o077 or self.path.parent.is_symlink() or
                parent.st_uid != os.geteuid() or stat.S_IMODE(parent.st_mode) & 0o077):
            raise PermissionError("Operator endpoint must be a private socket owned by this UID")
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(max(.001, deadline - monotonic()))
            connection.connect(str(self.path))
            if _peer(connection)["uid"] != os.geteuid():
                raise PermissionError("Unexpected operator server UID")
            _send(connection, dict(operation=operation, inputs={} if inputs is None else inputs,
                                   deadline=deadline, clock_domain=local_clock_domain()), deadline, REQUEST_BYTES)
            response = _receive(connection, deadline, RESPONSE_BYTES)
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error_type", "OperatorError") + ": " + response.get("error", "Invalid response"))
        return response["result"]
