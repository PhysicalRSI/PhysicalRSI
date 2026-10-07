"""Bounded mutually authenticated HTTP for controller and actuator services."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import socket
from threading import BoundedSemaphore, Lock, Timer
from time import monotonic
import uuid

from ..storage import atomic_json, canonical, digest
from .http_rpc import _NumpyEncoder, _from_json
from .rpc_facade import make_error_response
from .tls import ServerTLS, fingerprint


class SocketDeadline:
    """Close the current socket even while reads trickle or TLS handshakes stall."""

    def __init__(self, wire, seconds):
        self._wire, self._lock, self.expired = wire, Lock(), False
        self.timer = Timer(seconds, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        with self._lock:
            self.expired = True
            try:
                self._wire.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def wrap(self, context, **kwargs):
        # wrap_socket with handshake disabled performs no network I/O. Keep
        # descriptor transfer atomic with respect to the deadline watchdog.
        with self._lock:
            if self.expired:
                raise TimeoutError("Transport deadline reached before TLS setup")
            self._wire = context.wrap_socket(self._wire, do_handshake_on_connect=False, **kwargs)
            return self._wire

    def cancel(self):
        self.timer.cancel()
        self.timer.join()


def _strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate RPC JSON member")
            result[key] = value
        return result
    value = json.loads(body, object_pairs_hook=pairs)
    canonical(value)
    return value


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.close_connection = True
        try:
            if self.path != "/call" or self.headers.get("Transfer-Encoding") is not None:
                raise ValueError("Expected one length-delimited /call request")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise ValueError("Expected one explicit Content-Length")
            length = int(lengths[0])
            if not 0 < length <= self.server.policy.request_bytes:
                raise ValueError("RPC request exceeds the configured size limit")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("RPC request ended before its declared length")
            request = _strict_json(body)
            if (not isinstance(request, dict) or set(request) - {"method", "args", "kwargs", "session_id"}
                    or not isinstance(request.get("method"), str) or not re.fullmatch(r"[a-zA-Z0-9_.]{1,128}", request["method"])
                    or not isinstance(request.get("args", []), list)
                    or not isinstance(request.get("kwargs", {}), dict)):
                raise ValueError("Malformed RPC request")
            received_at = self.request._transport_received_at
            self.request._read_deadline.cancel()
            if self.request._read_deadline.expired:
                raise TimeoutError("RPC request expired while being read")
            self.request.settimeout(2)  # Bound response writes, not SDK execution.
            method = request["method"]
            peer = self.request._authenticated_peer
            grant = self.server.policy.peers[peer]
            allowed = method in grant.methods
            kwargs = request.get("kwargs", {})
            inputs = kwargs.get("inputs", {})
            request_id = kwargs.get("request_id") if "request_id" in kwargs else inputs.get("request_id") if isinstance(inputs, dict) else None
            if not isinstance(request_id, str) or len(request_id) > 128:
                request_id = None
            # Save authenticated provenance without persisting bearer tokens or
            # arbitrary request payloads. Audit failure prevents dispatch.
            atomic_json(self.server.audit_root / "access" / (uuid.uuid4().hex + ".json"),
                dict(schema="physicalrsi.transport-access/v1", principal=grant.principal, peer_sha256=peer,
                     policy_sha256=self.server.policy_sha256, method=method, request_id=request_id,
                     request_sha256=hashlib.sha256(body).hexdigest(), authorized=allowed, observed_at=monotonic(),
                     tls_version=self.request.version(), cipher=self.request.cipher()[0]))
            if not allowed:
                raise PermissionError("Authenticated principal is not allowed to call this RPC method")
            args = tuple(_from_json(value) for value in request.get("args", []))
            kwargs = {key: _from_json(value) for key, value in kwargs.items()}
            session = request.get("session_id")
            result = self.server.dispatch(method, args, kwargs, session_id=session, received_at=received_at)
            response = dict(ok=True, result=result)
        except Exception as error:
            response = make_error_response(error)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response, cls=_NumpyEncoder, allow_nan=False).encode())


class AuthenticatedHttpServer(ThreadingHTTPServer):
    scheme = "https"
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, dispatch, *, policy, audit_root):
        self.policy, self.dispatch, self.audit_root = policy, dispatch, Path(audit_root)
        self.policy_sha256 = digest(policy.identity())
        atomic_json(self.audit_root / "policies" / (self.policy_sha256 + ".json"), policy.identity())
        self._slots = BoundedSemaphore(policy.max_connections)
        super().__init__(address, _Handler)

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            return super().process_request(request, address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, address):
        received_at = monotonic()
        deadline = SocketDeadline(request, self.policy.read_seconds)
        try:
            request.settimeout(self.policy.read_seconds)
            request = deadline.wrap(self.policy.context, server_side=True)
            request.do_handshake()
            peer = fingerprint(request.getpeercert(binary_form=True))
            if deadline.expired or peer not in self.policy.peers:
                raise PermissionError("Client certificate is not admitted by this service")
            request._authenticated_peer, request._read_deadline = peer, deadline
            request._transport_received_at = received_at
            self.finish_request(request, address)
        except (OSError, ValueError, PermissionError):
            pass  # Rejected/truncated TLS connections never reach dispatch.
        finally:
            deadline.cancel()
            self.shutdown_request(request)
            self._slots.release()


class ControlTransport:
    """Only device hosts opt into this policy; generic model RPC is unchanged."""

    def configure_transport(self, tls, audit_root):
        if tls is not None and not isinstance(tls, ServerTLS):
            raise ValueError("Expected a ServerTLS policy")
        self._transport_tls, self._transport_audit_root = tls, audit_root

    def _make_rpc_server(self, transport, host, port, dispatch):
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        if self._transport_tls is None:
            if not local:
                raise ValueError("Non-loopback control services require explicit mutual TLS")
            return super()._make_rpc_server(transport, host, port, dispatch)
        if transport != "http":
            raise ValueError("Mutual TLS control services require the JSON HTTP transport")
        def authenticated(method, args, kwargs, *, session_id, received_at):
            return self._dispatch_authenticated(dispatch, method, args, kwargs, session_id=session_id, received_at=received_at)
        return AuthenticatedHttpServer((host, port), authenticated, policy=self._transport_tls, audit_root=self._transport_audit_root)

    def _dispatch_authenticated(self, dispatch, method, args, kwargs, *, session_id, received_at):
        return dispatch(method, args, kwargs, session_id=session_id)
