from contextlib import contextmanager
import json
import secrets
import shutil
import socket
import ssl
from threading import Event, Thread
from time import monotonic, sleep

import pytest

from PhysicalRSI_core.infra.actuation_rpc import ActuationHost
from PhysicalRSI_core.infra.controller_rpc import ControllerHost, controller_transport
from PhysicalRSI_core.infra.rpc.authenticated_http import AuthenticatedHttpServer
from PhysicalRSI_core.infra.rpc import RpcError
from PhysicalRSI_core.infra.rpc.tls import ClientTLS, ServerTLS, PeerGrant, certificate_fingerprint
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_demos.tls_fixture import certificates
from test_actuation import setup


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    if not shutil.which("openssl"):
        pytest.skip("OpenSSL CLI is required for ephemeral test certificates")
    return certificates(tmp_path_factory.mktemp("control-tls-certificates"))


def client_policy(pki, name="client", **overrides):
    return ClientTLS(**(dict(ca=pki["ca"], **pki[name], server_sha256=certificate_fingerprint(pki["server"]["certificate"])) | overrides))


def server_policy(pki, **overrides):
    return ServerTLS(**(dict(ca=pki["ca"], **pki["server"], read_seconds=.4,
        peers={certificate_fingerprint(pki["client"]["certificate"]): PeerGrant("executor", ("effect", "read")),
               certificate_fingerprint(pki["observer"]["certificate"]): PeerGrant("inspector", ("read",))}) | overrides))


@contextmanager
def serve(tmp_path, policy):
    effects = []

    def dispatch(method, args, kwargs, **metadata):
        if method == "effect":
            effects.append(kwargs)
        return dict(effects=len(effects), value="ok")

    server = AuthenticatedHttpServer(("127.0.0.1", 0), dispatch, policy=policy, audit_root=tmp_path / "transport")
    thread = Thread(target=server.serve_forever, kwargs=dict(poll_interval=.02), daemon=True)
    thread.start()
    endpoint = "https://127.0.0.1:" + str(server.server_address[1])
    try:
        yield server, endpoint, effects, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.fixture
def secured(tmp_path, pki, request):
    with serve(tmp_path, server_policy(pki, **getattr(request, "param", {}))) as value:
        yield value


def test_mutual_tls_permissions_and_audit_bind_actual_peer_without_credentials(secured, pki):
    _, endpoint, effects, root = secured
    observer = controller_transport(endpoint, tls=client_policy(pki, "observer"))
    assert observer.call("read", timeout_s=2)["effects"] == 0
    with pytest.raises(RpcError, match="not allowed"):
        observer.call("effect", kwargs=dict(request_id="denied"), timeout_s=2)
    with pytest.raises(RpcError, match="not allowed"):
        observer.call("request.execute", kwargs=dict(method="effect", kwargs={}), timeout_s=2)
    assert not effects
    token = secrets.token_hex(32)
    client = controller_transport(endpoint, tls=client_policy(pki))
    assert client.call("effect", kwargs=dict(request_id="owned", credential=token, principal="forged-name"), timeout_s=2)["effects"] == 1
    records = [read_json(path) for path in (root / "transport/access").glob("*.json")]
    owned = next(record for record in records if record["request_id"] == "owned")
    assert owned["principal"] == "executor" and owned["peer_sha256"] == certificate_fingerprint(pki["client"]["certificate"])
    assert owned["tls_version"] in {"TLSv1.2", "TLSv1.3"}
    assert token not in json.dumps(records)
    assert next(record for record in records if record["request_id"] == "denied")["authorized"] is False


def test_client_certificate_is_required_and_ca_membership_alone_does_not_grant_access(secured, pki):
    _, endpoint, effects, _ = secured
    with pytest.raises(RpcError):
        controller_transport(endpoint, tls=client_policy(pki, "outsider")).call("effect", timeout_s=2)
    context = ssl.create_default_context(cafile=pki["ca"])
    port = int(endpoint.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=2) as raw:
        try:
            with context.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
                tls.sendall(b'POST /call HTTP/1.0\r\nContent-Length: 19\r\n\r\n{"method":"effect"}')
                assert not tls.recv(4096)
        except ssl.SSLError:
            pass
    assert not effects


def test_untrusted_ca_and_wrong_server_pin_cannot_dispatch(secured, pki, tmp_path):
    _, endpoint, effects, _ = secured
    with pytest.raises(ValueError, match="pinned identity"):
        controller_transport(endpoint, tls=client_policy(pki, server_sha256="0" * 64)).call("effect", timeout_s=2)
    other = certificates(tmp_path / "foreign")
    with pytest.raises(RpcError):
        controller_transport(endpoint, tls=client_policy(other)).call("effect", timeout_s=2)
    assert not effects


def test_trusted_pinned_server_still_requires_matching_ip_san(tmp_path):
    if not shutil.which("openssl"):
        pytest.skip("OpenSSL CLI is required for ephemeral test certificates")
    pki = certificates(tmp_path / "certificates", server_ip="127.0.0.2")
    with serve(tmp_path, server_policy(pki)) as (_, endpoint, effects, _):
        with pytest.raises(RpcError, match="IP address mismatch"):
            controller_transport(endpoint, tls=client_policy(pki)).call("effect", timeout_s=2)
        assert not effects


def test_audit_failure_prevents_authorized_dispatch(secured, pki, monkeypatch):
    _, endpoint, effects, _ = secured

    def unavailable(*args, **kwargs):
        raise OSError("Audit storage unavailable")

    monkeypatch.setattr("PhysicalRSI_core.infra.rpc.authenticated_http.atomic_json", unavailable)
    with pytest.raises(RpcError, match="Audit storage unavailable"):
        controller_transport(endpoint, tls=client_policy(pki)).call("effect", timeout_s=2)
    assert not effects


def test_authentication_and_audit_time_count_toward_controller_budget(tmp_path, pki, monkeypatch):
    from PhysicalRSI_core.infra.rpc import authenticated_http
    from types import SimpleNamespace

    effects = []
    gateway = SimpleNamespace(root=tmp_path, identity=lambda: {}, acquire=lambda *args, **kwargs: effects.append("acquire"))
    policy = server_policy(pki, peers={certificate_fingerprint(pki["client"]["certificate"]):
        PeerGrant("executor", ("control.execute",))})
    host = ControllerHost(gateway, tls=policy)
    server = host._make_rpc_server("http", "127.0.0.1", 0, host._dispatch_main_thread)
    original = authenticated_http.atomic_json

    def slow_audit(path, value):
        sleep(.1)
        original(path, value)

    monkeypatch.setattr(authenticated_http, "atomic_json", slow_audit)
    thread = Thread(target=server.serve_forever, kwargs=dict(poll_interval=.02), daemon=True)
    thread.start()
    try:
        client = controller_transport("https://127.0.0.1:" + str(server.server_address[1]), tls=client_policy(pki))
        with pytest.raises(RpcError, match="deadline"):
            client.call("control.execute", kwargs=dict(request_id="expired", operation="acquire", inputs={},
                budget_seconds=.05, _received_at=monotonic() + 9999), timeout_s=2)
        assert not effects
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("secured", [dict(request_bytes=128, max_connections=1, read_seconds=.3)], indirect=True)
def test_request_size_and_slow_body_and_connection_admission_are_bounded(secured, pki):
    server, endpoint, effects, _ = secured
    client = controller_transport(endpoint, tls=client_policy(pki))
    with pytest.raises(RpcError, match="size limit"):
        client.call("effect", kwargs=dict(value="x" * 200), timeout_s=2)
    # Receiving the response can precede the worker's finally/release. Wait for
    # that completed request to return the sole slot before testing a new one.
    assert server._slots.acquire(timeout=2), "Rejected request did not return its connection slot"
    server._slots.release()
    # A fully authenticated but incomplete body occupies the one admitted slot.
    port = int(endpoint.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=2) as raw:
        with client_policy(pki).context.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
            tls.sendall(b'POST /call HTTP/1.0\r\nContent-Length: 100\r\n\r\n{"method":')
            with pytest.raises(RpcError):
                client.call("effect", timeout_s=2)
            assert tls.recv(4096) == b""
    assert server._slots.acquire(timeout=2), "Expired reader did not return its connection slot"
    server._slots.release()
    assert client.call("read", timeout_s=2)["effects"] == 0
    assert not effects


@pytest.mark.parametrize("methods", ["effect", ("*",), ("read", {}), ()])
def test_ambiguous_method_grants_are_rejected(methods):
    with pytest.raises(ValueError, match="exact RPC method"):
        PeerGrant("caller", methods)


def test_no_plaintext_downgrade_or_unprotected_nonlocal_control_listener(tmp_path, pki):
    with pytest.raises(ValueError, match="downgrade"):
        controller_transport("http://127.0.0.1:1", tls=client_policy(pki))
    with pytest.raises(ValueError, match="explicit mutual TLS"):
        controller_transport("https://127.0.0.1:1")
    with pytest.raises(ValueError, match="Non-loopback"):
        controller_transport("http://192.0.2.1:1")
    _, _, guard = setup(tmp_path)
    try:
        host = ActuationHost(guard)
        with pytest.raises(ValueError, match="Non-loopback"):
            host._make_rpc_server("http", "0.0.0.0", 0, host._dispatch)
        host = ActuationHost(guard, tls=server_policy(pki))
        with pytest.raises(ValueError, match="JSON HTTP"):
            host._make_rpc_server("socket", "127.0.0.1", 0, host._dispatch)
    finally:
        guard.close()


def test_stalled_handshake_cannot_block_accept_loop_or_extend_client_deadline(secured, pki):
    _, endpoint, effects, _ = secured
    port = int(endpoint.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=2) as stalled:
        assert controller_transport(endpoint, tls=client_policy(pki)).call("read", timeout_s=2)["effects"] == 0
        stalled.settimeout(2)
        assert stalled.recv(1) == b""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        release = Event()

        def peer():
            connection, _ = listener.accept()
            with connection:
                release.wait(3)

        thread = Thread(target=peer)
        thread.start()
        started = monotonic()
        try:
            client = controller_transport("https://127.0.0.1:" + str(listener.getsockname()[1]), tls=client_policy(pki))
            with pytest.raises(TimeoutError):
                client.call("effect", timeout_s=.15)
            assert monotonic() - started < 1
        finally:
            release.set()
            thread.join(2)
    assert not effects


def test_trickled_tls_response_cannot_extend_total_client_deadline(pki):
    release = Event()
    policy = server_policy(pki)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(2)

        def peer():
            try:
                raw, _ = listener.accept()
                with policy.context.wrap_socket(raw, server_side=True) as tls:
                    tls.settimeout(2)
                    request = b""
                    while not request.endswith(b"\r\n\r\n"):
                        chunk = tls.recv(1)
                        if not chunk:
                            return
                        request += chunk
                    body = b'{"ok": true, "result": {"value": "completed"}}'
                    tls.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n")
                    for value in body:
                        if release.wait(.035):
                            break
                        tls.sendall(bytes((value,)))
            except OSError:
                pass  # Expected client deadline shutdown.

        thread = Thread(target=peer)
        thread.start()
        started = monotonic()
        try:
            client = controller_transport("https://127.0.0.1:" + str(listener.getsockname()[1]), tls=client_policy(pki))
            with pytest.raises(TimeoutError):
                client.call("read", timeout_s=.18)
            assert monotonic() - started < 1
        finally:
            release.set()
            thread.join(3)
            assert not thread.is_alive()


@pytest.mark.parametrize("body", [b'{"method":"read","method":"effect"}', b'{"method":"effect","kwargs":{"x":NaN}}'])
def test_ambiguous_json_never_reaches_dispatch(secured, pki, body):
    _, endpoint, effects, _ = secured
    port = int(endpoint.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=2) as raw:
        with client_policy(pki).context.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
            tls.sendall(b'POST /call HTTP/1.0\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
            response = b""
            while data := tls.recv(4096):
                response += data
    assert b'"ok": false' in response and not effects
