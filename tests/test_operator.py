from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import tempfile
from threading import Event
from time import monotonic, sleep

import pytest

from PhysicalRSI_core.infra.actuation import local_clock_domain
from PhysicalRSI_core.infra.operator import (
    OperatorClient, OperatorServer, RecoveryTarget, REQUEST_BYTES, _receive, recovery_request,
)
from PhysicalRSI_core.infra.storage import atomic_json, canonical, digest, read_json
from test_actuation import acquire, setup, submit
from test_controller_rpc import remote, acquire as acquire_controller, context
from PhysicalRSI_core.infra.rpc import RpcError
from PhysicalRSI_core.timing import action_chunk


@pytest.fixture
def local_operator(tmp_path):
    clock, backend, guard = setup(tmp_path / "actuator")
    acquire(guard)
    with tempfile.TemporaryDirectory(prefix="prsi-op-") as directory:
        path = Path(directory) / "operator.sock"
        server = OperatorServer(path, RecoveryTarget(guard, kind="actuator"))
        try:
            yield OperatorClient(path), server, guard, backend, clock
        finally:
            server.close()
            guard.close()


def request(client, name="recover"):
    snapshot = client.call("inspect")
    return recovery_request(snapshot, request_id=name, reason="Inspect abandoned fixture owner",
                            evidence=dict(check="fixture output and measured velocity inspected"))


def test_recovery_receipt_binds_kernel_peer_and_measurement_without_repeated_stop(local_operator):
    client, server, guard, backend, _ = local_operator
    first, stale = request(client), request(client, "stale")
    assert stat.S_IMODE(server.path.stat().st_mode) == 0o600
    result = client.call("recover", first)
    assert result["state"] == "completed"
    assert result["peer"] == dict(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
    assert result["result"]["measurement"]["quiescent"] is True
    assert digest(result["result"]["measurement"]) == result["result"]["evidence_sha256"]
    assert guard.authority.inspect()["phase"] == "free"
    # A historical reply is safe even when a replacement currently has output.
    successor = acquire(guard, "replacement")
    submit(guard, successor)
    stops = backend.stops
    assert client.call("recover", first) == client.call("status", {"request_id": "recover"}) == result
    with pytest.raises(RuntimeError, match="obsolete ownership generation"):
        client.call("recover", stale)
    assert backend.stops == stops and backend.output == 1
    saved = read_json(guard.root / "operator/requests/recover.json")
    intent = read_json(guard.root / "intents" / (saved["core_request_id"] + ".json"))
    assert intent["inputs"]["operator"] == "unix-uid:" + str(os.geteuid())
    assert intent["inputs"]["evidence"]["authenticated_peer"] == result["peer"]
    assert "credential" not in canonical(result).decode()


def test_unknown_stop_requires_fresh_inspection_and_preserves_old_result(local_operator):
    client, _, guard, backend, _ = local_operator
    attempt = request(client)
    backend.speed = 1
    with pytest.raises(RuntimeError, match="motion has not settled"):
        client.call("recover", attempt)
    status = client.call("status", {"request_id": "recover"})
    assert status["state"] == "started" and status["core_status"]["state"] == "uncertain"
    assert guard.authority.inspect()["phase"] == "recovery_required"
    stops = backend.stops
    backend.speed = 0
    with pytest.raises(RuntimeError, match="uncertain device effect"):
        client.call("recover", attempt)
    with pytest.raises(RuntimeError, match="inspect the device again"):
        client.call("recover", dict(attempt, request_id="renamed"))
    assert backend.stops == stops
    recovered = client.call("recover", request(client, "settled-after-inspection"))
    assert recovered["state"] == "completed" and guard.authority.inspect()["phase"] == "free"
    assert client.call("status", {"request_id": "recover"}) == status


@pytest.mark.parametrize("field", ["target", "inspection", "request", "snapshot"])
def test_altered_requests_and_inspection_evidence_never_stop(local_operator, field):
    client, _, guard, backend, _ = local_operator
    attempt = request(client)
    stops = backend.stops
    if field == "target":
        attempt["target"]["authority_id"] = "another-device"
    elif field == "inspection":
        attempt["inspection"]["sha256"] = "0" * 64
    elif field == "request":
        attempt["operator"] = "root"
    else:
        path = guard.root / "operator/inspections" / (attempt["inspection"]["id"] + ".json")
        snapshot = read_json(path)
        snapshot["observation"]["quiescence"]["quiescent"] = False
        atomic_json(path, snapshot)
    with pytest.raises(RuntimeError):
        client.call("recover", attempt)
    assert backend.stops == stops and guard.authority.inspect()["phase"] == "held"


def test_lost_outer_checkpoint_recovers_core_receipt_with_original_peer(local_operator, monkeypatch):
    import PhysicalRSI_core.infra.operator as module

    client, _, guard, backend, _ = local_operator
    attempt = request(client)
    once = [True]

    def fail_checkpoint(path, value):
        if Path(path) == guard.root / "operator/requests/recover.json" and value["state"] == "completed" and once[0]:
            once[0] = False
            raise OSError("injected outer receipt failure")
        atomic_json(path, value)

    monkeypatch.setattr(module, "atomic_json", fail_checkpoint)
    with pytest.raises(RuntimeError, match="injected outer receipt failure"):
        client.call("recover", attempt)
    observed = client.call("status", {"request_id": "recover"})
    assert observed["state"] == "started" and observed["core_status"]["state"] == "completed"
    successor = acquire(guard, "new-owner")
    submit(guard, successor)
    stops = backend.stops
    # A new CLI PID must retain the original authenticated peer in core inputs.
    code = "from PhysicalRSI_core.infra.operator import OperatorClient; import json,sys; print(json.dumps(OperatorClient(sys.argv[1]).call('recover', json.loads(sys.argv[2]))))"
    child = subprocess.run([sys.executable, "-c", code, str(client.path), json.dumps(attempt)], capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    completed = json.loads(child.stdout)
    assert completed["state"] == "completed" and completed["peer"] == observed["peer"]
    assert backend.stops == stops and backend.output == 1


def test_lost_response_is_resolved_by_status_without_recovery_replay(local_operator, monkeypatch):
    client, server, _, backend, _ = local_operator
    attempt = request(client)
    dispatch = server.server.endpoint.dispatch
    release = Event()

    def delayed(message, peer, **kwargs):
        result = dispatch(message, peer, **kwargs)
        if message["operation"] == "recover":
            release.wait(5)
        return result

    monkeypatch.setattr(server.server.endpoint, "dispatch", delayed)
    try:
        with pytest.raises(TimeoutError):
            client.call("recover", attempt, timeout_seconds=.2)
        stops = backend.stops
        assert client.call("status", {"request_id": "recover"})["state"] == "completed"
        assert backend.stops == stops
    finally:
        release.set()


def test_late_listener_error_cannot_overwrite_completed_recovery_receipt(local_operator, monkeypatch):
    client, first, guard, backend, _ = local_operator
    attempt = request(client)
    target = first.server.endpoint.target
    recover = target.recover
    entered, release = Event(), Event()

    def delayed_error(*args):
        recover(*args)
        entered.set()
        assert release.wait(5)
        raise OSError("lost listener after core completion")

    monkeypatch.setattr(target, "recover", delayed_error)
    second = OperatorServer(client.path.parent / "replacement.sock", RecoveryTarget(guard, kind="actuator"))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.call, "recover", attempt)
            try:
                assert entered.wait(3)
                stops = backend.stops
                completed = OperatorClient(second.path).call("recover", attempt)
                assert completed["state"] == "completed" and backend.stops == stops
            finally:
                release.set()
            with pytest.raises(RuntimeError, match="lost listener"):
                pending.result(timeout=3)
        assert client.call("status", {"request_id": "recover"}) == completed
    finally:
        second.close()


def test_status_remains_live_while_backend_stop_is_blocked(local_operator, monkeypatch):
    client, _, _, backend, _ = local_operator
    attempt = request(client)
    entered, release = Event(), Event()
    stop = backend.stop

    def blocked():
        stop()
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(backend, "stop", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.call, "recover", attempt)
        try:
            assert entered.wait(3)
            started = client.call("status", {"request_id": "recover"}, timeout_seconds=.5)
            assert started["core_status"]["state"] == "started"
            with pytest.raises(TimeoutError):
                client.call("recover", attempt, timeout_seconds=.1)
        finally:
            release.set()
        assert future.result(timeout=3)["state"] == "completed"


@pytest.mark.parametrize("mode", ["oversize", "duplicate", "clock", "slow"])
def test_invalid_or_late_frames_never_dispatch_recovery(local_operator, mode):
    client, _, guard, backend, _ = local_operator
    attempt = request(client)
    stops = backend.stops
    message = dict(operation="recover", inputs=attempt, deadline=monotonic() + (0.1 if mode == "slow" else 5),
                   clock_domain=local_clock_domain())
    if mode == "clock":
        message["clock_domain"] = "another-host"
    payload = canonical(message)
    if mode == "duplicate":
        payload = b'{"operation":"inspect",' + payload[1:]
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(str(client.path))
        if mode == "oversize":
            connection.sendall(struct.pack("!I", REQUEST_BYTES + 1))
        elif mode == "slow":
            connection.sendall(struct.pack("!I", len(payload)) + payload[:1])
            sleep(.15)
            connection.sendall(payload[1:])
        else:
            connection.sendall(struct.pack("!I", len(payload)) + payload)
        response = _receive(connection, monotonic() + 2, 1024 * 1024)
    assert response["ok"] is False
    assert backend.stops == stops and guard.authority.inspect()["phase"] == "held"
    assert client.call("status", {"request_id": "recover"})["state"] == "missing"


def test_private_socket_paths_and_existing_listeners_are_preserved(local_operator):
    client, server, guard, _, _ = local_operator
    target = RecoveryTarget(guard, kind="actuator")
    with pytest.raises(BlockingIOError):
        OperatorServer(client.path, target)
    unrelated = client.path.parent / "important"
    unrelated.write_text("retain")
    with pytest.raises(PermissionError, match="unrelated"):
        OperatorServer(unrelated, target)
    assert unrelated.read_text() == "retain"
    with socket.socket(socket.AF_UNIX) as live:
        path = client.path.parent / "other.sock"
        live.bind(str(path))
        live.listen()
        with pytest.raises(RuntimeError, match="already live"):
            OperatorServer(path, target)
        assert path.is_socket()
    os.chmod(server.path, 0o666)
    with pytest.raises(PermissionError, match="private socket"):
        client.call("inspect")
    os.chmod(server.path, 0o600)
    os.chmod(server.path.parent, 0o755)
    with pytest.raises(PermissionError, match="private"):
        OperatorServer(server.path.parent / "open.sock", target)
    os.chmod(server.path.parent, 0o700)


@pytest.mark.skipif(os.geteuid() != 0, reason="Cross-UID process test requires setuid privilege")
def test_kernel_peer_credentials_reject_other_uid_even_if_filesystem_mode_is_open(local_operator):
    client, server, _, backend, _ = local_operator
    os.chmod(server.path.parent, 0o755)
    os.chmod(server.path, 0o666)
    stops = backend.stops
    # Use only stdlib from an unprivileged child; bypass the client's own mode
    # checks to verify that the server itself authenticates kernel credentials.
    code = "import socket,sys,json,struct; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); n=struct.unpack('!I',s.recv(4))[0]; print(s.recv(n).decode())"
    try:
        child = subprocess.run([sys.executable, "-c", code, str(client.path)], user=65534, group=65534,
                               extra_groups=(), cwd="/tmp", capture_output=True, text=True, timeout=5)
        assert child.returncode == 0, child.stderr
        assert json.loads(child.stdout)["error_type"] == "PermissionError"
        assert backend.stops == stops
    finally:
        os.chmod(server.path, 0o600)
        os.chmod(server.path.parent, 0o700)


def test_public_cli_prepares_and_executes_reviewed_artifacts(local_operator, tmp_path):
    client, _, _, backend, _ = local_operator
    launcher = Path(__file__).resolve().parents[1] / "physicalrsi"

    def cli(*args):
        result = subprocess.run([str(launcher), "operator", *map(str, args)], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    inspected = cli("inspect", "--socket", client.path, "--output", tmp_path / "inspection.json")
    atomic_json(tmp_path / "evidence.json", dict(inspected=inspected["id"], measured_stop=True))
    stops = backend.stops
    prepared = cli("prepare", "--inspection", tmp_path / "inspection.json", "--request-id", "cli-recovery",
                   "--reason", "Reviewed fixture stop", "--evidence", tmp_path / "evidence.json", "--output", tmp_path / "request.json")
    assert backend.stops == stops and prepared["target"]["generation"] == 1
    # A bad output destination is rejected before the recovery request is sent.
    blocked = subprocess.run([str(launcher), "operator", "recover", "--socket", str(client.path), "--request", str(tmp_path / "request.json"),
                              "--output", str(tmp_path / "request.json")], capture_output=True, text=True, timeout=10)
    assert blocked.returncode == 1 and backend.stops == stops
    completed = cli("recover", "--socket", client.path, "--request", tmp_path / "request.json", "--output", tmp_path / "receipt.json")
    assert completed["state"] == "completed"
    assert cli("status", "--socket", client.path, "--request-id", "cli-recovery") == completed
    assert read_json(tmp_path / "receipt.json") == completed


@pytest.mark.parametrize("remote", [{"operator": True}], indirect=True)
def test_controller_recovery_revokes_live_execution_and_stops_on_original_thread(remote):
    controller, rpc, entered, _, _, root, _ = remote
    client = OperatorClient(Path((root / "operator-path.txt").read_text()))
    claim = acquire_controller(controller)
    initial = controller.reset("reset", "episode", {}, context(), claim=claim)["observation"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(controller.step, "active-action", action_chunk(initial, [1], period_seconds=.1), context(), claim=claim)
        assert entered.wait(3)
        attempt = request(client)
        assert client.call("recover", attempt)["state"] == "completed"
        with pytest.raises(RpcError, match="revoked"):
            running.result(timeout=3)
    assert controller.status("active-action")["state"] == "uncertain"
    assert controller.state()["phase"] == "idle" and controller.ownership()["phase"] == "free"
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute", "quiesce"]
    successor = acquire_controller(controller, "successor")
    assert client.call("recover", attempt)["state"] == "completed"
    assert controller.ownership()["phase"] == "held"
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute", "quiesce"]
    with pytest.raises(RpcError, match="unknown RPC"):
        rpc.call("operator.recover", kwargs={"request": attempt}, timeout_s=2)
    with pytest.raises(RpcError, match="Unknown controller operation"):
        rpc.call("control.execute", kwargs=dict(request_id="forbidden", operation="recover", inputs={}, budget_seconds=2), timeout_s=2)
    assert controller.quiesce("release-successor", "episode", context(), claim=successor)["quiescent"]


@pytest.mark.parametrize("remote", [{"operator": True}], indirect=True)
def test_expired_operator_stop_is_not_executed_later_from_controller_queue(remote):
    controller, rpc, entered, release, _, root, _ = remote
    client = OperatorClient(Path((root / "operator-path.txt").read_text()))
    claim = acquire_controller(controller)
    initial = controller.reset("reset", "episode", {}, context(), claim=claim)["observation"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(controller.step, "blocked-action", action_chunk(initial, [2], period_seconds=.1), context(), claim=claim)
        assert entered.wait(3)
        attempt = request(client)
        try:
            with pytest.raises(TimeoutError):
                client.call("recover", attempt, timeout_seconds=.15)
            status = client.call("status", {"request_id": "recover"})
            assert status["state"] == "started" and status["core_status"]["state"] == "missing"
            assert controller.ownership()["phase"] == "recovery_required"
        finally:
            release.set()
        with pytest.raises(RpcError, match="revoked"):
            running.result(timeout=3)
    rpc.call("control.describe", timeout_s=3)  # Main-thread barrier after the expired callback.
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute"]
    # No device effect was ever dispatched. Explicit delivery of the same
    # immutable request is permitted; changing its ID is still forbidden.
    assert client.call("recover", attempt)["state"] == "completed"
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute", "quiesce"]
