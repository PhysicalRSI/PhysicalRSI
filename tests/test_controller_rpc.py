import multiprocessing
import gc
import os
import secrets
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep
from threading import get_ident

import pytest

from PhysicalRSI_core.contracts import Context, Contract
from PhysicalRSI_core.embodiment import Embodiment
from PhysicalRSI_core.infra.controller import ControllerGateway
from PhysicalRSI_core.infra.controller_rpc import ControllerClient, ControllerHost, controller_transport
from PhysicalRSI_core.infra.rpc import RpcError, wait_for_ready
from PhysicalRSI_core.infra.rpc.tls import ClientTLS, ServerTLS, certificate_fingerprint
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_core.timing import ControlTiming, SensorSample, action_chunk


def _serve(root, pipe, entered, release, response_release, lease_seconds, operator_socket=None, tls=None):
    root = Path(root)
    os.environ["PHYSICALRSI_ENDPOINT_FILE"] = str(root / "endpoint.json")
    driver_thread = get_ident()

    class Driver:
        sequence = -1

        def identity(self):
            return {"name": "RPC-fault-fixture", "revision": "1"}

        def record(self, event):
            assert get_ident() == driver_thread, "SDK operation moved off its owning thread"
            with (root / "effects.txt").open("a") as stream:
                stream.write(event + "\n")

        def reset(self, case, context):
            self.record("reset")
            return {"ready": True}

        def observe(self, context):
            self.sequence += 1
            return SensorSample({"position": 0}, self.sequence, monotonic())

        def validate(self, actions):
            pass

        def execute(self, actions, **kwargs):
            self.record("execute")
            if actions in ([1], [2]):
                entered.set()
                until = monotonic() + 15
                while not release.wait(.02):
                    if actions == [1]:
                        kwargs["context"].check()
                    if monotonic() >= until:
                        raise TimeoutError("Fixture was not released")
            return {"terminated": False, "executed_steps": len(actions)}

        def quiesce(self, context):
            if operator_socket:
                self.record("quiesce")
            return {"quiescent": True}

    class DelayedResponseHost(ControllerHost):
        def execute(self, *args, **kwargs):
            result = super().execute(*args, **kwargs)
            if kwargs.get("request_id") == "lost-response":
                response_release.wait(15)
            return result

    spec = Embodiment("rpc-fixture", "1", "software", Contract("state"), Contract("action"),
                      resources=("rpc:fixture",), timing=ControlTiming(.1, 2, 30, 20))
    gateway = ControllerGateway(root / "controller", driver=Driver(), embodiment=spec, lease_seconds=lease_seconds)
    pipe.send(gateway.identity())
    pipe.close()
    # multiprocessing gives the child /dev/null stdin; its lifecycle is owned
    # by the fixture, unlike ManagedProcess's parent-watch pipe.
    DelayedResponseHost(gateway, operator_socket=operator_socket, tls=ServerTLS(**tls) if tls else None).serve(
        transport="http", host="127.0.0.1", port=0)


@pytest.fixture
def remote(tmp_path, request):
    # Expected RPC exceptions can retain the preceding test's Future/frame
    # cycle, including its Events. Finalize those semaphores before registering
    # new ones: Python 3.12's resource tracker cannot reenter during cleanup.
    gc.collect()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    entered, release, response_release = (context.Event() for _ in range(3))
    options = getattr(request, "param", {})
    lease_seconds = options.get("lease_seconds", 30)
    operator_directory = tempfile.TemporaryDirectory(prefix="prsi-control-op-")
    server_tls, client_tls = None, None
    if options.get("tls"):
        if not shutil.which("openssl"):
            operator_directory.cleanup()
            pytest.skip("OpenSSL CLI required for ephemeral TLS fixture")
        from PhysicalRSI_demos.tls_fixture import certificates
        pki = certificates(Path(operator_directory.name) / "certificates", names=("server", "client"))
        server_tls = dict(ca=pki["ca"], **pki["server"], peers={certificate_fingerprint(pki["client"]["certificate"]):
            dict(principal="controller-fixture", methods=["healthz", "service.describe", "shutdown", "control.describe",
                "control.state", "control.status", "control.ownership", "control.execute"])})
        client_tls = ClientTLS(ca=pki["ca"], **pki["client"], server_sha256=certificate_fingerprint(pki["server"]["certificate"]))
    operator_socket = str(Path(operator_directory.name) / "op.sock") if options.get("operator") else None
    if operator_socket:
        (tmp_path / "operator-path.txt").write_text(operator_socket)
    process = context.Process(target=_serve, args=(str(tmp_path), child, entered, release, response_release, lease_seconds, operator_socket, server_tls))
    process.start()
    child.close()
    rpc = None
    try:
        assert parent.poll(15), "Controller did not initialize"
        expected = parent.recv()
        until = monotonic() + 15
        while not (tmp_path / "endpoint.json").exists():
            assert process.is_alive() and monotonic() < until
            sleep(.02)
        rpc = controller_transport(read_json(tmp_path / "endpoint.json")["endpoint"], tls=client_tls)
        wait_for_ready(rpc, timeout_s=5)
        client = ControllerClient(rpc, expected=expected)
        yield client, rpc, entered, release, response_release, tmp_path, process
    finally:
        # A killed process can leave a multiprocessing Event's condition in
        # the middle of a wait. Do not notify abandoned synchronization state.
        if process.is_alive():
            release.set()
            response_release.set()
        if rpc is not None and process.is_alive():
            try:
                rpc.call("shutdown", timeout_s=2)
            except Exception:
                pass
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        operator_directory.cleanup()
        assert not process.is_alive()
        process.close()


def context(seconds=20):
    return Context("trial", deadline=monotonic() + seconds, harness_revision="candidate-v1")


def acquire(client, request_id="claim"):
    credential = secrets.token_hex(32)
    result = client.acquire(request_id, "rpc-owner", credential, context())
    assert result["state"] == "completed"
    return dict(result["claim"], credential=credential)


@pytest.mark.parametrize("remote", [{}, {"tls": True}], indirect=True)
def test_status_remains_live_and_queued_expired_reset_never_executes(remote):
    client, rpc, entered, release, _, root, process = remote
    claim = acquire(client)
    initial = client.reset("reset", "episode", {}, context(), claim=claim)["observation"]
    action = action_chunk(initial, [1], period_seconds=.1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.step, "long-action", action, context(), claim=claim)
        try:
            assert entered.wait(10) and process.is_alive()
            assert client.status("long-action")["state"] == "started"
            assert client.state()["phase"] == "executing"
            with pytest.raises(TimeoutError):
                rpc.call("control.execute", kwargs=dict(request_id="queued-reset", operation="reset",
                         inputs=dict(episode="other-episode", case={}, claim=claim), budget_seconds=.05,
                         harness_revision="candidate-v1", _received_at=monotonic() + 9999), timeout_s=.05)
        finally:
            release.set()
        result = future.result(timeout=5)
    # A main-thread barrier follows the expired queued request. A supplied future
    # receiver timestamp cannot extend its life because the host overwrites it.
    rpc.call("control.describe", timeout_s=5)
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute"]
    assert client.step("long-action", action, context(), claim=claim) == result
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute"]
    with pytest.raises(RpcError, match="unknown RPC|not allowed"):
        rpc.call("env.step", kwargs={"action": 1}, timeout_s=5)
    assert client.quiesce("stop", "episode", context(), claim=claim)["quiescent"] is True


@pytest.mark.parametrize("remote", [{}, {"tls": True}], indirect=True)
def test_lost_response_is_resolved_from_durable_status_without_reexecution(remote):
    client, _, _, _, response_release, root, _ = remote
    claim = acquire(client)
    initial = client.reset("reset", "episode", {}, context(), claim=claim)["observation"]
    action = action_chunk(initial, [0], period_seconds=.1)
    try:
        with pytest.raises(TimeoutError) as failure:
            client.step("lost-response", action, context(.3), claim=claim)
        assert failure.value.request_id == "lost-response"
        status = client.status("lost-response")
        assert status["state"] == "completed"
        assert status["result"]["control"]["executed_steps"] == 1
        assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute"]
    finally:
        response_release.set()
    assert client.step("lost-response", action, context(), claim=claim) == status["result"]
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute"]
    client.quiesce("stop", "episode", context(), claim=claim)


@pytest.mark.parametrize("remote", [{}, {"tls": True}], indirect=True)
def test_independent_rpc_clients_cannot_bypass_controller_ownership(remote):
    client, rpc, _, _, _, root, _ = remote
    other_rpc = controller_transport(read_json(root / "endpoint.json")["endpoint"], tls=rpc.tls)
    other = ControllerClient(other_rpc, expected=client.identity())
    credentials = [secrets.token_hex(32), secrets.token_hex(32)]
    clients = [client, other]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda index: clients[index].acquire(
            "grant-" + str(index), "owner-" + str(index), credentials[index], context()), range(2)))
    assert sorted(result["state"] for result in results) == ["completed", "rejected"]
    winner = next(index for index, result in enumerate(results) if result["state"] == "completed")
    claim = dict(results[winner]["claim"], credential=credentials[winner])
    forged = dict(claim, credential=credentials[1 - winner])
    assert clients[1 - winner].reset("forged-reset", "episode", {}, context(), claim=forged)["state"] == "rejected"
    assert clients[1 - winner].reset("missing-claim", "episode", {}, context())["state"] == "rejected"
    clients[winner].reset("owned-reset", "episode", {}, context(), claim=claim)
    clients[winner].quiesce("owned-stop", "episode", context(), claim=claim)
    assert (root / "effects.txt").read_text().splitlines() == ["reset"]
    assert client.ownership()["phase"] == "free"
    with pytest.raises(RpcError, match="Unknown controller operation"):
        other_rpc.call("control.execute", kwargs=dict(request_id="remote-recovery", operation="recover",
                       inputs={}, budget_seconds=2), timeout_s=2)


def test_revocation_is_live_during_execution_and_fences_queued_actions(remote):
    client, rpc, entered, release, _, root, _ = remote
    claim = acquire(client)
    initial = client.reset("reset", "episode", {}, context(), claim=claim)["observation"]
    action = action_chunk(initial, [1], period_seconds=.1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.step, "active-action", action, context(), claim=claim)
        try:
            assert entered.wait(5)
            # This request remains queued with a live request deadline after
            # its client stops waiting. Revocation must prevent its dispatch.
            with pytest.raises(TimeoutError):
                rpc.call("control.execute", kwargs=dict(request_id="queued-old-action", operation="step",
                         inputs=dict(command=action, claim=claim), budget_seconds=10,
                         harness_revision="candidate-v1"), timeout_s=.05)
            assert client.renew("live-heartbeat", claim, context())["state"] == "completed"
            revoked = client.revoke("revoke", claim, "Abort interrupted trial", context())
            assert revoked["quiescent"] is False
            with pytest.raises(RpcError, match="revoked"):
                future.result(timeout=5)
        finally:
            release.set()
    rpc.call("control.describe", timeout_s=5)  # Main-thread queue barrier.
    assert client.status("queued-old-action")["result"]["state"] == "rejected"
    assert client.status("active-action")["state"] == "uncertain"
    assert client.ownership()["phase"] == "recovery_required"
    assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute"]
    client.quiesce("verified-stop", "episode", context(), claim=claim)
    assert client.ownership()["phase"] == "free"


@pytest.mark.parametrize("remote", [{"lease_seconds": .5}], indirect=True)
def test_expired_remote_claim_cannot_renew_or_be_automatically_taken_over(remote):
    client, _, _, _, _, root, _ = remote
    claim = acquire(client)
    client.reset("reset", "episode", {}, context(), claim=claim)
    sleep(.55)
    assert client.renew("expired-heartbeat", claim, context())["state"] == "rejected"
    assert client.acquire("takeover", "next-owner", secrets.token_hex(32), context())["state"] == "rejected"
    assert client.ownership()["phase"] == "recovery_required"
    assert (root / "effects.txt").read_text().splitlines() == ["reset"]
    client.quiesce("stop", "episode", context(), claim=claim)
    assert client.ownership()["phase"] == "free"


def test_lost_grant_response_can_be_read_without_allocating_another_generation(remote):
    client, _, _, _, response_release, root, _ = remote
    credential = secrets.token_hex(32)
    try:
        with pytest.raises(TimeoutError):
            client.acquire("lost-response", "owner", credential, context(.3))
        status = client.status("lost-response")
        assert status["state"] == "completed" and status["result"]["claim"]["generation"] == 1
        assert not (root / "effects.txt").exists()
    finally:
        response_release.set()
    assert client.acquire("lost-response", "owner", credential, context()) == status["result"]
    claim = dict(status["result"]["claim"], credential=credential)
    client.reset("reset", "episode", {}, context(), claim=claim)
    client.quiesce("stop", "episode", context(), claim=claim)
    assert client.ownership()["generation"] == 1


def test_killed_controller_preserves_unknown_effect_and_requires_recovery(remote):
    client, _, entered, _, _, root, process = remote
    claim = acquire(client)
    initial = client.reset("reset", "episode", {}, context(), claim=claim)["observation"]
    action = action_chunk(initial, [1], period_seconds=.1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.step, "interrupted-action", action, context(), claim=claim)
        assert entered.wait(5)
        process.terminate()
        process.join(5)
        assert not process.is_alive()
        with pytest.raises(Exception):
            future.result(timeout=5)
        assert future.done()

    class RecoveryDriver:
        stops = 0

        def identity(self):
            return client.identity()["driver"]

        def reset(self, case, context):
            pytest.fail("Restart cannot repeat or take over an interrupted reset")

        def quiesce(self, context):
            self.stops += 1
            return dict(quiescent=True, scope="software fixture recovery")

    gateway = ControllerGateway(root / "controller", driver=RecoveryDriver(), embodiment=client.embodiment)
    try:
        assert gateway.ownership()["phase"] == "recovery_required"
        assert gateway.status("interrupted-action")["state"] == "started"
        assert gateway.reset("old-owner-reset", "new-episode", {}, context(), claim=claim)["state"] == "rejected"
        assert gateway.acquire("early-takeover", "new-owner", secrets.token_hex(32), context())["state"] == "rejected"
        recovered = gateway.recover("operator-recovery", generation=claim["generation"], operator="fixture-operator",
                                    reason="Recover terminated controller", evidence=dict(process_exit_code=process.exitcode),
                                    context=context())
        assert recovered["quiescent"] and gateway.driver.stops == 1
        next_claim = acquire(gateway, "new-generation")
        assert next_claim["generation"] > claim["generation"]
        assert gateway.status("interrupted-action")["state"] == "started"
        assert (root / "effects.txt").read_text().splitlines() == ["reset", "execute"]
    finally:
        gateway.close()
