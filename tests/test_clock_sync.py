from dataclasses import replace
from itertools import product
import multiprocessing
import os
from pathlib import Path
import secrets
from time import monotonic, sleep

import pytest

from PhysicalRSI_core.contracts import Context, Contract
from PhysicalRSI_core.embodiment import Embodiment
from PhysicalRSI_core.infra.actuation import ActuationGuard, ActuationLimits
from PhysicalRSI_core.infra.actuation_rpc import ActuationClient, ActuationDriver, ActuationHost
from PhysicalRSI_core.infra.clock_sync import ClockMapping, ClockSyncPolicy, ClockUnsynchronized
from PhysicalRSI_core.infra.controller import ControllerGateway
from PhysicalRSI_core.infra.controller_rpc import controller_transport
from PhysicalRSI_core.infra.rpc import RpcError, wait_for_ready
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_core.timing import ControlTiming, SensorSample, action_chunk


def context(seconds=3):
    return Context("clock-trial", deadline=monotonic() + seconds, harness_revision="clock-fixture")


def test_bounds_contain_asymmetric_networks_and_both_drift_directions():
    policy = ClockSyncPolicy(max_round_trip_seconds=1, max_offset_width_seconds=1, max_relative_drift_ppm=200)
    for offset, drift, outbound, inbound, processing in product(
            [-90, 0, 10000], [-200e-6, 0, 200e-6], [.001, .08], [.002, .1], [0, .01]):
        remote = lambda local: local + offset + drift * (local - 100)
        received = 100 + outbound + processing + inbound
        mapping = ClockMapping(policy, "controller", "device", "probe", 100, remote(100 + outbound),
                               remote(100 + outbound + processing), received)
        lower, upper = mapping.offset_bounds(received + 2, now=received + 1)
        assert lower <= remote(received + 2) - (received + 2) <= upper
        assert mapping.remote_deadline(received + 2, now=received + 1) <= remote(received + 2)
        sent, captured, returned = received + .1, received + .11, received + .12
        earliest, latest = mapping.local_capture(remote(captured), remote(captured), local_sent=sent, local_received=returned)
        assert sent <= earliest <= captured <= latest <= returned


def test_mapping_rejects_excess_latency_uncertainty_and_old_or_contradictory_clocks():
    policy = ClockSyncPolicy(max_round_trip_seconds=.3, max_offset_width_seconds=.1, max_sample_age_seconds=1)
    mapping = ClockMapping(policy, "a", "b", "one", 10, 110.01, 110.011, 10.02)
    with pytest.raises(ClockUnsynchronized, match="round trip"):
        replace(mapping, local_received=10.4)
    with pytest.raises(ClockUnsynchronized, match="uncertainty"):
        replace(mapping, local_received=10.2)
    with pytest.raises(ClockUnsynchronized, match="backwards"):
        replace(mapping, remote_sent=110)
    with pytest.raises(ClockUnsynchronized, match="feasible"):
        replace(mapping, remote_sent=111)
    with pytest.raises(ClockUnsynchronized, match="expired"):
        mapping.remote_deadline(12, now=11.1)
    with pytest.raises(ClockUnsynchronized, match="uncertainty"):
        mapping.remote_deadline(10000, now=10.1)
    with pytest.raises(ClockUnsynchronized, match="causal interval"):
        mapping.local_capture(115, 115, local_sent=10.1, local_received=10.12)
    with pytest.raises(ClockUnsynchronized, match="drift bound"):
        ClockMapping(policy, "a", "b", "two", 10.1, 111.11, 111.111, 10.12).follows(mapping)
    with pytest.raises(ClockUnsynchronized, match="domain"):
        ClockMapping(policy, "a", "rebooted", "two", 10.1, 110.11, 110.111, 10.12).follows(mapping)


def _serve(root, pipe, offset, clock_delay, call_delay, sensor_shift, bad_reply):
    root = Path(root)
    os.environ["PHYSICALRSI_ENDPOINT_FILE"] = str(root / "endpoint.json")
    clock = lambda: monotonic() + offset.value

    class Backend:
        output = writes = resets = sequence = 0

        def identity(self):
            return dict(name="clock-transport-software-fixture", revision="1")

        def validate(self, actions):
            if any(type(value) is not int or not -2 <= value <= 2 for value in actions):
                raise ValueError("Fixture effort outside limits")

        def apply(self, value):
            self.output = value
            self.writes += 1

        def stop(self):
            self.output = 0

        def quiescence(self):
            return dict(quiescent=self.output == 0, writes=self.writes, resets=self.resets, output=self.output)

        def reset(self, case):
            self.resets += 1
            return dict(ready=True)

        def observe(self):
            self.sequence += 1
            return SensorSample(dict(position=0), self.sequence, clock() + sensor_shift.value)

        def tick(self, elapsed):
            pass

    class Host(ActuationHost):
        def clock_exchange(self, nonce):
            result = super().clock_exchange(nonce)
            if clock_delay.value:
                sleep(clock_delay.value)
            if bad_reply.value:
                result["nonce"] = "replayed-probe"
            return result

        def call(self, *args, **kwargs):
            if call_delay.value:
                sleep(call_delay.value)
            return super().call(*args, **kwargs)

    guard = ActuationGuard(root / "device", backend=Backend(), resources=("clock-device",), clock=clock,
        clock_domain="independent-device-clock", limits=ActuationLimits(lease_seconds=3, max_command_seconds=1))
    pipe.send(guard.identity())
    pipe.close()
    Host(guard).serve(transport="http", host="127.0.0.1", port=0)


@pytest.fixture
def remote_clock(tmp_path):
    process_context = multiprocessing.get_context("spawn")
    parent, child = process_context.Pipe()
    # These scalar fault controls have one writer (the parent). They do not
    # need multiprocessing semaphore locks or compound shared updates.
    values = [process_context.Value("d", value, lock=False) for value in (10000, 0, 0, 0)]
    bad_reply = process_context.Value("i", 0, lock=False)
    process = process_context.Process(target=_serve, args=(str(tmp_path), child, *values, bad_reply))
    process.start()
    child.close()
    rpc = None
    try:
        assert parent.poll(10)
        expected = parent.recv()
        until = monotonic() + 10
        while not (tmp_path / "endpoint.json").exists():
            assert process.is_alive() and monotonic() < until
            sleep(.01)
        rpc = controller_transport(read_json(tmp_path / "endpoint.json")["endpoint"])
        wait_for_ready(rpc, timeout_s=3)
        client = ActuationClient(rpc, expected=expected, clock_policy=ClockSyncPolicy())
        yield client, rpc, values, bad_reply, tmp_path
    finally:
        if rpc and process.is_alive():
            try:
                rpc.call("shutdown", timeout_s=2)
            except Exception:
                pass
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        assert not process.is_alive()
        process.close()


def acquire(client):
    token = secrets.token_hex(32)
    grant = client.call("acquire", context(), request_id="grant", owner="clock-owner", credential=token,
                        harness_revision="clock-fixture")
    assert grant["state"] == "completed"
    return dict(grant["claim"], credential=token)


def test_default_rejects_other_clock_and_explicit_mapping_preserves_source_evidence(remote_clock):
    client, rpc, _, _, root = remote_clock
    with pytest.raises(ValueError, match="clock domain mismatch"):
        ActuationClient(rpc, expected=client.identity())
    claim = acquire(client)
    before = monotonic()
    result = client.call("observe", context(), claim=claim, harness_revision="clock-fixture")
    after = monotonic()
    assert before <= result["captured_at"] <= result["captured_at"] + result["capture_uncertainty_seconds"] <= after
    evidence = result["timing_evidence"]
    assert evidence["source"]["captured_at"] > before + 9999
    assert evidence["mapping"]["remote_domain"] == "independent-device-clock"
    with pytest.raises(RpcError, match="different device clock"):
        rpc.call("actuation.call", kwargs=dict(operation="reset", deadline=monotonic() + 10001,
            clock_domain="other-device", inputs=dict(request_id="wrong-clock", case={}, claim=claim, harness_revision="clock-fixture")), timeout_s=2)
    assert not (root / "device/intents/wrong-clock.json").exists()
    assert client.inspect(context())["quiescence"]["resets"] == 0


def test_deadline_materialization_survives_resynchronization_and_client_restart(remote_clock):
    client, rpc, _, _, root = remote_clock
    claim = acquire(client)
    inputs = dict(request_id="motion", actions=[1, 2], period_seconds=.04, deadline=monotonic() + .6,
                  claim=claim, harness_revision="clock-fixture")
    accepted = client.call("submit", context(), **inputs)
    until = monotonic() + 2
    while client.status("motion", context())["state"] == "started":
        assert monotonic() < until
        sleep(.01)
    assert client.inspect(context())["quiescence"]["writes"] == 2
    intent = read_json(root / "device/intents/motion.json")
    evidence = intent["inputs"]["timing_evidence"]
    assert evidence["local_deadline"] == inputs["deadline"]
    assert inputs["deadline"] + 9999 < evidence["remote_deadline"] <= inputs["deadline"] + 10000
    successor = ActuationClient(rpc, expected=client.identity(), clock_policy=ClockSyncPolicy())
    successor.synchronize(context(), force=True)
    assert successor.call("submit", context(), **inputs) == accepted
    assert read_json(root / "device/intents/motion.json") == intent
    assert successor.inspect(context())["quiescence"]["writes"] == 2
    with pytest.raises(ValueError, match="different inputs"):
        successor.call("submit", context(), **dict(inputs, actions=[2, 1]))


def test_in_transit_delay_does_not_restart_command_deadline(remote_clock):
    client, _, values, _, root = remote_clock
    claim = acquire(client)
    values[2].value = .25
    with pytest.raises(RpcError, match="deadline/lease/horizon"):
        client.call("submit", context(), request_id="late", actions=[1], period_seconds=.04,
                    deadline=monotonic() + .15, claim=claim, harness_revision="clock-fixture")
    assert client.inspect(context())["quiescence"]["writes"] == 0
    assert read_json(root / "device/requests/late.json")["state"] == "uncertain"


def test_failed_clock_probe_never_dispatches_device_acquisition(remote_clock):
    client, _, values, _, root = remote_clock
    values[1].value = .09
    with pytest.raises(ClockUnsynchronized, match="uncertainty"):
        acquire(client)
    assert not (root / "device/intents/grant.json").exists()
    assert client.inspect(context())["ownership"]["phase"] == "free"


@pytest.mark.parametrize("shift", [-.1, .1])
def test_source_capture_cannot_be_relabelled_with_receipt_time(remote_clock, shift):
    client, _, values, _, _ = remote_clock
    claim = acquire(client)
    values[3].value = shift
    with pytest.raises(RpcError, match="not freshly captured"):
        client.call("observe", context(), claim=claim, harness_revision="clock-fixture")


def test_expired_mapping_demands_a_new_bound_reply_before_reset(remote_clock):
    client, rpc, _, bad_reply, root = remote_clock
    client = ActuationClient(rpc, expected=client.identity(), clock_policy=ClockSyncPolicy(max_sample_age_seconds=.03))
    claim = acquire(client)
    sleep(.04)
    bad_reply.value = 1
    with pytest.raises(ClockUnsynchronized, match="reply identity"):
        client.call("reset", context(), request_id="blocked-reset", case={}, claim=claim, harness_revision="clock-fixture")
    assert not (root / "device/intents/blocked-reset.json").exists()


def test_contradictory_device_clock_latches_session_fault(remote_clock):
    client, _, values, _, root = remote_clock
    client.synchronize(context())
    values[0].value += .5
    with pytest.raises(ClockUnsynchronized, match="drift bound"):
        client.synchronize(context(), force=True)
    with pytest.raises(ClockUnsynchronized, match="drift bound"):
        acquire(client)
    assert not (root / "device/intents/grant.json").exists()
    # Read-only recovery evidence does not require a usable device-clock map.
    assert client.status("grant", context())["state"] == "missing"
    assert client.inspect(context())["ownership"]["phase"] == "free"


def test_gateway_uses_mapped_capture_and_independently_releases_remote_actuator(remote_clock, tmp_path):
    client, _, _, _, _ = remote_clock
    driver = ActuationDriver(client, validator=lambda actions: None, validation_identity={"fixture": "1"})
    spec = Embodiment("clock-fixture", "1", "software", Contract("state"), Contract("effort"),
                      resources=("clock-device",), timing=ControlTiming(.04, 2, .5, .5))
    gateway = ControllerGateway(tmp_path / "gateway", driver=driver, embodiment=spec)
    try:
        credential = secrets.token_hex(32)
        grant = gateway.acquire("controller-grant", "owner", credential, context())
        claim = dict(grant["claim"], credential=credential)
        observation = gateway.reset("reset", "episode", {}, context(), claim=claim)["observation"]
        assert observation["capture_uncertainty_seconds"] > 0
        assert observation["timing_evidence"]["source"]["captured_at"] > observation["captured_at"] + 9999
        result = gateway.step("action", action_chunk(observation, [1, 2], period_seconds=.04), context(), claim=claim)
        assert result["control"]["executed_steps"] == 2
        actuation = result["driver"]["actuation"]
        assert actuation["state"] == "completed"
        assert actuation["deadline"] == actuation["timing_evidence"]["remote_deadline"]
        assert actuation["timing_evidence"]["local_deadline"] == result["control"]["deadline"]
        assert gateway.quiesce("stop", "episode", context(), claim=claim)["quiescent"]
        assert gateway.ownership()["phase"] == client.inspect(context())["ownership"]["phase"] == "free"
    finally:
        gateway.close()
