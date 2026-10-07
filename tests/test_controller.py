from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import secrets
from threading import Event
from time import monotonic

import pytest

from PhysicalRSI_core.contracts import Cancelled, Context, Contract, ReconciliationRequired
from PhysicalRSI_core.embodiment import Embodiment, ExecutionBinding, System1
from PhysicalRSI_core.infra import controller as controller_module
from PhysicalRSI_core.infra.controller import ControllerGateway, GatewayEnvironment, ControlRejected
from PhysicalRSI_core.infra.controller_rpc import ControllerHost
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.storage import digest, read_json
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.timing import ControlTiming, SensorSample, action_chunk


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class Driver:
    def __init__(self, clock):
        self.clock, self.sequence, self.position = clock, -1, 0
        self.resets = self.executions = self.stops = 0

    def identity(self):
        return {"name": "software-controller-fixture", "revision": "1"}

    def reset(self, case, context):
        self.resets += 1
        self.position = 0
        return {"ready": True}

    def observe(self, context):
        self.sequence += 1
        return SensorSample({"position": self.position}, self.sequence, self.clock())

    def validate(self, actions):
        if any(type(action) is not int or abs(action) > 2 for action in actions):
            raise ValueError("Action outside driver limits")

    def execute(self, actions, *, period_seconds, deadline, context):
        self.executions += 1
        self.position += sum(actions)
        self.clock.value += len(actions) * period_seconds
        return dict(terminated=False, executed_steps=len(actions))

    def quiesce(self, context):
        self.stops += 1
        return {"quiescent": True, "scope": "software fixture"}


TIMING = ControlTiming(.25, 4, 1, .125)
OBSERVATION = Contract("position", unit="count", embodiment="fixture")
ACTION = Contract("increment", unit="count", embodiment="fixture")
SPEC = Embodiment("fixture", "1", "software", OBSERVATION, ACTION,
                  resources=("fixture:one",), timing=TIMING)


@pytest.fixture
def gateway(tmp_path):
    clock = Clock()
    value = ControllerGateway(tmp_path, driver=Driver(clock), embodiment=SPEC, clock=clock)
    try:
        yield value
    finally:
        value.close()


def start(gateway):
    context = Context("trial", harness_revision="candidate-v1")
    claim = acquire(gateway, context)
    result = gateway.reset("reset", "episode", {}, context, claim=claim)
    return context, result["observation"], claim


def acquire(gateway, context, request_id="claim"):
    credential = secrets.token_hex(32)
    grant = gateway.acquire(request_id, "test-owner", credential, context)
    assert grant["state"] == "completed"
    return dict(grant["claim"], credential=credential)


def command(observation, actions=None):
    return action_chunk(observation, actions or [1], period_seconds=TIMING.period_seconds)


def test_action_is_bound_to_observation_and_executed_once(gateway):
    context, observation, claim = start(gateway)
    action = command(observation, [1, 1])
    result = gateway.step("action", action, context, claim=claim)
    assert result["observation"]["payload"]["position"] == 2
    assert result["control"]["executed_steps"] == 2
    gateway.clock.value += 100
    assert gateway.step("action", action, context, claim=claim) == result
    assert gateway.driver.executions == 1
    # A second ID cannot replay an action against a consumed observation.
    assert gateway.step("replay", action, context, claim=claim)["state"] == "rejected"
    with pytest.raises(ValueError, match="different inputs"):
        gateway.step("action", command(observation, [2]), context, claim=claim)
    assert gateway.state()["phase"] == "ready"


@pytest.mark.parametrize("change", ["expired", "wrong_period", "horizon", "source", "limit", "revision"])
def test_controller_rejects_invalid_commands_before_effects(gateway, change):
    context, observation, claim = start(gateway)
    action = command(observation)
    if change == "expired":
        gateway.clock.value += TIMING.max_observation_age_seconds
    elif change == "wrong_period":
        action["period_seconds"] *= 2
    elif change == "horizon":
        action["actions"] = [1] * (TIMING.max_chunk_steps + 1)
    elif change == "source":
        action["observation"]["sequence"] += 1
    elif change == "limit":
        action["actions"] = [9]
    else:
        context = replace(context, harness_revision="different-candidate")
    result = gateway.step("bad-action", action, context, claim=claim)
    assert result["state"] == "rejected" and result["dispatched"] is False
    assert gateway.driver.executions == 0
    assert gateway.status("bad-action")["result"] == result
    assert gateway.step("bad-action", action, context, claim=claim) == result


def test_write_latency_is_included_in_freshness_check(gateway, monkeypatch):
    context, observation, claim = start(gateway)
    original = controller_module.atomic_json

    def slow_write(path, value):
        original(path, value)
        if path.name == "controller.json" and value.get("last_request") == "slow-action":
            gateway.clock.value += 2

    monkeypatch.setattr(controller_module, "atomic_json", slow_write)
    result = gateway.step("slow-action", command(observation), context, claim=claim)
    assert result["state"] == "rejected"
    assert gateway.driver.executions == 0


def test_uncertain_capture_uses_earliest_time_for_freshness(gateway, monkeypatch):
    observe = gateway.driver.observe

    def uncertain(context):
        sample = observe(context)
        gateway.clock.value += .4
        return replace(sample, capture_uncertainty_seconds=.4, timing_evidence={"fixture": "bounded capture"})

    monkeypatch.setattr(gateway.driver, "observe", uncertain)
    context, observation, claim = start(gateway)
    assert observation["capture_uncertainty_seconds"] == .4
    gateway.clock.value += .7
    # Its latest possible capture would still be fresh; its earliest is old.
    assert gateway.clock() - (observation["captured_at"] + .4) < TIMING.max_observation_age_seconds
    result = gateway.step("old-interval", command(observation), context, claim=claim)
    assert result["state"] == "rejected" and gateway.driver.executions == 0


def test_capture_interval_extending_into_future_is_rejected(gateway, monkeypatch):
    observe = gateway.driver.observe
    monkeypatch.setattr(gateway.driver, "observe", lambda context: replace(observe(context), capture_uncertainty_seconds=.1))
    with pytest.raises(ReconciliationRequired, match="future"):
        start(gateway)


@pytest.mark.parametrize("fault", ["overrun", "stale_sensor", "sequence", "future_sensor", "disconnect", "clock_reversal"])
def test_uncertain_effects_block_future_actions_and_reset(gateway, monkeypatch, fault):
    context, observation, claim = start(gateway)
    original_execute, original_observe = gateway.driver.execute, gateway.driver.observe

    def execute(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        if fault == "overrun":
            gateway.clock.value += 10
        if fault == "disconnect":
            raise RuntimeError("Lost connection after dispatch")
        if fault == "clock_reversal":
            gateway.clock.value -= 100
        return result

    def observe(ctx):
        sample = original_observe(ctx)
        if fault == "stale_sensor":
            return replace(sample, captured_at=observation["captured_at"])
        if fault == "sequence":
            return replace(sample, sequence=observation["sequence"])
        if fault == "future_sensor":
            return replace(sample, captured_at=gateway.clock() + 1)
        return sample

    monkeypatch.setattr(gateway.driver, "execute", execute)
    monkeypatch.setattr(gateway.driver, "observe", observe)
    with pytest.raises((ReconciliationRequired, RuntimeError)):
        gateway.step("uncertain", command(observation), context, claim=claim)
    assert gateway.driver.executions == 1
    assert gateway.status("uncertain")["state"] == "uncertain"
    intent = read_json(gateway.root / "intents/uncertain.json")
    assert intent["inputs"] == command(observation)
    assert digest(intent) == gateway.status("uncertain")["binding"]
    assert gateway.state()["phase"] == "needs_reconciliation"
    with pytest.raises(RuntimeError, match="reconciliation"):
        gateway.step("uncertain", command(observation), context, claim=claim)
    assert gateway.step("new-id", command(observation), context, claim=claim)["state"] == "rejected"
    assert gateway.reset("another-reset", "new-episode", {}, context, claim=claim)["state"] == "rejected"
    assert gateway.driver.resets == 1 and gateway.driver.executions == 1


def test_restart_invalidates_clock_domain_without_losing_completed_receipts(gateway):
    context, observation, claim = start(gateway)
    result = gateway.step("action", command(observation), context, claim=claim)
    gateway.close()
    clock = Clock()
    restarted = ControllerGateway(gateway.root, driver=Driver(clock), embodiment=SPEC, clock=clock)
    try:
        assert restarted.step("action", command(observation), context, claim=claim) == result
        assert restarted.driver.executions == 0
        assert restarted.step("new", command(result["observation"]), context, claim=claim)["state"] == "rejected"
        assert restarted.reset("reset-new", "new-episode", {}, context, claim=claim)["state"] == "rejected"
        assert restarted.quiesce("recover", "episode", context, claim=claim)["quiescent"]
        claim = acquire(restarted, context, "fresh-claim")
        fresh = restarted.reset("fresh", "new-episode", {}, context, claim=claim)["observation"]
        assert fresh["clock_domain"] != observation["clock_domain"]
        assert restarted.step("stale-epoch", command(observation), context, claim=claim)["state"] == "rejected"
        assert restarted.step("fresh-action", command(fresh), context, claim=claim)["state"] == "completed"
    finally:
        restarted.close()


def test_live_gateway_cannot_be_replaced(gateway):
    with pytest.raises(RuntimeError, match="Another controller"):
        ControllerGateway(gateway.root, driver=Driver(Clock()), embodiment=SPEC)


def test_rpc_queue_delay_expires_before_reset(gateway):
    host = ControllerHost(gateway)
    with pytest.raises(Cancelled):
        host.execute("late-reset", "reset", dict(episode="episode", case={}),
                     budget_seconds=.1, _received_at=monotonic() - 1)
    assert gateway.driver.resets == 0


def test_timing_contract_must_match_before_start():
    policy = System1("policy", "1", OBSERVATION, ACTION, timing=replace(TIMING, period_seconds=.5))
    with pytest.raises(ValueError, match="control timing"):
        ExecutionBinding(SPEC, policy)


def test_experiment_preserves_controller_rejection_and_device_claim(gateway, tmp_path):
    class SlowPolicy:
        def identity(self):
            return {"name": "slow-fixture", "revision": "1"}

        def describe(self):
            return System1("slow-fixture", "1", OBSERVATION, ACTION, timing=TIMING)

        def begin_episode(self, *args):
            pass

        def act(self, observation, context):
            gateway.clock.value += 2
            return command(observation)

    class Verifier:
        def identity(self):
            return {"name": "unused", "revision": "1"}

        def verify(self, *args):
            pytest.fail("A rejected action must not become a scored trial")

    registry = DeviceRegistry(tmp_path / "devices")
    runtime = ExperimentRuntime(tmp_path / "trials", device_registry=registry)
    with pytest.raises(ControlRejected, match="expired"):
        runtime.run("slow", task="fixture", case={}, scope="software", environment=GatewayEnvironment(gateway),
                    policy=SlowPolicy(), verifier=Verifier(), budget=Budget(seconds=5))
    receipt = read_json(runtime.root / "slow/receipt.json")
    assert receipt["controller_rejection"]["dispatched"] is False
    assert gateway.driver.executions == 0 and registry.occupied()


def test_missing_or_forged_claim_cannot_reset_and_credentials_are_not_persisted(gateway):
    context = Context("trial", harness_revision="candidate-v1")
    assert gateway.reset("unowned", "episode", {}, context)["state"] == "rejected"
    claim = acquire(gateway, context)
    forged = dict(claim, credential=secrets.token_hex(32))
    assert gateway.reset("forged", "episode", {}, context, claim=forged)["state"] == "rejected"
    assert gateway.acquire("competing", "other-owner", secrets.token_hex(32), context)["state"] == "rejected"
    assert gateway.driver.resets == 0
    initial = gateway.reset("owned", "episode", {}, context, claim=claim)
    assert initial["control"]["authority"]["generation"] == claim["generation"]
    assert gateway.driver.resets == 1
    public = json.dumps([gateway.ownership(), gateway.state(), gateway.status("claim")])
    assert claim["credential"] not in public and forged["credential"] not in public
    for path in gateway.root.rglob("*.json"):
        assert claim["credential"] not in path.read_text()
        assert forged["credential"] not in path.read_text()


def test_expiry_requires_stop_before_transfer_and_old_owner_cannot_stop_new_owner(gateway):
    context, observation, old = start(gateway)
    gateway.clock.value += 30
    assert "expired" in gateway.step("expired", command(observation), context, claim=old)["reason"]
    assert gateway.renew("late-renewal", old, context)["state"] == "rejected"
    assert gateway.ownership()["phase"] == "recovery_required"
    assert gateway.acquire("takeover", "other", secrets.token_hex(32), context)["state"] == "rejected"
    assert gateway.driver.stops == 0
    released = gateway.quiesce("stop-old", "episode", context, claim=old)
    assert released["quiescent"] and gateway.driver.stops == 1
    current = acquire(gateway, context, "second-claim")
    assert current["generation"] > old["generation"]
    packet = gateway.reset("reset-new", "new-episode", {}, context, claim=current)["observation"]
    # Even knowledge of the new episode/observation is not ownership.
    assert gateway.step("late-old-action", command(packet), context, claim=old)["state"] == "rejected"
    assert gateway.quiesce("late-old-stop", "new-episode", context, claim=old)["state"] == "rejected"
    assert gateway.quiesce("stop-old", "episode", context, claim=old) == released
    assert gateway.driver.stops == 1 and gateway.driver.executions == 0
    assert gateway.ownership()["generation"] == current["generation"]


def test_replayed_renewal_does_not_extend_a_lease(gateway):
    context, _, claim = start(gateway)
    gateway.clock.value += 10
    result = gateway.renew("heartbeat", claim, context)
    expiry = gateway.ownership()["current"]["expires_at"]
    gateway.clock.value = expiry - .1
    assert gateway.renew("heartbeat", claim, context) == result
    assert gateway.ownership()["current"]["expires_at"] == expiry
    gateway.clock.value = expiry
    assert gateway.renew("new-heartbeat", claim, context)["state"] == "rejected"


def test_cancel_after_acquisition_can_release_an_unstarted_episode(gateway):
    context = Context("trial", harness_revision="candidate-v1")
    claim = acquire(gateway, context)
    result = gateway.quiesce("cancel-unstarted", "unstarted-episode", context, claim=claim)
    assert result["quiescent"] and gateway.ownership()["phase"] == "free"
    assert gateway.driver.resets == 0 and gateway.driver.stops == 1
    current = acquire(gateway, context, "next-claim")
    gateway.reset("reset", "active-episode", {}, context, claim=current)
    assert gateway.quiesce("wrong-episode", "unstarted-episode", context, claim=current)["state"] == "rejected"
    assert gateway.driver.stops == 1


def test_action_horizon_cannot_outlive_its_control_claim(tmp_path):
    clock = Clock()
    gateway = ControllerGateway(tmp_path, driver=Driver(clock), embodiment=SPEC, clock=clock, lease_seconds=.3)
    try:
        context, observation, claim = start(gateway)
        rejected = gateway.step("long-horizon", command(observation), context, claim=claim)
        assert rejected["state"] == "rejected" and "ownership" in rejected["reason"]
        assert gateway.driver.executions == 0
    finally:
        gateway.close()


def test_recovery_quarantines_until_verified_stop_and_cannot_release_replacement(gateway, monkeypatch):
    context, _, claim = start(gateway)
    quiesce = gateway.driver.quiesce
    monkeypatch.setattr(gateway.driver, "quiesce", lambda context: dict(quiescent=False))
    inputs = dict(generation=claim["generation"], operator="fixture-operator", reason="Inspect lost owner",
                  evidence=dict(inspection="fixture-check"), context=context)
    with pytest.raises(ReconciliationRequired, match="confirm quiescence"):
        gateway.recover("failed-recovery", **inputs)
    assert gateway.ownership()["phase"] == "recovery_required"
    assert gateway.acquire("blocked-takeover", "next", secrets.token_hex(32), context)["state"] == "rejected"
    monkeypatch.setattr(gateway.driver, "quiesce", quiesce)
    recovered = gateway.recover("verified-recovery", **inputs)
    assert recovered["quiescent"] and gateway.ownership()["phase"] == "free"
    current = acquire(gateway, context, "new-owner")
    assert gateway.recover("verified-recovery", **inputs) == recovered
    with pytest.raises(ReconciliationRequired, match="different controller generation"):
        gateway.recover("delayed-recovery", **inputs)
    assert gateway.ownership()["phase"] == "held" and gateway.ownership()["generation"] == current["generation"]
    assert gateway.driver.stops == 1


def test_authority_transaction_survives_a_lost_acknowledgment(gateway, monkeypatch):
    from PhysicalRSI_core.infra import control_authority as module
    write = module.atomic_json
    context = Context("trial", harness_revision="candidate-v1")
    credential = secrets.token_hex(32)

    def fail_after_replace(path, value):
        write(path, value)
        raise RuntimeError("Lost acknowledgment after metadata commit")

    monkeypatch.setattr(module, "atomic_json", fail_after_replace)
    with pytest.raises(RuntimeError, match="after metadata commit"):
        gateway.acquire("durable-grant", "owner", credential, context)
    assert gateway.status("durable-grant")["state"] == "completed"
    monkeypatch.setattr(module, "atomic_json", write)
    replay = gateway.acquire("durable-grant", "owner", credential, context)
    assert replay == gateway.status("durable-grant")["result"]
    assert gateway.ownership()["generation"] == 1 and gateway.driver.resets == 0


def test_context_cancellation_during_authority_check_prevents_dispatch():
    from PhysicalRSI_core.contracts import Operation
    effects = []
    context = Context("guarded", validity=lambda: context.cancelled.set())
    operation = Operation("guarded-reset", "1", Contract("input"), Contract("output"),
                          lambda value, ctx: effects.append(value))
    with pytest.raises(Cancelled):
        operation("must-not-dispatch", context)
    assert effects == []


def test_claim_for_another_controller_cannot_control_this_embodiment(gateway, tmp_path):
    context, _, claim = start(gateway)
    clock = Clock()
    other = ControllerGateway(tmp_path / "other", driver=Driver(clock), embodiment=SPEC, clock=clock)
    try:
        own_claim = acquire(other, context)
        assert own_claim["generation"] == claim["generation"]
        assert own_claim["authority_id"] != claim["authority_id"]
        assert other.reset("foreign-authority", "episode", {}, context, claim=claim)["state"] == "rejected"
        assert other.driver.resets == 0
    finally:
        other.close()


def test_gateway_does_not_release_its_process_lock_during_an_authority_commit(gateway, monkeypatch):
    from PhysicalRSI_core.infra import control_authority as module
    write = module.atomic_json
    entered, release, closing = Event(), Event(), Event()
    context = Context("trial", harness_revision="candidate-v1")

    def delayed_write(path, value):
        entered.set()
        assert release.wait(5)
        write(path, value)

    def close():
        closing.set()
        gateway.close()

    monkeypatch.setattr(module, "atomic_json", delayed_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        grant = executor.submit(gateway.acquire, "slow-grant", "owner", secrets.token_hex(32), context)
        assert entered.wait(5)
        closed = executor.submit(close)
        try:
            assert closing.wait(5)
            with pytest.raises(TimeoutError):
                closed.result(timeout=.05)
            with pytest.raises(RuntimeError, match="Another controller"):
                ControllerGateway(gateway.root, driver=Driver(Clock()), embodiment=SPEC)
        finally:
            release.set()
        assert grant.result(timeout=5)["state"] == "completed"
        closed.result(timeout=5)
    monkeypatch.setattr(module, "atomic_json", write)
    restarted = ControllerGateway(gateway.root, driver=Driver(Clock()), embodiment=SPEC)
    try:
        assert restarted.ownership()["phase"] == "recovery_required"
    finally:
        restarted.close()
