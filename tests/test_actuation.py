from pathlib import Path
import secrets

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.actuation import ActuationGuard, ActuationLimits
from PhysicalRSI_core.infra.control_authority import ControlClaimInvalid
from PhysicalRSI_core.infra.storage import digest, read_json
from PhysicalRSI_core.timing import SensorSample


class Clock:
    value = 10.

    def __call__(self):
        return self.value


class Backend:
    def __init__(self, clock):
        self.clock, self.output, self.speed = clock, 0, 0
        self.writes, self.stops, self.ticks, self.resets = [], 0, [], 0

    def identity(self):
        return dict(name="actuator-contract-fixture", revision="1")

    def validate(self, actions):
        if any(type(action) is not int or not -2 <= action <= 2 for action in actions):
            raise ValueError("Actuator effort outside limits")

    def apply(self, action):
        self.output = action
        self.writes.append(action)

    def stop(self):
        self.output = 0
        self.stops += 1

    def quiescence(self):
        return dict(quiescent=self.output == 0 and self.speed == 0, output=self.output, measured_speed=self.speed)

    def reset(self, case):
        self.resets += 1
        return dict(ready=True)

    def observe(self):
        return SensorSample(dict(speed=self.speed), len(self.ticks), self.clock())

    def tick(self, elapsed):
        self.ticks.append((elapsed, self.output))


def setup(tmp_path):
    clock = Clock()
    backend = Backend(clock)
    guard = ActuationGuard(tmp_path, backend=backend, resources=("device:a",), clock=clock,
                           limits=ActuationLimits(lease_seconds=2, max_command_seconds=1, max_poll_gap_seconds=.3))
    return clock, backend, guard


def acquire(guard, name="owner"):
    credential = secrets.token_hex(32)
    grant = guard.acquire(name, owner=name, credential=credential, harness_revision="candidate")
    assert grant["state"] == "completed"
    return dict(grant["claim"], credential=credential)


def submit(guard, claim, name="move", actions=None):
    return guard.submit(name, actions=actions or [1, 2], period_seconds=.1,
                        deadline=guard.clock() + .8, claim=claim, harness_revision="candidate")


def test_bounded_command_stops_output_without_claiming_measured_quiescence(tmp_path):
    clock, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        submit(guard, claim)
        backend.speed = 1
        clock.value += .11
        guard.poll()
        clock.value += .11
        guard.poll()
        assert backend.writes == [1, 2] and backend.output == 0
        result = guard.status("move")
        assert result["state"] == "completed" and result["applied_steps"] == 2
        assert result["quiescent"] is False
        assert guard.inspect()["ownership"]["phase"] == "held"
        with pytest.raises(ReconciliationRequired, match="not independently settled"):
            guard.release("too-early", claim=claim)
        backend.speed = 0
        release = guard.release("settled", claim=claim)
        measurement = read_json(tmp_path / "quiescence/settled.json")
        assert release["measurement"] == measurement
        assert release["evidence_sha256"] == digest(measurement)
        assert guard.inspect()["ownership"]["phase"] == "free"
    finally:
        guard.close()


@pytest.mark.parametrize("gap", [.22, .5, 3])
def test_missed_ticks_deadlines_and_owner_expiry_never_replay_late_actions(tmp_path, gap):
    clock, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        submit(guard, claim)
        clock.value += gap
        guard.poll()
        assert backend.writes == [1] and backend.output == 0
        assert backend.ticks[-1][1] == 0
        assert guard.status("move")["state"] == "interrupted"
        assert guard.inspect()["ownership"]["phase"] == "recovery_required"
        with pytest.raises(ControlClaimInvalid):
            submit(guard, claim, "late-command")
        assert backend.writes == [1]
    finally:
        guard.close()


def test_multiple_controllers_share_one_backend_and_stale_stop_cannot_touch_successor(tmp_path):
    _, backend, guard = setup(tmp_path)
    try:
        first = acquire(guard)
        rejected = guard.acquire("competitor", owner="elsewhere", credential=secrets.token_hex(32), harness_revision="candidate")
        assert rejected["state"] == "rejected"
        guard.stop("first-stop", claim=first)
        guard.release("first-release", claim=first)
        second = acquire(guard, "second")
        submit(guard, second)
        assert backend.output == 1
        stops = backend.stops
        with pytest.raises(ControlClaimInvalid):
            guard.stop("old-stop", claim=first)
        with pytest.raises(ControlClaimInvalid):
            submit(guard, first, "old-command")
        assert backend.output == 1 and backend.stops == stops
    finally:
        guard.close()


def test_restarted_backend_fences_old_claim_and_requires_local_measured_recovery(tmp_path):
    clock, backend, guard = setup(tmp_path)
    claim = acquire(guard)
    submit(guard, claim)
    guard.close()
    restarted = ActuationGuard(tmp_path, backend=backend, resources=("device:a",), clock=clock, limits=guard.limits)
    recovery = dict(generation=1, operator="fixture", reason="Inspect stopped device", evidence={"inspection": "fixture-1"})
    try:
        with pytest.raises(ControlClaimInvalid):
            submit(restarted, claim, "old-boot")
        assert restarted.recover("recover", **recovery)["quiescent"] is True
        stopped = backend.stops
        assert restarted.recover("recover", **recovery)["quiescent"] is True
        assert backend.stops == stopped
        second = acquire(restarted, "second")
        submit(restarted, second, "new-command")
        assert restarted.recover("recover", **recovery)["quiescent"] is True  # Historical receipt only.
        with pytest.raises(ControlClaimInvalid):
            restarted.recover("different-old-recovery", **recovery)
        assert backend.output == 1
    finally:
        restarted.close()


def test_completed_submission_replay_does_not_repeat_actuator_writes(tmp_path):
    clock, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        deadline = clock() + .8
        arguments = dict(actions=[1], period_seconds=.1, deadline=deadline, claim=claim, harness_revision="candidate")
        result = guard.submit("move", **arguments)
        clock.value += .11
        guard.poll()
        assert guard.submit("move", **arguments) == result
        assert backend.writes == [1] and backend.output == 0
    finally:
        guard.close()


def test_expired_queued_reset_does_not_touch_the_backend(tmp_path):
    clock, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        with pytest.raises(TimeoutError, match="expired before dispatch"):
            guard.reset("old-reset", case={}, claim=claim, harness_revision="candidate", request_deadline=clock() - .1)
        assert backend.resets == 0 and backend.output == 0
    finally:
        guard.close()


def test_actuator_side_validation_and_system1_binding_survive_driver_bypass(tmp_path):
    _, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        with pytest.raises(ControlClaimInvalid):
            guard.submit("other-candidate", actions=[1], period_seconds=.1, deadline=10.8,
                         claim=claim, harness_revision="different-candidate")
        assert not backend.writes
    finally:
        guard.close()


def test_raw_actuator_writes_still_validate_limits_and_keep_credentials_out_of_evidence(tmp_path):
    _, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        with pytest.raises(ValueError, match="outside limits"):
            submit(guard, claim, actions=[999])
        assert not backend.writes
        assert all(claim["credential"] not in path.read_text() for path in tmp_path.rglob("*.json"))
        assert (tmp_path / "intents/move.json").exists()
    finally:
        guard.close()


def test_lost_setpoint_write_is_stopped_and_quarantined(tmp_path):
    _, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        apply = backend.apply

        def lose_reply(action):
            apply(action)
            raise ConnectionError("Response lost after actuator effect")

        backend.apply = lose_reply
        with pytest.raises(ConnectionError, match="after actuator effect"):
            submit(guard, claim)
        assert backend.writes == [1] and backend.output == 0
        assert guard.status("move")["state"] == "interrupted"
        assert guard.inspect()["ownership"]["phase"] == "recovery_required"
        with pytest.raises(RuntimeError, match="requires reconciliation"):
            submit(guard, claim)
        assert backend.writes == [1]
    finally:
        guard.close()


def test_closed_backend_cannot_write_into_a_replacement_authority(tmp_path):
    clock, backend, guard = setup(tmp_path)
    guard.close()
    replacement = ActuationGuard(tmp_path, backend=backend, resources=("device:a",), clock=clock, limits=guard.limits)
    try:
        with pytest.raises(RuntimeError, match="closed"):
            acquire(guard, "stale-process")
        assert not (tmp_path / "intents/stale-process.json").exists()
        acquire(replacement)
    finally:
        replacement.close()


def test_release_requires_durable_measured_evidence_before_handoff(tmp_path, monkeypatch):
    import PhysicalRSI_core.infra.actuation as module
    _, backend, guard = setup(tmp_path)
    try:
        claim = acquire(guard)
        original = module.atomic_json

        def fail_evidence(path, value):
            if Path(path).parent.name == "quiescence":
                raise OSError("Cannot persist measured stop evidence")
            return original(path, value)

        monkeypatch.setattr(module, "atomic_json", fail_evidence)
        with pytest.raises(OSError, match="measured stop evidence"):
            guard.release("release", claim=claim)
        assert guard.inspect()["ownership"]["phase"] == "recovery_required"
        assert guard.authority.inspect()["current"] is not None
        assert backend.output == 0
    finally:
        guard.close()


def test_operator_recovery_cannot_reuse_an_unrelated_request_identity(tmp_path):
    _, backend, guard = setup(tmp_path)
    try:
        acquire(guard, "request")
        stops = backend.stops
        with pytest.raises(ValueError, match="reused with different inputs"):
            guard.recover("request", generation=1, operator="operator", reason="inspection", evidence={"record": 1})
        assert backend.stops == stops
        assert guard.inspect()["ownership"]["phase"] == "held"
    finally:
        guard.close()
