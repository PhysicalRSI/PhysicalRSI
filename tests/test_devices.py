import multiprocessing

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.devices import DeviceBusy, DeviceRegistry
from PhysicalRSI_core.infra.storage import digest


def _hold_devices(root, pipe):
    with DeviceRegistry(root).lease(["robot:arm", "camera:overhead"], owner="worker/trial") as lease:
        pipe.send(lease)
        pipe.recv()


@pytest.fixture
def live_owner(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold_devices, args=(str(tmp_path), child))
    process.start()
    child.close()
    try:
        assert parent.poll(15), "Device owner did not acquire its lease"
        yield process, parent, parent.recv(), DeviceRegistry(tmp_path)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(10)
        parent.close()
        assert not process.is_alive()


def test_live_process_exclusion_and_atomic_resource_set(live_owner):
    process, pipe, lease, registry = live_owner
    assert registry.inspect(lease["token"])["owner_active"] is True
    with pytest.raises(DeviceBusy):
        with registry.lease(["free-device", "robot:arm"], owner="another-workspace/trial"):
            pytest.fail("Conflicting resources were acquired")
    assert "free-device" not in registry.occupied()
    with registry.lease(["free-device"], owner="independent-trial"):
        assert set(registry.occupied()) == {"robot:arm", "camera:overhead", "free-device"}
    with pytest.raises(DeviceBusy, match="live device owner"):
        registry.reconcile(lease["token"], operator="test", reason="attempted takeover",
                           evidence={"idle": True})
    pipe.send("finish")
    process.join(10)
    assert process.exitcode == 0
    assert registry.occupied() == {}
    assert registry.inspect(lease["token"])["state"] == "released"


def test_process_death_retains_claim_until_evidenced_reconciliation(live_owner):
    process, _, lease, registry = live_owner
    process.kill()
    process.join(10)
    assert process.exitcode is not None and process.exitcode != 0
    # A fresh registry instance sees the old claim without trusting a PID or TTL.
    registry = DeviceRegistry(registry.root)
    record = registry.inspect(lease["token"])
    assert record["state"] == "held"
    assert record["owner_active"] is False
    with pytest.raises(ReconciliationRequired):
        with registry.lease(["robot:arm"], owner="fresh-run-id"):
            pytest.fail("A dead process's physical effects were assumed stopped")
    resolution = dict(operator="mock-operator", reason="Mock controller reports idle",
                      evidence={"controller": "software-fixture", "idle": True})
    recovered = registry.reconcile(lease["token"], **resolution)
    assert recovered["reconciliation"]["evidence_sha256"] == digest(resolution["evidence"])
    assert registry.occupied() == {}
    with registry.lease(["robot:arm"], owner="new-trial") as replacement:
        # Replaying an old recovery cannot release a newer owner's resources.
        assert registry.reconcile(lease["token"], **resolution) == recovered
        assert registry.occupied()["robot:arm"] == replacement["token"]
        with pytest.raises(ValueError, match="already recorded"):
            registry.reconcile(lease["token"], **dict(resolution, reason="different decision"))


@pytest.mark.parametrize("error", [RuntimeError("disconnected"), KeyboardInterrupt()])
def test_exception_keeps_device_quarantined(tmp_path, error):
    registry = DeviceRegistry(tmp_path)
    with pytest.raises(type(error)):
        with registry.lease(["robot"], owner="trial") as lease:
            raise error
    record = registry.inspect(lease["token"])
    assert record["state"] == "needs_reconciliation"
    assert record["owner_active"] is False
    assert registry.occupied() == {"robot": lease["token"]}
    with pytest.raises(ValueError, match="evidence"):
        registry.reconcile(lease["token"], operator="operator", reason="unknown", evidence={})


@pytest.mark.parametrize("resources", [[], ["robot", "robot"], [""], "robot"])
def test_invalid_resource_set_never_acquires(tmp_path, resources):
    registry = DeviceRegistry(tmp_path)
    with pytest.raises(ValueError):
        with registry.lease(resources, owner="trial"):
            pytest.fail("Invalid lease admitted")
    assert registry.occupied() == {}
