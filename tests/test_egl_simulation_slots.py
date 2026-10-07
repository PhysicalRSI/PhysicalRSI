import pytest

from PhysicalRSI_core.infra.devices import DeviceRegistry, DeviceBusy
from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots import owned_simulation_resource


def test_two_simulators_share_only_under_exclusive_outer_gpu_owner(tmp_path):
    registry = DeviceRegistry(tmp_path)
    with registry.lease(["GPU-example"], owner="supervisor") as outer:
        names = [owned_simulation_resource(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=i) for i in range(2)]
        assert names[0] != names[1]
        with registry.lease([names[0]], owner="simulator-0"), registry.lease([names[1]], owner="simulator-1"):
            with pytest.raises(DeviceBusy):
                with registry.lease(["GPU-example"], owner="unrelated-run"):
                    pass
    assert registry.occupied() == {}
    with pytest.raises(ValueError, match="active exclusive"):
        owned_simulation_resource(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=0)


def test_wrong_device_or_extra_slot_cannot_use_outer_lease(tmp_path):
    registry = DeviceRegistry(tmp_path)
    with registry.lease(["GPU-example"], owner="supervisor") as outer:
        with pytest.raises(ValueError, match="active exclusive"):
            owned_simulation_resource(registry, gpu_uuid="GPU-different", owner_token=outer["token"], slot=0)
        with pytest.raises(ValueError, match="two simulation slots"):
            owned_simulation_resource(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=2)


def test_three_slot_protocol_preserves_owner_and_separate_names(tmp_path):
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_three import owned_simulation_resource as three
    registry = DeviceRegistry(tmp_path)
    with registry.lease(["GPU-example"], owner="supervisor") as outer:
        names = [three(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=i) for i in range(3)]
        assert len(set(names)) == 3
        with pytest.raises(ValueError, match="three simulation slots"):
            three(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=3)
    with pytest.raises(ValueError, match="active exclusive"):
        three(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=2)


@pytest.mark.parametrize("count", [4, 6, 8, 10, 12, 24, 48])
def test_capacity_slots_preserve_physical_exclusion_and_reject_stale_owner(tmp_path, count):
    from contextlib import ExitStack
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_four import owned_simulation_resource as four
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_six import owned_simulation_resource as six
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_ten import owned_simulation_resource as ten
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_twelve import owned_simulation_resource as twelve
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_eight import owned_simulation_resource as eight
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_twenty_four import owned_simulation_resource as twenty_four
    from PhysicalRSI_baselines.embodied_goodharts_law.simulation_slots_forty_eight import owned_simulation_resource as forty_eight
    allocate = {4: four, 6: six, 8: eight, 10: ten, 12: twelve, 24: twenty_four, 48: forty_eight}[count]
    registry = DeviceRegistry(tmp_path)
    with registry.lease(["GPU-example"], owner="supervisor") as outer:
        names = [allocate(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=i) for i in range(count)]
        assert len(set(names)) == count
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(registry.lease([name], owner=name))
            with pytest.raises(DeviceBusy):
                with registry.lease(["GPU-example"], owner="unrelated-run"):
                    pass
        for slot in (-1, count, True):
            with pytest.raises(ValueError, match="simulation slots"):
                allocate(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=slot)
    assert registry.occupied() == {}
    with pytest.raises(ValueError, match="active exclusive"):
        allocate(registry, gpu_uuid="GPU-example", owner_token=outer["token"], slot=count-1)
