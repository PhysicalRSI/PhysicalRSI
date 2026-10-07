"""Three-slot capacity protocol after successful two-slot native validation."""
from .simulation_slots import owned_simulation_resource as _validate_two_slot_owner


def owned_simulation_resource(registry, *, gpu_uuid, owner_token, slot):
    if type(slot) is not int or slot not in (0, 1, 2):
        raise ValueError("This capacity protocol permits three simulation slots")
    # Preserve the same active physical GPU ownership check and namespace.
    zero = _validate_two_slot_owner(registry, gpu_uuid=gpu_uuid, owner_token=owner_token, slot=0)
    return zero.rsplit("/", 1)[0] + "/" + str(slot)
