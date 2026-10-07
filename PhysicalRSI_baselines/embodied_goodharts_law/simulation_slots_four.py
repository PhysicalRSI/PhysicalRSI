"""Four-slot capacity protocol with the existing exclusive GPU owner check."""
from .simulation_slots import owned_simulation_resource as _validate_owner


def owned_simulation_resource(registry, *, gpu_uuid, owner_token, slot):
    if type(slot) is not int or slot not in (0, 1, 2, 3):
        raise ValueError("This capacity protocol permits four simulation slots")
    zero = _validate_owner(registry, gpu_uuid=gpu_uuid, owner_token=owner_token, slot=0)
    return zero.rsplit("/", 1)[0] + "/" + str(slot)
