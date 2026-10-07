"""Forty-eight-slot simulation capacity protocol with exclusive physical GPU ownership."""
from .simulation_slots import owned_simulation_resource as _validate_owner


def owned_simulation_resource(registry, *, gpu_uuid, owner_token, slot):
    if type(slot) is not int or not 0 <= slot < 48:
        raise ValueError("This capacity protocol permits forty-eight simulation slots")
    zero = _validate_owner(registry, gpu_uuid=gpu_uuid, owner_token=owner_token, slot=0)
    return zero.rsplit("/", 1)[0] + "/" + str(slot)
