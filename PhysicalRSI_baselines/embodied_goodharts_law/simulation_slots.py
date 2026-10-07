"""Namespaced simulation leases under an actively owned physical GPU.

Only simulation workers may use these resources. The outer supervisor retains
the exclusive physical GPU lease; these names do not authorize physical robots.
"""


def owned_simulation_resource(registry, *, gpu_uuid, owner_token, slot):
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("A physical GPU UUID is required")
    if type(slot) is not int or slot not in (0, 1):
        raise ValueError("The initial capacity protocol permits two simulation slots")
    record = registry.inspect(owner_token)
    if (record["state"] != "held" or not record["owner_active"]
            or record["resources"] != [gpu_uuid]
            or registry.occupied().get(gpu_uuid) != owner_token):
        raise ValueError("An active exclusive outer GPU lease is required")
    return f"{gpu_uuid}/simulation-slot/{slot}"
