"""Helpers for carrying a bound skill implementation into a candidate."""

from copy import deepcopy

from PhysicalRSI_core.self_harness.skill_binding import attach_mutation


def bind_skill(skill, *, role):
    """Return the persisted binding identity for a loaded executable skill."""

    method = getattr(skill, "binding_identity", None)
    if not callable(method) or not getattr(skill, '_registry_bound', False):
        raise ValueError('Register the skill before creating an executable binding')
    value = method()
    value = deepcopy(value)
    value["role"] = role
    return value


def attach_skill_mutation(candidate, mutation, *, parent_skill, child_skill):
    """Attach executable parent/child bindings to a harness candidate."""

    return attach_mutation(
        candidate,
        mutation,
        parent_binding=bind_skill(parent_skill, role="parent"),
        child_binding=bind_skill(child_skill, role="child"),
    )


__all__ = ["attach_skill_mutation", "bind_skill"]
