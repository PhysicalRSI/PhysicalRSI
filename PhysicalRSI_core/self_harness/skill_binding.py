"""Portable skill bindings carried by Self-Harness candidates.

The harness does not import or execute a skill provider.  A proposer records
the registry binding that an evaluator can use to materialize the candidate,
while this module checks that the binding and mutation describe the same
skill revisions.  Candidates written before skill bindings existed remain
valid and continue through the legacy path.
"""

from copy import deepcopy

from ..infra.storage import identifier
from .skill_mutation import SkillMutation


SCHEMA = "physicalrsi.skill-binding/v2"
LEGACY_SCHEMA = "physicalrsi.skill-binding/v1"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _unit(binding):
    record = binding.get("record")
    if not isinstance(record, dict):
        raise ValueError("Skill binding record must be an object")
    descriptor = record.get("descriptor", record)
    if not isinstance(descriptor, dict):
        raise ValueError("Skill binding descriptor must be an object")
    skill_id = _text(descriptor.get("id"), "Skill binding skill id")
    revision = _text(descriptor.get("revision"), "Skill binding skill revision")
    identifier(skill_id)
    return skill_id, revision


def validate_binding(value, *, role=None):
    """Validate one persisted registry binding and return a detached copy."""

    if not isinstance(value, dict):
        raise ValueError("Skill binding must be an object")
    required = {"schema", "record", "implementation"}
    schema = value.get("schema")
    if not required <= set(value) or schema not in {SCHEMA, LEGACY_SCHEMA}:
        raise ValueError("Skill binding has an invalid schema")
    if schema == SCHEMA:
        if not isinstance(value.get('registry_id'), str) or not value['registry_id'].strip():
            raise ValueError('Portable skill binding needs a registry id')
    elif not isinstance(value.get('registry_root'), str) or not value['registry_root'].strip():
        raise ValueError('Legacy skill binding needs a registry root')
    if role is not None and value.get("role") != role:
        raise ValueError("Skill binding role differs from the mutation side")
    if value.get("implementation") is None:
        raise ValueError("Executable skill implementation binding is required")
    if not isinstance(value["implementation"], dict):
        raise ValueError("Skill implementation binding must be an object")
    _unit(value)
    return deepcopy(value)


def attach_mutation(candidate, mutation, *, parent_binding, child_binding):
    """Attach a mutation and its executable parent/child bindings to a candidate."""

    mutation = mutation if isinstance(mutation, SkillMutation) else SkillMutation.from_dict(mutation)
    parent = validate_binding(dict(parent_binding, role="parent"), role="parent")
    child = validate_binding(dict(child_binding, role="child"), role="child")
    if _unit(parent) != (mutation.parent["id"], mutation.parent["revision"]):
        raise ValueError("Parent skill binding differs from the mutation")
    if _unit(child) != (mutation.child["id"], mutation.child["revision"]):
        raise ValueError("Child skill binding differs from the mutation")
    result = deepcopy(candidate)
    result["skill_mutation"] = mutation.as_dict()
    result["skill_bindings"] = [parent, child]
    return result


def validate_candidate_skills(candidate):
    """Validate optional skill metadata on a candidate.

    The empty case is intentional for historical task-specific candidates.
    Once a mutation is declared, both executable sides must be persisted.
    """

    mutation_value = candidate.get("skill_mutation")
    bindings = candidate.get("skill_bindings")
    if mutation_value is None and bindings is None:
        return None
    if mutation_value is None or not isinstance(bindings, list) or len(bindings) != 2:
        raise ValueError("A skill mutation requires parent and child bindings")
    mutation = SkillMutation.from_dict(mutation_value)
    by_role = {}
    for value in bindings:
        role = value.get("role") if isinstance(value, dict) else None
        if role in by_role:
            raise ValueError("Skill bindings must contain one parent and one child")
        by_role[role] = validate_binding(value, role=role)
    if set(by_role) != {"parent", "child"}:
        raise ValueError("Skill bindings must contain one parent and one child")
    if _unit(by_role["parent"]) != (mutation.parent["id"], mutation.parent["revision"]):
        raise ValueError("Parent skill binding differs from the mutation")
    if _unit(by_role["child"]) != (mutation.child["id"], mutation.child["revision"]):
        raise ValueError("Child skill binding differs from the mutation")
    return dict(mutation=mutation, bindings=[by_role["parent"], by_role["child"]])


__all__ = ["SCHEMA", "attach_mutation", "validate_binding", "validate_candidate_skills"]
