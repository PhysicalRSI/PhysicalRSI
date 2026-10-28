"""Immutable mutation records for primitive and composed skills.

The Self-Harness still evaluates complete executable candidates.  A
``SkillMutation`` is the smaller research unit carried by a candidate so a
result can be attributed to a primitive or composition change without making
the selector understand skill internals.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infra.storage import digest, file_digest, identifier


SCHEMA = "physicalrsi.skill-mutation/v1"
UNIT_KINDS = frozenset({"primitive", "composition"})
OPERATIONS = frozenset({"replace", "insert", "remove", "rewire", "compose", "parameterize"})


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _reference(value, label, kind):
    if not isinstance(value, dict) or set(value) != {"id", "revision", "kind"}:
        raise ValueError(f"{label} must declare id, revision and kind")
    if value["kind"] != kind:
        raise ValueError(f"{label} kind differs from the mutation unit")
    identifier(_text(value["id"], f"{label} id"))
    _text(value["revision"], f"{label} revision")
    return {"id": value["id"], "revision": value["revision"], "kind": kind}


def _changes(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("A skill mutation needs at least one declared change")
    result = []
    for change in value:
        if not isinstance(change, dict) or not change:
            raise ValueError("Skill mutation changes must be nonempty objects")
        if "path" not in change or not isinstance(change["path"], str) or not change["path"].strip():
            raise ValueError("Every skill mutation change needs a path")
        if not ({"before", "after"} & set(change)):
            raise ValueError("Every skill mutation change needs before or after")
        result.append(deepcopy(change))
    return tuple(result)


def _evidence(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(path, str) or not path for path in value):
        raise ValueError("Skill mutation evidence must map paths to SHA256 digests")
    for sha in value.values():
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Skill mutation evidence must use SHA256 digests")
    return deepcopy(value)


def _payload(value, *, include_revision):
    payload = {
        "schema": SCHEMA,
        "unit_kind": value.unit_kind,
        "operation": value.operation,
        "parent": deepcopy(value.parent),
        "child": deepcopy(value.child),
        "operator": value.operator,
        "changes": deepcopy(list(value.changes)),
        "hypothesis": value.hypothesis,
        "expected_effect": value.expected_effect,
        "evidence": deepcopy(value.evidence),
        "implementation": value.implementation,
    }
    if include_revision:
        payload["revision"] = value.revision
    return payload


@dataclass(frozen=True)
class SkillMutation:
    """A content-addressed, candidate-independent skill mutation.

    ``parent`` and ``child`` refer to either a primitive descriptor or a
    ``SkillComposition`` declaration.  Candidate identity stays in
    ``PopulationRecord``; this makes the same mutation reusable across tasks,
    embodiments and candidate materializations.
    """

    unit_kind: str
    operation: str
    parent: dict[str, str]
    child: dict[str, str]
    operator: str
    changes: tuple[dict[str, Any], ...]
    hypothesis: str
    expected_effect: str
    evidence: dict[str, str] | None = None
    implementation: str | None = None

    def __post_init__(self):
        if self.unit_kind not in UNIT_KINDS:
            raise ValueError("Skill mutation unit_kind must be primitive or composition")
        if self.operation not in OPERATIONS:
            raise ValueError("Unknown skill mutation operation")
        object.__setattr__(self, "parent", _reference(self.parent, "parent", self.unit_kind))
        object.__setattr__(self, "child", _reference(self.child, "child", self.unit_kind))
        if self.parent["revision"] == self.child["revision"]:
            raise ValueError("A skill mutation must change the unit revision")
        identifier(_text(self.operator, "Skill mutation operator"))
        object.__setattr__(self, "operator", self.operator)
        object.__setattr__(self, "changes", _changes(self.changes))
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "Skill mutation hypothesis"))
        object.__setattr__(self, "expected_effect", _text(self.expected_effect, "Skill mutation expected_effect"))
        object.__setattr__(self, "evidence", _evidence(self.evidence))
        implementation = self.implementation or file_digest(Path(__file__))
        if (not isinstance(implementation, str) or len(implementation) != 64
                or any(c not in "0123456789abcdef" for c in implementation)):
            raise ValueError("Skill mutation implementation must be a SHA256 digest")
        object.__setattr__(self, "implementation", implementation)

    @property
    def revision(self):
        return digest(_payload(self, include_revision=False))

    def as_dict(self, *, include_revision=True):
        value = _payload(self, include_revision=False)
        if include_revision:
            value["revision"] = self.revision
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Skill mutation must be an object")
        required = {
            "schema", "revision", "unit_kind", "operation", "parent", "child",
            "operator", "changes", "hypothesis", "expected_effect", "evidence",
            "implementation",
        }
        if set(value) != required or value.get("schema") != SCHEMA:
            raise ValueError("Skill mutation has unknown or missing fields")
        result = cls(
            unit_kind=value["unit_kind"], operation=value["operation"],
            parent=value["parent"], child=value["child"], operator=value["operator"],
            changes=tuple(value["changes"]), hypothesis=value["hypothesis"],
            expected_effect=value["expected_effect"], evidence=value["evidence"],
            implementation=value["implementation"],
        )
        if value["revision"] != result.revision:
            raise ValueError("Skill mutation revision differs from its content")
        return result

    @classmethod
    def from_primitives(cls, parent, child, *, operation="replace", operator,
                        hypothesis, expected_effect, evidence=None):
        """Build a mutation from two ``SkillDescriptor``-like values."""
        before = _descriptor_payload(parent, "parent")
        after = _descriptor_payload(child, "child")
        return cls(
            unit_kind="primitive", operation=operation,
            parent={"kind": "primitive", "id": before["id"], "revision": before["revision"]},
            child={"kind": "primitive", "id": after["id"], "revision": after["revision"]},
            operator=operator, changes=_diff(before, after), hypothesis=hypothesis,
            expected_effect=expected_effect, evidence=evidence,
        )

    @classmethod
    def from_compositions(cls, parent, child, *, operation="compose", operator,
                          hypothesis, expected_effect, evidence=None):
        """Build a mutation from two serial ``SkillComposition`` values."""
        before = _composition_payload(parent)
        after = _composition_payload(child)
        return cls(
            unit_kind="composition", operation=operation,
            parent={"kind": "composition", "id": before["id"], "revision": before["revision"]},
            child={"kind": "composition", "id": after["id"], "revision": after["revision"]},
            operator=operator, changes=_diff(before, after), hypothesis=hypothesis,
            expected_effect=expected_effect, evidence=evidence,
        )


def _descriptor_payload(value, label):
    payload = value.as_dict() if hasattr(value, "as_dict") else deepcopy(value)
    if not isinstance(payload, dict) or not {"id", "revision"} <= set(payload):
        raise ValueError(f"{label} must be a skill descriptor")
    identifier(_text(payload["id"], f"{label} id"))
    _text(payload["revision"], f"{label} revision")
    return payload


def _composition_payload(value):
    from ..skill_graph.schema import normalize_composition
    composition = normalize_composition(value)
    return composition.as_dict()


def _diff(before, after, prefix=""):
    paths = sorted(set(before) | set(after))
    changes = []
    for key in paths:
        path = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changes.extend(_diff(left, right, path) if left != right else [])
        elif left != right:
            changes.append({"path": path, "before": deepcopy(left), "after": deepcopy(right)})
    if changes or prefix:
        return changes
    return [{"path": "revision", "before": before.get("revision"), "after": after.get("revision")}]


__all__ = ["SCHEMA", "UNIT_KINDS", "OPERATIONS", "SkillMutation"]
