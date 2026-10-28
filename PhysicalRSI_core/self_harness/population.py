"""Immutable per-round candidate populations.

Population records sit beside the existing Self-Harness receipts.  They do
not score or select candidates; they bind candidate identities and optional
skill-level mutations so later analysis can attribute a selected result to a
primitive or composition change.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from ..infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json
from .artifacts import verify_harness
from .skill_mutation import SkillMutation
from .skill_binding import validate_candidate_skills


SCHEMA = "physicalrsi.population-record/v1"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a SHA256 identity")
    return value


def _member(value, *, parent_sha, role):
    if not isinstance(value, dict):
        raise ValueError("Population members must be objects")
    required = {"candidate_id", "candidate_sha256", "role", "parent_sha256", "skill_mutation"}
    allowed = required | {"skill_bindings"}
    if not required <= set(value) or not set(value) <= allowed or value["role"] != role:
        raise ValueError("Population member has unknown fields or an invalid role")
    identifier(_text(value["candidate_id"], "Population candidate_id"))
    _sha(value["candidate_sha256"], "Population candidate_sha256")
    if value["parent_sha256"] != parent_sha:
        raise ValueError("Population member belongs to another parent")
    mutation = value["skill_mutation"]
    if mutation is not None:
        mutation = SkillMutation.from_dict(mutation).as_dict()
    result = dict(candidate_id=value["candidate_id"], candidate_sha256=value["candidate_sha256"],
                role=role, parent_sha256=parent_sha, skill_mutation=mutation)
    if "skill_bindings" in value:
        candidate = {"skill_mutation": mutation, "skill_bindings": value["skill_bindings"]}
        validate_candidate_skills(candidate)
        result["skill_bindings"] = deepcopy(value["skill_bindings"])
    return result


@dataclass(frozen=True)
class PopulationRecord:
    """Content-addressed candidate population for one Self-Harness round."""

    round_id: str
    generation: int
    scope: str
    parent: dict[str, str]
    members: tuple[dict, ...]
    implementation: str | None = None

    def __post_init__(self):
        identifier(_text(self.round_id, "Population round_id"))
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("Population generation must be a nonnegative integer")
        _text(self.scope, "Population scope")
        if (not isinstance(self.parent, dict)
                or set(self.parent) != {"candidate_id", "candidate_sha256"}):
            raise ValueError("Population parent must declare candidate_id and candidate_sha256")
        identifier(_text(self.parent["candidate_id"], "Population parent candidate_id"))
        parent_sha = _sha(self.parent["candidate_sha256"], "Population parent candidate_sha256")
        if not isinstance(self.members, (list, tuple)):
            raise ValueError("Population members must be a sequence")
        members = tuple(_member(row, parent_sha=parent_sha, role="child") for row in self.members)
        ids = [row["candidate_id"] for row in members]
        if len(set(ids)) != len(ids) or self.parent["candidate_id"] in ids:
            raise ValueError("Population candidate IDs must be unique and exclude the parent")
        object.__setattr__(self, "parent", deepcopy(self.parent))
        object.__setattr__(self, "members", tuple(sorted(members, key=lambda row: row["candidate_id"])))
        implementation = self.implementation or file_digest(Path(__file__))
        if (not isinstance(implementation, str) or len(implementation) != 64
                or any(c not in "0123456789abcdef" for c in implementation)):
            raise ValueError("Population implementation must be a SHA256 digest")
        object.__setattr__(self, "implementation", implementation)

    @property
    def revision(self):
        return digest(self.as_dict(include_revision=False))

    @property
    def candidate_ids(self):
        return (self.parent["candidate_id"],) + tuple(row["candidate_id"] for row in self.members)

    def as_dict(self, *, include_revision=True):
        value = {
            "schema": SCHEMA,
            "round_id": self.round_id,
            "generation": self.generation,
            "scope": self.scope,
            "parent": deepcopy(self.parent),
            "members": deepcopy(list(self.members)),
            "implementation": self.implementation,
        }
        if include_revision:
            value["revision"] = self.revision
        return value

    @classmethod
    def from_candidates(cls, *, round_id, generation, scope, parent, children):
        parent_sha = verify_harness(parent)
        if not isinstance(children, list):
            raise ValueError("Population children must be a list")
        members = []
        for child in children:
            child_sha = verify_harness(child)
            if child.get("parent_sha256") != parent_sha:
                raise ValueError("Population child does not descend from the parent")
            mutation = child.get("skill_mutation")
            if mutation is not None:
                mutation = SkillMutation.from_dict(mutation).as_dict()
            member = dict(candidate_id=child["id"], candidate_sha256=child_sha,
                          role="child", parent_sha256=parent_sha, skill_mutation=mutation)
            if "skill_bindings" in child:
                validate_candidate_skills(child)
                member["skill_bindings"] = deepcopy(child["skill_bindings"])
            members.append(member)
        return cls(round_id=round_id, generation=generation, scope=scope,
                   parent=dict(candidate_id=parent["id"], candidate_sha256=parent_sha),
                   members=tuple(members))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Population record must be an object")
        required = {"schema", "revision", "round_id", "generation", "scope", "parent",
                    "members", "implementation"}
        if set(value) != required or value.get("schema") != SCHEMA:
            raise ValueError("Population record has unknown or missing fields")
        result = cls(round_id=value["round_id"], generation=value["generation"], scope=value["scope"],
                     parent=value["parent"], members=tuple(value["members"]),
                     implementation=value["implementation"])
        if value["revision"] != result.revision:
            raise ValueError("Population record revision differs from its content")
        return result

    @classmethod
    def read(cls, path):
        return cls.from_dict(read_json(path))

    def write(self, path):
        """Publish once; an identical retry is accepted."""
        path = Path(path)
        value = self.as_dict()
        with locked(path.with_suffix(path.suffix + ".lock")):
            if path.exists():
                if read_json(path) != value:
                    raise ValueError("Population record already belongs to another candidate pool")
            else:
                atomic_json(path, value)
        return self.revision


__all__ = ["SCHEMA", "PopulationRecord"]
