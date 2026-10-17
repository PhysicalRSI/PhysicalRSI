"""Task-independent skill declarations and trusted execution adapters."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from PhysicalRSI_core.infra.storage import identifier, digest
from PhysicalRSI_core.contracts import Contract, Context, Operation
from dataclasses import asdict

SCHEMA = "physicalrsi.skill/v1"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


@dataclass(frozen=True)
class SkillDescriptor:
    id: str
    revision: str
    input: dict[str, Any]
    output: dict[str, Any]
    effects: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        identifier(_text(self.id, "Skill id"))
        _text(self.revision, "Skill revision")
        if not isinstance(self.input, dict) or not isinstance(self.output, dict):
            raise ValueError("Skill input and output contracts must be objects")
        Contract(**self.input)
        Contract(**self.output)
        effects = tuple(self.effects)
        if any(not isinstance(item, str) or not item.strip() for item in effects):
            raise ValueError("Skill effects must be nonempty strings")
        if len(set(effects)) != len(effects):
            raise ValueError("Skill effects must be unique")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("Skill metadata must be an object")
        object.__setattr__(self, "effects", tuple(sorted(effects)))

    def as_dict(self):
        return {
            "schema": SCHEMA,
            "id": self.id,
            "revision": self.revision,
            "input": deepcopy(self.input),
            "output": deepcopy(self.output),
            "effects": list(self.effects),
            "metadata": deepcopy(self.metadata),
        }


class CallableSkill:
    """A reviewed callable with an immutable, task-independent descriptor."""

    def __init__(self, descriptor: SkillDescriptor, function: Callable):
        if not isinstance(descriptor, SkillDescriptor) or not callable(function):
            raise ValueError("CallableSkill requires a descriptor and callable")
        self.descriptor = deepcopy(descriptor)
        self._identity = descriptor.as_dict()
        self._function = function

    @property
    def id(self):
        return self.descriptor.id

    @property
    def revision(self):
        return self.descriptor.revision

    def execute(self, value, context=None):
        self.identity()
        context = context or Context("skill-call")
        context.check()
        result = self._function(value, context)
        context.check()
        self.identity()
        return result

    def as_operation(self):
        d = self.descriptor
        return Operation(d.id, d.revision, Contract(**d.input), Contract(**d.output),
                         self.execute, frozenset(d.effects))

    def identity(self):
        if self.descriptor.as_dict() != self._identity:
            raise ValueError("Skill descriptor changed")
        return deepcopy(self._identity)

    @classmethod
    def from_operation(cls, operation):
        from dataclasses import asdict

        descriptor = SkillDescriptor(
            id=operation.name,
            revision=operation.revision,
            input=asdict(operation.input),
            output=asdict(operation.output),
            effects=tuple(sorted(operation.effects)),
            metadata={"adapter": "PhysicalRSI_core.contracts.Operation"},
        )
        return cls(descriptor, operation)


def descriptor_identity(skill):
    descriptor = getattr(skill, "descriptor", None)
    if not isinstance(descriptor, SkillDescriptor):
        raise ValueError("Expected a skill with a SkillDescriptor")
    return descriptor.as_dict()
