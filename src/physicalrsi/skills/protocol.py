"""Small, task-independent skill primitives.

The core only describes a callable and forwards execution.  Persistence,
legacy operation adaptation, and research policy belong to the surrounding
adapters.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
import json
from types import CodeType, FunctionType, MethodType, ModuleType
from typing import Any, Callable

from PhysicalRSI_core.contracts import Operation


SCHEMA = "physicalrsi.skill/v1"
RECORD_SCHEMA = "physicalrsi.skill-record/v2"


class CoreContext:
    """Minimal call context; adapters may provide richer runtime context."""

    def check(self):
        return None


@dataclass(frozen=True)
class SkillDescriptor:
    id: str
    revision: str
    input: dict[str, Any]
    output: dict[str, Any]
    effects: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, "effects", tuple(self.effects))

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
    """A callable plus a portable description of how it can be composed."""

    def __init__(self, descriptor: SkillDescriptor, function: Callable, *,
                 composition=None, executable_identity=None, children=()):
        self.descriptor = deepcopy(descriptor)
        self._identity = descriptor.as_dict()
        self._function = function
        self._children = tuple(children)
        if composition is None:
            self._composition = None
            self._composition_identity = None
        else:
            from PhysicalRSI_core.skill_graph.schema import normalize_composition
            self._composition = normalize_composition(composition)
            self._composition_identity = self._composition.as_dict()
        # A declaration of the initial implementation/configuration. Providers
        # with external assets can supply their own portable identity here.
        self._executable_identity = deepcopy(
            executable_identity if executable_identity is not None
            else _callable_identity(function)
        )

    @property
    def id(self):
        return self.descriptor.id

    @property
    def revision(self):
        return self.descriptor.revision

    @property
    def composition(self):
        return self._composition

    @composition.setter
    def composition(self, value):
        if value is None:
            self._composition = None
            self._composition_identity = None
            return
        from PhysicalRSI_core.skill_graph.schema import normalize_composition
        self._composition = normalize_composition(value)
        self._composition_identity = self._composition.as_dict()

    def execute(self, value, context=None):
        if context is None:
            context = CoreContext()
        return self._function(value, context)

    def as_operation(self):
        """Adapt this core skill to the historical Operation runtime."""
        from PhysicalRSI_core.contracts import Contract
        descriptor = self.descriptor
        children = tuple(child.as_operation() if isinstance(child, CallableSkill) else child
                         for child in self._children)
        return Operation(
            descriptor.id,
            descriptor.revision,
            Contract(**descriptor.input),
            Contract(**descriptor.output),
            self.execute,
            frozenset(descriptor.effects),
            children,
        )

    def identity(self):
        return deepcopy(self._identity)

    def composition_identity(self):
        if self._composition is None:
            return None
        return deepcopy(self._composition.as_dict())

    def executable_identity(self):
        return deepcopy(self._executable_identity)

    execution_identity = executable_identity

    def registry_record(self):
        return {
            "schema": RECORD_SCHEMA,
            "descriptor": self.identity(),
            "composition": self.composition_identity(),
            "implementation": self.executable_identity(),
        }

    @classmethod
    def from_operation(cls, operation, *, composition=None):
        descriptor = SkillDescriptor(
            id=operation.name,
            revision=operation.revision,
            input=asdict(operation.input),
            output=asdict(operation.output),
            effects=tuple(sorted(operation.effects)),
            metadata={"adapter": "PhysicalRSI_core.contracts.Operation"},
        )

        def execute(value, context):
            # The historical Operation runtime needs its own context fields.
            if isinstance(context, CoreContext):
                from PhysicalRSI_core.contracts import Context
                context = Context("skill-call")
            return operation(value, context)

        return cls(
            descriptor,
            execute,
            composition=composition,
            executable_identity=_operation_executable_identity(operation),
            children=operation.children,
        )


def descriptor_identity(skill):
    return skill.descriptor.as_dict()


def _callable_identity(function, seen=None):
    seen = set() if seen is None else seen
    if id(function) in seen:
        return {"reference": f"{getattr(function, '__module__', type(function).__module__)}."
                             f"{getattr(function, '__qualname__', type(function).__qualname__)}"}
    seen = seen | {id(function)}
    value = {
        "schema": "physicalrsi.executable/v1",
        "kind": "python-callable",
        "module": getattr(function, "__module__", type(function).__module__),
        "qualname": getattr(function, "__qualname__", type(function).__qualname__),
    }
    if isinstance(function, partial):
        value.update(kind="partial", function=_callable_identity(function.func, seen),
                     args=_stable_identity_value(function.args, seen),
                     keywords=_stable_identity_value(function.keywords, seen))
        return value
    if isinstance(function, MethodType):
        value.update(kind="bound-method", function=_callable_identity(function.__func__, seen),
                     owner=_stable_identity_value(function.__self__, seen))
        return value
    code = getattr(function, "__code__", None)
    if code is not None:
        value["code"] = _stable_identity_value(code, seen)
        value["defaults"] = _stable_identity_value(function.__defaults__, seen)
        value["kwdefaults"] = _stable_identity_value(function.__kwdefaults__, seen)
        closure = getattr(function, "__closure__", None)
        if closure is not None:
            value["closure"] = [
                _stable_identity_value(cell.cell_contents, seen)
                for cell in closure
            ]
    elif not isinstance(function, type) and hasattr(type(function).__call__, "__code__"):
        value.update(kind="callable-object", call=_callable_identity(type(function).__call__, seen),
                     state=_object_state(function, seen))
    return value


def _stable_identity_value(value, seen=None):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, ModuleType):
        return {"module": value.__name__}
    if isinstance(value, CallableSkill):
        return {"skill": value.registry_record()}
    seen = set() if seen is None else seen
    if isinstance(value, (FunctionType, MethodType, partial)):
        return _callable_identity(value, seen)
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    if id(value) in seen:
        return {"reference": name}
    seen = seen | {id(value)}
    if isinstance(value, CodeType):
        return dict(bytecode=value.co_code.hex(), names=list(value.co_names),
                    constants=[_stable_identity_value(item, seen) for item in value.co_consts],
                    argcount=value.co_argcount, posonlyargcount=value.co_posonlyargcount,
                    kwonlyargcount=value.co_kwonlyargcount, flags=value.co_flags,
                    varnames=list(value.co_varnames), freevars=list(value.co_freevars),
                    cellvars=list(value.co_cellvars),
                    exceptiontable=getattr(value, 'co_exceptiontable', b'').hex())
    if isinstance(value, list):
        return [_stable_identity_value(item, seen) for item in value]
    if isinstance(value, tuple):
        return {"tuple": [_stable_identity_value(item, seen) for item in value]}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _stable_identity_value(item, seen) for key, item in sorted(value.items())}
        return {"mapping": [[_stable_identity_value(key, seen), _stable_identity_value(item, seen)]
                            for key, item in value.items()]}
    if isinstance(value, (set, frozenset)):
        return {name: sorted((_stable_identity_value(item, seen) for item in value),
                             key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, (bytes, bytearray)):
        return {name: value.hex()}
    return {"type": name, "state": _object_state(value, seen)}


def _object_state(value, seen):
    state = dict(getattr(value, "__dict__", {}))
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot not in {"__dict__", "__weakref__"} and hasattr(value, slot):
                state[slot] = getattr(value, slot)
    return _stable_identity_value(state, seen)


def _operation_executable_identity(operation):
    value = {
        "schema": "physicalrsi.executable/v1",
        "kind": "operation",
        "name": operation.name,
        "revision": operation.revision,
        "children": [_operation_executable_identity(child)
                      for child in operation.children],
    }
    owner = getattr(operation.call, "__self__", None)
    if isinstance(owner, CallableSkill):
        value["implementation"] = owner.executable_identity()
    elif not operation.children:
        value["callable"] = _callable_identity(operation.call)
    return value
