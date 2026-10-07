"""Declared embodiment and System 1 interfaces, plus System 2 trial attribution.

These declarations bind adapters; they do not implement a controller, convert
units, inspect checkpoints, or qualify hardware. Revisions are provider-owned
except memory and candidate identities, which must be content-addressed.
"""

from dataclasses import dataclass

from .contracts import Contract
from .timing import ControlTiming


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Expected a SHA256 revision")


def resource_names(values):
    if isinstance(values, str):
        raise ValueError("Resources must be a collection of names")
    values = tuple(values)
    for value in values:
        _text(value, "Resource name")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate resource name")
    return tuple(sorted(values))


@dataclass(frozen=True)
class Revision:
    name: str
    revision: str

    def __post_init__(self):
        _text(self.name, "Name")
        _text(self.revision, "Revision")


@dataclass(frozen=True)
class Dependency(Revision):
    kind: str

    def __post_init__(self):
        super().__post_init__()
        if self.kind not in {"model", "skill", "tool", "memory"}:
            raise ValueError("Unknown System 1 dependency kind")
        if self.kind == "memory":
            _sha(self.revision)


@dataclass(frozen=True)
class Embodiment(Revision):
    mode: str
    observation: Contract
    action: Contract
    resources: tuple[str, ...] = ()
    timing: ControlTiming | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in {"software", "simulation", "physical"}:
            raise ValueError("Declare software, simulation or physical execution")
        if not isinstance(self.observation, Contract) or not isinstance(self.action, Contract):
            raise ValueError("Embodiment requires observation and action contracts")
        object.__setattr__(self, "resources", resource_names(self.resources))
        if self.mode == "physical" and not self.resources:
            raise ValueError("Physical execution requires named device resources")
        if self.timing is not None and not isinstance(self.timing, ControlTiming):
            raise ValueError("Invalid embodiment control timing")


@dataclass(frozen=True)
class System1(Revision):
    observation: Contract
    action: Contract
    dependencies: tuple[Dependency, ...] = ()
    timing: ControlTiming | None = None

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.observation, Contract) or not isinstance(self.action, Contract):
            raise ValueError("System 1 requires observation and action contracts")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, Dependency) for item in dependencies):
            raise ValueError("System 1 dependencies must be versioned")
        if len({(item.kind, item.name) for item in dependencies}) != len(dependencies):
            raise ValueError("Duplicate System 1 dependency")
        object.__setattr__(self, "dependencies", tuple(sorted(
            dependencies, key=lambda item: (item.kind, item.name))))
        if self.timing is not None and not isinstance(self.timing, ControlTiming):
            raise ValueError("Invalid System 1 control timing")


@dataclass(frozen=True)
class ExecutionBinding:
    embodiment: Embodiment
    system1: System1

    def __post_init__(self):
        if not isinstance(self.embodiment, Embodiment) or not isinstance(self.system1, System1):
            raise ValueError("An embodiment and System 1 declaration are required")
        for field in ("observation", "action"):
            if getattr(self.embodiment, field) != getattr(self.system1, field):
                raise ValueError(f"System 1 {field} contract differs from the embodiment")
        if self.embodiment.timing != self.system1.timing:
            raise ValueError("System 1 control timing differs from the embodiment")


@dataclass(frozen=True)
class System2Trial:
    """A trial belongs to a frozen candidate; selection remains in Self-Harness.

    Validation/test trials require the comparison and cohort identities. This
    is attribution, not a proof of split independence or a selection decision.
    """

    system2: Revision
    candidate_id: str
    candidate_sha256: str
    split: str
    comparison_sha256: str | None = None
    cohort_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.system2, Revision):
            raise ValueError("Declare the System 2 implementation revision")
        _text(self.candidate_id, "Candidate ID")
        _sha(self.candidate_sha256)
        if self.split not in {"development", "validation", "test"}:
            raise ValueError("Declare the trial split")
        for value in (self.comparison_sha256, self.cohort_sha256):
            if value is not None:
                _sha(value)
        if self.split != "development" and (
            self.comparison_sha256 is None or self.cohort_sha256 is None
        ):
            raise ValueError("Validation/test trials require a frozen comparison and cohort")

    def bind(self, system1):
        if system1.revision != self.candidate_sha256:
            raise ValueError("System 2 candidate differs from the executing System 1 revision")
