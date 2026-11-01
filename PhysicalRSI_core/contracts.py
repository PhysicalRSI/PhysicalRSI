"""Shared execution contracts; core has no dependency on concrete capabilities."""

from dataclasses import dataclass, field
import math
from threading import Event
from time import monotonic
from typing import Any, Callable


@dataclass(frozen=True)
class Contract:
    """Exact semantic type. Unit/frame/embodiment conversions must be explicit."""

    name: str
    unit: str = ""
    frame: str = ""
    embodiment: str = ""
    shape: tuple[int | None, ...] | None = None
    dtype: str = ""
    rate_hz: float | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("A named contract is required")
        if self.shape is not None:
            if not isinstance(self.shape, (tuple, list)):
                raise ValueError("Contract shape must be a list or tuple")
            shape = tuple(self.shape)
            if any(dim is not None and (isinstance(dim, bool) or not isinstance(dim, int) or dim < 0)
                   for dim in shape):
                raise ValueError("Contract shape entries must be nonnegative integers or None")
            object.__setattr__(self, "shape", shape)
        if not isinstance(self.dtype, str):
            raise ValueError("Contract dtype must be a string")
        if self.rate_hz is not None:
            if isinstance(self.rate_hz, bool) or not isinstance(self.rate_hz, (int, float)) \
                    or not math.isfinite(self.rate_hz) or self.rate_hz <= 0:
                raise ValueError("Contract rate_hz must be finite and positive")

    def accepts(self, producer: "Contract") -> bool:
        return not mismatches(producer, self)


def mismatches(producer: Contract, consumer: Contract) -> list[str]:
    if not isinstance(producer, Contract) or not isinstance(consumer, Contract):
        raise ValueError("mismatches requires Contract values")
    result = []
    if producer.name != consumer.name:
        result.append(f"name: producer {producer.name!r} != consumer {consumer.name!r}")
    for field_name in ("unit", "frame", "embodiment", "dtype", "rate_hz"):
        produced = getattr(producer, field_name)
        consumed = getattr(consumer, field_name)
        if consumed in ("", None):
            continue
        if produced in ("", None):
            result.append(f"{field_name}: producer is unspecified, consumer requires {consumed!r}")
        elif produced != consumed:
            result.append(f"{field_name}: producer {produced!r} != consumer {consumed!r}")
    if consumer.shape is not None:
        if producer.shape is None:
            result.append("shape: producer is unspecified, consumer requires " + repr(consumer.shape))
        elif len(producer.shape) != len(consumer.shape):
            result.append(f"shape: producer rank {len(producer.shape)} != consumer rank {len(consumer.shape)}")
        else:
            for index, (produced, consumed) in enumerate(zip(producer.shape, consumer.shape)):
                if consumed is None:
                    continue
                if produced is None:
                    result.append(f"shape[{index}]: producer wildcard does not satisfy consumer {consumed}")
                elif produced != consumed:
                    result.append(f"shape[{index}]: producer {produced} != consumer {consumed}")
    return result


def contract_from(value) -> Contract:
    if isinstance(value, Contract):
        return value
    if not isinstance(value, dict):
        raise ValueError("Contract must be a Contract or dict")
    fields = {"name", "unit", "frame", "embodiment", "shape", "dtype", "rate_hz"}
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"Contract has unknown fields: {sorted(unknown)}")
    if "name" not in value:
        raise ValueError("Contract dict requires a name")
    try:
        return Contract(**value)
    except TypeError as exc:
        raise ValueError(f"Invalid contract fields: {exc}") from exc


def contract_payload(contract: Contract) -> dict:
    contract = contract_from(contract)
    return {
        "name": contract.name,
        "unit": contract.unit,
        "frame": contract.frame,
        "embodiment": contract.embodiment,
        "shape": list(contract.shape) if contract.shape is not None else None,
        "dtype": contract.dtype,
        "rate_hz": contract.rate_hz,
    }


class Cancelled(RuntimeError):
    """Cooperative cancellation, including deadline expiry."""


class InfrastructureFault(RuntimeError):
    """The trial could not be executed faithfully because infrastructure failed.

    Ports may raise this exception for a lost device, sensor timeout, or another
    explicitly detected execution fault.  The evaluator may retry the same
    deterministic case after the runtime closes the faulted attempt.
    """


class ReconciliationRequired(RuntimeError):
    """An uncertain external effect must be inspected before reuse or retry."""


@dataclass(frozen=True)
class Context:
    episode: str
    cancelled: Event = field(default_factory=Event)
    deadline: float | None = None
    execution: Callable | None = field(default=None, compare=False, repr=False)
    harness_revision: str = ""
    # A controller can attach a cooperative ownership/liveness check. It is
    # transient execution context, never part of a frozen harness artifact.
    validity: Callable[[], None] | None = field(default=None, compare=False, repr=False)
    # Per-trial content addressed sensor access.  This is transient execution
    # context and is intentionally excluded from equality and representation.
    blobs: Any | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        from PhysicalRSI_core.infra.storage import identifier

        identifier(self.episode)

    def check(self) -> None:
        if self.cancelled.is_set() or (
            self.deadline is not None and monotonic() >= self.deadline
        ):
            raise Cancelled("Execution cancelled or deadline exceeded")
        if self.validity is not None:
            self.validity()
            if self.cancelled.is_set() or (self.deadline is not None and monotonic() >= self.deadline):
                raise Cancelled("Execution cancelled or deadline exceeded while checking validity")


@dataclass(frozen=True)
class Operation:
    """A versioned executable endpoint shared by memory, tools and skills.

    `revision` binds the implementation and its dependencies by the provider.
    `effects` name exclusive physical/state resources, not success conditions.
    The caller owns completion validation; returning a value is not task success.
    """

    name: str
    revision: str
    input: Contract
    output: Contract
    call: Callable[[Any, Context], Any] = field(compare=False, repr=False)
    effects: frozenset[str] = frozenset()
    children: tuple = ()

    def __post_init__(self):
        if not self.name or not self.revision or not callable(self.call):
            raise ValueError("Operation requires a name, revision and callable")
        object.__setattr__(self, "effects", frozenset(self.effects))

    def __call__(self, value: Any, context: Context) -> Any:
        if context.execution is not None:
            return context.execution(self, value, context)
        context.check()
        result = self.call(value, context)
        context.check()
        return result
