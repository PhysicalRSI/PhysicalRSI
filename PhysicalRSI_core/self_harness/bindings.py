"""Separate stable research rules from immutable per-round bindings.

Identity declarations remain trusted adapter contracts, not process isolation.
An adapter without policy_identity is fixed in its entirety across rounds.
"""
from copy import deepcopy
from typing import Protocol

from ..infra.storage import digest


class ResearchProposer(Protocol):
    def identity(self) -> dict: ...
    def develop(self, parent: dict, output) -> dict: ...
    def propose_with_context(self, parent: dict, feedback: dict,
                             research_context: dict, output) -> list[dict]: ...


def validate_round_context(context):
    from .protocol import _sha
    fields = {"schema", "index", "parent_revision", "memory_revision", "previous_round", "scope", "lessons", "budget"}
    if (not isinstance(context, dict) or set(context) != fields
            or context["schema"] != "physicalrsi.round-context/v1"
            or type(context["index"]) is not int or context["index"] < 0
            or not isinstance(context["lessons"], dict) or not isinstance(context["scope"], dict)
            or not context["scope"]
            or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
                   for k, v in context["scope"].items())):
        raise ValueError("Expected a complete versioned research round context")
    _sha(context["parent_revision"])
    _sha(context["memory_revision"])
    if context["previous_round"] is not None:
        _sha(context["previous_round"])
    budget = context["budget"]
    if (not isinstance(budget, dict) or set(budget) != {"max_rounds", "remaining_rounds"}
            or type(budget["max_rounds"]) is not int or not 1 <= budget["max_rounds"] <= 10000
            or type(budget["remaining_rounds"]) is not int
            or budget["remaining_rounds"] != budget["max_rounds"] - context["index"]
            or budget["remaining_rounds"] < 1):
        raise ValueError("Round context has an invalid research budget")
    digest(context)


def policy_identity(port):
    method = getattr(port, "policy_identity", None)
    value = method() if callable(method) else port.identity()
    if not isinstance(value, dict) or not value:
        raise ValueError("A nonempty implementation policy identity is required")
    digest(value)
    return deepcopy(value)


def validate_admission(receipt, freeze):
    """A normal rejection is evidence, while a malformed receipt is an error."""
    if not isinstance(receipt, dict) or receipt.get("freeze_sha256") != freeze or not receipt.get("evidence"):
        raise ValueError("Admission is not bound to candidate evidence")
    if type(receipt.get("accepted")) is not bool:
        raise ValueError("Admission must complete explicitly")
    if not receipt["accepted"] and (not isinstance(receipt.get("reason"), str) or not receipt["reason"].strip()):
        raise ValueError("Keep rejection reasons")
    return receipt
