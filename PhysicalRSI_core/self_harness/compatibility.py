"""Separate producer provenance from the execution rules of a campaign."""
from copy import deepcopy

from ..infra.storage import canonical


# Source-only legacy configurations implement this same execution protocol.
# Change this version when stage/recovery semantics become incompatible.
ENGINE_VERSION = 1
ARCHIVE_SCHEMA = "physicalrsi.candidate-archive/v1"
CAMPAIGN_SCHEMAS = {"physicalrsi.campaign/v1", "physicalrsi.research-campaign/v1"}


def archive_reference(identity):
    """Reader identity is the storage format and location, not its producer."""
    return {"schema": identity["schema"], "root": identity["root"]}


def same_archive(saved, current):
    return (saved.get("schema") == current.get("schema") == ARCHIVE_SCHEMA
            and archive_reference(saved) == archive_reference(current))


def _archive_references(value):
    if isinstance(value, dict):
        if value.get("schema") == ARCHIVE_SCHEMA:
            return archive_reference(value)
        return {key: _archive_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_archive_references(item) for item in value]
    return deepcopy(value)


def execution_config(config):
    """Project a frozen config onto the rules required to continue execution.

    Only orchestration and archive source provenance is omitted. Selectors,
    scoring, providers, evaluation protocols, budgets and scope remain bound.
    The stored configuration itself is never converted or rewritten.
    """
    value = _archive_references(config)
    value.setdefault("engine_version", 1)
    if "identities" in value:
        value["identities"].pop("core", None)
    if value.get("schema") in CAMPAIGN_SCHEMAS:
        value.pop("implementation", None)
    for key in ("policy", "campaign_policy"):
        if key in value:
            value[key] = execution_config(value[key])
    return value


def same_execution(saved, current):
    return canonical(execution_config(saved)) == canonical(execution_config(current))
