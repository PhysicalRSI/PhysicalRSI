"""Small public facade for the general-purpose PhysicalRSI framework.

The original ``PhysicalRSI`` package remains the compatibility application.
This lowercase package exposes stable, task-independent building blocks and
adapters to the existing Core and Autoresearch implementations.
"""

from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path
import tomllib


def _version():
    """Read the installed distribution version, with a source-tree fallback."""
    source_metadata = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if source_metadata.exists():
        with source_metadata.open("rb") as stream:
            return tomllib.load(stream)["project"]["version"]
    try:
        return _distribution_version("physicalrsi")
    except PackageNotFoundError:
        return "0.3.0"


__version__ = _version()

from PhysicalRSI_core.contracts import Contract
from .skills.protocol import CallableSkill, SkillDescriptor, CompositionTypeError
from .skills.registry import SkillRegistry
from .skills.compose import compose_sequence, adapt, check_composition

__all__ = ["__version__", "Contract", "CallableSkill", "SkillDescriptor", "SkillRegistry",
           "compose_sequence", "adapt", "CompositionTypeError", "check_composition"]
