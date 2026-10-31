"""Small public facade for the general-purpose PhysicalRSI framework.

The original ``PhysicalRSI`` package remains the compatibility application.
This lowercase package exposes stable, task-independent building blocks and
adapters to the existing Core and Autoresearch implementations.
"""

from PhysicalRSI_core.contracts import Contract
from .skills.protocol import CallableSkill, SkillDescriptor, CompositionTypeError
from .skills.registry import SkillRegistry
from .skills.compose import compose_sequence, adapt, check_composition

__all__ = ["Contract", "CallableSkill", "SkillDescriptor", "SkillRegistry",
           "compose_sequence", "adapt", "CompositionTypeError", "check_composition"]
