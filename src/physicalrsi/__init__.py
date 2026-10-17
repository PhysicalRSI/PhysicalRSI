"""Small public facade for the general-purpose PhysicalRSI framework.

The original ``PhysicalRSI`` package remains the compatibility application.
This lowercase package exposes stable, task-independent building blocks and
adapters to the existing Core and Autoresearch implementations.
"""

from .skills.protocol import CallableSkill, SkillDescriptor
from .skills.registry import SkillRegistry
from .skills.compose import compose_sequence

__all__ = ["CallableSkill", "SkillDescriptor", "SkillRegistry", "compose_sequence"]
