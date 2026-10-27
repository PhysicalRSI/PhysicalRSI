"""Separate upward retreat from lateral motion in a gravity-aligned world frame."""
import math


def retreat_waypoint(current_m, destination_m):
    for vector in (current_m, destination_m):
        if (not isinstance(vector, (list, tuple)) or len(vector) != 3 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
            raise ValueError('Expected finite world-frame xyz metres')
    return [current_m[0], current_m[1], max(current_m[2], destination_m[2])]


def vertical_retreat():
    from pathlib import Path
    from PhysicalRSI_core.infra.storage import file_digest
    from .protocol import CallableSkill, SkillDescriptor
    return CallableSkill(SkillDescriptor('vertical_retreat', file_digest(Path(__file__)),
        dict(name='retreat_geometry/v1', unit='SI', frame='world'),
        dict(name='retreat_waypoint/v1', unit='SI', frame='world'),
        metadata=dict(qualification='Assumes world +Z is a suitable retreat direction; not a collision-free guarantee')),
        lambda value, context: retreat_waypoint(**value))
