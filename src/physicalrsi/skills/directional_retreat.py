"""Generate a retreat waypoint opposite a caller-observed contact direction.

The caller owns triggering, collision checks, execution and completion checks.
This geometry primitive neither verifies an object effect nor executes motion.
"""
import math
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest
from .protocol import CallableSkill, SkillDescriptor


def _xyz(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('Expected finite world-frame xyz')
    return value


def directional_retreat(*, distance_m=.15):
    if type(distance_m) not in (int, float) or not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError('Retreat distance must be positive finite metres')
    parameters = dict(distance_m=float(distance_m))

    def run(value, context):
        if not isinstance(value, dict) or set(value) != {'current_m', 'contact_direction'}:
            raise ValueError('Expected current_m and contact_direction only')
        point = _xyz(value['current_m'])
        direction = _xyz(value['contact_direction'])
        norm = math.hypot(*direction)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('Contact direction must have a finite nonzero norm')
        waypoint = [p - distance_m * (d / norm) for p, d in zip(point, direction)]
        _xyz(waypoint)
        return dict(waypoint_m=waypoint)

    revision = digest(dict(source=file_digest(Path(__file__)), parameters=parameters))
    return CallableSkill(SkillDescriptor(
        'directional_retreat', revision,
        dict(name='directional_retreat_input/v1', unit='SI', frame='world'),
        dict(name='directional_retreat_waypoint/v1', unit='SI', frame='world'),
        metadata=dict(parameters=parameters,
                      qualification='Geometry only; caller must verify clearance and observed effect')),
        run)
