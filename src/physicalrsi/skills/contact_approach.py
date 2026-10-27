"""Bounded contact approach clearance; no task, seed or success-state inputs."""
import math


def scaled_clearance(nominal_clearance_m, clearance_scale=1.0):
    for value in (nominal_clearance_m, clearance_scale):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Clearance values must be finite numbers')
    if nominal_clearance_m <= 0 or not 0 < clearance_scale <= 1:
        raise ValueError('Require positive nominal clearance and scale in (0, 1]')
    return nominal_clearance_m * clearance_scale


def contact_approach(*, clearance_scale=1.0):
    from pathlib import Path
    from PhysicalRSI_core.infra.storage import digest, file_digest
    from .protocol import CallableSkill, SkillDescriptor
    scaled_clearance(1.0, clearance_scale)
    metadata = dict(clearance_scale=clearance_scale, state='stateless',
                    qualification='Geometric clearance only; no collision or reachability guarantee')
    revision = digest(dict(source=file_digest(Path(__file__)), metadata=metadata))
    return CallableSkill(SkillDescriptor('contact_approach', revision,
        dict(name='nominal_contact_clearance/v1', unit='m'),
        dict(name='contact_clearance/v1', unit='m'), metadata=metadata),
        lambda value, context: dict(clearance_m=scaled_clearance(value['nominal_clearance_m'], clearance_scale)))
