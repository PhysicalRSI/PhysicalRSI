"""Observed constant-velocity motion crossing a declared support boundary."""
import math
from pathlib import Path
from PhysicalRSI_core.infra.storage import digest, file_digest
from .protocol import CallableSkill, SkillDescriptor


def support_exit(*, max_horizon_s=30.):
    """Estimate a reference-point crossing, without inventing a release impulse.

    Boundary normal points out of the supported region. The caller supplies
    visible static geometry, not a future release position. The model assumes
    locally constant observed velocity until crossing and continuous velocity
    at release. It does not model tipping, changing support, or an exit impulse.
    """
    if isinstance(max_horizon_s, bool) or not math.isfinite(max_horizon_s) or max_horizon_s <= 0:
        raise ValueError('Positive finite horizon required')

    def vector(value):
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError('World xyz required')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
            raise ValueError('Finite vector required')
        return tuple(float(x) for x in value)

    def run(value, context):
        p, v = vector(value['position_m']), vector(value['velocity_m_s'])
        boundary = value['boundary']
        point, normal = vector(boundary['point_m']), vector(boundary['outward_normal'])
        length = math.sqrt(sum(x*x for x in normal))
        if length <= 1e-9:
            raise ValueError('Nonzero boundary normal required')
        normal = tuple(x/length for x in normal)
        distance = sum((point[i]-p[i])*normal[i] for i in range(3))
        speed = sum(v[i]*normal[i] for i in range(3))
        if distance < 0:
            return dict(valid=False, reason='outside_supported_region')
        if speed <= 1e-9:
            return dict(valid=False, reason='not_approaching_support_boundary')
        duration = distance/speed
        if duration > max_horizon_s:
            return dict(valid=False, reason='support_exit_horizon_exceeded')
        return dict(valid=True, reason=None, time_to_exit_s=duration,
                    exit_position_m=[p[i]+duration*v[i] for i in range(3)],
                    exit_velocity_m_s=list(v),
                    model='Observed constant velocity, continuous reference-point release')

    metadata = dict(parameters=dict(max_horizon_s=max_horizon_s),
                    assumptions='Static boundary, constant observed velocity, no release impulse')
    revision = digest(dict(source=file_digest(Path(__file__)), metadata=metadata))
    return CallableSkill(SkillDescriptor('support_exit', revision,
        dict(name='observed_support_motion/v1', unit='SI', frame='world'),
        dict(name='predicted_support_exit/v1', unit='SI', frame='world'), metadata=metadata), run)
