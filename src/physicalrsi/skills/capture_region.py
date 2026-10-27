"""Select a catcher centre within a declared goal region and capture aperture.

The region is an axis-aligned world-space box supplied by an adapter. Projection
onto its intersection with the workspace minimizes planar offset from the
predicted crossing. An empty capture intersection is an infeasible proposal,
never permission to move an object directly or ignore the task goal.
"""
from copy import deepcopy
from pathlib import Path
import math

from PhysicalRSI_core.infra.storage import digest, file_digest
from .protocol import CallableSkill, SkillDescriptor


def _vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Expected world xyz metres')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Expected finite world coordinates')
    return list(map(float, value))


def capture_region_planning(*, aperture_margin_m=.002):
    if (isinstance(aperture_margin_m, bool) or not isinstance(aperture_margin_m, (int,float))
            or not math.isfinite(aperture_margin_m) or aperture_margin_m < 0):
        raise ValueError('Aperture margin must be finite and nonnegative')

    def run(value, context):
        plan = deepcopy(value['plan'])
        region = value.get('region')
        if region is None or plan.get('predicted_position_m') is None:
            return plan
        # Prediction-model failures must not be erased by target selection.
        if plan.get('reason') not in (None, 'outside_workspace', 'insufficient_time'):
            return plan
        predicted = _vector(plan['predicted_position_m'])
        robot = value['robot']
        current = _vector(robot['target_position_m'])
        lower = [max(a,b) for a,b in zip(_vector(region['minimum_m']),_vector(robot['workspace_min_m']))]
        upper = [min(a,b) for a,b in zip(_vector(region['maximum_m']),_vector(robot['workspace_max_m']))]
        radius = region['capture_radius_m']
        speed, latency = robot['max_speed_m_s'], robot.get('command_latency_s',0.)
        for x in (radius,speed,latency):
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
                raise ValueError('Capture and motion parameters must be finite')
        if radius <= 0 or speed <= 0 or latency < 0:
            raise ValueError('Invalid capture or motion parameters')
        reason = None
        target = list(predicted)
        if any(a>b for a,b in zip(lower,upper)) or not lower[2] <= predicted[2] <= upper[2]:
            reason = 'empty_goal_workspace_intersection'
        else:
            target[:2] = [min(upper[i], max(lower[i],predicted[i])) for i in range(2)]
            if math.dist(target[:2],predicted[:2]) >= radius-aperture_margin_m:
                reason = 'capture_aperture_misses_goal_region'
        travel = math.dist(current,target)/speed+latency
        if travel > plan['time_to_intercept_s']:
            reason = reason or 'insufficient_time'
        plan.update(target_position_m=target if reason is None else None,
                    feasible=reason is None, reason=reason, required_motion_time_s=travel,
                    capture_offset_m=math.dist(target[:2],predicted[:2]))
        return plan

    metadata = dict(parameters=dict(aperture_margin_m=aperture_margin_m),
                    observation='predicted crossing, static goal region, capture aperture, reachability; no task or seed')
    revision = digest(dict(source=file_digest(Path(__file__)),metadata=metadata))
    return CallableSkill(SkillDescriptor('capture_region_planning',revision,
        dict(name='capture_region_request/v1',unit='SI',frame='world'),
        dict(name='dynamic_target/v1',unit='SI',frame='world'),metadata=metadata),run)
