"""Observed object progress after contact actions; caller owns episode history.

Planner completion does not establish contact or object movement. This skill
compares each observed displacement with its requested target direction. It
returns a bounded recovery recommendation, never a simulator state mutation.
"""
from pathlib import Path
import math

from PhysicalRSI_core.infra.storage import digest, file_digest
from .protocol import CallableSkill, SkillDescriptor


def _vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Expected world xyz metres')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Coordinates must be finite numbers')
    return tuple(float(x) for x in value)


def contact_progress(*, minimum_progress_m=0.001, consecutive_stalls=2,
                     target_tolerance_m=0.015, max_recoveries=2, minimum_direction_cosine=0.0):
    """Recommend continue, stop, replan, recontact or abort from observed attempts.

    An attempt contains before_m, after_m, target_m and plan_success. Attempts
    must be chronological and belong to one contact episode. The caller clears
    history after recontact and supplies the number of recoveries already used.
    Progress is projected onto each attempt's requested direction, so sideways
    slip or motion away from the target does not count as successful pushing.
    An optional direction cosine rejects sustained glancing motion even when
    its small forward component passes the minimum progress threshold.
    """
    for value in (minimum_progress_m, target_tolerance_m):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Progress and target tolerances must be positive finite metres')
    for value in (consecutive_stalls, max_recoveries):
        if type(value) is not int or value < 1:
            raise ValueError('Stall and recovery bounds must be positive integers')
    if (isinstance(minimum_direction_cosine, bool) or
            not isinstance(minimum_direction_cosine, (int, float)) or
            not math.isfinite(minimum_direction_cosine) or not 0 <= minimum_direction_cosine <= 1):
        raise ValueError('Minimum direction cosine must be finite in [0, 1]')
    parameters = dict(minimum_progress_m=minimum_progress_m, consecutive_stalls=consecutive_stalls,
                      target_tolerance_m=target_tolerance_m, max_recoveries=max_recoveries,
                      minimum_direction_cosine=minimum_direction_cosine)

    def run(value, context):
        attempts = value['attempts']
        recoveries = value['recoveries_used']
        if not isinstance(attempts, list) or type(recoveries) is not int or recoveries < 0:
            raise ValueError('Expected attempt list and nonnegative recovery count')
        if not attempts:
            return dict(decision='observe', reason='no_contact_evidence', stalled_attempts=0)
        measurements = []
        for attempt in attempts:
            before, after, target = (_vector(attempt[k]) for k in ('before_m', 'after_m', 'target_m'))
            if type(attempt['plan_success']) is not bool:
                raise ValueError('plan_success must be a boolean')
            distance = math.dist(before, target)
            progress = (sum((after[i]-before[i])*(target[i]-before[i]) for i in range(3))/distance
                        if distance else 0.)
            movement = math.dist(before, after)
            alignment = min(1., max(-1., progress/movement)) if movement else 0.
            measurements.append((progress, math.dist(after, target), attempt['plan_success'], alignment))
        progress, error, planned, alignment = measurements[-1]
        stalls = 0
        for amount, remaining, success, direction in reversed(measurements):
            if (not success or remaining <= target_tolerance_m or
                    (amount >= minimum_progress_m and direction >= minimum_direction_cosine)):
                break
            stalls += 1
        if error <= target_tolerance_m:
            decision, reason = 'stop', 'target_reached'
        elif not planned:
            decision, reason = 'replan', 'planner_failed'
        elif stalls >= consecutive_stalls:
            decision, reason = ('abort', 'recovery_budget_exhausted') if recoveries >= max_recoveries else ('recontact', 'insufficient_object_progress')
        else:
            decision, reason = 'continue', 'insufficient_stall_evidence' if stalls else 'observed_progress'
        return dict(decision=decision, reason=reason, stalled_attempts=stalls,
                    projected_progress_m=progress, target_error_m=error, direction_cosine=alignment)

    metadata = dict(parameters=parameters, state='caller-owned contact history, reset after recovery',
                    observation='measured object positions and requested targets; no task or seed')
    revision = digest(dict(source=file_digest(Path(__file__)), metadata=metadata))
    return CallableSkill(SkillDescriptor('contact_progress', revision,
        dict(name='contact_attempts/v1', unit='SI', frame='world'),
        dict(name='contact_feedback/v1', unit='SI', frame='world'), metadata=metadata), run)
