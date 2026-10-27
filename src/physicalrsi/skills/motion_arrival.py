"""Observed position arrival for bounded free-space motion; no task or seed input."""
import math


def arrival_decision(target_m, observed_m, *, plan_success, attempts,
                     position_tolerance_m=0.015, max_attempts=3):
    if (isinstance(position_tolerance_m, bool) or
            not isinstance(position_tolerance_m, (int, float)) or
            not math.isfinite(position_tolerance_m) or position_tolerance_m <= 0):
        raise ValueError('Position tolerance must be positive finite metres')
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('Maximum attempts must be a positive integer')
    if type(attempts) is not int or attempts < 1 or type(plan_success) is not bool:
        raise ValueError('Expected positive attempt count and boolean planner status')
    for vector in (target_m, observed_m):
        if (not isinstance(vector, (list, tuple)) or len(vector) != 3 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                    not math.isfinite(x) for x in vector)):
            raise ValueError('Expected finite world-frame xyz metres')
    error = math.dist(target_m, observed_m)
    if not plan_success:
        decision, reason = 'abort', 'planner_failed'
    elif error <= position_tolerance_m:
        decision, reason = 'complete', 'observed_position_reached'
    elif attempts >= max_attempts:
        decision, reason = 'abort', 'arrival_attempt_budget_exhausted'
    else:
        decision, reason = 'retry', 'observed_position_not_reached'
    return dict(decision=decision, reason=reason, position_error_m=error, attempts=attempts)


def motion_arrival(*, position_tolerance_m=0.015, max_attempts=3):
    from pathlib import Path
    from PhysicalRSI_core.infra.storage import digest, file_digest
    from .protocol import CallableSkill, SkillDescriptor
    parameters = dict(position_tolerance_m=position_tolerance_m, max_attempts=max_attempts)
    arrival_decision([0., 0., 0.], [0., 0., 0.], plan_success=True, attempts=1, **parameters)
    metadata = dict(parameters=parameters, state='Caller owns attempts for a fixed target',
                    qualification='Position only; free-space motion, not contact completion, orientation or task success')
    revision = digest(dict(source=file_digest(Path(__file__)), metadata=metadata))
    return CallableSkill(SkillDescriptor('motion_arrival', revision,
        dict(name='observed_motion_arrival/v1', unit='SI', frame='world'),
        dict(name='motion_arrival_decision/v1', unit='SI', frame='world'), metadata=metadata),
        lambda value, context: arrival_decision(**value, **parameters))
