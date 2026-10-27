"""Retrospective guide-exit identification from observed positions only."""
import math
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest
from .protocol import CallableSkill, SkillDescriptor


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite numeric input required')
    return float(value)


def _vec(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('World xyz vector required')
    return [_number(x) for x in value]


def exit_transition(*, max_residual_m=0.001):
    """Bound along-tangent exit displacement over an observed phase-time gap.

    Caller supplies the last guided observation and the first 3–4 flight
    observations, plus a static guide endpoint/tangent and positive gravity.
    Fit constant-gravity flight, then back-project over the unresolved exit
    time. Ranges are conditional on that model, NOT confidence intervals.
    Low residual does not prove absence of contact or an unobserved impulse.
    This is retrospective evidence for memory, not a pre-exit observation.
    """
    tolerance = _number(max_residual_m)
    if tolerance <= 0:
        raise ValueError('Positive residual tolerance required')

    def run(value, context):
        before = value['last_guided_sample']
        last_t = _number(before['timestamp_s'])
        _vec(before['position_m'])
        if last_t < 0:
            raise ValueError('Nonnegative timestamps required')
        samples = value['flight_samples']
        if len(samples) < 3:
            return dict(valid=False, reason='insufficient_transition_samples')
        if len(samples) > 4:
            raise ValueError('Use only the first 3–4 flight samples')
        times = [_number(s['timestamp_s']) for s in samples]
        if times[0] <= last_t or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Strictly increasing transition timestamps required')
        positions = [_vec(s['position_m']) for s in samples]
        geometry = value['geometry']
        end = _vec(geometry['exit_position_m'])
        axis = _vec(geometry['exit_tangent'])
        length = math.sqrt(sum(x*x for x in axis))
        if length <= 1e-9:
            raise ValueError('Nonzero exit tangent required')
        axis = [x/length for x in axis]
        gravity = _number(geometry['gravity_m_s2'])
        if gravity <= 0:
            raise ValueError('Positive gravity magnitude required')
        ts = [t-times[0] for t in times]
        mean = sum(ts)/len(ts)
        denominator = sum((t-mean)**2 for t in ts)
        if denominator <= 1e-18:
            return dict(valid=False, reason='ill_conditioned_history')
        coords = [[p[i]+(0.5*gravity*t*t if i == 2 else 0.)
                   for t, p in zip(ts, positions)] for i in range(3)]
        velocity = [sum((t-mean)*x for t, x in zip(ts, c))/denominator for c in coords]
        origin = [sum(c)/len(c)-velocity[i]*mean for i, c in enumerate(coords)]
        residual = math.sqrt(sum(sum((c[j]-origin[i]-velocity[i]*ts[j])**2
            for i, c in enumerate(coords)) for j in range(len(ts)))/len(ts))
        if residual > tolerance:
            return dict(valid=False, reason='nonballistic_or_contact_contaminated',
                        residual_rms_m=residual)
        gap = times[0]-last_t
        along = sum((origin[i]-end[i])*axis[i] for i in range(3))
        speed = sum(velocity[i]*axis[i] for i in range(3))
        taus = [0., gap]
        if abs(gravity*axis[2]) > 1e-12:
            stationary = -speed/(gravity*axis[2])
            if 0 < stationary < gap:
                taus.append(stationary)
        offsets = [along-speed*t-0.5*gravity*axis[2]*t*t for t in taus]
        return dict(valid=True, reason=None, residual_rms_m=residual,
                    position_offset_range_m=[min(offsets), max(offsets)],
                    transition_time_range_s=[last_t, times[0]],
                    first_flight_velocity_m_s=velocity,
                    uncertainty='Timing range conditional on unperturbed constant-gravity flight; not a confidence interval',
                    available_after_s=times[-1])

    metadata = dict(parameters=dict(max_residual_m=tolerance),
                    purpose='Retrospective transition evidence; caller manages episode history',
                    valid_means='Ballistic residual passes; contact-free dynamics remain an assumption')
    revision = digest(dict(source=file_digest(Path(__file__)), metadata=metadata))
    return CallableSkill(SkillDescriptor('exit_transition', revision,
        dict(name='observed_exit_transition/v1', unit='SI', frame='world'),
        dict(name='exit_transition_range/v1', unit='SI', frame='world'), metadata=metadata), run)
