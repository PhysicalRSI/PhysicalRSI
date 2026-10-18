"""Observed-motion → static-guide prediction → reachable interception target.

All vectors use world xyz metres and timestamps use episode simulation seconds.
Inputs are plain dictionaries (see ``dynamic_interception``). The caller supplies
and persists observation history, and clears it on reset; skills retain no state.
Static guide geometry is a declared privileged input, not an oracle trajectory.
The model assumes travel along a guide followed by gravity-only free flight; it
cannot anticipate unobserved collisions or simulator-specific launch impulses.
"""
from copy import deepcopy
import math
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest
from .protocol import CallableSkill, SkillDescriptor
from .compose import compose_sequence


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _vec(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} must be world xyz')
    return tuple(_number(x, name) for x in value)


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _skill(name, before, after, function, parameters):
    metadata = dict(parameters=parameters, state='stateless; caller clears history on episode reset',
                    time='episode simulation seconds', frame='world xyz', units='metres, seconds',
                    observation='position history and explicitly declared static geometry')
    revision = digest(dict(source=file_digest(Path(__file__)), name=name, metadata=metadata))
    return CallableSkill(SkillDescriptor(name, revision,
        dict(name=before, unit='SI', frame='world'), dict(name=after, unit='SI', frame='world'),
        metadata=metadata), function)


def motion_estimation(*, window=3):
    """Least-squares velocity over strictly ordered observations; minimum two."""
    if isinstance(window, bool) or not isinstance(window, int) or not 2 <= window <= 20:
        raise ValueError('window must be an integer in [2, 20]')

    def run(value, context):
        samples = value['samples']
        phase = value['phase']
        if phase not in ('held', 'dropping', 'rolling', 'flight'):
            raise ValueError('Unknown motion phase')
        if len(samples) < 2 or phase in ('held', 'dropping'):
            result = deepcopy(value)
            result['motion'] = {'reason': 'insufficient_observed_motion'}
            return result
        times = [_number(s['timestamp_s'], 'timestamp_s') for s in samples]
        positions = [_vec(s['position_m'], 'position_m') for s in samples]
        if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Samples must have nonnegative strictly increasing timestamps')
        times, positions = times[-window:], positions[-window:]
        mean_t = sum(times)/len(times)
        denominator = sum((t-mean_t)**2 for t in times)
        velocity = [sum((t-mean_t)*p[axis] for t, p in zip(times, positions))/denominator
                    for axis in range(3)]
        mean_p = [sum(p[i] for p in positions)/len(positions) for i in range(3)]
        residual = math.sqrt(sum(sum((p[i]-mean_p[i]-velocity[i]*(t-mean_t))**2
            for i in range(3)) for t, p in zip(times, positions))/len(times))
        result = deepcopy(value)
        result['motion'] = dict(residual_rms_m=residual, position_m=list(positions[-1]), velocity_m_s=velocity,
                                timestamp_s=times[-1], sample_dt_s=times[-1]-times[-2])
        return result
    return _skill('motion_estimation', 'dynamic_observation/v1', 'dynamic_motion/v1', run, dict(window=window))


def intercept_prediction(*, velocity_gain=1.0, rolling_energy_factor=5/7, max_horizon_s=10.0):
    """Predict the descending crossing of a horizontal capture plane.

    ``geometry.path_m`` is an optional ordered static ball-centre guide. Empty
    means already in free flight. The guide model transports observed tangential
    speed using v² + 2*k*g*height_drop, and preserves observed lateral drift.
    Evolution is bounded to this prediction primitive's parameters.
    """
    gain = _number(velocity_gain, 'velocity_gain')
    factor = _number(rolling_energy_factor, 'rolling_energy_factor')
    horizon = _number(max_horizon_s, 'max_horizon_s')
    if not 0.25 <= gain <= 2 or not 0 <= factor <= 1 or not 0.05 <= horizon <= 30:
        raise ValueError('Prediction parameters outside bounded domain')

    def run(value, context):
        motion, geometry = value['motion'], value['geometry']
        if motion.get('reason'):
            result = deepcopy(value)
            result['prediction'] = dict(position_m=None, interception_time_s=None,
                time_to_intercept_s=None, feasible=False, reason=motion['reason'], residual_rms_m=None)
            return result
        p = _vec(motion['position_m'], 'position_m')
        v = tuple(gain*x for x in _vec(motion['velocity_m_s'], 'velocity_m_s'))
        g = _number(geometry.get('gravity_m_s2', 9.81), 'gravity_m_s2')
        z = _number(geometry['capture_height_m'], 'capture_height_m')
        if g <= 0:
            raise ValueError('Gravity must be positive')
        path = [_vec(x, 'path_m') for x in geometry.get('path_m', [])] if value['phase'] == 'rolling' else []
        if value['phase'] == 'rolling' and not path:
            raise ValueError('Rolling observations require static guide geometry')
        guide_time = 0.0
        reason = None
        if path:
            if len(path) < 2:
                raise ValueError('A static guide needs at least two points')
            segments = []
            for a, b in zip(path, path[1:]):
                delta = _sub(b, a)
                length = _norm(delta)
                if length <= 1e-9:
                    raise ValueError('Static guide has a zero-length segment')
                tangent = tuple(x/length for x in delta)
                u = min(length, max(0., _dot(_sub(p, a), tangent)))
                projection = tuple(a[i]+u*tangent[i] for i in range(3))
                segments.append((length, tangent, u, projection, _norm(_sub(p, projection))))
            index = min(range(len(segments)), key=lambda i: segments[i][4])
            length, tangent, u, projection, _ = segments[index]
            speed = _dot(v, tangent)
            energy = speed*speed + 2*factor*g*(projection[2]-path[-1][2])
            # Check intermediate uphill barriers, not just the exit height.
            barrier = max(point[2] for point in path[index+1:])
            if speed <= 1e-6 or speed*speed + 2*factor*g*(projection[2]-barrier) <= 0:
                reason = 'insufficient_observed_forward_motion'
            else:
                exit_speed = math.sqrt(max(0., energy))
                distance = length-u + sum(item[0] for item in segments[index+1:])
                guide_time = 2*distance/max(speed+exit_speed, 1e-9)
                lateral = tuple(v[i]-speed*tangent[i] for i in range(3))
                offset = _sub(p, projection)
                p = tuple(path[-1][i]+offset[i]+lateral[i]*guide_time for i in range(3))
                v = tuple(exit_speed*segments[-1][1][i]+lateral[i] for i in range(3))
        disc = v[2]*v[2]+2*g*(p[2]-z)
        flight_time = (v[2]+math.sqrt(max(0., disc)))/g
        if disc < 0 or flight_time <= 0:
            reason = reason or 'no_future_descending_crossing'
        duration = guide_time + max(0., flight_time)
        if duration > horizon:
            reason = reason or 'prediction_horizon_exceeded'
        predicted = [p[0]+v[0]*max(0., flight_time), p[1]+v[1]*max(0., flight_time), z]
        result = deepcopy(value)
        result['prediction'] = dict(position_m=predicted, time_to_intercept_s=duration,
            interception_time_s=motion['timestamp_s']+duration, feasible=reason is None,
            reason=reason, residual_rms_m=motion['residual_rms_m'],
            model='observed-guide-energy-ballistic/v1')
        return result
    return _skill('intercept_prediction', 'dynamic_motion/v1', 'dynamic_prediction/v1', run,
                  dict(velocity_gain=gain, rolling_energy_factor=factor, max_horizon_s=horizon))


def intercept_planning():
    """Check a target-origin workspace and a declared speed/latency envelope.

    Robot fields describe the controlled catcher target origin, not its joints.
    The adapter retains grasp, IK, contact and collision responsibility. A false
    feasible flag is a failed proposal, never permission to teleport or clamp.
    """
    def run(value, context):
        prediction, robot = value['prediction'], value['robot']
        if prediction['position_m'] is None:
            return dict(target_position_m=None, predicted_position_m=None, interception_time_s=None,
                time_to_intercept_s=None, required_motion_time_s=None, feasible=False,
                reason=prediction['reason'], residual_rms_m=prediction['residual_rms_m'])
        target = _vec(prediction['position_m'], 'predicted position')
        current = _vec(robot['target_position_m'], 'target_position_m')
        lower = _vec(robot['workspace_min_m'], 'workspace_min_m')
        upper = _vec(robot['workspace_max_m'], 'workspace_max_m')
        speed = _number(robot['max_speed_m_s'], 'max_speed_m_s')
        latency = _number(robot.get('command_latency_s', 0.), 'command_latency_s')
        if speed <= 0 or latency < 0 or any(a > b for a, b in zip(lower, upper)):
            raise ValueError('Invalid robot reachability envelope')
        travel = _norm(_sub(target, current))/speed+latency
        reason = prediction['reason']
        if any(not a <= p <= b for a, p, b in zip(lower, target, upper)):
            reason = reason or 'outside_workspace'
        if travel > prediction['time_to_intercept_s']:
            reason = reason or 'insufficient_time'
        return dict(target_position_m=list(target), predicted_position_m=list(target),
                    interception_time_s=prediction['interception_time_s'],
                    time_to_intercept_s=prediction['time_to_intercept_s'],
                    required_motion_time_s=travel, feasible=reason is None, reason=reason,
                    residual_rms_m=prediction['residual_rms_m'])
    return _skill('intercept_planning', 'dynamic_prediction/v1', 'dynamic_target/v1', run, {})


def dynamic_interception(**prediction_parameters):
    """Compose three frozen primitives; repeated calls implement target tracking.

    Input: ``phase`` is held/dropping/rolling/flight; history must be cleared
    on phase transitions, to avoid fitting a launch discontinuity. ``samples=[{timestamp_s, position_m:[x,y,z]}]``,
    ``geometry={path_m: optional xyz list, capture_height_m, gravity_m_s2:9.81}``,
    ``robot={target_position_m, workspace_min_m, workspace_max_m,
    max_speed_m_s, command_latency_s:0}``. No task identifiers, seeds or future
    launch states are accepted as modeling inputs. Output is a target proposal
    with absolute/relative interception time and an explicit feasibility reason.
    """
    return compose_sequence('dynamic_interception', motion_estimation(),
                            intercept_prediction(**prediction_parameters), intercept_planning())
