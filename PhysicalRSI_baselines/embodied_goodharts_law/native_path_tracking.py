"""Bounded native path tracking with separate intermediate and final criteria.

Intermediate joint proximity is a tracking heuristic, not a collision proof.
The native adapter still owns action limits, physics budget and observation.
"""
import math
import time

import numpy as np


def track_native_path(api, plan, native_target, *, deadline,
                      intermediate_tolerance_rad=.03, max_steps=600):
    if api._deadline is not None:
        raise RuntimeError('Cannot nest native trajectory tracking')
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise TimeoutError('Native trajectory deadline reached')
    if (type(max_steps) is not int or not 1 <= max_steps <= 1000
            or not math.isfinite(intermediate_tolerance_rad)
            or not .01 <= intermediate_tolerance_rad <= .05):
        raise ValueError('Invalid native trajectory tracking limits')
    path = np.asarray(plan.get('trajectory'), dtype=float)
    goal = np.asarray(native_target, dtype=float)
    if (plan.get('status') != 'planned' or path.ndim != 2
            or path.shape[1] != 7 or not 1 <= len(path) <= 200
            or not np.isfinite(path).all() or goal.shape != (7,)
            or not np.isfinite(goal).all()
            or abs(np.linalg.norm(goal[3:])-1) > 1e-5):
        raise ValueError('Expected bounded joint path and unit native hand pose')
    limits = np.asarray(api._joint_limits)
    if limits.shape != (7, 2) or np.any(path < limits[:, 0]) or np.any(path > limits[:, 1]):
        raise ValueError('Path exceeds native joint limits')
    def joints():
        q = np.asarray(api.get_robot_state()['robot_joint_pos'][:7], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Invalid measured joints')
        return q
    if np.linalg.norm(joints()-path[0]) > .01:
        return {'status': 'stale_plan_start', 'execution_verified': False}
    before = api.physics_steps
    measurement = {}
    reached = False
    index = 0
    def at_goal():
        actual = np.asarray(api.get_robot_state()['robot_cartesian_pos'][:7], dtype=float)
        if (actual.shape != (7,) or not np.isfinite(actual).all()
                or abs(np.linalg.norm(actual[3:])-1) > 1e-5):
            raise ValueError('Invalid measured native hand pose')
        position = float(np.linalg.norm(actual[:3]-goal[:3]))
        angle = float(2*np.arccos(np.clip(abs(np.dot(actual[3:], goal[3:])), 0, 1)))
        measurement.update(position_error_m=position, orientation_error_rad=angle)
        return position <= .001 and angle <= .005
    api._deadline = deadline
    try:
        for index, target in enumerate(path[1:], 1):
            while np.linalg.norm(target-joints()) > intermediate_tolerance_rad:
                if api.physics_steps-before >= max_steps:
                    break
                delta = np.clip((target-joints())/.05, -1, 1)
                api._step(np.r_[delta, 1-2*api._gripper_fraction])
            if api.physics_steps-before >= max_steps:
                break
        remaining = max_steps-(api.physics_steps-before)
        if remaining:
            try:
                api._track_joint_target(path[-1], tolerance=.002,
                                        max_steps=min(120, remaining), goal=at_goal)
                reached = True
            except RuntimeError as error:
                if str(error) != 'Joint primitive did not converge within its step allowance':
                    raise
        at_goal()
    finally:
        api._deadline = None
    return {'status': 'reached' if reached else 'tracking_limit_reached',
            'execution_verified': reached, 'tracking_steps': api.physics_steps-before,
            'last_intermediate_index': index, **measurement,
            'intermediate_tolerance_rad': intermediate_tolerance_rad,
            'continuous_collision_verified': False, 'grasp_verified': False}
