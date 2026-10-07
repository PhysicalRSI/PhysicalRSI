"""Trusted native endpoint settling; no new IK or object-state access."""
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation


def settle_native_endpoint(api, joints, target_pose, *, deadline, max_steps=120):
    if type(max_steps) is not int or not 1 <= max_steps <= 120:
        raise ValueError('Invalid endpoint settling step bound')
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Endpoint settling deadline reached')
    target = np.asarray(target_pose, dtype=float)
    if (target.shape != (7,) or not np.isfinite(target).all() or
            abs(np.linalg.norm(target[3:])-1) > 1e-5):
        raise ValueError('Expected finite native hand xyz and unit wxyz')
    if api._deadline is not None:
        raise RuntimeError('Cannot nest native endpoint settling')
    before = api.physics_steps
    measurement = {}

    def at_goal():
        actual = np.asarray(api.get_robot_state()['robot_cartesian_pos'][:7])
        position = float(np.linalg.norm(actual[:3]-target[:3]))
        orientation = float((Rotation.from_quat(actual[[4, 5, 6, 3]]).inv() *
                             Rotation.from_quat(target[[4, 5, 6, 3]])).magnitude())
        measurement.update(position_error_m=position, orientation_error_rad=orientation)
        return position <= .001 and orientation <= .005

    api._deadline = deadline
    try:
        # Native tracking requires two consecutive post-step Cartesian checks.
        api._track_joint_target(joints, tolerance=.002, max_steps=max_steps, goal=at_goal)
        reached = True
    except RuntimeError as error:
        if str(error) != 'Joint primitive did not converge within its step allowance':
            raise
        reached = False
    finally:
        api._deadline = None
    return {'status': 'reached' if reached else 'endpoint_not_reached',
            'execution_verified': reached, 'settling_steps': api.physics_steps-before,
            **measurement, 'grasp_verified': False}
