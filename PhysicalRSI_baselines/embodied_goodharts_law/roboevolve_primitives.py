"""Host-owned RoboEvolve joint controls through the official policy interface.

A supervisor must bound native calls. These callbacks check deadlines before
and after control, and never infer task success from a returned action.
"""
import copy
import math
import time

import numpy as np

from .roboevolve_proprioception import measured_robot_state


class RoboEvolveJointPrimitives:
    def __init__(self, environment, *, joint_limits, max_control_actions, action_applier=None):
        if type(max_control_actions) is not int or max_control_actions < 1:
            raise ValueError('A positive native control budget is required')
        self._environment = environment
        self._arms = tuple(str(x) for x in environment.robot.available_arms)
        if not self._arms or len(set(self._arms)) != len(self._arms) or set(self._arms)-{'left', 'right'}:
            raise ValueError('Unsupported robot arm configuration')
        if set(joint_limits) != set(self._arms):
            raise ValueError('Declare joint limits for every configured arm')
        self._limits = {}
        for arm in self._arms:
            n = len(environment.robot.get_arm_info(arm).arm_joint_names)
            limits = np.asarray(joint_limits[arm], dtype=float)
            if (n < 1 or limits.shape != (n, 2) or not np.isfinite(limits).all()
                    or np.any(limits[:, 0] >= limits[:, 1])):
                raise ValueError('Invalid declared robot joint limits')
            self._limits[arm] = limits.copy()
        if action_applier is None:
            from roboevolve.policy.xpolicylab import apply_policy_action
            action_applier = apply_policy_action
        self._apply = action_applier
        self._budget = max_control_actions
        self._trace = []

    @property
    def trace(self):
        return copy.deepcopy(self._trace)

    @staticmethod
    def _deadline(deadline):
        if not math.isfinite(deadline):
            raise ValueError('A finite control deadline is required')
        if time.monotonic() >= deadline:
            raise TimeoutError('RoboEvolve control deadline reached')

    @property
    def handlers(self):
        def state(args, kwargs, *, deadline):
            self._deadline(deadline)
            if args or kwargs:
                raise ValueError('Robot state takes no arguments')
            result = measured_robot_state(self._environment.robot)
            self._deadline(deadline)
            return result

        def control(args, kwargs, *, deadline):
            self._deadline(deadline)
            if len(args) != 1 or kwargs:
                raise ValueError('Expected one complete packed joint action')
            values = np.asarray(args[0], dtype=float)
            expected = sum(len(self._limits[a])+1 for a in self._arms)
            if values.shape != (expected,) or not np.isfinite(values).all():
                raise ValueError('Invalid packed joint action')
            offset = 0
            for arm in self._arms:
                limits = self._limits[arm]
                joints = values[offset:offset+len(limits)]
                gripper = values[offset+len(limits)]
                if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
                    raise ValueError('Joint action exceeds declared limits')
                if not 0 <= gripper <= 1:
                    raise ValueError('Gripper action must be normalized')
                offset += len(limits)+1
            if len(self._trace) >= self._budget:
                raise RuntimeError('Native control-action budget exhausted')
            row = {'attempt': len(self._trace)+1, 'action_type': 'joint',
                   'action': values.tolist(), 'state': 'started'}
            self._trace.append(row)
            try:
                applied = bool(self._apply(self._environment, values.tolist(), 'joint', render=True))
                row.update(state='returned', applied=applied)
                self._deadline(deadline)
                measured = measured_robot_state(self._environment.robot)
                self._deadline(deadline)
            except BaseException:
                row['state'] = 'raised'
                raise
            return {'control_applied': applied, 'control_attempt': row['attempt'],
                    'robot_state': measured, 'convergence_verified': False,
                    'task_success_claimed': False}

        return {'get_robot_state': state, 'apply_joint_action': control}
