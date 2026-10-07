"""Bounded robot-only Cartesian planning over the native joint policy interface."""
import copy
import math
import numpy as np

from .roboevolve_primitives import RoboEvolveJointPrimitives
from .roboevolve_proprioception import measured_robot_state


def bounded_joint_waypoint_indices(positions, *, maximum_arc=.02):
    """Retain endpoints with bounded accumulated L-infinity joint arc per edge.

    This bounds skipped joint motion, including reversals; it does not certify
    collision clearance or dynamic tracking. Reject coarse input rather than
    inventing intermediate planner states.
    """
    values = np.asarray(positions, dtype=float)
    if (values.ndim != 2 or not 1 <= len(values) <= 4096
            or not np.isfinite(values).all() or not 0 < maximum_arc <= .03):
        raise ValueError('Invalid trajectory sampling input')
    indices = [0]
    arc = 0.
    for i in range(1, len(values)):
        step = float(np.max(np.abs(values[i]-values[i-1])))
        if step > maximum_arc:
            raise ValueError('Planner joint increment exceeds sampling bound')
        if arc+step > maximum_arc:
            if indices[-1] != i-1:
                indices.append(i-1)
            arc = step
        else:
            arc += step
    if indices[-1] != len(values)-1:
        indices.append(len(values)-1)
    return indices


class RoboEvolvePosePrimitives(RoboEvolveJointPrimitives):
    def __init__(self, environment, *, planners, joint_limits, max_control_actions,
                 action_applier=None, max_waypoints=140, settle_attempts=60):
        super().__init__(environment, joint_limits=joint_limits,
                         max_control_actions=max_control_actions, action_applier=action_applier)
        if (not planners or set(planners) - set(self._arms)
                or any(p.world_config is not None for p in planners.values())):
            raise ValueError('Declare robot-only planners with no scene collision map')
        if (type(max_waypoints) is not int or not 1 <= max_waypoints <= 200
                or type(settle_attempts) is not int or not 0 <= settle_attempts <= 100):
            raise ValueError('Invalid trajectory budget')
        self._planners = dict(planners)
        self._max_waypoints = max_waypoints
        self._settle_attempts = settle_attempts
        self._planning_trace = []

    @property
    def planning_trace(self):
        return copy.deepcopy(self._planning_trace)

    @property
    def handlers(self):
        handlers = super().handlers
        control = handlers['apply_joint_action']

        def move(args, kwargs, *, deadline):
            self._deadline(deadline)
            if len(args) != 2 or kwargs or args[0] not in self._planners:
                raise ValueError('Expected configured arm and world-frame xyz+wxyz pose')
            arm, raw_pose = args
            target = np.asarray(raw_pose, dtype=float)
            if (target.shape != (7,) or not np.isfinite(target).all()
                    or abs(np.linalg.norm(target[3:])-1) > 1e-5):
                raise ValueError('Invalid Cartesian target')
            robot = self._environment.robot
            state = measured_robot_state(robot)
            planner = self._planners[arm]
            if planner.world_config is not None:
                raise ValueError('Planner acquired a scene collision map')
            plan = planner.plan_path(state['arms'][arm]['measured_joint_positions'],
                                     robot._trans_from_gripper_to_endlink(target.copy(), arm),
                                     arms_tag=arm, max_attempts=2)
            self._deadline(deadline)
            diagnostic = {'arm': arm, 'target_pose': target.tolist(),
                          'planner_status': plan['status'],
                          'robot_planner_statuses': plan.get('robot_planner_statuses'),
                          'control_attempts_before': len(self._trace)}
            self._planning_trace.append(diagnostic)
            if plan['status'] != 'Success':
                return {'reached': False, 'status': 'planning_failed',
                        'task_success_claimed': False, 'scene_collision_checked': False}
            positions = np.asarray(plan['position'], dtype=float)
            limits = self._limits[arm]
            diagnostic.update(shape=list(positions.shape), maximum_waypoints=self._max_waypoints,
                              finite=bool(np.isfinite(positions).all()))
            if positions.ndim != 2 or positions.shape[1] != len(limits) or len(positions) < 1:
                diagnostic['rejection'] = 'invalid_shape'
                raise ValueError('Invalid planned trajectory shape: '+str(positions.shape))
            if not np.isfinite(positions).all():
                diagnostic['rejection'] = 'nonfinite_joints'
                raise ValueError('Nonfinite planned joint positions')
            violation = float(max(0., np.max(limits[:, 0]-positions), np.max(positions-limits[:, 1])))
            diagnostic['maximum_joint_limit_violation_rad'] = violation
            if violation > 0:
                diagnostic['rejection'] = 'joint_limits'
                raise ValueError('Planned joints exceed limits by '+str(violation)+' radians')
            if len(positions) > self._max_waypoints:
                indices = bounded_joint_waypoint_indices(positions)
                diagnostic.update(original_waypoints=len(positions), sampled_waypoints=len(indices),
                                  retained_indices=indices, maximum_skipped_joint_arc_rad=.02)
                if len(indices) > self._max_waypoints:
                    diagnostic['rejection'] = 'waypoint_budget'
                    raise ValueError('Sampled trajectory has '+str(len(indices))+' waypoints; budget is '+str(self._max_waypoints))
                positions = positions[indices]
            if len(positions) + self._settle_attempts > self._budget - len(self._trace):
                raise RuntimeError('Insufficient control budget for planned trajectory')
            reached = False
            for index in range(len(positions) + self._settle_attempts):
                joints = positions[min(index, len(positions)-1)]
                action = []
                for side in self._arms:
                    action.extend(joints.tolist() if side == arm else state['arms'][side]['measured_joint_positions'])
                    action.append(state['arms'][side]['gripper_command_fraction'])
                reply = control([action], {}, deadline=deadline)
                if not reply['control_applied']:
                    return {'reached': False, 'status': 'control_rejected',
                            'task_success_claimed': False, 'scene_collision_checked': False}
                measured = np.asarray(reply['robot_state']['arms'][arm]['measured_end_pose'])
                distance = float(np.linalg.norm(measured[:3]-target[:3]))
                angle = float(2*math.acos(np.clip(abs(np.dot(measured[3:]/np.linalg.norm(measured[3:]),
                                                            target[3:]/np.linalg.norm(target[3:]))), 0, 1)))
                if index >= len(positions)-1 and distance < .003 and angle < .03:
                    reached = True
                    break
            return {'reached': reached, 'status': 'reached' if reached else 'tracking_failed',
                    'measured_pose': measured.tolist(), 'position_error_m': distance,
                    'orientation_error_rad': angle, 'control_attempts': index+1,
                    'task_success_claimed': False, 'scene_collision_checked': False}

        handlers['move_to_pose'] = move
        return handlers
