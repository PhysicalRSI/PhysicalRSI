"""Robot-only predicted joint paths for choosing an arm before grasping."""
import numpy as np

from .roboevolve_pose_primitives import RoboEvolvePosePrimitives, bounded_joint_waypoint_indices
from .roboevolve_proprioception import measured_robot_state


class RoboEvolveTransferPreview(RoboEvolvePosePrimitives):
    @property
    def handlers(self):
        handlers = super().handlers

        def preview(args, kwargs, *, deadline):
            self._deadline(deadline)
            if len(args) != 2 or kwargs or args[0] not in self._planners:
                raise ValueError('Expected configured arm and a bounded world-pose sequence')
            arm, raw = args
            poses = np.asarray(raw, dtype=float)
            if (poses.ndim != 2 or poses.shape[1] != 7 or not 1 <= len(poses) <= 8
                    or not np.isfinite(poses).all()
                    or np.any(np.abs(np.linalg.norm(poses[:, 3:], axis=1)-1) > 1e-5)):
                raise ValueError('Invalid transfer preview poses')
            planner = self._planners[arm]
            robot = self._environment.robot
            joints = measured_robot_state(robot)['arms'][arm]['measured_joint_positions']
            result = dict(status='sequence_planned', arm=arm, stages=[],
                          execution_verified=False, scene_collision_checked=False,
                          held_object_collision_checked=False, task_success_claimed=False)
            total = 0
            for index, pose in enumerate(poses):
                self._deadline(deadline)
                if planner.world_config is not None:
                    raise ValueError('Transfer preview requires a robot-only planner')
                plan = planner.plan_path(joints, robot._trans_from_gripper_to_endlink(pose.copy(), arm),
                                         arms_tag=arm, max_attempts=2)
                self._deadline(deadline)
                row = dict(index=index, target_pose=pose.tolist(), planner_status=plan['status'],
                           robot_planner_statuses=plan.get('robot_planner_statuses'))
                result['stages'].append(row)
                if plan['status'] != 'Success':
                    result.update(status='planning_failed', failed_stage=index)
                    break
                positions = np.asarray(plan['position'], dtype=float)
                limits = self._limits[arm]
                if (positions.ndim != 2 or positions.shape[1] != len(limits)
                        or not 1 <= len(positions) <= 4096 or not np.isfinite(positions).all()
                        or np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1])):
                    raise ValueError('Invalid preview joint trajectory')
                count = len(positions)
                if count > self._max_waypoints:
                    count = len(bounded_joint_waypoint_indices(positions))
                row['sampled_waypoints'] = count
                total += count + self._settle_attempts
                if count > self._max_waypoints or total > self._budget-len(self._trace):
                    result.update(status='trajectory_budget_exceeded', failed_stage=index)
                    break
                # This is a predicted endpoint, not measured execution evidence.
                joints = positions[-1].tolist()
            result['estimated_waypoint_and_settle_actions'] = total
            result['gripper_actions_included'] = False
            self._planning_trace.append(dict(kind='transfer_preview', **result))
            return result

        handlers['preview_transfer'] = preview
        return handlers
