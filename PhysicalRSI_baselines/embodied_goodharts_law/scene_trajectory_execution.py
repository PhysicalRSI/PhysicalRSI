"""Execute a bounded planned path through public native primitive RPCs.

The host still owns the episode physics-step and wall-clock limits. Following
waypoints is not continuous collision verification or proof of grasp success.
"""
import math


def execute_scene_trajectory(robot, plan, native_target_pose, *, max_waypoints=200,
                             waypoint_tolerance_rad=.01, subsample=1):
    if type(subsample) is not int or subsample not in (1, 2):
        raise ValueError('Expected trajectory subsample 1 or 2')
    # ASPIRE's trajectory executor defaults to .01 rad for intermediate
    # waypoints. Final Cartesian acceptance below remains independent.
    if not math.isfinite(waypoint_tolerance_rad) or not 0 < waypoint_tolerance_rad <= .01:
        raise ValueError('Invalid joint waypoint tolerance')
    if plan.get('status') != 'planned':
        return {'status': 'no_plan', 'execution_verified': False}
    path = plan.get('trajectory')
    if (type(max_waypoints) is not int or not 1 <= max_waypoints <= 200 or
            not isinstance(path, list) or not 1 <= len(path) <= max_waypoints or
            any(len(q) != 7 or not all(math.isfinite(x) for x in q) for q in path)):
        raise ValueError('Expected a bounded finite seven-joint trajectory')
    target = list(native_target_pose)
    if (len(target) != 7 or not all(math.isfinite(x) for x in target) or
            abs(sum(x*x for x in target[3:])-1) > 1e-5):
        raise ValueError('Expected native hand xyz and unit wxyz target')
    initial = robot.get_robot_state()['robot_joint_pos'][:7]
    if len(initial) != 7 or not all(math.isfinite(x) for x in initial):
        raise ValueError('Invalid measured start state')
    if math.dist(initial, path[0]) > .01:
        return {'status': 'stale_plan_start', 'execution_verified': False}
    indices = list(range(0, len(path), subsample))
    if indices[-1] != len(path)-1:
        indices.append(len(path)-1)
    for index in indices:
        joints = path[index]
        try:
            reply = robot.move_to_joints(joints, tolerance=waypoint_tolerance_rad, max_steps=120)
        except RuntimeError as error:
            if str(error) != 'Joint primitive did not converge within its step allowance':
                raise
            measured = robot.get_robot_state()
            return {'status': 'joint_tracking_failed', 'waypoint': index,
                    'target_joints': joints, 'measured_robot_state': measured,
                    'reason': str(error), 'execution_verified': False}
        if not reply.get('converged', False):
            return {'status': 'joint_tracking_failed', 'waypoint': index,
                    'execution_verified': False}
    actual = robot.get_robot_state()['robot_cartesian_pos'][:7]
    if (len(actual) != 7 or not all(math.isfinite(x) for x in actual) or
            abs(sum(x*x for x in actual[3:])-1) > 1e-5):
        raise ValueError('Invalid measured native hand pose')
    position_error = math.dist(actual[:3], target[:3])
    dot = abs(sum(a*b for a, b in zip(actual[3:], target[3:])))
    angle = 2*math.acos(min(1., dot))
    reached = position_error <= .001 and angle <= .005
    return {'status': 'reached' if reached else 'endpoint_not_reached',
            'execution_verified': reached, 'position_error_m': position_error,
            'orientation_error_rad': angle, 'waypoints': len(path),
            'executed_waypoints': len(indices), 'subsample': subsample,
            'grasp_verified': False}
