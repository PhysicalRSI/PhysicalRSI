"""CAP source fragment for a bounded, public basket-relative wrist-camera scan.

Compose with the reviewed merge_visible helper. View positions are exploration
memory, not known object positions; successful motion does not prove visibility.
"""


def explore_bread_views(robot, memory, bounds, pieces, weak_pieces):
    center = [(bounds['lower_05'][i]+bounds['upper_95'][i])/2 for i in range(2)]
    attempts = []
    for arm, sign in [('left', -1), ('right', 1)]:
        state = robot.get_proprioception()['endpose'][arm]
        q = 2**-.5
        orientations = [[q, 0., q, 0.], [0., -q, 0., q]]
        orientation = max(orientations, key=lambda v: abs(sum(v[i]*state[i+3] for i in range(4))))
        pose = [center[0]+sign*memory['active_scan_lateral_m'],
                center[1]+memory['active_scan_forward_m'],
                bounds['upper_95'][2]+memory['active_scan_height_m']]+orientation
        moved = robot.move_to_pose(arm, pose)
        attempt = {'arm': arm, 'pose': pose, 'motion': moved, 'accepted_surfaces': 0}
        attempts.append(attempt)
        if (moved['position_error'] < memory['position_tolerance_m']
                and moved['orientation_error_radians'] < memory['orientation_tolerance_rad']):
            for prompt in ['bread', 'toast', 'slice of bread']:
                reply = robot.locate_objects(prompt, camera=arm+'_camera')
                accepted = []
                for item in reply['objects']:
                    if item['surface_quantiles']['median'][2] > bounds['upper_95'][2]+.08:
                        continue
                    if item['detection_score'] >= .35:
                        merge_visible(weak_pieces, item)
                    if item['detection_score'] >= memory['minimum_detection_score']:
                        accepted.append(item)
                for item in accepted:
                    merge_visible(pieces, item)
                attempt['accepted_surfaces'] += len(accepted)
                if accepted:
                    break
        returned = robot.goto_home_joint_position(arm)
        attempt['home'] = returned
        if returned['joint_error'] >= .02:
            return {'status': 'scan_return_failed', 'attempts': attempts,
                    'task_success_claimed': False}
    return {'status': 'scan_completed', 'attempts': attempts,
            'task_success_claimed': False, 'semantic_identity_verified': False}
