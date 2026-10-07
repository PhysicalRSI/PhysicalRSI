"""Candidate wrist-view refinement of a clipped public target observation.

Association is geometric and remains unqualified; never infer full object extent
from either view. Failed motion or ambiguous observations provide no target.
"""


def refine_clipped_target(robot, memory, located):
    if not located.get('touches_image_border', True):
        return located
    bounds = located['surface_quantiles']
    xy = [(bounds['lower_05'][i]+bounds['upper_95'][i])/2 for i in range(2)]
    state = robot.get_robot_state()['robot_cartesian_pos']
    height = max(state[2], bounds['upper_95'][2]+memory['wrist_view_clearance_m'])
    for pose in [state[:2]+[height]+state[3:7], xy+[height]+[0., 1., 0., 0.]]:
        moved = robot.try_move_to_pose(pose)
        if not moved['reached']:
            return {'status': 'target_view_motion_failed', 'semantic_identity_verified': False}
    candidates = []
    for item in robot.locate_objects(memory['target'], camera='robot0_eye_in_hand')['objects']:
        if (item.get('status') != 'estimated_surface'
                or item.get('touches_image_border', True)
                or item.get('detection_score', 0) < memory['wrist_view_minimum_score']):
            continue
        center = item['surface_quantiles']['median']
        if sum((center[i]-bounds['median'][i])**2 for i in range(2)) > memory['wrist_view_association_xy_m']**2:
            continue
        if abs(center[2]-bounds['median'][2]) > memory['wrist_view_association_z_m']:
            continue
        candidates.append(item)
    if len(candidates) != 1:
        return {'status': 'target_view_ambiguous' if candidates else 'target_view_not_located',
                'semantic_identity_verified': False}
    return dict(candidates[0], semantic_identity_verified=False,
                view_refinement={'source_camera': memory['camera'],
                                 'camera': 'robot0_eye_in_hand',
                                 'association_verified': False,
                                 'full_object_extent_known': False})
