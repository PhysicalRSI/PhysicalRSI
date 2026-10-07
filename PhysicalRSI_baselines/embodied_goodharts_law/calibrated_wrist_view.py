"""Candidate observation poses from public camera and measured hand calibration.

IK preview is robot-only and does not certify collision clearance. Target
association remains an estimate; no simulator object state is queried.
"""
import math


def wrist_rotation(q):
    w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]


def calibrated_view_poses(hand, camera, target, distance, tilts):
    rotation = wrist_rotation(hand[3:7])
    offset = [sum(rotation[j][i]*(camera[j][3]-hand[j]) for j in range(3)) for i in range(3)]
    optical = [sum(rotation[j][i]*camera[j][2] for j in range(3)) for i in range(3)]
    poses = []
    for tilt in tilts:
        quaternion = [0., math.cos(tilt/2), 0., -math.sin(tilt/2)]
        desired = wrist_rotation(quaternion)
        direction = [sum(desired[i][j]*optical[j] for j in range(3)) for i in range(3)]
        camera_offset = [sum(desired[i][j]*offset[j] for j in range(3)) for i in range(3)]
        position = [target[i]-distance*direction[i]-camera_offset[i] for i in range(3)]
        poses.append(position+quaternion)
    return poses


def refine_clipped_target(robot, memory, located):
    if not located.get('touches_image_border', True):
        return located
    initial = robot.locate_objects('bowl', camera='robot0_eye_in_hand')
    calibration = initial['public_camera_calibration']['pose_mat']
    hand = robot.get_robot_state()['robot_cartesian_pos']
    anchor = located['surface_quantiles']['median']
    poses = calibrated_view_poses(hand, calibration, anchor,
                                 memory['wrist_observation_distance_m'],
                                 memory['wrist_observation_tilts_rad'])
    attempted = []
    for pose in poses:
        preview = robot.preview_poses([pose])
        if not preview['kinematically_solved']:
            continue
        moved = robot.try_move_to_pose(pose)
        attempted.append({'pose': pose, 'motion': moved})
        for prompt in memory['wrist_observation_prompts']:
            reply = robot.locate_objects(prompt, camera='robot0_eye_in_hand')
            eligible = []
            for item in reply['objects']:
                if (item.get('status') != 'estimated_surface'
                        or item.get('touches_image_border', True)
                        or item.get('detection_score', 0) < memory['wrist_view_minimum_score']):
                    continue
                center = item['surface_quantiles']['median']
                if sum((center[i]-anchor[i])**2 for i in range(2)) > memory['wrist_view_association_xy_m']**2:
                    continue
                if abs(center[2]-anchor[2]) > memory['wrist_view_association_z_m']:
                    continue
                eligible.append(item)
            if len(eligible) > 1:
                return {'status': 'target_view_ambiguous', 'semantic_identity_verified': False}
            if eligible:
                return dict(eligible[0], semantic_identity_verified=False,
                            view_refinement={'camera': 'robot0_eye_in_hand',
                                             'association_verified': False,
                                             'attempts': attempted})
    return {'status': 'target_view_not_located', 'attempts': attempted,
            'semantic_identity_verified': False}
