"""Initial CAP proposal for the public green-on-red, table-center instruction.

Visible table center and block heights are estimates, not full-object geometry.
This proposal is unqualified until native evaluation; no evaluator is a policy
capability. Motion orientation comes from measured robot proprioception.
"""

MEMORY = {'camera': 'head_camera', 'fallback_camera': 'front_camera',
          'confidence': .5, 'clearance_m': .12, 'grasp_below_surface_m': .015,
          'release_clearance_m': .01, 'minimum_visible_lift_m': .03,
          'gripper_steps': 15, 'table_confidence': .35,
          'table_maximum_z_span_m': .03, 'table_xy_margin_m': .1,
          'table_block_gap_min_m': -.01, 'table_block_gap_max_m': .12,
          'task_ee_to_tcp_m': .12, 'transfer_clearance_m': .06}

SOURCE = '''def locate(robot, memory, prompt):
    for camera in [memory['camera'], memory['fallback_camera']]:
        value = robot.locate_object(prompt, camera=camera)
        if value.get('status') == 'estimated_surface' and value.get('detection_score', 0) >= memory['confidence']:
            return value
    return None

def locate_table(robot, memory, block):
    anchor = block['surface_quantiles']
    for camera in [memory['camera'], memory['fallback_camera']]:
        for prompt in ['table', 'tabletop', 'work surface']:
            reply = robot.locate_objects(prompt, camera=camera)
            eligible = []
            for value in reply.get('objects', []):
                if value.get('status') != 'estimated_surface' or value.get('detection_score', 0) < memory['table_confidence']:
                    continue
                bounds = value['surface_quantiles']
                gap = anchor['lower_05'][2] - bounds['upper_95'][2]
                flat = 0 <= bounds['upper_95'][2] - bounds['lower_05'][2] <= memory['table_maximum_z_span_m']
                under = all(bounds['lower_05'][i] - memory['table_xy_margin_m'] <= anchor['median'][i] <= bounds['upper_95'][i] + memory['table_xy_margin_m'] for i in range(2))
                if flat and under and memory['table_block_gap_min_m'] <= gap <= memory['table_block_gap_max_m']:
                    eligible.append(value)
            if len(eligible) > 1:
                return None
            if len(eligible) == 1:
                return eligible[0]
    return None

def set_gripper(robot, memory, arm, opening):
    state = robot.get_robot_state()
    action = []
    for side in ['left', 'right']:
        action.extend(state['arms'][side]['measured_joint_positions'])
        action.append(opening if side == arm else state['arms'][side]['gripper_command_fraction'])
    for step in range(memory['gripper_steps']):
        reply = robot.apply_joint_action(action)
        if not reply['control_applied']:
            return False
    return True

def transfer(robot, memory, prompt, destination):
    item = locate(robot, memory, prompt)
    if item is None:
        return {'status': 'target_not_located', 'task_success_claimed': False}
    geometry = item['surface_quantiles']
    center = geometry['median']
    state = robot.get_robot_state()
    arms = sorted(['left', 'right'], key=lambda side: sum((state['arms'][side]['measured_end_pose'][i]-center[i])**2 for i in range(2)))
    q = 2**-.5
    # Public robot convention: TCP is +X from task EE; both poses aim +X down.
    orientations = [[q, 0., q, 0.], [0., -q, 0., q]]
    grasp = [center[0], center[1], geometry['upper_95'][2]-memory['grasp_below_surface_m']+memory['task_ee_to_tcp_m']]
    hover = [grasp[0], grasp[1], grasp[2]+memory['clearance_m']]
    for arm in arms:
        measured_orientation = state['arms'][arm]['measured_end_pose'][3:]
        orientation = max(orientations, key=lambda value: abs(sum(value[i]*measured_orientation[i] for i in range(4))))
        if not set_gripper(robot, memory, arm, 1):
            return {'status': 'open_failed', 'task_success_claimed': False}
        approach = robot.move_to_pose(arm, hover+orientation)
        if approach['reached']:
            break
        if approach.get('status') != 'planning_failed':
            return {'status': 'pick_approach_failed', 'task_success_claimed': False}
    else:
        return {'status': 'both_arms_planning_failed', 'task_success_claimed': False}
    if not robot.move_to_pose(arm, grasp+orientation)['reached']:
        return {'status': 'pick_approach_failed', 'task_success_claimed': False}
    if not set_gripper(robot, memory, arm, 0):
        return {'status': 'close_failed', 'task_success_claimed': False}
    if not robot.move_to_pose(arm, hover+orientation)['reached']:
        return {'status': 'lift_motion_failed', 'task_success_claimed': False}
    lifted = locate(robot, memory, prompt)
    if lifted is None or lifted['surface_quantiles']['median'][2]-center[2] < memory['minimum_visible_lift_m']:
        return {'status': 'visible_lift_not_verified', 'task_success_claimed': False}
    height = max(.01, geometry['upper_95'][2]-geometry['lower_05'][2])
    dest = destination['surface_quantiles']
    place = [dest['median'][0], dest['median'][1], dest['upper_95'][2]+height-memory['grasp_below_surface_m']+memory['release_clearance_m']+memory['task_ee_to_tcp_m']]
    above = [place[0], place[1], place[2]+memory['transfer_clearance_m']]
    transfer_orientations = orientations + [[.5, -.5, .5, .5], [.5, .5, .5, -.5]]
    for alternative in [orientation] + [value for value in transfer_orientations if value != orientation]:
        transit = robot.move_to_pose(arm, above+alternative)
        if transit['reached']:
            orientation = alternative
            break
        if transit.get('status') != 'planning_failed':
            return {'status': 'placement_motion_failed', 'task_success_claimed': False}
    else:
        return {'status': 'placement_planning_failed', 'task_success_claimed': False}
    if not robot.move_to_pose(arm, place+orientation)['reached']:
        return {'status': 'placement_motion_failed', 'task_success_claimed': False}
    if not set_gripper(robot, memory, arm, 1):
        return {'status': 'release_failed', 'task_success_claimed': False}
    retreat = robot.move_to_pose(arm, above+orientation)
    return {'status': 'transfer_completed' if retreat['reached'] else 'retreat_failed',
            'task_success_claimed': False}

def policy(robot, memory):
    anchor = locate(robot, memory, 'red block')
    if anchor is None:
        return {'status': 'red_anchor_not_located', 'task_success_claimed': False}
    table = locate_table(robot, memory, anchor)
    if table is None:
        return {'status': 'table_not_located', 'task_success_claimed': False}
    first = transfer(robot, memory, 'red block', table)
    if first['status'] != 'transfer_completed':
        return first
    red = locate(robot, memory, 'red block')
    if red is None:
        return {'status': 'base_block_not_reacquired', 'task_success_claimed': False}
    second = transfer(robot, memory, 'green block', red)
    return {'status': 'stack_sequence_completed' if second['status'] == 'transfer_completed' else second['status'],
            'task_success_claimed': False}
'''
