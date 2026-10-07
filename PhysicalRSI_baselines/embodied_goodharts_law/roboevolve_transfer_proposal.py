"""Reviewed source edit to screen a transfer before closing the gripper."""
import ast


def add_transfer_preview(source):
    marker = "    for arm in arms:\n"
    opening = "        if not set_gripper(robot, memory, arm, 1):\n"
    if source.count(marker) != 1 or source.count(opening) != 1 or 'preview_transfer(' in source:
        raise ValueError('Expected the reviewed unmodified transfer loop')
    geometry = '''    preview_height = max(.01, geometry['upper_95'][2]-geometry['lower_05'][2])
    preview_dest = destination['surface_quantiles']
    preview_place = [preview_dest['median'][0], preview_dest['median'][1], preview_dest['upper_95'][2]+preview_height-memory['grasp_below_surface_m']+memory['release_clearance_m']+memory['task_ee_to_tcp_m']]
    preview_above = [preview_place[0], preview_place[1], preview_place[2]+memory['transfer_clearance_m']]
    transfer_previews = []
'''
    check = '''        preview = robot.preview_transfer(arm, [hover+orientation, grasp+orientation,
                                                       hover+orientation, preview_above+orientation,
                                                       preview_place+orientation, preview_above+orientation])
        transfer_previews.append(preview)
        if preview['status'] != 'sequence_planned':
            continue
'''
    result = source.replace(marker, geometry+marker).replace(opening, check+opening)
    result = result.replace("{'status': 'both_arms_planning_failed', 'task_success_claimed': False}",
                            "{'status': 'both_arms_planning_failed', 'previews': transfer_previews, 'task_success_claimed': False}")
    ast.parse(result)
    return result
