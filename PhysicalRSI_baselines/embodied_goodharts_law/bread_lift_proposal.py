"""Build reviewed post-lift feedback candidates without executing their code."""
import ast
from pathlib import Path


_WRAPPER = '''
def surface_pick(robot, memory):
    picked = surface_pick_without_feedback(robot, memory)
    if picked.get('status') != 'lift_motion_returned':
        return picked
    before_hand = picked['descent']['measured_pose']
    after_hand = picked['lift']['measured_pose']
    if (picked['lift']['position_error'] >= memory['position_tolerance_m']
            or after_hand[2]-before_hand[2] < .05):
        return dict(picked, lift_feedback={'status':'hand_lift_not_confirmed',
                                          'grasp_verified':False})
    views = []
    for camera in ['head_camera', 'left_camera', 'right_camera']:
        observed = robot.locate_objects('bread', camera=camera)
        feedback = bread_lift_feedback(
            picked['location'], observed['objects'], after_hand,
            minimum_score=memory.get('lift_feedback_minimum_score', .5),
            stationary_radius_m=memory.get('lift_feedback_stationary_radius_m', .02),
            minimum_rise_m=memory.get('lift_feedback_minimum_rise_m', .05),
            association_radius_m=memory.get('lift_feedback_association_radius_m', .06))
        views.append(dict(camera=camera, feedback=feedback))
    stationary = sum(v['feedback']['status']=='stationary_surface_observed' for v in views)
    raised = sum(v['feedback']['status']=='raised_surface_near_hand' for v in views)
    result = dict(picked, lift_feedback={'views':views, 'grasp_verified':False,
                                        'semantic_identity_verified':False})
    if memory.get('lift_feedback_gate', True) and stationary >= 2 and raised == 0:
        result['status'] = 'target_remained_at_pick_location'
    return result
'''


def add_bread_lift_feedback(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    found = [n for n in functions if n.name == 'surface_pick']
    if len(found) != 1 or any(n.name in ['surface_pick_without_feedback', 'bread_lift_feedback']
                              for n in functions):
        raise ValueError('Expected an unwrapped surface_pick primitive')
    lines = source.splitlines(keepends=True)
    index = found[0].lineno-1
    if not lines[index].startswith('def surface_pick(robot, memory):'):
        raise ValueError('Unexpected primitive signature')
    lines[index] = lines[index].replace('def surface_pick(', 'def surface_pick_without_feedback(', 1)
    helper = Path(__file__).with_name('bread_lift_feedback.py').read_text()
    result = ''.join(lines) + '\n' + helper + '\n' + _WRAPPER
    ast.parse(result)
    return result
