from pathlib import Path

from PhysicalRSI_baselines.embodied_goodharts_law import active_bread_view


MEMORY = {'active_scan_lateral_m': .25, 'active_scan_forward_m': .2,
          'active_scan_height_m': .3, 'position_tolerance_m': .015,
          'orientation_tolerance_rad': .12, 'minimum_detection_score': .5}
BOUNDS = {'lower_05': [-.1, -.3, .7], 'upper_95': [.1, -.1, .8]}


class Robot:
    def __init__(self, motion_error=0, home_error=0):
        self.moves, self.homes, self.queries = [], [], []
        self.motion_error, self.home_error = motion_error, home_error

    def get_proprioception(self):
        return {'endpose': {arm: [0, 0, 1, 2**-.5, 0, 2**-.5, 0] for arm in ['left', 'right']}}

    def move_to_pose(self, arm, pose):
        self.moves.append((arm, pose))
        return {'position_error': self.motion_error, 'orientation_error_radians': 0}

    def locate_objects(self, prompt, *, camera):
        self.queries.append((prompt, camera))
        return {'objects': [{'detection_score': .8, 'surface_quantiles': {'median': [0, 0, .78]}}]}

    def goto_home_joint_position(self, arm):
        self.homes.append(arm)
        return {'joint_error': self.home_error}


def run(robot):
    scope = {'merge_visible': lambda objects, item: objects.append(item)}
    exec(Path(active_bread_view.__file__).read_text(), scope)
    pieces, weak = [], []
    return scope['explore_bread_views'](robot, MEMORY, BOUNDS, pieces, weak), pieces


def test_viewpoints_are_relative_to_public_basket_and_return_home():
    robot = Robot()
    reply, pieces = run(robot)
    assert reply['status'] == 'scan_completed' and len(pieces) == 2
    assert robot.moves[0][1][:3] == [-.25, 0., 1.1]
    assert robot.moves[1][1][:3] == [.25, 0., 1.1]
    assert robot.homes == ['left', 'right']
    assert robot.queries == [('bread', 'left_camera'), ('bread', 'right_camera')]
    assert not reply['task_success_claimed']


def test_failed_viewpoint_is_not_used_for_perception():
    robot = Robot(motion_error=.1)
    reply, pieces = run(robot)
    assert reply['status'] == 'scan_completed' and not pieces
    assert not robot.queries and robot.homes == ['left', 'right']


def test_failed_return_stops_before_second_arm_view():
    robot = Robot(home_error=.1)
    reply, _ = run(robot)
    assert reply['status'] == 'scan_return_failed'
    assert len(robot.moves) == 1
