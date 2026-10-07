from pathlib import Path

from PhysicalRSI_baselines.embodied_goodharts_law import active_bread_view_measured
from test_egl_active_bread_view import Robot, MEMORY, BOUNDS


def run(robot):
    scope = {'merge_visible': lambda objects, item: objects.append(item)}
    exec(Path(active_bread_view_measured.__file__).read_text(), scope)
    pieces, weak = [], []
    return scope['explore_bread_views'](robot, MEMORY, BOUNDS, pieces, weak), pieces


def test_incomplete_view_motion_still_reads_actual_camera_without_reached_claim():
    robot = Robot(motion_error=.023)
    reply, pieces = run(robot)
    assert len(pieces) == 2
    assert robot.queries == [('bread', 'left_camera'), ('bread', 'right_camera')]
    assert all(not row['requested_view_reached'] for row in reply['attempts'])
    assert robot.homes == ['left', 'right']
    assert not reply['task_success_claimed']


def test_bad_return_still_stops_before_next_view():
    robot = Robot(motion_error=.023, home_error=.1)
    reply, _ = run(robot)
    assert reply['status'] == 'scan_return_failed'
    assert len(robot.moves) == 1 and robot.homes == ['left']


def test_empty_current_view_is_not_fabricated_into_detection():
    robot = Robot(motion_error=.023)
    robot.locate_objects = lambda prompt, camera: {'objects': []}
    reply, pieces = run(robot)
    assert reply['status'] == 'scan_completed' and not pieces
    assert all(row['accepted_surfaces'] == 0 for row in reply['attempts'])
