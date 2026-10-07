import copy

from PhysicalRSI_baselines.embodied_goodharts_law.clipped_target_view import refine_clipped_target
from test_egl_support_reference import surface

MEMORY = {'camera': 'agentview', 'target': 'black bowl', 'wrist_view_clearance_m': .16,
          'wrist_view_minimum_score': .3, 'wrist_view_association_xy_m': .12,
          'wrist_view_association_z_m': .08}


class Robot:
    def __init__(self, candidates, reached=True):
        self.candidates, self.reached = candidates, reached
        self.moves, self.queries = [], []

    def get_robot_state(self):
        return {'robot_cartesian_pos': [.4, 0, .35, 0, 1, 0, 0, 1]}

    def try_move_to_pose(self, pose):
        self.moves.append(pose)
        return {'reached': self.reached}

    def locate_objects(self, prompt, *, camera):
        self.queries.append((prompt, camera))
        return {'objects': self.candidates}


def test_refinement_uses_unique_associated_current_wrist_surface():
    original = surface(.6, .25, clipped=True)
    target = surface(.62, .26)
    robot = Robot([surface(.9, .26), target])
    before = copy.deepcopy(original)
    result = refine_clipped_target(robot, MEMORY, original)
    assert result['surface_quantiles'] == target['surface_quantiles']
    assert not result['view_refinement']['association_verified']
    assert original == before
    assert robot.queries == [('black bowl', 'robot0_eye_in_hand')]


def test_unclipped_original_skips_motion():
    target = surface(.6, .25)
    robot = Robot([])
    assert refine_clipped_target(robot, MEMORY, target) is target
    assert not robot.moves


def test_failed_motion_stops_before_perception_or_grasp():
    robot = Robot([], reached=False)
    assert refine_clipped_target(robot, MEMORY, surface(.6, .25, clipped=True))['status'] == 'target_view_motion_failed'
    assert len(robot.moves) == 1 and not robot.queries


def test_ambiguous_clipped_and_vertical_distractors_are_rejected():
    for candidates, expected in [([surface(.6, .25), surface(.61, .25)], 'target_view_ambiguous'),
                                 ([surface(.6, .25, clipped=True), surface(.6, .5)], 'target_view_not_located')]:
        assert refine_clipped_target(Robot(candidates), MEMORY, surface(.6, .25, clipped=True))['status'] == expected
