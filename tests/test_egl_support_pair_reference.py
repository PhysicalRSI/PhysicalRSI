import copy

from PhysicalRSI_baselines.embodied_goodharts_law.support_pair_reference import locate_supported_target
from test_egl_support_reference import MEMORY, Robot, surface


def test_unrelated_support_detections_do_not_hide_unique_supported_target():
    target = surface(.5, .25)
    robot = Robot([target, surface(.8, .05)], [surface(.5, .2), surface(.9, .1)])
    before = copy.deepcopy((robot.targets, robot.supports, MEMORY))
    result = locate_supported_target(robot, MEMORY)
    assert result['surface_quantiles'] == target['surface_quantiles']
    assert not result['public_relation']['contact_verified']
    assert not result['public_relation']['support_identity_verified']
    assert before == (robot.targets, robot.supports, MEMORY)


def test_two_supported_targets_remain_ambiguous():
    robot = Robot([surface(.5, .25), surface(.9, .25)], [surface(.5, .2), surface(.9, .2)])
    assert locate_supported_target(robot, MEMORY)['status'] == 'supported_target_ambiguous'


def test_multiple_support_masks_do_not_duplicate_the_same_target():
    robot = Robot([surface(.5, .25)], [surface(.5, .2), surface(.51, .2)])
    result = locate_supported_target(robot, MEMORY)
    assert result['status'] == 'estimated_surface'
    assert result['public_relation']['support_candidates_matched'] == 2


def test_clipping_confidence_and_vertical_gates_are_preserved():
    memory = dict(MEMORY, support_allow_clipped=False)
    assert locate_supported_target(Robot([surface(.5, .25)]), memory)['status'] == 'support_not_located'
    assert locate_supported_target(Robot([surface(.5, .25, score=.2)]), MEMORY)['status'] == 'supported_target_not_located'
    assert locate_supported_target(Robot([surface(.5, -.1)]), MEMORY)['status'] == 'supported_target_not_located'
