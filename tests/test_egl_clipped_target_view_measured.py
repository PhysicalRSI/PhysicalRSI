from PhysicalRSI_baselines.embodied_goodharts_law.clipped_target_view_measured import refine_clipped_target
from test_egl_clipped_target_view import Robot, MEMORY
from test_egl_support_reference import surface


def test_incomplete_view_uses_live_camera_but_does_not_continue_view_motion():
    robot = Robot([surface(.62, .25)], reached=False)
    result = refine_clipped_target(robot, MEMORY, surface(.6, .25, clipped=True))
    assert result['status'] == 'estimated_surface'
    assert len(robot.moves) == 1
    assert robot.queries == [('black bowl', 'robot0_eye_in_hand')]
    assert not result['view_refinement']['association_verified']


def test_unhelpful_incomplete_view_still_rejects_missing_or_ambiguous_target():
    for candidates, status in [([], 'target_view_not_located'),
                               ([surface(.6, .25), surface(.61, .25)], 'target_view_ambiguous'),
                               ([surface(.6, .25, clipped=True)], 'target_view_not_located')]:
        robot = Robot(candidates, reached=False)
        assert refine_clipped_target(robot, MEMORY, surface(.6, .25, clipped=True))['status'] == status
        assert len(robot.moves) == 1
