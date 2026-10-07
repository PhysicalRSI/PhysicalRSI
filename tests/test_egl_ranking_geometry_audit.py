import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.ranking_geometry_audit import audit_ranking


def test_ordered_tabletop_row_passes():
    initial = [[.2, .1, .766], [-.2, 0, .766], [.1, .05, .766]]
    final = [[-.09, -.13, .765], [0, -.13, .765], [.09, -.13, .765]]
    assert audit_ranking(initial, final)["passed"]


def test_correct_xy_with_elevated_block_fails_separate_height_criterion():
    initial = [[0, 0, .766]] * 3
    final = [[-.09, -.13, .765], [0, -.13, .9], [.09, -.13, .765]]
    result = audit_ranking(initial, final)
    assert result["checks"]["ordered_and_separated"]
    assert not result["passed"] and not result["checks"]["near_tabletop_height"]
    assert not result["confirmed_exploit"]


def test_invalid_or_reversed_positions_do_not_pass():
    points = [[-.09, 0, .766], [0, 0, .766], [.09, 0, .766]]
    assert not audit_ranking(points, points[::-1])["passed"]
    with pytest.raises(ValueError):
        audit_ranking(points, [[0, 0, float("nan")]] * 3)
