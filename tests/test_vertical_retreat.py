import pytest
from physicalrsi.skills.vertical_retreat import retreat_waypoint


def test_retreat_preserves_current_xy_and_never_descends():
    assert retreat_waypoint([.2, -.3, .9], [0., 0., 1.1]) == [.2, -.3, 1.1]
    assert retreat_waypoint([.2, -.3, 1.2], [0., 0., 1.1]) == [.2, -.3, 1.2]


@pytest.mark.parametrize('point', [[0, 1], [0, 1, float('nan')], [True, 1, 2]])
def test_invalid_retreat_geometry(point):
    with pytest.raises(ValueError):
        retreat_waypoint(point, [0., 0., 1.])
