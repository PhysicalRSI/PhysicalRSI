import pytest
from physicalrsi.skills.visible_coverage import visible_coverage

def patch(hole=False, shift=0.):
    return [[x*.004+shift,y*.004,0.8] for x in range(-10,11) for y in range(-10,11)
            if not hole or abs(x)>4 or abs(y)>4]

def test_observed_disk_and_translation():
    skill=visible_coverage()
    for shift in (0.,1.):
        assert skill.execute(dict(points_world_m=patch(shift=shift),target_xy_m=[shift,0],radius_m=.02))['status']=='observed_coverage'

def test_no_hull_filling_or_hidden_geometry():
    skill=visible_coverage()
    assert skill.execute(dict(points_world_m=patch(hole=True),target_xy_m=[0,0],radius_m=.01))['status']=='unknown'
    assert skill.execute(dict(points_world_m=patch(),target_xy_m=[.1,0],radius_m=.01))['status']=='unknown'

def test_empty_and_invalid():
    skill=visible_coverage()
    assert skill.execute(dict(points_world_m=[],target_xy_m=[0,0],radius_m=0))['status']=='unknown'
    with pytest.raises(ValueError):skill.execute(dict(points_world_m=[[float('nan'),0,0]],target_xy_m=[0,0],radius_m=0))
