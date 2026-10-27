import math
import pytest

from physicalrsi.skills.directional_retreat import directional_retreat
from physicalrsi.skills.registry import SkillRegistry


def test_retreat_is_translation_equivariant_and_direction_scale_invariant():
    skill = directional_retreat(distance_m=.15)
    a = skill.execute(dict(current_m=[1., 2., 3.], contact_direction=[3., 4., 0.]))['waypoint_m']
    b = skill.execute(dict(current_m=[3., 1., 8.], contact_direction=[6., 8., 0.]))['waypoint_m']
    assert a == pytest.approx([.91, 1.88, 3.])
    assert b == pytest.approx([a[0]+2, a[1]-1, a[2]+5])
    assert math.dist(a, [1., 2., 3.]) == pytest.approx(.15)


def test_restart_requires_provider_and_configuration_has_distinct_revision(tmp_path):
    skill = directional_retreat()
    revision = SkillRegistry(tmp_path).register(skill)
    restarted = SkillRegistry(tmp_path)
    value = dict(current_m=[0., 0., 0.], contact_direction=[0., 0., 1.])
    with pytest.raises(RuntimeError):
        restarted.load(revision).execute(value)
    assert restarted.load(revision, implementation=skill).execute(value) == dict(waypoint_m=[0., 0., -.15])
    assert directional_retreat(distance_m=.2).revision != skill.revision
    with pytest.raises(ValueError):
        restarted.load(revision, implementation=directional_retreat(distance_m=.2))


@pytest.mark.parametrize('direction', [[0, 0, 0], [1, 2], [True, 0, 1], [float('nan'), 0, 1]])
def test_invalid_direction_is_rejected(direction):
    with pytest.raises(ValueError):
        directional_retreat().execute(dict(current_m=[0, 0, 0], contact_direction=direction))


@pytest.mark.parametrize('distance', [0, -1, True, float('inf')])
def test_invalid_distance_is_rejected(distance):
    with pytest.raises(ValueError):
        directional_retreat(distance_m=distance)


def test_unknown_inputs_are_not_accepted():
    with pytest.raises(ValueError):
        directional_retreat().execute(dict(current_m=[0, 0, 0], contact_direction=[1, 0, 0], task='catch_cup'))
