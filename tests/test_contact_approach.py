import pytest

from physicalrsi.skills.contact_approach import contact_approach, scaled_clearance


@pytest.mark.parametrize('value', [0, -1, 1.1, True, float('nan'), float('inf')])
def test_invalid_scale_rejected(value):
    with pytest.raises(ValueError):
        contact_approach(clearance_scale=value)


def test_clearance_is_scale_equivariant_and_versioned():
    parent = contact_approach()
    child = contact_approach(clearance_scale=1/3)
    assert parent.revision != child.revision
    for nominal in (.09, .1):
        assert parent.execute({'nominal_clearance_m': nominal})['clearance_m'] == nominal
        assert child.execute({'nominal_clearance_m': nominal})['clearance_m'] == pytest.approx(nominal/3)
    assert scaled_clearance(.2, .5) == .1
