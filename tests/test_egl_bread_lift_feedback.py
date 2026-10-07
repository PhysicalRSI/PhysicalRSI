from copy import deepcopy
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.bread_lift_feedback import bread_lift_feedback


def surface(x=0., z=.8):
    return dict(status='estimated_surface', frame='world', detection_score=.8,
                touches_image_border=False, surface_quantiles={'median':[x,0.,z]})


def test_stationary_surface_does_not_become_grasp_success():
    d = bread_lift_feedback(surface(), [surface()], [0,0,1.04])
    assert d['status'] == 'stationary_surface_observed'
    assert not d['grasp_verified']


def test_visible_rise_requires_nearby_hand_and_does_not_certify_identity():
    d = bread_lift_feedback(surface(), [surface(z=.9)], [0,0,1.04])
    assert d['status'] == 'raised_surface_near_hand'
    assert not d['semantic_identity_verified']
    assert bread_lift_feedback(surface(), [surface(z=.9)], [.2,0,1.04])['status'] == 'unknown'


@pytest.mark.parametrize('objects', [[], [surface(), surface(x=.04)],
                                   [dict(surface(), touches_image_border=True)],
                                   [dict(surface(), detection_score=.2)],
                                   [surface(z=float('nan'))]])
def test_absence_ambiguity_occlusion_and_invalid_geometry_are_unknown(objects):
    assert bread_lift_feedback(surface(), objects, [0,0,1.04])['status'] == 'unknown'


def test_threshold_contract():
    with pytest.raises(ValueError):
        bread_lift_feedback(surface(), [], [0,0,1.04], stationary_radius_m=.1)
