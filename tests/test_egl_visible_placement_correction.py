import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.visible_placement_correction import propose_visible_correction


def geometry(x, clipped=False):
    return dict(status='estimated_surface', touches_image_border=clipped,
                surface_quantiles={'lower_05':[x-.02,-.02,0], 'median':[x,0,.01], 'upper_95':[x+.02,.02,.02]})


def run(bowl, plate=None, **kwargs):
    return propose_visible_correction(bowl, plate or geometry(0), supported_points=kwargs.get('support',300), reference_points=1000)


def test_translation_centers_visible_extent_without_changing_height():
    result=run(geometry(.03))
    assert result['translation_m']==pytest.approx([-.03,0,0])
    assert result['status']=='correction_proposed' and not result['full_object_center_verified']


def test_rejects_occluded_unverified_and_large_corrections():
    assert run(geometry(.03,True))['status']=='incomplete_visible_surface'
    assert run(geometry(.03),support=50)['status']=='insufficient_plate_support'
    assert run(geometry(.2))['status']=='correction_exceeds_limit'
    assert run(geometry(.001))['status']=='not_needed'


def test_invalid_geometry_is_not_a_motion_command():
    g=geometry(.03);g['surface_quantiles']['median'][0]=99
    with pytest.raises(ValueError,match='Unordered'):run(g)
