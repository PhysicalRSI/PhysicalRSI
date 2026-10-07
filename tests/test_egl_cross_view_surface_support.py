import numpy as np
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.cross_view_surface_support import cross_view_surface_support


def camera():
    return dict(images={'depth':np.ones((5,5))}, intrinsics=np.array([[2.,0,2],[0,2,2],[0,0,1]]),pose_mat=np.eye(4))


def test_depth_agreement_occlusion_and_foreground_are_distinct():
    d=cross_view_surface_support([[0,0,1],[0,0,.8],[0,0,1.2],[0,0,-1],[9,0,1]],camera(),np.ones((5,5)))
    assert d['states'].tolist()==['supported','in_front_of_reference','behind_reference','outside_view','outside_view']
    assert not d['temporal_stability_verified'] and not d['semantic_identity_verified']


def test_world_transform_and_reference_mask_are_used():
    c=camera();c['pose_mat'][:3,3]=[1,2,3]
    mask=np.zeros((5,5));mask[2,2]=1
    d=cross_view_surface_support([[1,2,4],[1.5,2,4]],c,mask)
    assert d['states'].tolist()==['supported','unmasked_surface']


def test_invalid_depth_and_projection_edges_do_not_get_clipped_into_support():
    c=camera();c['images']['depth'][2,2]=np.nan
    d=cross_view_surface_support([[0,0,1],[-1.01,0,1]],c,np.ones((5,5)))
    assert d['states'].tolist()==['invalid_reference_depth','outside_view']


def test_invalid_queries_rejected():
    with pytest.raises(ValueError):cross_view_surface_support([[0,0,float('nan')]],camera(),np.ones((5,5)))
