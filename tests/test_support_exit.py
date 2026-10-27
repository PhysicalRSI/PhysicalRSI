import copy
import pytest
from physicalrsi.skills.support_exit import support_exit


def sample():
    return dict(position_m=[0,.5,1.],velocity_m_s=[.1,-.2,0.],
                boundary=dict(point_m=[0,0,0],outward_normal=[0,-1,0]))


def test_intersects_support_plane_and_preserves_velocity():
    x=sample();saved=copy.deepcopy(x);r=support_exit().execute(x)
    assert r['valid'] and r['time_to_exit_s']==pytest.approx(2.5)
    assert r['exit_position_m']==pytest.approx([.25,0,1])
    assert r['exit_velocity_m_s']==x['velocity_m_s']
    assert x==saved


def test_rotation_translation_and_normal_scale():
    x=sample();x.update(position_m=[2.5,2,2],velocity_m_s=[.2,.1,0])
    x['boundary']=dict(point_m=[3,2,1],outward_normal=[7,0,0])
    r=support_exit().execute(x)
    assert r['time_to_exit_s']==pytest.approx(2.5)
    assert r['exit_position_m']==pytest.approx([3,2.25,2])


@pytest.mark.parametrize('velocity',[[0,0,0],[.1,0,0],[0,.2,0]])
def test_no_unobserved_motion_toward_boundary(velocity):
    x=sample();x['velocity_m_s']=velocity
    assert support_exit().execute(x)['reason']=='not_approaching_support_boundary'


def test_outside_domain_and_horizon_abstain():
    x=sample();assert not support_exit(max_horizon_s=1).execute(x)['valid']
    x['position_m'][1]=-.1
    assert support_exit().execute(x)['reason']=='outside_supported_region'


def test_degenerate_and_nonfinite_geometry_rejected():
    x=sample();x['boundary']['outward_normal']=[0,0,0]
    with pytest.raises(ValueError):support_exit().execute(x)
    x=sample();x['position_m'][0]=float('nan')
    with pytest.raises(ValueError):support_exit().execute(x)
