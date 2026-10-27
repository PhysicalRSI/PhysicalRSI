from copy import deepcopy
import pytest
from physicalrsi.skills.capture_region import capture_region_planning


def request():
    return dict(plan=dict(predicted_position_m=[.1,0,.8],target_position_m=[.1,0,.8],
        feasible=True,reason=None,time_to_intercept_s=3.),
        robot=dict(target_position_m=[0,0,.8],workspace_min_m=[-1,-1,0],
                   workspace_max_m=[1,1,2],max_speed_m_s=1.),
        region=dict(minimum_m=[.12,-1,0],maximum_m=[1,1,2],capture_radius_m=.04))


def test_selects_nearest_feasible_catcher_without_changing_prediction_or_input():
    value=request();before=deepcopy(value)
    result=capture_region_planning().execute(value)
    assert result['feasible']
    assert result['target_position_m']==[.12,0,.8]
    assert result['predicted_position_m']==[.1,0,.8]
    assert value==before
    assert result['capture_offset_m']==pytest.approx(.02)


def test_no_clamping_through_empty_aperture_workspace_or_time():
    value=request();value['region']['minimum_m'][0]=.2
    result=capture_region_planning().execute(value)
    assert not result['feasible'] and result['target_position_m'] is None
    assert result['reason']=='capture_aperture_misses_goal_region'
    value=request();value['robot']['workspace_max_m'][0]=.11
    assert capture_region_planning().execute(value)['reason']=='empty_goal_workspace_intersection'
    value=request();value['plan']['time_to_intercept_s']=.01
    assert capture_region_planning().execute(value)['reason']=='insufficient_time'


def test_mirrored_region_and_prediction_abstention():
    value=request()
    value['plan']['predicted_position_m'][0]=-.1
    value['region']['minimum_m'][0]=-1
    value['region']['maximum_m'][0]=-.12
    assert capture_region_planning().execute(value)['target_position_m'][0]==-.12
    value['plan'].update(feasible=False,reason='insufficient_observed_forward_motion')
    assert capture_region_planning().execute(value)==value['plan']
    value['region']=None
    assert capture_region_planning().execute(value)==value['plan']
