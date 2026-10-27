import pytest
from physicalrsi.skills.moving_region import select_moving_region


def history(two_moving=False):
    result=[]
    for i in range(5):
        candidates=[dict(uv=[.1+.02*i,.2],confidence=.8,measurement='box_center'),
                    dict(uv=[.8-.02*i if two_moving else .8,.8],confidence=.9,measurement='box_center')]
        result.append(dict(schema='physicalrsi.visual-regions/v1',camera='head',calibration_revision='1',
            image_size_px=[320,240],provider={'revision':'1'},timestamp_s=i*.2,observation_sequence=i,
            regions={'object':dict(status='ambiguous',uv=None,confidence=None,candidates=candidates)}))
    return dict(observations=result,feature='object',stationary_camera=True)


def test_motion_selects_lower_confidence_moving_region_over_stationary_distractor():
    result=select_moving_region().execute(history())
    assert result['status']=='selected'
    assert result['candidate']['uv']==pytest.approx([.18,.2])
    assert result['motion']['velocity_uv_s']==pytest.approx([.1,0])


def test_multiple_moving_objects_do_not_become_a_target_identity():
    result=select_moving_region().execute(history(True))
    assert result['reason']=='multiple_moving_regions'


def test_duplicate_detections_and_occlusion_do_not_create_confident_tracks():
    value=history()
    for row in value['observations']:
        row['regions']['object']['candidates'].append(dict(uv=[.101+.1*row['timestamp_s'],.2],confidence=.8,measurement='box_center'))
    assert select_moving_region().execute(value)['status']=='unavailable'
    value=history();value['observations'][-2]['regions']['object']['candidates']=[]
    assert select_moving_region().execute(value)['status']=='unavailable'


def test_moving_camera_requires_another_estimator():
    value=history();value['stationary_camera']=False
    with pytest.raises(ValueError):select_moving_region().execute(value)
