from copy import deepcopy
import pytest
from physicalrsi.skills.visual_motion import estimate_visual_motion


def frame(t, sequence, status='visible'):
    return dict(schema='physicalrsi.visual-regions/v1', camera='camera', calibration_revision='1',
                image_size_px=[320,240], provider={'revision':'1'}, timestamp_s=t,
                observation_sequence=sequence, regions={'object':dict(status=status,
                    uv=[.2+.1*t,.5-.2*t] if status=='visible' else None,
                    confidence=.8 if status=='visible' else None, measurement='box_center')})


def test_irregular_cadence_estimates_image_velocity_without_phase_labels():
    rows=[frame(t,i) for i,t in enumerate([0.,.1,.3,.5])]
    result=estimate_visual_motion().execute(dict(observations=rows,feature='object'))
    assert result['velocity_uv_s']==pytest.approx([.1,-.2])
    assert result['residual_rms_uv']<1e-12
    assert 'velocity_m_s' not in result


@pytest.mark.parametrize('reason',['ambiguous','unobserved','low_confidence','gap','measurement'])
def test_uncertain_discontinuous_history_cannot_create_a_velocity(reason):
    rows=[frame(0,0),frame(.1,1),frame(.2,2)]
    if reason in ('ambiguous','unobserved'): rows[1]=frame(.1,1,reason)
    if reason=='low_confidence': rows[1]['regions']['object']['confidence']=.1
    if reason=='gap': rows[2]=frame(1.,2)
    if reason=='measurement': rows[2]['regions']['object']['measurement']='mask_centroid'
    result=estimate_visual_motion().execute(dict(observations=rows,feature='object'))
    assert result['status']=='unavailable'


@pytest.mark.parametrize('field,value',[('camera','other'),('calibration_revision','2'),
    ('image_size_px',[640,480]),('provider',{'revision':'2'}),('timestamp_s',0.),('observation_sequence',0)])
def test_incompatible_observations_are_rejected(field,value):
    rows=[frame(0,0),frame(.1,1)];rows[1][field]=deepcopy(value)
    with pytest.raises(ValueError):
        estimate_visual_motion().execute(dict(observations=rows,feature='object'))
