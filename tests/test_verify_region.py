from copy import deepcopy
import numpy as np
import pytest
from physicalrsi.runtime.sensors import SCHEMA, encode_rgb
from physicalrsi.skills.verify_region import verify_region

class Provider:
    def __init__(self):
        self.calls=[];self.revision=1;self.mutate=False
    def identity(self):return {'revision':self.revision}
    def segment(self,rgb,text,*,threshold):
        self.calls.append((rgb.shape,text,threshold))
        if self.mutate:self.revision+=1
        return [dict(bbox_xyxy=[0,0,rgb.shape[1],rgb.shape[0]],score=.9)]

def value():
    obs=dict(schema=SCHEMA,sequence=1,timestamp_s=.1,embodiment_revision='robot',
        calibration_revision='cal',cameras={'head':dict(captured_at_s=.1,
        rgb_png_b64=encode_rgb(np.zeros((240,320,3),dtype=np.uint8)))},actuator_position=[0.])
    return dict(observation=obs,proposal=dict(observation_sequence=1,calibration_revision='cal',
        camera='head',image_size_px=[320,240],selected=dict(uv=[.99,.01],area_px=100)))

def test_crop_is_clamped_and_support_is_not_identity():
    p=Provider();r=verify_region(p,'cup',camera='head').execute(value())
    assert r['crop_xyxy']==[224,0,320,96]
    assert r['accepted'] and r['semantic_identity'] is None
    assert p.calls==[((96,96,3),'cup',.25)]

def test_missing_candidate_does_not_invoke_model():
    p=Provider();v=value();v['proposal']['selected']=None
    assert verify_region(p,'cup',camera='head').execute(v)['status']=='unobserved'
    assert not p.calls

@pytest.mark.parametrize('field,bad',[('observation_sequence',0),('camera','other'),
    ('calibration_revision','other'),('image_size_px',[640,480])])
def test_stale_or_incompatible_proposal_rejected(field,bad):
    v=value();v['proposal'][field]=bad;p=Provider()
    with pytest.raises(ValueError):verify_region(p,'cup',camera='head').execute(v)
    assert not p.calls

@pytest.mark.parametrize('level',['observation','proposal','selected','outer'])
def test_hidden_state_rejected(level):
    v=value();target=v if level=='outer' else (v['proposal']['selected'] if level=='selected' else v[level])
    target['object_pose']=[0,0,0]
    with pytest.raises(ValueError):verify_region(Provider(),'cup',camera='head').execute(v)

def test_changed_provider_fails():
    p=Provider();skill=verify_region(p,'cup',camera='head');p.mutate=True
    with pytest.raises(ValueError,match='Provider changed'):skill.execute(value())

def test_configuration_versions_category_and_crop():
    p=Provider();a=verify_region(p,'cup',camera='head')
    assert a.revision!=verify_region(p,'ball',camera='head').revision
    assert a.revision!=verify_region(p,'cup',camera='head',minimum_side_px=100).revision

@pytest.mark.parametrize('box,score',[([0,0,97,96],.9),([0,0,96,96],float('nan'))])
def test_bad_provider_geometry_and_confidence_fail(box,score):
    p=Provider();p.segment=lambda *args,**kwargs:[dict(bbox_xyxy=box,score=score)]
    with pytest.raises(ValueError,match='Invalid detection'):
        verify_region(p,'cup',camera='head').execute(value())

def test_category_evidence_must_cover_candidate_and_pass_threshold():
    p=Provider();p.segment=lambda *args,**kwargs:[dict(bbox_xyxy=[0,0,20,20],score=.9),
        dict(bbox_xyxy=[0,0,96,96],score=.1)]
    r=verify_region(p,'cup',camera='head').execute(value())
    assert not r['accepted'] and r['selected'] is None
    assert len(r['boxes'])==1
