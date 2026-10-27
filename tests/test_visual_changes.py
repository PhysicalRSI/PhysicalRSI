from copy import deepcopy
import numpy as np
import pytest
pytest.importorskip('cv2')
from physicalrsi.runtime.sensors import SCHEMA, encode_rgb
from physicalrsi.skills.visual_changes import locate_visual_changes


def observation(image, sequence):
    t=sequence*.1
    return dict(schema=SCHEMA, sequence=sequence, timestamp_s=t,
        embodiment_revision='robot', calibration_revision='camera-v1',
        cameras={'head':dict(captured_at_s=t,rgb_png_b64=encode_rgb(image))},
        actuator_position=[0.])


def pair():
    image=np.zeros((32,32,3),dtype=np.uint8)
    later=image.copy();later[8:12,8:12]=255
    return dict(previous=observation(image,0),current=observation(later,1),stationary_camera=True)


def test_image_change_is_not_promoted_to_semantic_identity():
    result=locate_visual_changes(camera='head').execute(pair())
    assert result['status']=='selected'
    assert result['selected']['uv']==pytest.approx([10/32,10/32])
    assert result['semantic_identity'] is None
    assert result['measurement']=='image_change_centroid'


def test_equal_changes_abstain_and_unchanged_frames_are_unobserved():
    value=pair();image=np.zeros((32,32,3),dtype=np.uint8)
    image[8:12,8:12]=255;image[20:24,20:24]=255
    value['current']=observation(image,1)
    skill=locate_visual_changes(camera='head')
    assert skill.execute(value)['status']=='ambiguous'
    value['current']=observation(np.zeros_like(image),1)
    assert skill.execute(value)['status']=='unobserved'


@pytest.mark.parametrize('change',['camera_motion','calibration','time','resolution','hidden_state','gap'])
def test_incompatible_or_privileged_inputs_fail(change):
    value=pair()
    if change=='camera_motion':value['stationary_camera']=False
    if change=='calibration':value['current']['calibration_revision']='other'
    if change=='time':value['current']=deepcopy(value['previous'])
    if change=='resolution':value['current']=observation(np.zeros((16,16,3),dtype=np.uint8),1)
    if change=='hidden_state':value['current']['object_pose']=[0,0,0]
    if change=='gap':value['current']=observation(np.zeros((32,32,3),dtype=np.uint8),9)
    with pytest.raises(ValueError):locate_visual_changes(camera='head').execute(value)


def test_configuration_is_versioned():
    a=locate_visual_changes(camera='head').identity()
    b=locate_visual_changes(camera='head',threshold=31).identity()
    assert a!=b
