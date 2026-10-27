import numpy as np
import pytest
cv2=pytest.importorskip('cv2')
from physicalrsi.runtime.sensors import SCHEMA,encode_rgb
from physicalrsi.skills.track_rgb_point import track_rgb_point

def obs(im,seq):
 return dict(schema=SCHEMA,sequence=seq,timestamp_s=seq*.04,embodiment_revision='robot',calibration_revision='cal',cameras={'head':dict(captured_at_s=seq*.04,rgb_png_b64=encode_rgb(im))},actuator_position=[0.])
def pair():
 rng=np.random.default_rng(9);im=rng.integers(0,256,(80,80,3),dtype=np.uint8);later=cv2.warpAffine(im,np.float32([[1,0,3],[0,1,2]]),(80,80))
 return dict(previous=obs(im,0),current=obs(later,1),uv=[40.5/80,40.5/80])
def test_translation_tracks_pixel_centres():
 r=track_rgb_point(camera='head').execute(pair())
 assert r['status']=='tracked';assert r['uv']==pytest.approx([43.5/80,42.5/80],abs=.001)
 assert r['semantic_identity'] is None

def test_textureless_point_abstains():
 im=np.zeros((80,80,3),dtype=np.uint8)
 assert track_rgb_point(camera='head').execute(dict(previous=obs(im,0),current=obs(im,1),uv=[.5,.5]))['status']=='lost'

@pytest.mark.parametrize('change',['hidden','gap','calibration','sequence'])
def test_invalid_sensor_pair_rejected(change):
 v=pair()
 if change=='hidden':v['current']['object_velocity']=[0,0,0]
 if change=='gap':v['current']['cameras']['head']['captured_at_s']=.2;v['current']['timestamp_s']=.2
 if change=='calibration':v['current']['calibration_revision']='other'
 if change=='sequence':v['current']['sequence']=0
 with pytest.raises(ValueError):track_rgb_point(camera='head').execute(v)

def test_tracking_configuration_versioned():
 assert track_rgb_point(camera='head').revision!=track_rgb_point(camera='head',max_roundtrip_px=2).revision
