from copy import deepcopy
import numpy as np
import pytest
pytest.importorskip('cv2')
trimesh=pytest.importorskip('trimesh', minversion='4.4')
from physicalrsi.perception.self_model import RobotSelfModel
from physicalrsi.runtime.sensors import SCHEMA,encode_rgb
from physicalrsi.skills.visual_changes import locate_visual_changes


def sensor(seq=0):
    return dict(schema=SCHEMA,sequence=seq,timestamp_s=seq*.1,
        embodiment_revision='test',calibration_revision='v1',actuator_position=[0.],
        cameras={'head':dict(captured_at_s=seq*.1,rgb_png_b64=encode_rgb(np.zeros((32,32,3),np.uint8)))})


def make_model(tmp_path):
    trimesh.creation.box(extents=[.2,.2,.2]).export(str(tmp_path/'link.stl'))
    urdf=tmp_path/'robot.urdf'
    urdf.write_text('''<robot name="fixture"><link name="base"/><link name="link"><visual><geometry><mesh filename="link.stl"/></geometry></visual></link><joint name="slide" type="prismatic"><parent link="base"/><child link="link"/><origin xyz="0 0 2"/><axis xyz="1 0 0"/></joint></robot>''')
    return RobotSelfModel(urdf,mounts={'robot':dict(base_to_world=np.eye(4).tolist(),bindings={'slide':dict(index=0,scale=1.,offset=0.)})},intrinsic_cv=[[30,0,16],[0,30,16],[0,0,1]],extrinsic_cv=np.eye(4).tolist(),image_size_px=[32,32],camera='head',embodiment_revision='test',calibration_revision='v1')


def test_model_projection_moves_with_measured_joint_and_identity_is_defensive(tmp_path):
    model=make_model(tmp_path);obs=sensor();a=model.mask(obs,camera='head')
    assert a.dtype==np.bool_ and a[16,16]
    obs['actuator_position']=[.5];b=model.mask(obs,camera='head')
    assert np.nonzero(b)[1].mean()>np.nonzero(a)[1].mean()
    assert model.forward_kinematics(obs)['robot']['link'][0,3]==pytest.approx(.5)
    identity=model.identity();identity['mounts'].clear()
    assert model.identity()['mounts']


@pytest.mark.parametrize('change',['calibration','embodiment','resolution','layout','hidden'])
def test_incompatible_sensor_input_is_rejected(tmp_path,change):
    model=make_model(tmp_path);obs=sensor()
    if change=='calibration':obs['calibration_revision']='other'
    if change=='embodiment':obs['embodiment_revision']='other'
    if change=='resolution':obs['cameras']['head']['rgb_png_b64']=encode_rgb(np.zeros((16,16,3),np.uint8))
    if change=='layout':obs['actuator_position']=[0., 1.]
    if change=='hidden':obs['object_pose']=[0,0,0]
    with pytest.raises(ValueError):model.mask(obs,camera='head')


def test_composition_excludes_own_projection_without_semantic_identity(tmp_path):
    model=make_model(tmp_path);previous=sensor();current=sensor(1)
    pixels=np.zeros((32,32,3),np.uint8);pixels[15:18,15:18]=255
    current['cameras']['head']['rgb_png_b64']=encode_rgb(pixels)
    pair=dict(previous=previous,current=current,stationary_camera=True)
    assert locate_visual_changes(camera='head').execute(pair)['status']=='selected'
    result=locate_visual_changes(camera='head',self_model=model).execute(pair)
    assert result['status']=='unobserved' and result['semantic_identity'] is None


def test_calibration_change_changes_skill_identity(tmp_path):
    model=make_model(tmp_path)
    assert locate_visual_changes(camera='head').identity()!=locate_visual_changes(camera='head',self_model=model).identity()
