import numpy as np
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.public_geometry import masked_surface_geometry


def test_visible_depth_plane_transforms_without_claiming_whole_object():
    pose=np.eye(4);pose[:3,3]=[2,3,4]
    camera={'images':{'depth':np.ones((8,8))},'intrinsics':np.array([[4,0,3.5],[0,4,3.5],[0,0,1]]),'pose_mat':pose}
    mask=np.zeros((8,8),dtype=bool);mask[1:7,1:7]=True
    result=masked_surface_geometry(camera,mask)
    assert result['surface_quantiles']['median']==[2,3,5]
    assert result['points']==36 and not result['touches_image_border']
    assert result['full_object_extent_known'] is False
    camera['images']['depth'][:]=float('nan')
    assert masked_surface_geometry(camera,mask)['status']=='insufficient_depth'
    camera['pose_mat'][0,0]=2
    with pytest.raises(ValueError,match='geometry'):masked_surface_geometry(camera,mask)


def test_near_field_surface_requires_explicit_sensor_range():
    depth = np.full((8, 8), .1)
    depth[0, :4] = [0., -.1, float('nan'), 3.]
    camera = {'images': {'depth': depth}, 'intrinsics': np.eye(3), 'pose_mat': np.eye(4)}
    mask = np.ones((8, 8), dtype=bool)
    assert masked_surface_geometry(camera, mask)['status'] == 'insufficient_depth'
    result = masked_surface_geometry(camera, mask, depth_range_m=(.015, 2.))
    assert result['status'] == 'estimated_surface'
    assert result['points'] == 60
    assert result['surface_quantiles']['median'][2] == .1
    assert result['full_object_extent_known'] is False


@pytest.mark.parametrize('bounds', [(0., 2.), (.2, .1), (.015, 21.), (float('nan'), 2.), (.2,)])
def test_invalid_sensor_depth_range_is_rejected(bounds):
    with pytest.raises(ValueError, match='depth range'):
        masked_surface_geometry({}, [], depth_range_m=bounds)


def test_location_callback_obeys_deadline_and_only_passes_public_observation(monkeypatch):
    from PhysicalRSI_baselines.embodied_goodharts_law import located_libero_primitives as module
    api = object.__new__(module.LocatedLiberoPrimitives)
    api._pose_solver = api._segmenter = None
    public = {'agentview': {'images': {'rgb': 'public-image'}}}
    api.get_observation = lambda: public
    received = []
    def locator(observation, prompt, *, deadline):
        received.append((observation, prompt, deadline))
        return {'status': 'not_detected', 'semantic_identity_verified': False}
    api._object_locator = locator
    monkeypatch.setattr(module.time, 'monotonic', lambda: 10.)
    callback = api.handlers['locate_object']
    with pytest.raises(TimeoutError):
        callback(['basket'], {}, deadline=9.)
    with pytest.raises(ValueError):
        callback(['basket'], {}, deadline=float('inf'))
    assert not received
    assert callback(['basket'], {}, deadline=11.)['status'] == 'not_detected'
    assert received == [(public, 'basket', 11.)]
