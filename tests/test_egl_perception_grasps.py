import time
import numpy as np
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.perception_grasps import PublicGraspPlanner


def test_public_grasp_camera_transform_and_empty_segmentation():
    transform=np.eye(4);transform[:3,3]=[1,2,3]
    transform[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    observation={'agentview':{'pose_mat':transform,'images':{'rgb':np.zeros((8,8,3),dtype=np.uint8),'depth':np.ones((8,8,1))},'intrinsics':np.eye(3)}}
    detections=[{'score':.5,'mask':np.ones((8,8))}]
    calls=[]
    def grasp(*args,**kwargs):
        calls.append(args)
        return {'frame':'camera-optical','grasps':[{'matrix':np.eye(4),'score':.7,'contact_point':[1,0,1]}]}
    planner=PublicGraspPlanner(segmenter=lambda *a,**k:{'detections':detections},grasp_client=grasp)
    result=planner(observation,'can',deadline=time.monotonic()+5)
    assert np.array_equal(np.asarray(result['grasps'][0]['matrix']),transform)
    assert result['grasps'][0]['contact_point']==[1,3,4]
    assert result['semantic_identity_verified'] is False
    detections.clear()
    assert planner(observation,'can',deadline=time.monotonic()+5)['status']=='not_detected'
    assert len(calls)==1
    observation['agentview']['pose_mat'][0,0]=2
    with pytest.raises(ValueError,match='transform'):
        planner(observation,'can',deadline=time.monotonic()+5)
