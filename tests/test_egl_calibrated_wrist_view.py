import math
import numpy as np
from PhysicalRSI_baselines.embodied_goodharts_law.calibrated_wrist_view import calibrated_view_poses, wrist_rotation, refine_clipped_target
from test_egl_support_reference import surface


def test_view_distance_and_optical_axis_use_rigid_camera_hand_calibration():
    hand = [.4, .1, .5, 0., 1., 0., 0.]
    r = np.array(wrist_rotation(hand[3:]));offset=np.array([.03, -.02, .1])
    cam=np.eye(4);cam[:3,:3]=r;cam[:3,3]=np.array(hand[:3])+r@offset
    target=np.array([.7, -.2, .25])
    poses=calibrated_view_poses(hand,cam.tolist(),target.tolist(),.25,[-math.pi/3,-math.pi/4,0])
    for pose in poses:
        desired=np.array(wrist_rotation(pose[3:]));camera_position=np.array(pose[:3])+desired@offset
        np.testing.assert_allclose(camera_position+.25*desired[:,2],target,atol=1e-12)
        assert abs(np.linalg.norm(target-camera_position)-.25)<1e-12


class Robot:
    def __init__(self, reachable=True, ambiguous=False):
        self.reachable,self.ambiguous=reachable,ambiguous
        self.moves=[];self.queries=[]
    def get_robot_state(self):return {'robot_cartesian_pos':[.4,0,.5,0,1,0,0,1]}
    def preview_poses(self, poses):return {'kinematically_solved':self.reachable}
    def try_move_to_pose(self, pose):self.moves.append(pose);return {'reached':False}
    def locate_objects(self,prompt,*,camera):
        self.queries.append(prompt)
        return {'public_camera_calibration':{'pose_mat':np.eye(4).tolist()},'objects':([] if not self.moves else [surface(.6,.25)]*(2 if self.ambiguous else 1))}


MEMORY={'wrist_observation_distance_m':.25,'wrist_observation_tilts_rad':[-math.pi/3,-math.pi/4,0],
        'wrist_observation_prompts':['bowl','patterned bowl'],'wrist_view_minimum_score':.3,
        'wrist_view_association_xy_m':.12,'wrist_view_association_z_m':.08}


def test_unreachable_views_never_cause_motion():
    robot=Robot(reachable=False)
    assert refine_clipped_target(robot,MEMORY,surface(.6,.25,clipped=True))['status']=='target_view_not_located'
    assert not robot.moves and robot.queries==['bowl']


def test_actual_view_can_refine_target_without_claiming_motion_reached():
    robot=Robot()
    result=refine_clipped_target(robot,MEMORY,surface(.6,.25,clipped=True))
    assert result['status']=='estimated_surface'
    assert not result['view_refinement']['association_verified']
    assert not result['view_refinement']['attempts'][0]['motion']['reached']


def test_multiple_associated_surfaces_stop():
    assert refine_clipped_target(Robot(ambiguous=True),MEMORY,surface(.6,.25,clipped=True))['status']=='target_view_ambiguous'
