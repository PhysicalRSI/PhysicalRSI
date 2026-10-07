import time
from types import SimpleNamespace
import pytest
from test_egl_roboevolve_primitives import Robot
from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_transfer_preview import RoboEvolveTransferPreview


def make(plans, budget=20):
    starts=[]
    def plan(joints, *a, **k):
        starts.append(list(joints))
        return plans[len(starts)-1]
    robot=Robot(); robot._trans_from_gripper_to_endlink=lambda pose, arm: pose
    api=RoboEvolveTransferPreview(SimpleNamespace(robot=robot),
        planners={'left':SimpleNamespace(world_config=None,plan_path=plan)},
        joint_limits={'left':[[-1,1],[-2,2]]},max_control_actions=budget,
        settle_attempts=1,action_applier=lambda *a,**k:pytest.fail('Preview actuated robot'))
    return api,starts


def preview(api, count=2):
    return api.handlers['preview_transfer'](['left',[[1,2,3,1,0,0,0]]*count],{},deadline=time.monotonic()+10)


def test_predicted_endpoint_feeds_next_stage_without_actuation():
    api,starts=make([{'status':'Success','position':[[.2,.3]]}, {'status':'Success','position':[[.4,.5]]}])
    result=preview(api)
    assert starts[1]==[.2,.3] and result['status']=='sequence_planned'
    assert not api.trace and not result['execution_verified']
    assert not result['held_object_collision_checked']


def test_unreachable_destination_is_reported_before_any_motion():
    api,_=make([{'status':'Success','position':[[.2,.3]]},{'status':'Fail','robot_planner_statuses':['IK_FAIL']}])
    result=preview(api)
    assert result['failed_stage']==1 and result['status']=='planning_failed'
    assert not api.trace


def test_combined_budget_and_joint_limits_are_checked():
    api,_=make([{'status':'Success','position':[[.2,.3]]}]*2,budget=3)
    assert preview(api)['status']=='trajectory_budget_exceeded'
    api,_=make([{'status':'Success','position':[[9,.3]]}])
    with pytest.raises(ValueError,match='joint trajectory'):preview(api,1)


def test_scene_map_and_unbounded_pose_count_rejected():
    api,_=make([])
    with pytest.raises(ValueError,match='poses'):preview(api,9)
    api._planners['left'].world_config={}
    with pytest.raises(ValueError,match='robot-only'):preview(api,1)
