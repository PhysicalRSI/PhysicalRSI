import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_stack_policy import SOURCE, MEMORY
from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_transfer_proposal import add_transfer_preview


def test_all_unreachable_transfers_stop_before_gripper_or_motion():
    namespace={}
    exec(add_transfer_preview(SOURCE),namespace)
    item={'surface_quantiles':{'median':[0,0,.8],'upper_95':[.02,.02,.82],'lower_05':[-.02,-.02,.78]}}
    namespace['locate']=lambda *args:item
    class Robot:
        def __init__(self): self.previews=[]
        def get_robot_state(self):
            return {'arms':{arm:{'measured_end_pose':[0,0,1,1,0,0,0]} for arm in ['left','right']}}
        def preview_transfer(self,arm,poses):
            self.previews.append((arm,poses))
            return {'status':'planning_failed','failed_stage':3}
        def __getattr__(self,name):
            raise AssertionError('Unexpected effect: '+name)
    robot=Robot()
    result=namespace['transfer'](robot,MEMORY,'green block',item)
    assert result['status']=='both_arms_planning_failed'
    assert len(robot.previews)==2
    assert all(len(poses)==6 for arm,poses in robot.previews)
    assert not result['task_success_claimed']


def test_unrecognized_or_already_wrapped_policy_is_rejected():
    with pytest.raises(ValueError): add_transfer_preview('def policy(): pass')
    with pytest.raises(ValueError): add_transfer_preview(add_transfer_preview(SOURCE))
