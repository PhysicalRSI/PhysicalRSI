import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.bread_lift_proposal import add_bread_lift_feedback


def test_gate_requires_two_stationary_views_and_measured_hand_rise():
    # Reviewed fixture code only; production materialization never executes source.
    source = 'def surface_pick(robot, memory):\n    return memory["picked"]\n'
    namespace = {}
    exec(add_bread_lift_feedback(source), namespace)
    item = dict(status='estimated_surface', frame='world', detection_score=.8,
                touches_image_border=False, surface_quantiles={'median':[0,0,.8]})
    picked = dict(status='lift_motion_returned', location=item,
                  descent={'measured_pose':[0,0,.92]},
                  lift={'measured_pose':[0,0,1.04], 'position_error':.001})
    class Robot:
        def __init__(self, count): self.count=count; self.calls=[]
        def locate_objects(self, prompt, *, camera):
            self.calls.append(camera)
            return {'objects':[item] if len(self.calls)<=self.count else []}
    settings = dict(picked=picked, position_tolerance_m=.015)
    assert namespace['surface_pick'](Robot(1),settings)['status']=='lift_motion_returned'
    assert namespace['surface_pick'](Robot(2),settings)['status']=='target_remained_at_pick_location'
    assert namespace['surface_pick'](Robot(3),dict(settings,lift_feedback_gate=False))['status']=='lift_motion_returned'
    picked['lift']['measured_pose']=[0,0,.94]
    robot=Robot(3)
    assert namespace['surface_pick'](robot,settings)['lift_feedback']['status']=='hand_lift_not_confirmed'
    assert robot.calls==[]


def test_reject_double_wrapping():
    source=add_bread_lift_feedback('def surface_pick(robot, memory):\n return {}\n')
    with pytest.raises(ValueError): add_bread_lift_feedback(source)
