import pytest
from test_egl_calibrated_wrist_view import Robot, MEMORY
from test_egl_support_reference import surface
from PhysicalRSI_baselines.embodied_goodharts_law.calibrated_wrist_return import refine_clipped_target

class ReturnRobot(Robot):
    def __init__(self, error=0., query_error=False):
        super().__init__(); self.error=error; self.returned=False; self.query_error=query_error
    def get_robot_state(self):
        return dict(super().get_robot_state(),robot_joint_pos=[self.error if self.returned else 0.]*7+[1.])
    def move_to_joints(self,joints):
        assert joints == [0.]*7
        self.returned=True
        return {'converged':True}
    def locate_objects(self,*args,**kwargs):
        if self.query_error: raise RuntimeError('camera unavailable')
        return super().locate_objects(*args,**kwargs)

def test_refinement_restores_measured_seven_joints_before_return():
    robot=ReturnRobot()
    result=refine_clipped_target(robot,MEMORY,surface(.6,.25,clipped=True))
    assert robot.returned and result['status']=='estimated_surface'
    assert result['observation_return']['joint_error']==0.

def test_measured_return_failure_overrides_successful_detection_and_command():
    robot=ReturnRobot(error=.02)
    assert refine_clipped_target(robot,MEMORY,surface(.6,.25,clipped=True))['status']=='target_view_return_failed'

def test_camera_exception_still_restores_joints_and_remains_failure():
    robot=ReturnRobot(query_error=True)
    with pytest.raises(RuntimeError,match='camera unavailable'):
        refine_clipped_target(robot,MEMORY,surface(.6,.25,clipped=True))
    assert robot.returned
