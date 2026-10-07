from types import SimpleNamespace
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_planning_diagnostics import RobotPlannerDiagnostics


class Planner:
    world_config = None
    def __init__(self):
        self.motion_gen = SimpleNamespace(plan_single=lambda *a, **kw: SimpleNamespace(status='IK_FAIL'))
    def plan_path(self, *args, **kwargs):
        self.motion_gen.plan_single(*args, **kwargs)
        return {'status': 'Fail'}


def test_native_failure_status_is_retained_without_changing_result():
    planner = Planner();original = planner.motion_gen
    result = RobotPlannerDiagnostics(planner).plan_path([0], [1], max_attempts=2)
    assert result['status'] == 'Fail' and result['robot_planner_statuses'] == ['IK_FAIL']
    assert planner.motion_gen is original


def test_planner_is_restored_after_exception_and_scene_maps_rejected():
    planner = Planner();original = planner.motion_gen
    def fail(*args, **kwargs):raise RuntimeError('planner error')
    planner.plan_path = fail
    with pytest.raises(RuntimeError):RobotPlannerDiagnostics(planner).plan_path()
    assert planner.motion_gen is original
    planner.world_config = {'scene': True}
    with pytest.raises(ValueError):RobotPlannerDiagnostics(planner)
