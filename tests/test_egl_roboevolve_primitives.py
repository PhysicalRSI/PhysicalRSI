import time
from types import SimpleNamespace

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_primitives import RoboEvolveJointPrimitives


class Robot:
    available_arms = ['left']
    control_callbacks = {'get_joint_positions': lambda *args: [.1, .2],
                         'get_link_world_pose': lambda *args: [1, 2, 3, 1, 0, 0, 0]}

    def get_arm_info(self, arm):
        return SimpleNamespace(robot_name='robot', arm_joint_names=['a', 'b'], move_group='hand')

    def _control_is_ready(self, arm):
        return True

    def end_link_pose_to_task_pose(self, pose, arm):
        return pose

    def get_left_gripper_val(self):
        return .5


def api(applier, budget=1):
    return RoboEvolveJointPrimitives(SimpleNamespace(robot=Robot()),
        joint_limits={'left': [[-1, 1], [-2, 2]]}, max_control_actions=budget,
        action_applier=applier)


def call(primitive, values):
    return primitive.handlers['apply_joint_action']([values], {}, deadline=time.monotonic()+10)


def test_false_native_return_is_not_convergence_or_success_and_consumes_budget():
    received = []
    def apply(env, values, kind, **kwargs):
        received.append((values, kind, kwargs))
        return False
    primitive = api(apply)
    result = call(primitive, [.2, .3, .5])
    assert received == [([.2, .3, .5], 'joint', {'render': True})]
    assert not result['control_applied']
    assert not result['convergence_verified'] and not result['task_success_claimed']
    with pytest.raises(RuntimeError, match='budget'):
        call(primitive, [.2, .3, .5])


def test_invalid_actions_never_reach_host_control():
    primitive = api(lambda *a, **k: pytest.fail('Invalid action reached native control'))
    for values in [[2, 0, .5], [0, 0, 2], [0, 0], [float('nan'), 0, .5]]:
        with pytest.raises(ValueError):
            call(primitive, values)
    assert not primitive.trace


def test_exception_keeps_attempt_and_trace_is_immutable_copy():
    def fail(*args, **kwargs):
        raise RuntimeError('native failure')
    primitive = api(fail)
    with pytest.raises(RuntimeError, match='native failure'):
        call(primitive, [0, 0, 1])
    trace = primitive.trace
    assert trace[0]['state'] == 'raised'
    trace.clear()
    assert len(primitive.trace) == 1
    with pytest.raises(RuntimeError, match='budget'):
        call(primitive, [0, 0, 1])
