from types import SimpleNamespace

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_proprioception import measured_robot_state


class Robot:
    available_arms = ['left']

    def __init__(self):
        self.control_callbacks = {'get_joint_positions': lambda *a: [.1, .2],
                                  'get_link_world_pose': lambda *a: [1, 2, 3, 1, 0, 0, 0]}

    def _control_is_ready(self, side):
        return True

    def get_arm_info(self, side):
        return SimpleNamespace(robot_name='robot', arm_joint_names=['a', 'b'],
                               move_group='hand', private_scene_data='excluded')

    def end_link_pose_to_task_pose(self, pose, side):
        pose[0] += .1
        return pose

    def get_left_gripper_val(self):
        return .5

    def get_arm_state(self, side):
        raise AssertionError('Cached fallback accessor must not be used')

    def get_arm_pose(self, side):
        raise AssertionError('Cached fallback accessor must not be used')


def test_live_sensor_values_and_command_distinction():
    result = measured_robot_state(Robot())
    arm = result['arms']['left']
    assert arm['measured_joint_positions'] == [.1, .2]
    assert arm['measured_end_pose'] == [1.1, 2., 3., 1., 0., 0., 0.]
    assert arm['gripper_command_fraction'] == .5
    assert not arm['gripper_aperture_measured']
    assert 'private_scene_data' not in str(result)


@pytest.mark.parametrize('key', ['get_joint_positions', 'get_link_world_pose'])
def test_missing_live_sensor_does_not_fall_back(key):
    robot = Robot()
    robot.control_callbacks[key] = lambda *args: None
    with pytest.raises(RuntimeError, match='unavailable'):
        measured_robot_state(robot)


def test_invalid_joint_measurement_and_unready_backend_rejected():
    robot = Robot()
    robot.control_callbacks['get_joint_positions'] = lambda *args: [float('nan'), 0]
    with pytest.raises(ValueError, match='measurement'):
        measured_robot_state(robot)
    robot._control_is_ready = lambda side: False
    with pytest.raises(RuntimeError, match='not ready'):
        measured_robot_state(robot)
