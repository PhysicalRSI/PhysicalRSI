"""Read live robot sensors without RoboEvolve's cached-state fallbacks."""
import numpy as np


def measured_robot_state(robot):
    """Trusted host adapter; robot model/calibration are declared public inputs.

    The native gripper accessor is a command cache, not a measured aperture.
    Label it explicitly and never use it as grasp-success evidence.
    """
    get_joints = robot.control_callbacks.get('get_joint_positions')
    get_pose = robot.control_callbacks.get('get_link_world_pose')
    if not callable(get_joints) or not callable(get_pose):
        raise RuntimeError('Live robot sensor callbacks are required')
    arms = {}
    for side in robot.available_arms:
        name = str(side)
        if name not in ('left', 'right') or name in arms:
            raise ValueError('Unexpected robot arm configuration')
        if not robot._control_is_ready(side):
            raise RuntimeError('Robot control backend is not ready')
        info = robot.get_arm_info(side)
        link = info.move_group or robot._link_name_from_joint(info.ee_joint_name)
        if not link:
            raise RuntimeError('Robot end-link sensor is not configured')
        joint_values = get_joints(info.robot_name, list(info.arm_joint_names))
        link_pose = get_pose(info.robot_name, link)
        if joint_values is None or link_pose is None:
            raise RuntimeError('Live robot measurement is unavailable')
        joints = np.asarray(joint_values, dtype=float)
        pose = np.asarray(link_pose, dtype=float)
        if (joints.shape != (len(info.arm_joint_names),) or not len(joints)
                or not np.isfinite(joints).all() or pose.shape != (7,)
                or not np.isfinite(pose).all()
                or abs(np.linalg.norm(pose[3:])-1) > 1e-5):
            raise ValueError('Invalid live robot measurement')
        endpose = np.asarray(robot.end_link_pose_to_task_pose(pose.copy(), side), dtype=float)
        if (endpose.shape != (7,) or not np.isfinite(endpose).all()
                or abs(np.linalg.norm(endpose[3:])-1) > 1e-5):
            raise ValueError('Invalid calibrated robot end pose')
        gripper = float(getattr(robot, 'get_'+name+'_gripper_val')())
        if not np.isfinite(gripper) or not 0 <= gripper <= 1:
            raise ValueError('Invalid gripper command state')
        arms[name] = {'measured_joint_positions': joints.tolist(),
                      'measured_end_pose': endpose.tolist(),
                      'gripper_command_fraction': gripper,
                      'gripper_aperture_measured': False}
    if not arms:
        raise ValueError('No configured robot arms')
    return {'arms': arms, 'pose_frame': 'world', 'pose_format': 'xyz+wxyz',
            'scope': 'robot-sensors-and-declared-calibration'}
