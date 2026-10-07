from PhysicalRSI_baselines.embodied_goodharts_law.scene_trajectory_execution import execute_scene_trajectory


class Robot:
    def __init__(self, *, start=0., converged=True, endpoint=.5):
        self.start, self.converged, self.endpoint = start, converged, endpoint
        self.moves = []

    def get_robot_state(self):
        return {'robot_joint_pos': [self.start]*7,
                'robot_cartesian_pos': [self.endpoint, 0., .2, 0., 1., 0., 0.]}

    def move_to_joints(self, joints, **kwargs):
        assert kwargs['tolerance'] == .01
        self.moves.append(joints)
        return {'converged': self.converged}


PLAN = {'status': 'planned', 'trajectory': [[0.]*7, [.1]*7]}
TARGET = [.5, 0., .2, 0., 1., 0., 0.]


def test_stale_plan_never_moves_robot():
    robot = Robot(start=.1)
    assert execute_scene_trajectory(robot, PLAN, TARGET)['status'] == 'stale_plan_start'
    assert robot.moves == []


def test_tracking_failure_stops_before_next_waypoint():
    robot = Robot(converged=False)
    result = execute_scene_trajectory(robot, PLAN, TARGET)
    assert result['status'] == 'joint_tracking_failed' and len(robot.moves) == 1


def test_joint_success_does_not_override_cartesian_error():
    result = execute_scene_trajectory(Robot(endpoint=.51), PLAN, TARGET)
    assert result['status'] == 'endpoint_not_reached'
    assert not result['execution_verified']


def test_reached_endpoint_is_not_a_verified_grasp():
    result = execute_scene_trajectory(Robot(), PLAN, TARGET)
    assert result['execution_verified'] and not result['grasp_verified']


def test_native_nonconvergence_preserves_measurement_and_stops():
    class NonconvergingRobot(Robot):
        def move_to_joints(self, joints, **kwargs):
            self.moves.append(joints)
            raise RuntimeError('Joint primitive did not converge within its step allowance')
    robot = NonconvergingRobot()
    result = execute_scene_trajectory(robot, PLAN, TARGET)
    assert result['status'] == 'joint_tracking_failed'
    assert result['waypoint'] == 0 and len(robot.moves) == 1
    assert result['measured_robot_state'] == robot.get_robot_state()


def test_subsample_keeps_final_waypoint_and_cartesian_acceptance():
    robot = Robot(endpoint=.51)
    path = [[i*.01]*7 for i in range(6)]
    result = execute_scene_trajectory(robot, {'status': 'planned', 'trajectory': path}, TARGET, subsample=2)
    assert robot.moves == [path[i] for i in [0, 2, 4, 5]]
    assert result['executed_waypoints'] == 4
    assert not result['execution_verified']
