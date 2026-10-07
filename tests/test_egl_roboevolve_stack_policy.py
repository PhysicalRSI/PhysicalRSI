import copy

from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_stack_policy import MEMORY, SOURCE


def surface(z):
    return {'status': 'estimated_surface', 'detection_score': .9,
            'surface_quantiles': {'median': [.1, .1, z],
                                  'lower_05': [.08, .08, z-.02],
                                  'upper_95': [.12, .12, z+.02]}}


class Robot:
    def __init__(self, *, motion=True, visible_lift=True, detected=True):
        self.motion, self.visible_lift, self.detected = motion, visible_lift, detected
        self.moves = []
        self.actions = []
        self.queries = []
        self.counts = {}

    def locate_object(self, prompt, *, camera):
        self.queries.append((prompt, camera))
        if not self.detected:
            return {'status': 'ambiguous'}
        count = self.counts.get(prompt, 0)
        self.counts[prompt] = count+1
        lifted = len(self.moves) % 6 == 3 and self.visible_lift
        return surface(.85 if lifted else .7)

    def locate_objects(self, prompt, *, camera):
        self.queries.append((prompt, camera))
        value = surface(.66)
        value['surface_quantiles']['lower_05'][2] = .655
        value['surface_quantiles']['upper_95'][2] = .665
        value['detection_score'] = .385
        return {'objects': [value]}

    def get_robot_state(self):
        return {'arms': {side: {'measured_joint_positions': [0.],
                               'measured_end_pose': [0, 0, 1, 1, 0, 0, 0],
                               'gripper_command_fraction': 1.}
                         for side in ['left', 'right']}}

    def move_to_pose(self, arm, pose):
        self.moves.append((arm, pose))
        return {'reached': self.motion}

    def apply_joint_action(self, action):
        self.actions.append(action)
        return {'control_applied': True}


def run(robot):
    scope = {}
    exec(SOURCE, scope)
    before = copy.deepcopy(MEMORY)
    result = scope['policy'](robot, MEMORY)
    assert MEMORY == before
    assert not result['task_success_claimed']
    return result


def test_missing_or_ambiguous_table_never_causes_actuation():
    robot = Robot(detected=False)
    assert run(robot)['status'] == 'red_anchor_not_located'
    assert not robot.moves and not robot.actions


def test_failed_motion_stops_before_close_or_lift():
    robot = Robot(motion=False)
    assert run(robot)['status'] == 'pick_approach_failed'
    assert len(robot.moves) == 1
    assert all(action[1] == 1 for action in robot.actions)


def test_missing_visible_lift_does_not_continue_to_placement():
    robot = Robot(visible_lift=False)
    assert run(robot)['status'] == 'visible_lift_not_verified'
    assert len(robot.moves) == 3
    assert not any(prompt == 'green block' for prompt, _ in robot.queries)


def test_sequence_reacquires_red_before_green_transfer_without_success_claim():
    robot = Robot()
    assert run(robot)['status'] == 'stack_sequence_completed'
    prompts = [prompt for prompt, _ in robot.queries]
    assert prompts == ['red block', 'table', 'red block', 'red block', 'red block', 'green block', 'green block']
    assert len(robot.moves) == 12


def test_table_grounding_rejects_floor_and_nonplanar_surface():
    for height in [.0, .66]:
        robot = Robot()
        robot.locate_objects = lambda prompt, camera: {'objects': [surface(height)]}
        assert run(robot)['status'] == 'table_not_located'
        assert not robot.moves and not robot.actions


def test_table_grounding_rejects_multiple_eligible_surfaces():
    robot = Robot()
    original = robot.locate_objects
    robot.locate_objects = lambda prompt, camera: {'objects': original(prompt, camera=camera)['objects'] * 2}
    assert run(robot)['status'] == 'table_not_located'
    assert not robot.moves and not robot.actions


def test_table_grounding_rejects_plane_outside_visible_block():
    robot = Robot()
    original = robot.locate_objects
    def shifted(prompt, *, camera):
        reply = original(prompt, camera=camera)
        for point in reply['objects'][0]['surface_quantiles'].values():
            point[0] += 1
        return reply
    robot.locate_objects = shifted
    assert run(robot)['status'] == 'table_not_located'
    assert not robot.moves and not robot.actions


def test_downward_tcp_calibration_is_applied_to_pick_and_place():
    robot = Robot()
    assert run(robot)['status'] == 'stack_sequence_completed'
    import pytest
    for index in [1, 4]:
        pose = robot.moves[index][1]
        w, x, y, z = pose[3:]
        axis = [1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)]
        assert axis == pytest.approx([0, 0, -1])
        tcp_z = pose[2] + MEMORY['task_ee_to_tcp_m'] * axis[2]
        expected = .72 - MEMORY['grasp_below_surface_m'] if index == 1 else .665 + .04 - MEMORY['grasp_below_surface_m'] + MEMORY['release_clearance_m']
        assert tcp_z == pytest.approx(expected)


def test_first_hover_planning_failure_tries_other_arm():
    class FallbackRobot(Robot):
        def __init__(self):
            super().__init__()
            self.attempted_arms = []

        def move_to_pose(self, arm, pose):
            self.attempted_arms.append(arm)
            if len(self.attempted_arms) == 1:
                return {'reached': False, 'status': 'planning_failed'}
            return super().move_to_pose(arm, pose)

    robot = FallbackRobot()
    assert run(robot)['status'] == 'stack_sequence_completed'
    assert robot.attempted_arms[:2] == ['left', 'right']


def test_both_arms_planning_failure_stops_without_close():
    robot = Robot()
    attempted = []
    def move(arm, pose):
        attempted.append(arm)
        return {'reached': False, 'status': 'planning_failed'}
    robot.move_to_pose = move
    assert run(robot)['status'] == 'both_arms_planning_failed'
    assert attempted == ['left', 'right']
    assert all(action[1] == 1 and action[3] == 1 for action in robot.actions)


def test_transfer_planning_failure_tries_other_downward_orientation():
    class TransferRobot(Robot):
        def __init__(self):
            super().__init__()
            self.failed_pose = None
            self.alternate_pose = None

        def move_to_pose(self, arm, pose):
            if len(self.moves) == 3 and self.failed_pose is None:
                self.failed_pose = pose
                return {'reached': False, 'status': 'planning_failed'}
            if len(self.moves) == 3:
                self.alternate_pose = pose
            return super().move_to_pose(arm, pose)

    robot = TransferRobot()
    assert run(robot)['status'] == 'stack_sequence_completed'
    assert robot.failed_pose[:3] == robot.alternate_pose[:3]
    assert robot.failed_pose[3:] != robot.alternate_pose[3:]
    assert robot.moves[4][1][3:] == robot.alternate_pose[3:]


def test_transfer_tracking_failure_does_not_try_another_orientation():
    robot = Robot()
    original = robot.move_to_pose
    failures = []
    def move(arm, pose):
        if len(robot.moves) == 3:
            failures.append(pose)
            return {'reached': False, 'status': 'tracking_failed'}
        return original(arm, pose)
    robot.move_to_pose = move
    assert run(robot)['status'] == 'placement_motion_failed'
    assert len(failures) == 1


def test_transfer_clearance_changes_only_destination_hover_and_retreat():
    robot = Robot()
    assert run(robot)['status'] == 'stack_sequence_completed'
    import pytest
    for offset in [0, 6]:
        poses = [pose for arm, pose in robot.moves[offset:offset+6]]
        assert poses[0][2] - poses[1][2] == pytest.approx(MEMORY['clearance_m'])
        assert poses[3][2] - poses[4][2] == pytest.approx(MEMORY['transfer_clearance_m'])
        assert poses[5] == poses[3]
    assert MEMORY['transfer_clearance_m'] == .06


def test_transfer_can_use_quarter_turn_after_two_ik_failures():
    robot=Robot()
    original=robot.move_to_pose
    attempted=[]
    def move(arm,pose):
        if len(robot.moves)==3:
            attempted.append(pose)
            if len(attempted)<=2:
                return {'reached':False,'status':'planning_failed'}
        return original(arm,pose)
    robot.move_to_pose=move
    assert run(robot)['status']=='stack_sequence_completed'
    assert len(attempted)==3
    assert all(p[:3]==attempted[0][:3] for p in attempted)
    w,x,y,z=attempted[-1][3:]
    assert [1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)]==[0.,0.,-1.]
    assert robot.moves[4][1][3:]==attempted[-1][3:]
