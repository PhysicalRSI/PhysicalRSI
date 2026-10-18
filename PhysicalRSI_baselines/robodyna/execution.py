"""Native motion execution for shared world-frame interception targets.

Loaded directly by the isolated Python 3.10 worker: no PhysicalRSI/Core import.
Targets are metres in the simulator world frame. Actor poses and tool geometry
are privileged simulator observations, not camera estimates. No trajectory,
landing oracle, success latch, actor pose, or body velocity is written here.
Each call runs one motion phase (a native grasp is a small action sequence).
A new executor must be constructed after every environment reset.
"""
from math import isfinite, sqrt


class InterceptionExecutor:
    """Cup pick/place or closed-gripper contact push; prediction stays upstream.

    Valley requires native interactive setup to make its box dynamic. Set
    ``env._interactive_robot_mode = True`` before ``setup_demo``; the official
    setup performs the transition once. This executor never changes physics.
    """

    TASKS = ("catch_ramp_ball", "catch_valley_ball")

    def __init__(self, env, task):
        if task not in self.TASKS:
            raise ValueError("Unsupported native execution task: " + str(task))
        self.env, self.task = env, task
        self.phase = "grasp" if task == self.TASKS[0] else "close"
        actor = env.cup if task == self.TASKS[0] else env.bowl
        self.arm = "right" if float(actor.get_pose().p[0]) >= 0 else "left"
        self.calls = 0
        self._behind = None
        self._push_z = None
        if task == self.TASKS[1] and not getattr(env, "_push_active", False):
            raise RuntimeError("Valley box is frozen: use native interactive setup before reset")

    @staticmethod
    def _position(actor):
        return [float(x) for x in actor.get_pose().p]

    def _move(self, action):
        self.env.move(action)
        if not self.env.plan_success:
            self.phase = "failed"
            return False
        return True

    def _tcp(self, xy, z):
        # Official native top-down gripper orientation, not an interception rule.
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
        pose = [float(xy[0]), float(xy[1]), float(z)] + list(GRASP_DIRECTION_DIC["top_down"])
        return self._move(self.env.move_to_pose(self.arm, pose))

    def _motion_evidence(self):
        actor = self.env.cup if self.task == self.TASKS[0] else self.env.bowl
        pose = actor.get_pose()
        tcp = (self.env.robot.get_left_tcp_pose() if self.arm == "left"
               else self.env.robot.get_right_tcp_pose())
        return {"catcher_position_m": [float(x) for x in pose.p],
                "catcher_quaternion_wxyz": [float(x) for x in pose.q],
                "tcp_pose_world": [float(x) for x in tcp]}

    def execute_target(self, target_position_m):
        target = [float(x) for x in target_position_m]
        if len(target) != 3 or not all(isfinite(x) for x in target):
            raise ValueError("Target must contain three finite world-frame metres")
        before = self.phase
        if self.task == self.TASKS[0] and self.calls == 0:
            # Choose the work side from the supplied prediction, never a native
            # landing oracle. Keep that arm for the entire pick/place sequence.
            self.arm = "right" if target[0] > 0 else "left"
        evidence_before = self._motion_evidence()
        self.calls += 1
        if self.phase not in ("failed", "done"):
            if not self.env.plan_success:
                self.phase = "failed"
            elif self.task == self.TASKS[0]:
                self._cup(target)
            else:
                self._box(target)
        evidence_after = self._motion_evidence()
        return {"phase_before": before, "phase": self.phase, "arm": self.arm,
                "calls": self.calls, "plan_success": bool(self.env.plan_success),
                "target_position_m": target,
                "motion_before": evidence_before, "motion_after": evidence_after,
                "catcher_displacement_m": [evidence_after["catcher_position_m"][i] -
                    evidence_before["catcher_position_m"][i] for i in range(3)],
                "grasp_verified": None,
                "grasp_evidence_note": "Planner completion does not verify physical attachment",
                "privileged_state": ["catcher_pose", "robot_tcp_pose", "tool_geometry"]}

    def _cup(self, target):
        env = self.env
        if self.phase == "grasp":
            if self._move(env.grasp_actor(env.cup, arm_tag=self.arm, pre_grasp_dis=0.08)):
                self.phase = "lift"
        elif self.phase == "lift":
            if self._move(env.move_by_displacement(arm_tag=self.arm, z=0.12, move_axis="arm")):
                self.phase = "place"
        elif self.phase == "place":
            now = self._position(env.cup)
            # Cup rests on the table. Target z is the predicted ball centre, not
            # a command to raise the receptacle through the ramp or floor.
            aim = [target[0], target[1], float(env.table_top + env.CUP_CENTER_Z)]
            delta = [aim[i] - now[i] for i in range(3)]
            # Follow the official low-level place/release semantics once. A
            # slipped cup must not cause an unbounded chase by the robot TCP.
            if self._move(env.move_by_displacement(arm_tag=self.arm,
                         x=delta[0], y=delta[1], z=delta[2], move_axis="world")):
                self.phase = "release"
        elif self.phase == "release":
            if self._move(env.open_gripper(self.arm)):
                self.phase = "retreat"
        elif self.phase == "retreat":
            if self._move(env.move_by_displacement(arm_tag=self.arm, z=0.12, move_axis="arm")):
                self.phase = "done"

    def _box(self, target):
        env = self.env
        now = self._position(env.bowl)
        delta = [target[i] - now[i] for i in range(2)]
        distance = sqrt(sum(x*x for x in delta))
        if self.phase == "close":
            if self._move(env.close_gripper(self.arm)):
                self.phase = "approach"
        elif self.phase == "approach":
            if distance <= 0.015:
                return
            direction = [x / distance for x in delta]
            half = sum(abs(direction[i]) * float(env._box_half_xy[i]) for i in range(2))
            self._behind = [now[i] - direction[i]*(half + float(env.PUSH_BEHIND_STANDOFF)) for i in range(2)]
            # Native fingertip geometry measurement does not change task state.
            self._push_z = float(env.table_top + env.PUSH_FINGER_HEIGHT_FRAC * env._box_height + env._finger_drop(self.arm))
            if self._tcp(self._behind, env.table_top + 0.18):
                self.phase = "lower"
        elif self.phase == "lower":
            if self._tcp(self._behind, self._push_z):
                self.phase = "push"
        elif self.phase == "push":
            if distance <= 0.015:
                # Keep the body fully dynamic; no snap, brake, freeze, or score flag.
                if self._tcp(now[:2], env.table_top + 0.18):
                    self.phase = "approach"
                return
            direction = [x / distance for x in delta]
            half = sum(abs(direction[i]) * float(env._box_half_xy[i]) for i in range(2))
            advance = min(0.04, distance)
            aim = [now[i] - direction[i]*(half + float(env.PUSH_CONTACT_GAP) - advance) for i in range(2)]
            # A single small shove; the next target uses new observed motion.
            self._tcp(aim, self._push_z)
