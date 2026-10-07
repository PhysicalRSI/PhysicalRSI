"""Trusted CAP callbacks using RoboTwin's unmodified evaluation action surface.

Keep the task object in the host. This is not a sandbox or a success evaluator.
A process supervisor must bound native planners and take_action, whose internal
physics loop cannot be interrupted by these before/after deadline checks.
"""
import math
import time


class NativeRoboTwinPrimitives:
    def __init__(self, task, *, max_native_actions):
        import numpy as np

        if type(max_native_actions) is not int or max_native_actions < 1:
            raise ValueError("Declare a positive native-action budget")
        self._task = task
        self._budget = max_native_actions
        self._attempts = 0
        self._trace = []
        self._deadline = None
        self._limits = {}
        self._home = {}
        for arm in ("left", "right"):
            joints = getattr(task.robot, arm + "_arm_joints")
            limits = np.asarray([joint.get_limits()[0] for joint in joints])
            if limits.shape != (len(joints), 2) or np.isnan(limits).any():
                raise ValueError("Invalid native joint limits")
            self._limits[arm] = limits
            self._home[arm] = self._measured_joints(arm)

    @property
    def trace(self):
        import copy
        return copy.deepcopy(self._trace)

    def _check_deadline(self):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("RoboTwin primitive deadline reached")

    def _arm(self, arm):
        if arm not in ("left", "right"):
            raise ValueError("Expected left or right arm")
        return arm

    def _measured_joints(self, arm):
        # Upstream jointState reads drive targets; real_jointState reads qpos.
        return [float(value) for value in getattr(self._task.robot, f"get_{arm}_arm_real_jointState")()[:-1]]

    def _joint_target(self):
        return list(self._task.robot.get_left_arm_jointState()) + list(self._task.robot.get_right_arm_jointState())

    def _take_action(self, values, action_type):
        import numpy as np

        self._check_deadline()
        if self._attempts >= self._budget:
            raise RuntimeError("Native-action budget exhausted")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite action")
        self._attempts += 1
        before = self._task.take_action_cnt
        row = {"attempt": self._attempts, "action_type": action_type,
               "action": [float(x) for x in values], "native_counter_before": before,
               "state": "started"}
        self._trace.append(row)
        try:
            self._task.take_action(values, action_type=action_type)
            row.update(state="returned", native_counter_after=self._task.take_action_cnt)
        except BaseException:
            row["state"] = "raised"
            raise
        self._check_deadline()
        # Do not expose task success or interpret a native return as convergence.

    def move_to_joints(self, arm, joints):
        import numpy as np

        arm = self._arm(arm)
        target = np.asarray(joints, dtype=float)
        limits = self._limits[arm]
        if target.shape != (len(limits),) or not np.isfinite(target).all():
            raise ValueError("Invalid target joint vector")
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise ValueError("Target exceeds native joint limits")
        action = self._joint_target()
        start = 0 if arm == "left" else len(self._limits["left"]) + 1
        action[start:start + len(target)] = target.tolist()
        self._take_action(action, "qpos")
        measured = self._measured_joints(arm)
        return {"measured_joint_positions": measured,
                "joint_error": float(np.linalg.norm(np.asarray(measured) - target))}

    def move_to_pose(self, arm, pose):
        import numpy as np

        arm = self._arm(arm)
        target = np.asarray(pose, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all() or abs(np.linalg.norm(target[3:]) - 1) > 1e-3:
            raise ValueError("Expected finite xyz and unit wxyz quaternion")
        values = []
        for side in ("left", "right"):
            values.extend(target.tolist() if side == arm else self._task.get_arm_pose(side))
            values.append(getattr(self._task.robot, f"get_{side}_gripper_val")())
        self._take_action(values, "ee")
        measured = np.asarray(self._task.get_arm_pose(arm))
        return {"measured_pose": measured.tolist(),
                "position_error": float(np.linalg.norm(measured[:3] - target[:3])),
                "orientation_error_radians": float(2 * np.arccos(np.clip(abs(np.dot(measured[3:], target[3:])), 0, 1)))}

    def set_gripper(self, arm, fraction):
        arm = self._arm(arm)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Expected opening fraction in [0,1]")
        action = self._joint_target()
        index = len(self._limits["left"]) if arm == "left" else len(action) - 1
        action[index] = float(fraction)
        self._take_action(action, "qpos")
        # Upstream's normalized gripper value is commanded, not measured.
        return {"commanded_opening_fraction": float(getattr(self._task.robot, f"get_{arm}_gripper_val")())}

    def get_observation(self):
        observation = self._task.get_obs()
        cameras = {name: {key: value for key, value in channels.items()
                          if key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv", "cam2world_gl")}
                   for name, channels in observation["observation"].items()}
        return {"cameras": cameras, **self.get_proprioception()}

    def get_proprioception(self):
        return {"measured_joint_positions": {arm: self._measured_joints(arm) for arm in ("left", "right")},
                "endpose": {arm: self._task.get_arm_pose(arm) for arm in ("left", "right")},
                "commanded_gripper_fraction": {arm: getattr(self._task.robot, f"get_{arm}_gripper_val")() for arm in ("left", "right")}}

    @property
    def handlers(self):
        def plain(value):
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value.tolist() if hasattr(value, "tolist") else value

        def callback(function):
            def call(args, kwargs, *, deadline):
                if not math.isfinite(deadline):
                    raise ValueError("A finite primitive deadline is required")
                self._deadline = deadline
                try:
                    self._check_deadline()
                    result = plain(function(*args, **kwargs))
                    self._check_deadline()
                    return result
                finally:
                    self._deadline = None
            return call

        return {"get_observation": callback(self.get_observation),
                "get_proprioception": callback(self.get_proprioception),
                "move_to_joints": callback(self.move_to_joints),
                "move_to_pose": callback(self.move_to_pose),
                "open_gripper": callback(lambda arm: self.set_gripper(arm, 1)),
                "close_gripper": callback(lambda arm: self.set_gripper(arm, 0)),
                "goto_home_joint_position": callback(lambda arm: self.move_to_joints(self._arm(arm), self._home[arm]))}
