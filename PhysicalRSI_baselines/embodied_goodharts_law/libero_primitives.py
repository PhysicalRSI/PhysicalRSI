"""Trusted native primitives for original LIBERO and LIBERO-Plus.

Keep this object in the simulator process. Generated Python must receive only
admitted RPC results, never this object or its simulator handle. This module is
not a sandbox, an ASPIRE environment replacement, or a benchmark evaluator.
"""
import math
import time


class NativeLiberoPrimitives:
    """Bounded joint/gripper control and ASPIRE-shaped public observations.

    Camera conventions follow ASPIRE's reviewed FrankaLiberoEnv
    (f4c8939aab0af9b97690c561bd80e282940f7886). The Cartesian pose
    uses the native right_hand body, not a fixed offset from the gripper EEF.
    Object poses, segmentation labels and evaluator outcomes are excluded.
    """

    def __init__(self, environment, observation, *, max_physics_steps, pose_solver=None, segmenter=None,
                 grasp_planner=None):
        import numpy as np

        if type(max_physics_steps) is not int or max_physics_steps < 1:
            raise ValueError("Declare a positive physics-step budget")
        self._environment = environment
        self._observation = observation
        self._remaining = max_physics_steps
        self._steps = 0
        self._gripper_fraction = 1.0
        self._trace = []
        self._deadline = None
        self._pose_solver = pose_solver
        self._segmenter = segmenter
        self._grasp_planner = grasp_planner
        self._pose_trace = []
        self.home_joint_position = np.asarray(observation["robot0_joint_pos"]).copy()
        controller = environment.env.robots[0].controller
        # The delta formula below is valid only for this admitted controller.
        if (controller.name != "JOINT_POSITION" or controller.impedance_mode != "fixed" or controller.control_dim != 7 or
                not np.allclose(controller.input_max, 1) or not np.allclose(controller.input_min, -1) or
                not np.allclose(controller.output_max, 0.05) or not np.allclose(controller.output_min, -0.05)):
            raise ValueError("Unsupported joint-controller contract")
        model = environment.sim.model
        self._joint_limits = np.asarray([
            model.jnt_range[model.joint_name2id(f"robot0_joint{i}")]
            for i in range(1, 8)
        ])

    @property
    def physics_steps(self):
        return self._steps

    @property
    def trace(self):
        # Return plain copies for evidence storage, not mutable controller state.
        return [{**row, "action": list(row["action"])} for row in self._trace]

    def _step(self, action):
        import numpy as np

        self._check_deadline()
        if self._remaining <= 0:
            raise RuntimeError("Primitive physics-step budget exhausted")
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (8,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
            raise ValueError("Expected eight finite normalized controls")
        self._remaining -= 1
        observation, _, _, _ = self._environment.step(value)
        self._steps += 1
        self._observation = observation
        self._trace.append({"step": self._steps, "action": value.tolist()})
        self._check_deadline()

    def _check_deadline(self):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("Native primitive deadline reached")

    @property
    def pose_trace(self):
        import copy
        return copy.deepcopy(self._pose_trace)

    @property
    def handlers(self):
        """Explicit host callbacks for physicalRSI's isolated code-policy bridge.

        The surrounding simulator supervisor must also impose a process deadline
        for a native call that hangs. These callbacks check between physics steps.
        """
        def plain(value):
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if hasattr(value, "tolist"):
                return value.tolist()
            return value

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

        handlers = {"get_observation": callback(self.get_observation),
                "get_robot_state": callback(self.get_robot_state),
                "move_to_joints": callback(self.move_to_joints_blocking),
                "open_gripper": callback(lambda: self.set_gripper(1)),
                "close_gripper": callback(lambda: self.set_gripper(0)),
                "goto_home_joint_position": callback(lambda: self.move_to_joints_blocking(self.home_joint_position))}
        if self._pose_solver is not None:
            handlers["move_to_pose"] = callback(self.move_to_pose_blocking)
            handlers["try_move_to_pose"] = callback(self.try_move_to_pose)
        if self._segmenter is not None:
            handlers["segment_objects"] = callback(self.segment_objects)
        if getattr(self, '_grasp_planner', None) is not None:
            handlers["plan_grasps"] = callback(self.plan_grasps)
        return handlers

    def get_robot_state(self):
        """Compact public proprioception for policies using host perception tools."""
        observation = self.get_observation()
        return {key: observation[key] for key in ('robot_joint_pos', 'robot_cartesian_pos')}

    def plan_grasps(self, prompt, camera='agentview'):
        if self._grasp_planner is None or self._deadline is None:
            raise RuntimeError('A configured grasp planner and callback deadline are required')
        self._check_deadline()
        result = self._grasp_planner(self.get_observation(), prompt,
                                     camera=camera, deadline=self._deadline)
        self._check_deadline()
        return result

    def segment_objects(self, prompt, camera="agentview"):
        """Segment a public camera image by text; detections are fallible estimates."""
        if camera not in {"agentview", "robot0_eye_in_hand"}:
            raise ValueError("Unknown public camera")
        if self._segmenter is None or self._deadline is None:
            raise RuntimeError("A configured segmenter and callback deadline are required")
        self._check_deadline()
        image = self.get_observation()[camera]["images"]["rgb"]
        return self._segmenter(image, prompt, deadline=self._deadline)

    def move_to_pose_blocking(self, pose):
        """Reach native hand xyz+wxyz in robot base frame; no collision planning.

        Solver configuration and convergence limits are host-owned. Failure to
        reach the Cartesian tolerance remains an error even if joints converge.
        """
        import numpy as np
        from scipy.spatial.transform import Rotation

        target = np.asarray(pose, dtype=float)
        if (target.shape != (7,) or not np.isfinite(target).all()
                or abs(np.linalg.norm(target[3:]) - 1) > 1e-5):
            raise ValueError("Expected finite hand xyz and unit quaternion wxyz")
        if self._pose_solver is None or self._deadline is None:
            raise RuntimeError("A configured solver and callback deadline are required")
        self._check_deadline()
        if self._remaining <= 0:
            raise RuntimeError("Primitive physics-step budget exhausted")
        current = self.get_observation()
        solution = self._pose_solver(pose=target, joints=current["robot_joint_pos"][:7],
                                    gripper_fraction=current["robot_joint_pos"][7],
                                    deadline=self._deadline)
        self._check_deadline()
        row = {"target": target.tolist(), "solver_result": solution,
               "start_physics_step": self._steps, "state": "pending"}
        self._pose_trace.append(row)
        def at_cartesian_goal():
            actual = self.get_observation()["robot_cartesian_pos"]
            position_error = float(np.linalg.norm(actual[:3] - target[:3]))
            orientation_error = float((Rotation.from_quat(target[[4, 5, 6, 3]]).inv() *
                Rotation.from_quat(actual[[4, 5, 6, 3]])).magnitude())
            return position_error <= .001 and orientation_error <= .005

        try:
            try:
                motion = self._track_joint_target(solution["joints"], tolerance=.002,
                                                  goal=at_cartesian_goal)
            except RuntimeError as exc:
                if str(exc) != "Joint primitive did not converge within its step allowance":
                    raise
                # Joint tracking is a means of reaching the requested hand pose.
                # Preserve the exhausted joint criterion, then measure the actual
                # Cartesian goal before deciding whether this pose call failed.
                motion = {"joint_converged": False, "physics_steps": self._steps}
            actual = self.get_observation()["robot_cartesian_pos"]
            position_error = float(np.linalg.norm(actual[:3] - target[:3]))
            orientation_error = float((Rotation.from_quat(target[[4, 5, 6, 3]]).inv() *
                Rotation.from_quat(actual[[4, 5, 6, 3]])).magnitude())
            row.update(position_error_m=position_error, orientation_error_rad=orientation_error,
                       actual_pose=actual[:7].tolist(), end_physics_step=self._steps,
                       joint_converged=motion["joint_converged"])
            if position_error > .001 or orientation_error > .005:
                raise RuntimeError("Native hand did not reach the Cartesian tolerance")
            row["state"] = "completed"
            return {**motion, "converged": True, "position_error_m": position_error,
                    "orientation_error_rad": orientation_error}
        except BaseException as exc:
            row.update(state="failed", error=type(exc).__name__, end_physics_step=self._steps)
            raise

    def try_move_to_pose(self, pose):
        """Expose a bounded reach failure as data so a skill can decide to retry.

        Invalid requests, solver errors, exhausted budgets and deadlines remain
        exceptions. A failed reach retains its original pose trace and effects.
        """
        try:
            return {"reached": True, **self.move_to_pose_blocking(pose)}
        except RuntimeError as exc:
            if str(exc) not in (
                "Joint primitive did not converge within its step allowance",
                "Native hand did not reach the Cartesian tolerance",
            ):
                raise
            return {"reached": False, "reason": str(exc), "physics_steps": self._steps,
                    "robot_state": self.get_robot_state()}

    def move_to_joints_blocking(self, joints, *, tolerance=0.01, max_steps=120):
        return self._track_joint_target(joints, tolerance=tolerance, max_steps=max_steps)

    def _track_joint_target(self, joints, *, tolerance=0.01, max_steps=120, goal=None):
        """Track joints, optionally stopping at a host-owned Cartesian goal.

        Early Cartesian stopping requires two consecutive post-step goal
        measurements. Joint convergence alone does not end Cartesian tracking.
        """
        import numpy as np

        target = np.asarray(joints, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Expected seven finite target joint angles")
        if np.any(target < self._joint_limits[:, 0]) or np.any(target > self._joint_limits[:, 1]):
            raise ValueError("Target exceeds native robot joint limits")
        if (type(max_steps) is not int or not 1 <= max_steps <= 120 or
                not math.isfinite(tolerance) or not 0 < tolerance <= 0.01):
            raise ValueError("Invalid primitive convergence limits")
        consecutive = 0
        for _ in range(max_steps):
            current = np.asarray(self._observation["robot0_joint_pos"])
            delta = np.clip((target - current) / 0.05, -1, 1)
            self._step(np.concatenate([delta, [1 - 2 * self._gripper_fraction]]))
            error = float(np.linalg.norm(self._observation["robot0_joint_pos"] - target))
            if goal is not None:
                consecutive = consecutive + 1 if goal() else 0
                reached = consecutive >= 2
            else:
                reached = error < tolerance
            if reached:
                return {"converged": True, "joint_converged": error < tolerance,
                        "joint_error": error, "physics_steps": self._steps}
        raise RuntimeError("Joint primitive did not converge within its step allowance")

    def set_gripper(self, fraction, *, steps=30):
        import numpy as np

        if not math.isfinite(fraction) or not 0 <= fraction <= 1 or type(steps) is not int or not 1 <= steps <= 30:
            raise ValueError("Expected opening fraction in [0,1] and 1 to 30 steps")
        self._gripper_fraction = float(fraction)
        for _ in range(steps):
            self._step(np.concatenate([np.zeros(7), [1 - 2 * fraction]]))
        return {"opening_fraction": float(self._observation["robot0_gripper_qpos"][0] / 0.04),
                "physics_steps": self._steps}

    def get_observation(self):
        """Return camera calibration, RGB-D and robot proprioception in base frame.

        Calibration and robot kinematics are trusted adapter metadata. No object
        state is queried or returned. Images follow ASPIRE's vertical flip and
        optical-axis convention; depth is measured in metres.
        """
        import numpy as np
        from robosuite.utils.camera_utils import get_real_depth_map
        from robosuite.utils.transform_utils import mat2quat

        sim = self._environment.sim
        base = sim.model.body_name2id("robot0_base")
        base_rotation = sim.data.body_xmat[base].reshape(3, 3)
        base_position = sim.data.body_xpos[base]
        result = {}
        for camera in ("agentview", "robot0_eye_in_hand"):
            rgb = np.asarray(self._observation[camera + "_image"])[::-1].copy()
            depth = get_real_depth_map(sim, np.asarray(self._observation[camera + "_depth"])[::-1]).copy()
            height, width = rgb.shape[:2]
            fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
            focal = 0.5 * height / np.tan(fovy * np.pi / 360)
            pose = np.eye(4)
            pose[:3, :3] = base_rotation.T @ sim.data.get_camera_xmat(camera) @ np.diag([1, -1, -1])
            pose[:3, 3] = base_rotation.T @ (sim.data.get_camera_xpos(camera) - base_position)
            quat_xyzw = mat2quat(pose[:3, :3])
            result[camera] = {"images": {"rgb": rgb, "depth": depth},
                              "intrinsics": np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]),
                              "pose_mat": pose,
                              "pose": np.concatenate([pose[:3, 3], quat_xyzw[[3, 0, 1, 2]]])}
        hand = sim.model.body_name2id("robot0_right_hand")
        position = base_rotation.T @ (sim.data.body_xpos[hand] - base_position)
        rotation = base_rotation.T @ sim.data.body_xmat[hand].reshape(3, 3)
        quat_xyzw = mat2quat(rotation)
        fraction = self._observation["robot0_gripper_qpos"][0] / 0.04
        result["robot_joint_pos"] = np.concatenate([self._observation["robot0_joint_pos"], [fraction]])
        result["robot_cartesian_pos"] = np.concatenate([position, quat_xyzw[[3, 0, 1, 2]], [fraction]])
        return result
