"""ASPIRE scene-aware grasp planning from an explicit public sensor snapshot.

This plans only; callers must execute and verify the returned trajectory in a
bounded native trial. A deadline check cannot interrupt a running CUDA call.
No simulator object handles, object poses or evaluator measurements are used.
"""
import time

import numpy as np


class PublicSceneGraspPlanner:
    def __init__(self, curobo_api, *, position_tolerance_m=.001,
                 orientation_tolerance_rad=.005, use_grasp_approach=False,
                 allow_target_contact=True, scene_mesh_pitch_m=.03,
                 target_mesh_pitch_m=None):
        if not isinstance(use_grasp_approach, bool) or not isinstance(allow_target_contact, bool):
            raise ValueError('Expected boolean grasp planning options')
        for pitch in (scene_mesh_pitch_m, target_mesh_pitch_m):
            if pitch is not None and (not np.isfinite(pitch) or not .001 <= pitch <= .1):
                raise ValueError('Expected mesh pitch between .001 and .1 metres')
        if scene_mesh_pitch_m is None:
            raise ValueError('Scene mesh pitch is required')
        if (not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0 or
                not np.isfinite(orientation_tolerance_rad) or orientation_tolerance_rad <= 0):
            raise ValueError('Expected finite positive planning tolerances')
        self.api = curobo_api
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.use_grasp_approach = use_grasp_approach
        self.allow_target_contact = allow_target_contact
        self.scene_mesh_pitch_m = float(scene_mesh_pitch_m)
        self.target_mesh_pitch_m = target_mesh_pitch_m

    def __call__(self, *, depth, intrinsics, camera_to_base, object_mask,
                 robot_joints, grasp_poses, pose_reference, deadline):
        """Plan to robot-base xyz+wxyz poses; declare hand versus fingertip.

        The mask and grasps must describe the same observed object and snapshot.
        This function validates numeric contracts, not semantic association.
        Scene geometry is a partial RGB-D reconstruction, not a complete world.
        """
        def check_deadline():
            if not np.isfinite(deadline) or time.monotonic() >= deadline:
                raise TimeoutError('Public scene grasp planning deadline reached')

        check_deadline()
        if pose_reference not in ('hand', 'fingertip'):
            raise ValueError('Declare hand or fingertip grasp pose reference')
        depth = np.asarray(depth, dtype=float)
        mask = np.asarray(object_mask)
        k = np.asarray(intrinsics, dtype=float)
        transform = np.asarray(camera_to_base, dtype=float)
        joints = np.asarray(robot_joints, dtype=float)
        poses = np.asarray(grasp_poses, dtype=float)
        if (depth.ndim != 2 or mask.shape != depth.shape or
                mask.dtype != np.bool_):
            raise ValueError('Expected matching 2D depth and boolean object mask')
        valid = np.isfinite(depth) & (depth >= .015) & (depth <= 20.)
        if not (valid & mask).any() or not (valid & ~mask).any():
            raise ValueError('Object and scene both require valid measured depth')
        if (k.shape != (3, 3) or not np.isfinite(k).all() or
                k[0, 0] <= 0 or k[1, 1] <= 0 or
                not np.allclose(k[2], [0, 0, 1]) or
                not np.allclose([k[0, 1], k[1, 0]], 0)):
            raise ValueError('Expected pinhole camera intrinsics without skew')
        if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
                not np.allclose(transform[3], [0, 0, 0, 1]) or
                not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
            raise ValueError('Expected rigid camera-to-robot-base transform')
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError('Expected seven measured arm joints')
        if (poses.ndim != 2 or poses.shape[1] != 7 or not len(poses) or
                not np.isfinite(poses).all() or
                not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-5)):
            raise ValueError('Expected finite robot-base xyz and unit wxyz poses')
        world = self.api.create_curobo_world_from_depth_with_object(
            depth, mask, k, camera_pose=transform,
            robot_joint_position=joints, object_name='observed_target',
            scene_name='observed_scene', object_pose_override=None,
            marching_cubes_pitch=self.scene_mesh_pitch_m)
        check_deadline()
        # ASPIRE can return a partial/empty reconstruction. Never silently
        # degrade this reference into the existing obstacle-free IK path.
        meshes = getattr(world, 'mesh', None) or []
        if not {'observed_target', 'observed_scene'}.issubset(
                {mesh.name for mesh in meshes}):
            return {'status': 'incomplete_observed_world', 'trajectory': None}
        if self.target_mesh_pitch_m is not None:
            # Keep the scene and robot filtering unchanged. Reconstruct only
            # the target from the same public mask/depth, without a pose label.
            v, u = np.nonzero(valid & mask)
            z = depth[v, u]
            camera_points = np.c_[(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z]
            points = camera_points @ transform[:3, :3].T + transform[:3, 3]
            target = self.api.Mesh.from_pointcloud(
                points, pitch=self.target_mesh_pitch_m, name='observed_target')
            world = self.api.WorldConfig(mesh=[m for m in meshes if m.name != 'observed_target'] + [target])
            check_deadline()
        success, trajectory, index = self.api.plan_to_grasp_poses(
            world, joints, [(p[:3], p[3:]) for p in poses],
            use_world_collision=True,
            ignore_obstacle_names=['observed_target'] if self.allow_target_contact else [],
            grasp_pose_is_fingertip=pose_reference == 'fingertip',
            position_threshold=self.position_tolerance_m,
            position_threshold_z=None,
            rotation_threshold=self.orientation_tolerance_rad,
            relax_orientation=False, use_grasp_approach=self.use_grasp_approach)
        check_deadline()
        if not success:
            return {'status': 'planning_failed', 'trajectory': None}
        trajectory = np.asarray(trajectory, dtype=float)
        if (trajectory.ndim != 2 or trajectory.shape[1] != 7 or
                not len(trajectory) or not np.isfinite(trajectory).all() or
                not isinstance(index, (int, np.integer)) or isinstance(index, bool) or
                not 0 <= index < len(poses)):
            raise ValueError('Planner returned invalid trajectory or grasp index')
        return {'status': 'planned', 'trajectory': trajectory.tolist(),
                'grasp_index': int(index), 'pose_reference': pose_reference,
                'frame': 'robot-base', 'world_source': 'public RGB-D reconstruction',
                'position_tolerance_m': self.position_tolerance_m,
                'orientation_tolerance_rad': self.orientation_tolerance_rad,
                'use_grasp_approach': self.use_grasp_approach,
                'allow_target_contact': self.allow_target_contact,
                'scene_mesh_pitch_m': self.scene_mesh_pitch_m,
                'target_mesh_pitch_m': self.target_mesh_pitch_m,
                'semantic_identity_verified': False, 'execution_verified': False}
