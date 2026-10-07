"""Trusted subprocess boundary for ASPIRE's pinned Panda hand IK.

The configured interpreter owns ASPIRE/JAX dependencies. Candidate programs
receive neither the interpreter nor a simulator handle. This solves kinematics,
not collision-free motion; an outer supervisor must still bound native actions.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

PANDA_SHA256 = '63792d2679f22c4e41cda2c4d9d903644e628970ad0656b9eec8485113941555'


def gripper_seed_fraction(measured):
    """Project a finite measured fraction into the IK model's seed domain.

    Native soft constraints can put a measured joint outside its model limits.
    The optimization initial value must be inside those limits. This projection
    does not validate, change or conceal the public measurement: the solve
    receipt records both values and clipping, and hand convergence is measured
    independently after actuation. Non-finite measurements remain errors.
    """
    value = float(measured)
    if not math.isfinite(value):
        raise ValueError(f'Invalid measured gripper fraction: {value}')
    return min(1., max(0., value))


class AspireHandIK:
    def __init__(self, *, interpreter, urdf, native_hand_translation,
                 native_hand_quaternion_wxyz, timeout_seconds=45):
        # A venv's Python is commonly a symlink. Resolving it loses pyvenv.cfg
        # discovery and silently launches the base interpreter instead.
        executable = Path(interpreter).absolute()
        if not executable.is_file():
            raise ValueError('Missing IK interpreter')
        self.interpreter = str(executable)
        self.urdf = str(Path(urdf).resolve(strict=True))
        if hashlib.sha256(Path(self.urdf).read_bytes()).hexdigest() != PANDA_SHA256:
            raise ValueError('Unexpected Panda model identity')
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
            raise ValueError('Invalid IK timeout')
        self.timeout_seconds = timeout_seconds
        self.translation = list(native_hand_translation)
        self.quaternion = list(native_hand_quaternion_wxyz)

    def identity(self):
        return {'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'urdf_sha256': PANDA_SHA256, 'interpreter': self.interpreter,
                'native_hand_translation': list(self.translation),
                'native_hand_quaternion_wxyz': list(self.quaternion),
                'timeout_seconds': self.timeout_seconds,
                'solver': 'ASPIRE solve_ik_rest; current joint seed; zero rest cost',
                'collision_planning': False}

    def __call__(self, *, pose, joints, gripper_fraction, deadline):
        remaining = min(self.timeout_seconds, deadline - time.monotonic())
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError('IK deadline reached')
        seed_fraction = gripper_seed_fraction(gripper_fraction)
        request = {'urdf': self.urdf, 'pose': list(pose), 'joints': list(joints),
                   'gripper_fraction': seed_fraction,
                   'translation': self.translation, 'quaternion': self.quaternion}
        import os
        env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([self.interpreter, str(Path(__file__).resolve()), '--worker'],
                                input=json.dumps(request, allow_nan=False), text=True,
                                capture_output=True, timeout=remaining, env=env)
        if result.returncode:
            raise RuntimeError('ASPIRE IK worker failed: ' + result.stderr[-2000:])
        response = json.loads(result.stdout)
        response['gripper_seed'] = {'measured_fraction': float(gripper_fraction),
                                    'model_fraction': seed_fraction,
                                    'clipped': seed_fraction != float(gripper_fraction)}
        if time.monotonic() >= deadline:
            raise TimeoutError('IK deadline reached')
        return response


def _worker(request):
    import numpy as np
    import jax.numpy as jnp
    import yourdfpy
    import pyroki as pk
    from scipy.spatial.transform import Rotation
    from aspire.sim.cap.serving.launch_pyroki_server import set_min_distance_from_limits
    from aspire.sim.cap.integrations.motion.pyroki_snippets._solve_ik_with_rest_cost import solve_ik_rest

    path = Path(request['urdf'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != PANDA_SHA256:
        raise ValueError('Unexpected Panda model identity')
    pose = np.asarray(request['pose'], dtype=float)
    joints = np.asarray(request['joints'], dtype=float)
    translation = np.asarray(request['translation'], dtype=float)
    quaternion = np.asarray(request['quaternion'], dtype=float)
    if (pose.shape != (7,) or joints.shape != (7,) or translation.shape != (3,)
            or quaternion.shape != (4,) or not all(np.isfinite(x).all()
            for x in [pose, joints, translation, quaternion])
            or abs(np.linalg.norm(pose[3:]) - 1) > 1e-5
            or abs(np.linalg.norm(quaternion) - 1) > 1e-5):
        raise ValueError('Invalid IK inputs')
    urdf = set_min_distance_from_limits(yourdfpy.URDF.load(
        str(path), load_meshes=False, build_scene_graph=True))
    robot = pk.Robot.from_urdf(urdf)
    if robot.joints.actuated_names != tuple([f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1']):
        raise ValueError('Unexpected joint order')
    standard = np.eye(4)
    standard[:3, :3] = Rotation.from_euler('z', -np.pi / 4).as_matrix()
    standard[2, 3] = .107
    native = np.eye(4)
    native[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    native[:3, 3] = translation
    correction = np.linalg.inv(standard) @ native
    target_native = np.eye(4)
    target_native[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    target_native[:3, 3] = pose[:3]
    target = target_native @ np.linalg.inv(correction)
    fraction = float(request['gripper_fraction'])
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid measured gripper fraction')
    initial = jnp.asarray(np.r_[joints, fraction * .04])
    solution = solve_ik_rest(robot, 'panda_hand',
        Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]], target[:3, 3],
        rest_cost_weights=0., initial_q=initial)
    if (not np.isfinite(solution).all()
            or np.any(solution < np.asarray(robot.joints.lower_limits) - 1e-5)
            or np.any(solution > np.asarray(robot.joints.upper_limits) + 1e-5)):
        raise ValueError('Invalid IK solution')
    fk = np.asarray(robot.forward_kinematics(jnp.asarray(solution)))[robot.links.names.index('panda_hand')]
    position_error = float(np.linalg.norm(fk[4:] - target[:3, 3]))
    angle = float(Rotation.from_matrix(target[:3, :3].T @
        Rotation.from_quat(fk[[1, 2, 3, 0]]).as_matrix()).magnitude())
    if position_error > .001 or angle > .005:
        raise RuntimeError('IK target did not converge')
    return {'joints': solution[:7].tolist(), 'fk_position_error_m': position_error,
            'fk_orientation_error_rad': angle, 'urdf_sha256': PANDA_SHA256}


if __name__ == '__main__':
    import contextlib
    import sys
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Use the configured AspireHandIK host interface')
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        response = _worker(request)
    print(json.dumps(response, allow_nan=False))
