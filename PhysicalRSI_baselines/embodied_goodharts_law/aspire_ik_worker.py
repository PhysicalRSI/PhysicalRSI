"""Persistent CPU ASPIRE IK worker, with one cached pinned Panda model.

The solver and native hand correction match aspire_ik._worker. Keeping the
Robot object stable also preserves its JAX-compiled joint-variable type.
"""
import contextlib
import ast
import ctypes
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys

from PhysicalRSI_baselines.embodied_goodharts_law.aspire_ik import PANDA_SHA256


@functools.lru_cache(maxsize=1)
def _upstream_math():
    """Load pinned math without optional simulator integration initializers.

    ASPIRE's integration package eagerly imports unrelated simulator adapters;
    some request interactive dataset configuration. Neither is part of IK.
    """
    import aspire
    import yourdfpy
    root = Path(aspire.__file__).resolve().parent
    paths = {
        "helper": (root / "sim/cap/serving/launch_pyroki_server.py",
                   "e85dcfd80cf66beaed0040310fc0328a50af73ae46714b0d4f99b6bca0efaf49"),
        "solver": (root / "sim/cap/integrations/motion/pyroki_snippets/_solve_ik_with_rest_cost.py",
                   "cddd223f77a62fb2289e89ac4a3503623d215825ad4f9e529c7069045e443fd3"),
    }
    for path, sha in paths.values():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Unexpected ASPIRE mathematical source identity")
    helper_path = paths["helper"][0]
    tree = ast.parse(helper_path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "set_min_distance_from_limits"]
    if len(functions) != 1:
        raise ValueError("Missing pinned joint-limit helper")
    namespace = {"yourdfpy": yourdfpy}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(helper_path), "exec"), namespace)
    spec = importlib.util.spec_from_file_location("_physicalrsi_pinned_aspire_ik", paths["solver"][0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return namespace["set_min_distance_from_limits"], module.solve_ik_rest


@functools.lru_cache(maxsize=1)
def _model(path):
    import yourdfpy
    import pyroki as pk
    set_min_distance_from_limits, _ = _upstream_math()
    urdf = set_min_distance_from_limits(yourdfpy.URDF.load(
        path, load_meshes=False, build_scene_graph=True))
    return pk.Robot.from_urdf(urdf)


def _worker(request):
    import numpy as np
    import jax.numpy as jnp
    import yourdfpy
    import pyroki as pk
    from scipy.spatial.transform import Rotation
    _, solve_ik_rest = _upstream_math()

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
    robot = _model(str(path))
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


def main():
    owner = int(os.environ["PHYSICALRSI_IK_OWNER_PID"])
    # Linux worker ownership survives an outer supervisor killing the client.
    # Checking twice closes the race where the parent dies before prctl runs.
    if os.getppid() != owner:
        raise SystemExit("IK owner already exited")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot bind IK worker to owner lifetime")
    if os.getppid() != owner:
        raise SystemExit("IK owner exited during startup")
    for index in range(1000):
        line = sys.stdin.buffer.readline(4097)
        if not line:
            return
        if len(line) > 4096 or not line.endswith(b"\n"):
            raise ValueError("IK request exceeds allowance")
        request = json.loads(line)
        if request.pop("id") != index:
            raise ValueError("IK request sequence mismatch")
        try:
            with contextlib.redirect_stdout(sys.stderr):
                value = _worker(request)
            response = {"id": index, "status": "solved", "value": value}
        except RuntimeError as error:
            if str(error) != "IK target did not converge":
                raise
            response = {"id": index, "status": "nonconverged"}
        print(json.dumps(response, allow_nan=False), flush=True)
    raise RuntimeError("IK worker request allowance exhausted")


if __name__ == "__main__":
    main()
