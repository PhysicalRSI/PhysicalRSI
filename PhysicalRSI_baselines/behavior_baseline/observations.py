"""R1Pro policy input boundary for the pinned official Challenge interface.

Indices follow wensi-ai/openpi 0cc8e355, configs/robots/b1k.py. The native
evaluator's full proprio vector is NOT the public policy observation contract.
This projection must run before passing observations to any evolved program.
It does not provide process isolation or certify arbitrary imported code.
"""
import numpy as np

CAMERAS = {
    "head": "robot::robot:zed_link:Camera:0",
    "left_wrist": "robot::robot:left_realsense_link:Camera:0",
    "right_wrist": "robot::robot:right_realsense_link:Camera:0",
}
PROPRIO = {
    "base_qvel": (0, 1, 2),
    "trunk_qpos": (53, 54, 55, 56),
    "left_arm_qpos": tuple(range(3, 10)),
    "left_gripper_qpos": (24, 25),
    "right_arm_qpos": tuple(range(28, 35)),
    "right_gripper_qpos": (49, 50),
}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def project(raw, *, task, depth=False):
    """Copy only onboard fields; task is frozen by episode initialization.

    Returns batched arrays even for a single unbatched observation. Extra native
    fields, camera world poses, evaluator metrics, and raw task IDs are omitted.
    The full vector is never retained in the returned object.
    """
    if not isinstance(task, str) or not task:
        raise ValueError("Declare the public task instruction")
    # The pinned evaluator defaults to r1pro.yaml's robot_r1 name; the
    # baseline model wrapper uses legacy robot keys. Select one complete
    # namespace, never mix proprioception and cameras from different robots.
    names = [name for name in ("robot", "robot_r1") if name + "::proprio" in raw]
    if len(names) != 1:
        raise ValueError("Expected exactly one supported robot observation namespace")
    robot_name = names[0]
    proprio = _array(raw[robot_name + "::proprio"])
    if proprio.ndim == 1:
        proprio = proprio[None, :]
    if proprio.ndim != 2 or proprio.shape[1] < 57 or proprio.shape[0] < 1:
        raise ValueError("Expected the pinned R1Pro proprio layout")
    public = {"task": task, "proprio": {}, "rgb": {}}
    for name, indices in PROPRIO.items():
        values = proprio[:, indices]
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("Invalid onboard proprioception")
        public["proprio"][name] = values.copy()
    if depth:
        public["depth"] = {}
    for role, prefix in CAMERAS.items():
        prefix = prefix.replace("robot::robot:", robot_name + "::" + robot_name + ":", 1)
        native_rgb = raw[prefix + "::rgb"]
        rgb = _array(native_rgb)
        # Core trajectories are JSON. Validate integer pixels before restoring
        # their uint8 representation; never truncate arbitrary floating values.
        if isinstance(native_rgb, list):
            if not np.issubdtype(rgb.dtype, np.integer) or (rgb < 0).any() or (rgb > 255).any():
                raise ValueError("JSON RGB pixels must be integers in [0, 255]")
            rgb = rgb.astype(np.uint8)
        if rgb.ndim == 3:
            rgb = rgb[None, ...]
        if rgb.ndim != 4 or rgb.shape[0] != proprio.shape[0] or rgb.shape[-1] not in (3, 4) or rgb.dtype != np.uint8:
            raise ValueError("Expected matching batched uint8 RGB camera observations")
        public["rgb"][role] = rgb[..., :3].copy()
        if depth:
            metric_depth = _array(raw[prefix + "::depth_linear"])
            if metric_depth.ndim == 2:
                metric_depth = metric_depth[None, ...]
            if metric_depth.shape != rgb.shape[:3]:
                raise ValueError("Depth and RGB must have matching batch and image dimensions")
            # Native background rays can be infinite. Preserve their semantics
            # for the perception backend instead of fabricating a finite range.
            if not np.issubdtype(metric_depth.dtype, np.floating) or np.isnan(metric_depth).any() or (metric_depth < 0).any():
                raise ValueError("Invalid metric depth")
            public["depth"][role] = metric_depth.copy()
    return public


def validate_action(action, *, batch_size):
    """Check the official 23-dimensional command shape without changing control."""
    value = _array(action)
    if value.ndim == 1 and batch_size == 1:
        value = value[None, :]
    if value.shape != (batch_size, 23) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("Expected finite floating R1Pro actions with shape (batch, 23)")
    return value.copy()
