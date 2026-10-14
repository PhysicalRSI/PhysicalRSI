"""Project public R1Pro observations into the 2025 champion input layout.

The 2026 evaluator changed the proprioception layout. Reconstruct only the
onboard coordinates consumed by the published policy; privileged coordinates
in the legacy 256-vector stay zero. Model normalization remains upstream.
"""
import numpy as np

from .observations import CAMERAS

SOURCE_REVISION = "ca556f74a455cef7987a2be4537b5ac85cc56dd7"
OPENPI_REVISION = "01177e0242a1c7e8fad2547caa0e987def614cda"
BEHAVIOR_REVISION = "684a83050ddd398de231e6aa7fc605bc34458d4b"
CHECKPOINT_REVISION = "89545bc1b7aa7f2e687bc0032d091f132d715d4e"

LEGACY_PROPRIO = {
    "base_qvel": slice(253, 256),
    "trunk_qpos": slice(236, 240),
    "left_arm_qpos": slice(158, 165),
    "left_gripper_qpos": slice(193, 195),
    "right_arm_qpos": slice(197, 204),
    "right_gripper_qpos": slice(232, 234),
}


def champion_observation(public):
    """Return one unbatched legacy observation for the single-policy wrapper."""
    vector = np.zeros(256, dtype=np.float32)
    for name, target in LEGACY_PROPRIO.items():
        value = np.asarray(public["proprio"][name])
        if value.shape != (1, target.stop - target.start) or not np.isfinite(value).all():
            raise ValueError("Champion wrapper requires one finite onboard observation")
        vector[target] = value[0]
    raw = {"robot_r1::proprio": vector}
    for role, prefix in CAMERAS.items():
        pixels = np.asarray(public["rgb"][role])
        if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
            raise ValueError("Champion wrapper requires one uint8 RGB image per camera")
        key = prefix.replace("robot::robot:", "robot_r1::robot_r1:", 1)
        raw[key + "::rgb"] = pixels[0].copy()
    return raw
