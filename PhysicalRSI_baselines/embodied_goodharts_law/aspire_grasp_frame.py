"""Match the reviewed ASPIRE CGN -> fingertip -> Panda hand conversion.

ASPIRE's libero.py adds .12 m along raw CGN local Z; curobo_api.py then
subtracts .1168 m along the same axis for panda_hand. The net local offset is
.0032 m. This is an upstream convention, not a sensor-derived object offset
or a universal calibration for other grippers. Native hand correction is
separate and must be applied after this conversion.
"""
import numpy as np


def contact_grasp_to_model_hand(matrix):
    pose = np.asarray(matrix, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0, 0, 0, 1]) or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
        raise ValueError('Expected a finite rigid raw CGN pose')
    result = pose.copy()
    result[:3, 3] += result[:3, 2] * (.12 - .1168)
    return result


def contact_grasp_hand_variants(matrix):
    """Return original and local-Z half-turn poses for a parallel gripper.

    The latter exchanges ideal fingers while preserving contact center and
    approach axis. Robot/palm collision and reachability must be checked again.
    """
    original = contact_grasp_to_model_hand(matrix)
    rotated = original @ np.diag([-1., -1., 1., 1.])
    return [original, rotated]
