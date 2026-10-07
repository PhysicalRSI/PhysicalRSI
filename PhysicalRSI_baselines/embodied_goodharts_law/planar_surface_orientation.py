"""Planar orientation from a visible, calibrated RGB-D surface only."""
import math

import numpy as np

from .public_geometry import masked_surface_geometry


def masked_planar_orientation(camera, mask):
    """Estimate an unoriented XY axis in the coordinate frame of pose_mat.

    This describes the visible point distribution, not the object's true pose.
    Clipped, nearly circular, thin, and insufficient surfaces are rejected.
    """
    geometry = masked_surface_geometry(camera, mask)
    if geometry["status"] != "estimated_surface":
        return {"status": "insufficient_depth"}
    if geometry["touches_image_border"]:
        return {"status": "clipped_surface"}
    depth = np.asarray(camera["images"]["depth"], dtype=float).squeeze()
    selected = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > .2) & (depth < 2.)
    v, u = np.nonzero(selected)
    k = np.asarray(camera["intrinsics"], dtype=float)
    transform = np.asarray(camera["pose_mat"], dtype=float)
    z = depth[v, u]
    points = np.c_[(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z]
    xy = (points @ transform[:3, :3].T + transform[:3, 3])[:, :2]
    values, vectors = np.linalg.eigh(np.cov(xy, rowvar=False))
    if values[0] < .002**2 or values[1] < 1.69*values[0]:
        return {"status": "ambiguous_orientation", "points": len(xy)}
    axis = vectors[:, 1]
    return {"status": "estimated_visible_axis", "yaw_rad": math.atan2(axis[1], axis[0]) % math.pi,
            "variance_ratio": float(values[1]/values[0]), "points": len(xy),
            "semantic_identity_verified": False, "full_object_orientation_known": False}
