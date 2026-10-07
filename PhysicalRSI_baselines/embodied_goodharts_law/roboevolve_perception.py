"""RoboEvolve camera boundary; simulated RGB-D, not learned depth.

The native generic depth channel is ambiguous. The trusted sensor adapter must
identify its annotator before calling this conversion; policies cannot choose it.
"""
import numpy as np


def public_world_camera(channels, *, depth_kind):
    """Whitelist camera data and convert metre-valued depth to optical Z."""
    if depth_kind not in ("distance_to_image_plane", "distance_to_camera"):
        raise ValueError("An explicit native depth annotator is required")
    config = channels["config"]
    k = np.asarray(config["intrinsic"], dtype=float)
    pose = np.asarray(config["camera_to_world"], dtype=float)
    depth = np.asarray(channels["depth"], dtype=float)
    rgb = np.asarray(channels["rgb"])
    if (depth.ndim != 2 or rgb.shape != (*depth.shape, 3)
            or k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose([k[0, 1], k[1, 0]], 0)
            or pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Invalid public RoboEvolve camera")
    if depth_kind == "distance_to_camera":
        v, u = np.indices(depth.shape)
        depth = depth / np.sqrt(1 + ((u-k[0, 2])/k[0, 0])**2
                                + ((v-k[1, 2])/k[1, 1])**2)
    return {"images": {"rgb": rgb.copy(), "depth": depth.copy()},
            "intrinsics": k.copy(),
            "pose_mat": pose @ np.diag([1., -1., -1., 1.])}
