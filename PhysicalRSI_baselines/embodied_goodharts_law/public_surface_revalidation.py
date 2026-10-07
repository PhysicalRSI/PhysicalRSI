"""Compare shared visible target pixels in a fixed public RGB-D camera.

This checks a partial surface, not semantic identity or the full object pose.
It does not infer stability for occluded geometry or qualify a grasp.
"""
import numpy as np

from .public_geometry import masked_surface_geometry


def compare_projected_surface(reference, mask, before, after, *,
                              visibility_tolerance_m=.005, change_tolerance_m=.001):
    for value in (visibility_tolerance_m, change_tolerance_m):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Expected finite positive depth tolerances')
    masked_surface_geometry(reference, mask, depth_range_m=(.015, 20.))
    arrays = []
    for camera in (before, after):
        depth = np.asarray(camera['images']['depth'], dtype=float).squeeze()
        masked_surface_geometry(camera, np.ones(depth.shape, dtype=bool), depth_range_m=(.015, 20.))
        rgb = np.asarray(camera['images']['rgb'])
        if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError('Expected matching uint8 public RGB')
        arrays.append((depth, rgb))
    if (arrays[0][0].shape != arrays[1][0].shape
            or not np.array_equal(before['intrinsics'], after['intrinsics'])
            or not np.array_equal(before['pose_mat'], after['pose_mat'])):
        raise ValueError('Comparison requires unchanged camera calibration and dimensions')
    depth = np.asarray(reference['images']['depth'], dtype=float).squeeze()
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > .015) & (depth < 20.)
    v, u = np.nonzero(valid); z = depth[v, u]
    k = np.asarray(reference['intrinsics']); pose = np.asarray(reference['pose_mat'])
    points = np.c_[(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z]
    points = points @ pose[:3, :3].T + pose[:3, 3]
    pose = np.asarray(before['pose_mat']); k = np.asarray(before['intrinsics'])
    points = (points-pose[:3, 3]) @ pose[:3, :3]
    points = points[points[:, 2] > .015]
    projected = points[:, :2] / points[:, 2, None] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    height, width = arrays[0][0].shape
    inside = ((projected[:, 0] >= 0) & (projected[:, 0] < width-.5)
              & (projected[:, 1] >= 0) & (projected[:, 1] < height-.5))
    uv = np.rint(projected[inside]).astype(int); points = points[inside]
    baseline = arrays[0][0][uv[:, 1], uv[:, 0]]
    visible = np.isfinite(baseline) & (np.abs(baseline-points[:, 2]) < visibility_tolerance_m)
    uv = np.unique(uv[visible], axis=0)
    result = {'compared_pixels': len(uv), 'full_object_pose_verified': False,
              'semantic_identity_verified': False, 'grasp_verified': False}
    if len(uv) < 20:
        return {**result, 'status': 'insufficient_shared_surface'}
    a, b = (item[0][uv[:, 1], uv[:, 0]] for item in arrays)
    finite = np.isfinite(b)
    changed = ~finite | (np.abs(b-a) >= change_tolerance_m)
    same_rgb = np.all(arrays[0][1][uv[:, 1], uv[:, 0]] == arrays[1][1][uv[:, 1], uv[:, 0]], axis=1)
    return {**result, 'status': 'compared', 'changed_depth_pixels': int(changed.sum()),
            'invalid_after_depth_pixels': int((~finite).sum()),
            'fraction_identical_rgb': float(same_rgb.mean()),
            'visibility_tolerance_m': visibility_tolerance_m,
            'change_tolerance_m': change_tolerance_m}
