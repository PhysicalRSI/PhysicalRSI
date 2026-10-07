"""Visible surface estimates from public calibrated RGB-D; never simulator poses."""
import numpy as np


def masked_surface_geometry(camera, mask, *, depth_range_m=(.2, 2.)):
    """Estimate a visible surface within an explicitly chosen sensor depth range."""
    bounds = np.asarray(depth_range_m, dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not .015 <= bounds[0] < bounds[1] <= 20.):
        raise ValueError('Invalid public depth range')
    depth = np.asarray(camera['images']['depth'], dtype=float).squeeze()
    mask = np.asarray(mask)
    k = np.asarray(camera['intrinsics'], dtype=float)
    pose = np.asarray(camera['pose_mat'], dtype=float)
    if (depth.ndim != 2 or min(depth.shape) < 2 or max(depth.shape) > 512
            or mask.shape != depth.shape or not np.isin(mask, [0, 1]).all()
            or k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0, 1], k[1, 0]], 0)
            or pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Invalid public depth geometry')
    selected = mask.astype(bool) & np.isfinite(depth) & (depth > bounds[0]) & (depth < bounds[1])
    v, u = np.nonzero(selected)
    if len(v) < 20:
        return {'status':'insufficient_depth', 'points':len(v)}
    z = depth[v, u]
    points = np.c_[(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z]
    base = points@pose[:3, :3].T+pose[:3, 3]
    lower, median, upper = np.quantile(base, [.05, .5, .95], axis=0)
    return {'status':'estimated_surface', 'frame':'robot-base', 'points':len(v),
            'surface_quantiles':{'lower_05':lower.tolist(), 'median':median.tolist(), 'upper_95':upper.tolist()},
            'touches_image_border':bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
            'full_object_extent_known':False}


class PublicObjectLocator:
    def __init__(self, *, segmenter):
        self.segmenter = segmenter

    def __call__(self, observation, prompt, *, camera='agentview', deadline):
        if camera not in ('agentview', 'robot0_eye_in_hand'):
            raise ValueError('Unsupported public camera')
        source = observation[camera]
        result = self.segmenter(source['images']['rgb'], prompt, deadline=deadline)
        if not result['detections']:
            return {'status':'not_detected', 'semantic_identity_verified':False}
        detection = max(result['detections'], key=lambda row:row['score'])
        return {**masked_surface_geometry(source, detection['mask']),
                'detection_score':detection['score'], 'semantic_identity_verified':False}
