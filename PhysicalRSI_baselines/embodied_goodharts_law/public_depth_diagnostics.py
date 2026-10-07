"""Counts from a public segmentation mask and camera depth, without scene state."""
import numpy as np


def mask_depth_diagnostics(camera, mask):
    depth = np.asarray(camera['images']['depth'], dtype=float).squeeze()
    mask = np.asarray(mask)
    if depth.ndim != 2 or mask.shape != depth.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError('Invalid public mask/depth diagnostic input')
    values = depth[mask.astype(bool)]
    positive = values[np.isfinite(values) & (values > 0)]
    return {'masked_pixels': int(values.size),
            'finite_positive_depth_pixels': int(positive.size),
            'within_legacy_depth_gate_pixels': int(((positive > .2) & (positive < 2.)).sum()),
            'positive_depth_at_most_20cm_pixels': int((positive <= .2).sum()),
            'positive_depth_at_least_2m_pixels': int((positive >= 2.).sum()),
            'positive_depth_quantiles_m': np.quantile(positive, [.05, .5, .95]).tolist() if positive.size else None,
            'scope': 'public camera depth and segmentation mask only'}
