import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.public_depth_diagnostics import mask_depth_diagnostics
from PhysicalRSI_baselines.embodied_goodharts_law.public_geometry import masked_surface_geometry


def test_close_visible_plane_is_distinguished_from_missing_detection():
    camera = {'images': {'depth': np.full((8, 8), .15)}, 'intrinsics': np.eye(3), 'pose_mat': np.eye(4)}
    mask = np.ones((8, 8), dtype=bool)
    assert masked_surface_geometry(camera, mask)['status'] == 'insufficient_depth'
    result = mask_depth_diagnostics(camera, mask)
    assert result['finite_positive_depth_pixels'] == 64
    assert result['within_legacy_depth_gate_pixels'] == 0
    assert result['positive_depth_at_most_20cm_pixels'] == 64
    assert mask_depth_diagnostics(camera, np.zeros_like(mask))['masked_pixels'] == 0


def test_invalid_depth_is_not_reported_as_close_surface():
    depth = np.array([[np.nan, np.inf, -1], [0, .5, 3.]])
    result = mask_depth_diagnostics({'images': {'depth': depth}}, np.ones_like(depth))
    assert result['finite_positive_depth_pixels'] == 2
    assert result['positive_depth_at_most_20cm_pixels'] == 0
    assert result['within_legacy_depth_gate_pixels'] == 1
    with pytest.raises(ValueError):
        mask_depth_diagnostics({'images': {'depth': depth}}, np.full_like(depth, .5))
