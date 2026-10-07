import math

import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.planar_surface_orientation import masked_planar_orientation


def fixture():
    source = {"images": {"depth": np.ones((32, 32))},
              "intrinsics": np.array([[100, 0, 16], [0, 100, 16], [0, 0, 1]]),
              "pose_mat": np.eye(4)}
    mask = np.zeros((32, 32), dtype=bool)
    mask[13:19, 4:28] = True
    return source, mask


def test_axis_transforms_with_public_camera_calibration():
    source, mask = fixture()
    angle = math.pi/4
    c, s = math.cos(angle), math.sin(angle)
    source["pose_mat"][:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result = masked_planar_orientation(source, mask)
    assert result["status"] == "estimated_visible_axis"
    assert result["yaw_rad"] == pytest.approx(angle)
    assert not result["full_object_orientation_known"]


def test_square_surface_has_no_preferred_axis():
    source, mask = fixture()
    mask[:] = False
    mask[8:24, 8:24] = True
    assert masked_planar_orientation(source, mask)["status"] == "ambiguous_orientation"


def test_clipped_and_insufficient_surfaces_are_rejected():
    source, mask = fixture()
    mask[13:19, :4] = True
    assert masked_planar_orientation(source, mask)["status"] == "clipped_surface"
    mask[:] = False
    mask[15, 15] = True
    assert masked_planar_orientation(source, mask)["status"] == "insufficient_depth"
