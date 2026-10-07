import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.roboevolve_perception import public_world_camera


def camera():
    return {"rgb": np.zeros((3, 3, 3), dtype=np.uint8),
            "depth": np.ones((3, 3)), "object_pose": "private",
            "segmentation": "private", "config": {
                "intrinsic": np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
                "camera_to_world": np.eye(4), "prim_path": "private"}}


def test_metric_plane_depth_and_private_fields():
    result = public_world_camera(camera(), depth_kind="distance_to_image_plane")
    assert set(result) == {"images", "intrinsics", "pose_mat"}
    assert set(result["images"]) == {"rgb", "depth"}
    np.testing.assert_allclose(result["images"]["depth"], 1)
    np.testing.assert_allclose(result["pose_mat"], np.diag([1, -1, -1, 1]))


def test_ray_depth_off_axis_conversion():
    result = public_world_camera(camera(), depth_kind="distance_to_camera")
    assert result["images"]["depth"][1, 1] == 1
    assert result["images"]["depth"][0, 0] == pytest.approx(1 / np.sqrt(3))


def test_ambiguous_native_depth_rejected():
    with pytest.raises(ValueError, match="annotator"):
        public_world_camera(camera(), depth_kind="depth")
