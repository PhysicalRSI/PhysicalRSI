import time

import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.surface_matched_grasps import SurfaceMatchedGraspPlanner


def fixture():
    left, right = np.zeros((10, 20), dtype=bool), np.zeros((10, 20), dtype=bool)
    left[:, :5] = True
    right[:, 15:] = True
    source = {"images": {"rgb": np.zeros((10, 20, 3)), "depth": np.ones((10, 20))},
              "intrinsics": np.diag([100., 100., 1.]), "pose_mat": np.eye(4)}
    calls = []
    def grasp(depth, k, mask, *, deadline):
        calls.append(mask)
        return {"frame": "camera-optical", "grasps": [{"matrix": np.eye(4), "score": .8}]}
    planner = SurfaceMatchedGraspPlanner(segmenter=lambda *a, **k: {
        "detections": [{"score": .99, "mask": right}, {"score": .6, "mask": left}]},
        grasp_client=grasp)
    return planner, {"agentview": source, "object_poses": "unused private sentinel"}, calls, left


def test_surface_match_overrides_wrong_instance_with_higher_confidence():
    planner, obs, calls, left = fixture()
    result = planner(obs, "bowl", [.02, .045, 1.], deadline=time.monotonic()+5)
    assert result["status"] == "estimated"
    assert result["detection_score"] == .6
    assert not result["semantic_identity_verified"]
    np.testing.assert_array_equal(calls[0], left)


def test_mismatch_does_not_invoke_grasp_model():
    planner, obs, calls, _ = fixture()
    result = planner(obs, "bowl", [1., 1., 1.], deadline=time.monotonic()+5)
    assert result["status"] == "surface_not_matched"
    assert calls == []


def test_ambiguous_nearby_surfaces_are_rejected():
    planner, obs, calls, _ = fixture()
    obs["agentview"]["intrinsics"] = np.diag([300., 100., 1.])
    result = planner(obs, "bowl", [.095/3, .045, 1.], deadline=time.monotonic()+5)
    assert result["status"] == "ambiguous_surface"
    assert calls == []


def test_expired_deadline_does_not_invoke_models():
    planner, obs, calls, _ = fixture()
    with pytest.raises(TimeoutError):
        planner(obs, "bowl", [0, 0, 1], deadline=time.monotonic()-1)
    assert calls == []
