"""Associate an estimated RGB-D surface with a SAM3 mask before grasp inference.

The reference point is supplied by the code policy from public perception.
Neither object identities nor simulator/evaluator handles enter this module.
"""
import math
import time

import numpy as np

from .perception_grasps import PublicGraspPlanner
from .public_geometry import masked_surface_geometry


class SurfaceMatchedGraspPlanner:
    def __init__(self, *, segmenter, grasp_client):
        self.segmenter = segmenter
        self.grasp_client = grasp_client

    def __call__(self, observation, prompt, reference_point, *,
                 camera="agentview", deadline):
        point = np.asarray(reference_point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Expected a finite public surface point")
        if camera not in ("agentview", "robot0_eye_in_hand"):
            raise ValueError("Unsupported public camera")
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("Public surface association deadline reached")
        source = observation[camera]
        detected = self.segmenter(source["images"]["rgb"], prompt, deadline=deadline)
        surfaces = []
        for detection in sorted(detected["detections"], key=lambda x: -x["score"])[:16]:
            if not math.isfinite(detection["score"]) or detection["score"] < .4:
                continue
            geometry = masked_surface_geometry(source, detection["mask"])
            if geometry["status"] != "estimated_surface":
                continue
            bounds = geometry["surface_quantiles"]
            center = (np.asarray(bounds["lower_05"])+bounds["upper_95"])/2
            if any(np.linalg.norm(center-row[1]) < .02 for row in surfaces):
                continue
            surfaces.append((float(np.linalg.norm(center-point)), center, detection))
        surfaces.sort(key=lambda row: row[0])
        empty = {"frame": "robot-base", "grasps": [], "semantic_identity_verified": False}
        if time.monotonic() >= deadline:
            raise TimeoutError("Public surface association deadline reached")
        if not surfaces or surfaces[0][0] > .04:
            return {**empty, "status": "surface_not_matched"}
        if len(surfaces) > 1 and surfaces[1][0]-surfaces[0][0] < .01:
            return {**empty, "status": "ambiguous_surface"}
        distance, center, detection = surfaces[0]
        # Reuse the camera transform and grasp validation path without a second
        # segmentation request that could change the selected image/mask pair.
        planner = PublicGraspPlanner(
            segmenter=lambda image, text, **kwargs: {"detections": [detection]},
            grasp_client=self.grasp_client)
        result = planner(observation, prompt, camera=camera, deadline=deadline)
        return {**result, "surface_match_distance_m": distance,
                "matched_visible_center": center.tolist()}
