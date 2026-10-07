"""Multiple public RGB-D surface estimates; no simulator object identities."""
import math
import time

from .robotwin_perception import PublicRoboTwinLocator, PerceivedRoboTwinPrimitives, public_world_camera
from .public_geometry import masked_surface_geometry


class MultiSurfaceLocator(PublicRoboTwinLocator):
    def all(self, observation, prompt, *, camera="head_camera", deadline):
        source = public_world_camera(observation, camera)
        detected = self.segmenter(source["images"]["rgb"], prompt, deadline=deadline)
        objects = []
        for detection in sorted(detected["detections"], key=lambda x: -x["score"])[:16]:
            geometry = masked_surface_geometry(source, detection["mask"])
            if geometry["status"] != "estimated_surface":
                continue
            bounds = geometry["surface_quantiles"]
            center = [(bounds["lower_05"][i] + bounds["upper_95"][i]) / 2 for i in range(2)]
            if any(math.dist(center, old["visible_xy_center"]) < .02
                   and abs(bounds["upper_95"][2] - old["surface_quantiles"]["upper_95"][2]) < .012
                   for old in objects):
                continue
            objects.append({**geometry, "frame": "world", "visible_xy_center": center,
                            "detection_score": float(detection["score"]),
                            "semantic_identity_verified": False})
        return {"status": "estimated_surfaces", "objects": objects,
                "frame": "world", "semantic_identity_verified": False}


class MultiSurfaceRoboTwinPrimitives(PerceivedRoboTwinPrimitives):
    @property
    def handlers(self):
        handlers = super().handlers
        def locate_all(args, kwargs, *, deadline):
            if not math.isfinite(deadline):
                raise ValueError("A finite perception deadline is required")
            if time.monotonic() >= deadline:
                raise TimeoutError("RoboTwin location deadline reached")
            result = self._object_locator.all(self.get_observation(), *args, **kwargs, deadline=deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError("RoboTwin location deadline reached")
            return result
        handlers["locate_objects"] = locate_all
        return handlers
