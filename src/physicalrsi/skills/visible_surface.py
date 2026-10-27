"""Measure a seeded visible RGB-D patch; never infer hidden object geometry."""
from PhysicalRSI_core.infra.storage import digest
from physicalrsi.runtime.sensors import decode_rgb
from physicalrsi.runtime.rgbd import decode_depth_mm
from .depth_point import lift_depth_pixel
from .protocol import CallableSkill, SkillDescriptor


def visible_surface(**calibration):
    """Use a frame-bound image proposal and calibrated depth to measure a patch."""
    import cv2
    import numpy as np

    lift = lift_depth_pixel(**calibration)
    config = dict(lift.descriptor.metadata, color_distance=60,
                  depth_distance_m=.12, minimum_pixels=9, quantiles=[.02, .98])
    camera = config["camera"]
    width, height = config["image_size_px"]
    inverse = np.linalg.inv(np.array(config["intrinsic_cv"]))
    extrinsic = np.array(config["extrinsic_cv"])

    def run(value, context):
        if set(value) != {"observation", "proposal"}:
            raise ValueError("Unexpected visible-surface inputs")
        packet, proposal = value["observation"], value["proposal"]
        if set(proposal) != {"sequence", "camera", "uv", "bbox_xyxy"}:
            raise ValueError("Unexpected proposal fields")
        if proposal["sequence"] != packet["rgb"]["sequence"] or proposal["camera"] != camera:
            raise ValueError("Proposal does not belong to this camera frame")
        seed = lift.execute(dict(observation=packet, uv=proposal["uv"]), context)
        bounds = np.asarray(proposal["bbox_xyxy"], dtype=float)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or not 0 <= bounds[0] < bounds[2] <= width
                or not 0 <= bounds[1] < bounds[3] <= height):
            raise ValueError("Invalid image region")
        x, y = seed["pixel_xy"]
        if not bounds[0] <= x+.5 < bounds[2] or not bounds[1] <= y+.5 < bounds[3]:
            raise ValueError("Seed lies outside proposal")
        result = dict(status="unavailable", observation_sequence=proposal["sequence"],
                      camera=camera, region_flat_indices=[], bounds_world_m=None,
                      centroid_world_m=None, semantic_identity=None,
                      measurement="visible_surface_extent_not_complete_object_geometry")
        if seed["status"] != "measured":
            return result
        rgb = decode_rgb(packet["rgb"]["cameras"][camera]["rgb_png_b64"])
        depth = decode_depth_mm(packet["depth"][camera]["png_b64"]).astype(float)/1000
        yy, xx = np.indices(depth.shape)
        mask = ((np.linalg.norm(rgb.astype(float)-rgb[y, x], axis=2) <= 60)
                & (depth > 0) & (np.abs(depth-depth[y, x]) <= .12)
                & (xx+.5 >= bounds[0]) & (xx+.5 < bounds[2])
                & (yy+.5 >= bounds[1]) & (yy+.5 < bounds[3]))
        _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        label = labels[y, x]
        if not label:
            return result
        ys, xs = np.nonzero(labels == label)
        if len(xs) < 9:
            return result
        context.check()
        points = (inverse @ np.stack([xs+.5, ys+.5, np.ones(len(xs))])) * depth[ys, xs]
        world = (extrinsic[:, :3].T @ (points-extrinsic[:, 3, None])).T
        result.update(status="measured", region_flat_indices=(ys*width+xs).tolist(),
                      bounds_world_m=np.quantile(world, [.02, .98], axis=0).tolist(),
                      centroid_world_m=world.mean(axis=0).tolist())
        return result

    return CallableSkill(SkillDescriptor(
        "visible_surface", digest(config),
        {"name": "physicalrsi.rgbd-region/v1"},
        {"name": "physicalrsi.visible-surface/v1"}, metadata=config), run)
