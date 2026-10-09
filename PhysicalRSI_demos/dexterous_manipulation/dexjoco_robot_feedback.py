"""Robot-only FK and fixed task rules, with no per-scene object state."""
import importlib

def robot_geometry(core):
    """Read robot link transforms determined by the measured robot joints."""
    import mujoco
    model,data=core.model,core.data
    hands={}
    for side in ("left","right"):
        palm=model.body("allegro_palm_"+side).id
        tips={}
        for finger,prefix in (("index","ff"),("middle","mf"),("ring","rf"),("thumb","th")):
            body=model.body(prefix+"_tip_"+side).id
            ids=[i for i in range(model.ngeom) if model.geom_bodyid[i]==body
                 and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE
                 and (model.geom_contype[i] or model.geom_conaffinity[i])]
            if len(ids)!=1:
                raise ValueError(f"Expected one robot fingertip collision capsule: {side}/{finger}")
            gid=ids[0]
            tips[finger]=dict(center_world_m=data.geom_xpos[gid].tolist(),
                axis_world=data.geom_xmat[gid].reshape(3,3)[:,2].tolist(),
                radius_m=float(model.geom_size[gid,0]),half_length_m=float(model.geom_size[gid,1]))
        hands[side]=dict(palm_position_world_m=data.xpos[palm].tolist(),
            palm_rotation_to_world=data.xmat[palm].reshape(3,3).tolist(),fingertips=tips)
    return dict(source="Measured robot joints and fixed robot geometry only; no object pose or contact oracle",
                frame="world",hands=hands)


def fixed_task_context(task):
    context=dict(source="Fixed task definition shared across scenes; infer scene landmarks from RGB",
                 scene_object_ground_truth=False)
    if task=="bimanual_photograph":
        native=importlib.import_module("dexjoco.sim.envs.panda_bimanual_photograph_env")
        context.update(
            instruction="Support the camera with the left hand. Move its viewpoint into the capture region "
                        "relative to the visible logo, aim the lens forward, and use the right hand to press "
                        "the shutter while preserving camera support and alignment.",
            capture_region=dict(center_offset_from_visible_logo_world_m=native._TARGET_REGION_OFFSET.tolist(),
                cylinder_axis_world=[1.,0.,0.],radius_m=float(native._TARGET_REGION_RADIUS),
                half_depth_m=float(native._TARGET_REGION_HALF_HEIGHT)),
            viewing_direction_world=[1.,0.,0.],maximum_viewing_angle_degrees=float(native._TARGET_ANGLE_DEG),
            requirement="Capture position, viewing direction and physical shutter contact must hold together. "
                        "Pointing toward the logo from an arbitrary position is insufficient. "
                        "The logo's current position is not supplied; estimate it from the actual RGB views.")
    elif task=="bimanual_assembly":
        context.update(instruction="Support the tray with the left hand and insert the peg with the right hand.",
            requirement="The peg must seat against the socket bottom and remain there continuously for "
                        "30 native control steps (0.6 seconds). Hole-rim contact alone is insufficient. "
                        "Estimate the peg and hole from RGB; no contact oracle or current object pose is supplied.")
    return context
