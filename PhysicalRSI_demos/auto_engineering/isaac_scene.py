"""Native Isaac Sim worker. Build the lab from primitives; actuate real joints.

Run only with an Isaac Sim Python environment. No surrogate physics fallback.
"""
import argparse
import json
from pathlib import Path
import time


def capture_sensor_frame():
    """Render active annotators at the settled scene's current simulation time."""
    import omni.replicator.core as rep
    rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=4)


def initial_sensor_frames(world, overhead, view, *, timeout=120):
    """Wait for complete native frames without advancing the settled physics.

    Shader compilation can outlast a fixed number of physics steps. Empty
    annotator arrays are not sensor observations and must never reach System 1.
    """
    started = time.monotonic()
    physics_state = (world.current_time_step_index, world.current_time)
    renders = 0
    width, height = overhead.get_resolution()
    view_width, view_height = view.get_resolution()
    while True:
        rgb, depth, presentation = overhead.get_rgba(), overhead.get_depth(), view.get_rgba()
        shapes = tuple(getattr(value, 'shape', None) for value in (rgb, depth, presentation))
        if shapes == ((height, width, 4), (height, width), (view_height, view_width, 4)):
            return rgb, depth, presentation, dict(renders=renders, seconds=time.monotonic()-started,
                capture_mode='replicator-static/v1', physics_step=physics_state[0],
                physics_time=physics_state[1])
        if time.monotonic()-started >= timeout:
            raise TimeoutError('Native RGB-D/presentation sensors did not become ready: ' + repr(shapes))
        capture_sensor_frame()
        if (world.current_time_step_index, world.current_time) != physics_state:
            raise RuntimeError('Native sensor warmup advanced settled physics')
        renders += 1
        time.sleep(.05)


def run(args):
    from importlib.metadata import version
    runtime_version = version('isaacsim')
    if runtime_version not in {'5.1.0', '5.1.0.0'}:
        raise ValueError('This native protocol requires Isaac Sim 5.1.0')
    from isaacsim import SimulationApp
    extra_args = []
    if getattr(args, 'runtime_root', None):
        extra_args = ['--portable', '--portable-root', str(Path(args.runtime_root).resolve())]
    app = SimulationApp({'headless':not args.gui, 'disable_viewport_updates':not args.gui,
                         'width':1920, 'height':1080,
                         'renderer':'RayTracedLighting', 'anti_aliasing':2,
                         'multi_gpu':False, 'extra_args':extra_args})
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation
        import imageio.v2 as imageio
        from PIL import Image
        from pxr import UsdLux, UsdPhysics, UsdGeom, UsdShade, Gf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.api.objects import FixedCuboid, DynamicCylinder
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
        from isaacsim.sensors.camera import Camera
        from PhysicalRSI_core.infra.storage import atomic_json, file_digest, digest
        from .task import layout, judge, SCOPE
        from .perception import detect
        from .policy import plan

        out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
        memory=json.loads(Path(args.memory).read_text()) if args.memory else None
        spec=layout(args.seed, getattr(args, 'scenario', 'standard'))
        atomic_json(out/'layout.json',spec)
        world=World(stage_units_in_meters=1.,physics_dt=1/60,rendering_dt=1/60)
        world.scene.add(FixedCuboid('/World/Floor',name='floor',position=np.array([0.,0.,-.87]),
            scale=np.array([3.,3.,.08]),color=np.array([.16,.16,.16])))
        stage=world.stage
        light=UsdLux.DomeLight.Define(stage,'/World/LabLight');light.CreateIntensityAttr(150)
        key=UsdLux.DistantLight.Define(stage,'/World/Key');key.CreateIntensityAttr(220)
        material=PhysicsMaterial('/World/GripMaterial',static_friction=1.5,dynamic_friction=1.2,restitution=0.)
        def block(name,position,scale,color):
            return world.scene.add(FixedCuboid('/World/'+name,name=name,position=np.array(position),
                                               scale=np.array(scale),color=np.array(color)))
        block('Bench',[.38,0,-.035],[1.0,.85,.07],[.24,.24,.24])
        block('BenchFront',[.38,-.42,-.08],[1.,.012,.10],[.08,.13,.18])
        # A receiving rack with a visible green landing pad and low guard rails.
        gx,gy,gz=spec['receiver']
        block('Receiver',[gx,gy,gz],[.13,.13,.018],[.08,.65,.25])
        for i,dx in enumerate([-.07,.07]):
            block('RackRail'+str(i),[gx+dx,gy,.042],[.009,.14,.065],[.34,.39,.45])
        tx,ty,tz=spec['tube']
        block('SourceRack',[tx,ty,.006],[.10,.10,.012],[.48,.52,.58])
        def cylinder(name,position,radius,height,color,mass):
            path='/World/'+name
            UsdGeom.Xform.Define(stage,path)
            collider=UsdGeom.Cylinder.Define(stage,path+'/Collision')
            collider.CreateRadiusAttr(radius);collider.CreateHeightAttr(height);collider.CreateAxisAttr('Z')
            collider.CreatePurposeAttr('guide');UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(material.material,materialPurpose='physics')
            return world.scene.add(SingleRigidPrim(path,name=name,position=np.array(position),mass=mass))
        tube=cylinder('BlueSample',spec['tube'],spec['tube_radius'],spec['tube_height'],[.025,.13,.85],.025)
        other=cylinder('RedSample',spec['distractor'],.018,.12,[.85,.08,.08],.025)
        # Opaque beaker-shaped obstruction: rigid-body demo, no fluid claim.
        beaker=cylinder('Beaker',spec['obstacle'],spec['obstacle_radius'],spec['obstacle_height'],[.8,.43,.08],.3)
        robot=world.scene.add(Franka('/World/Franka',name='franka'))
        from .lab_style import dress
        atomic_json(out/'design.json',dress(stage,spec,out/'assets'))
        overhead=Camera('/World/Overhead',position=np.array([.43,0,1.25]),resolution=(640,640),frequency=60)
        overhead.set_world_pose(position=np.array([.43,0,1.25]),orientation=np.array([1.,0,0,0]),camera_axes='usd')
        overhead.set_focal_length(18.);overhead.set_horizontal_aperture(20.955)
        overhead.set_clipping_range(.05,5.)
        view=Camera('/World/Presentation',position=np.array([1.25,-1.5,.92]),resolution=(1920,1080),frequency=60)
        eye=np.array([1.25,-1.5,.92]);target=np.array([.31,.10,.22]);back=(eye-target)/np.linalg.norm(eye-target)
        right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        xyzw=Rotation.from_matrix(np.column_stack([right,up,back])).as_quat()
        view.set_world_pose(position=eye,orientation=xyzw[[3,0,1,2]],camera_axes='usd')
        view.set_focal_length(35.);view.set_horizontal_aperture(36.);view.set_clipping_range(.05,5.)
        world.reset();overhead.initialize();view.initialize();overhead.add_distance_to_image_plane_to_frame()
        from isaacsim.core.utils.types import ArticulationAction
        home=np.array([-1.3,-.57,0.,-2.81,0.,3.04,.74,.04,.04])
        robot.set_joint_positions(home)
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=home))
        def render_state():
            transforms=robot._articulation_view._physics_view.get_link_transforms()[0]
            poses={}
            for name,t in zip(robot._articulation_view.body_names,transforms):
                poses['/World/Franka/'+name]=[t[:3].tolist(),t[[6,3,4,5]].tolist()]
            for path,body in [('/World/BlueSample',tube),('/World/RedSample',other),('/World/Beaker',beaker)]:
                poses[path]=[p.tolist() for p in body.get_world_pose()]
            return poses
        # Settle physics before the explicit static sensor capture below.
        for _ in range(90):world.step(render=False, update_fabric=True)
        rgb,depth,presentation,sensor_warmup=initial_sensor_frames(world,overhead,view)
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        Image.fromarray(presentation[...,:3].astype('uint8')).save(out/'overview.png')
        Image.fromarray(rgb[...,:3].astype('uint8')).save(out/'observation.png')
        np.save(out/'depth.npy',depth,allow_pickle=False)
        atomic_json(out/'camera.json',dict(intrinsics=overhead.get_intrinsics_matrix().tolist(),
            world_to_optical=overhead.get_view_matrix_ros().tolist(),depth_source='rendered distance_to_image_plane',
            physics_hz=60,render_hz=60,video_hz=20,sensor_warmup=sensor_warmup,
            presentation={'resolution':[1920,1080],'intrinsics':view.get_intrinsics_matrix().tolist(),
                          'world_to_optical':view.get_view_matrix_ros().tolist()}))
        public=detect(rgb,depth,overhead.get_world_points_from_image_coords)
        atomic_json(out/'public-observation.json',public)
        if getattr(args, 'bridge', None):
            # Reset is complete before the parent runtime dispatches System 1's
            # macro action. Simulator object state never enters this message.
            from .trial_bridge import receive_plan
            request=receive_plan(Path(args.bridge), dict(public_observation=public,
                joint_positions=robot.get_joint_positions().tolist()), timeout=120)
            memory=request['memory'];instructions=request['plan']
            if instructions != plan(public,memory):
                raise ValueError('Dispatched plan differs from the public observation and memory')
        else:
            instructions=plan(public,memory)
        atomic_json(out/'plan.json',instructions)
        controller=PickPlaceController(name='sample_transfer',gripper=robot.gripper,
            robot_articulation=robot,end_effector_initial_height=instructions['transit_height_m'],
            events_dt=[.008,.005,.05,.01,.005,.005,.005,.01,.008,.008])
        articulation=robot.get_articulation_controller()
        # Robot/scene assets may contain external references; do not claim a full asset closure.
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        atomic_json(out/'robot.json',dict(name='Franka',dof_names=list(robot.dof_names),
            public_inputs=['RGB','rendered metric depth','camera calibration','joint state'],
            runtime='Isaac Sim 5.1',scene='procedurally built lab',external_asset_closure_attested=False))
        start=time.monotonic();trace=[];done_at=None
        public_trace=[dict(observation=dict(public_observation=public,
            joint_positions=robot.get_joint_positions().tolist()))]
        with imageio.get_writer(out/'episode.mp4',fps=20,codec='libx264',quality=7,macro_block_size=8) as video:
            for step in range(2400):
                if time.monotonic()-start>600:raise TimeoutError('Native episode wall-time limit')
                applied={'kind':'hold'}
                if not controller.is_done():
                    action=controller.forward(picking_position=np.array(instructions['pick']),
                        placing_position=np.array(instructions['place']),current_joint_positions=robot.get_joint_positions(),
                        end_effector_offset=np.array([0.,0.,.005]))
                    articulation.apply_action(action)
                    applied={'kind':'joint_targets', **{name:np.asarray(getattr(action,name)).tolist()
                        for name in ('joint_positions','joint_velocities','joint_efforts','joint_indices')
                        if getattr(action,name,None) is not None}}
                elif done_at is None:done_at=step
                world.step(render=True)
                public_trace.append(dict(step=step, action=applied,
                    observation=dict(joint_positions=robot.get_joint_positions().tolist()),
                    phase=int(controller.get_current_event())))
                if step%3==0:
                    frame=view.get_rgba()
                    if frame is not None and frame.size:video.append_data(frame[...,:3].astype('uint8'))
                    trace.append(dict(step=step,phase=int(controller.get_current_event()),
                        joint_positions=robot.get_joint_positions().tolist(),
                        render_poses=render_state()))
                if done_at is not None and step-done_at>=180:break
        position,orientation=tube.get_world_pose()
        measured=dict(tube_position=position.tolist(),tube_orientation_wxyz=orientation.tolist(),
            tube_velocity=tube.get_linear_velocity().tolist(),obstacle_position=beaker.get_world_pose()[0].tolist(),
            distractor_position=other.get_world_pose()[0].tolist(),gripper_opening=float(sum(robot.gripper.get_joint_positions())))
        verdict=judge(spec,measured)
        if not controller.is_done():verdict['success']=False;verdict['checks']['controller_completed']=False
        Image.fromarray(view.get_rgba()[...,:3].astype('uint8')).save(out/'final.png')
        atomic_json(out/'trajectory.json',trace);atomic_json(out/'private-evaluation.json',measured)
        atomic_json(out/'public-rollout.json',public_trace)
        result=dict(state='completed',**verdict,seed=args.seed,layout_sha256=digest(spec),
            public_observation=public,memory=memory,elapsed_seconds=time.monotonic()-start,
            video='episode.mp4',physics_steps=step+1,backend='isaacsim-5.1',
            runtime_version=runtime_version,
            controller_completed=bool(controller.is_done()),
            evidence={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()})
        atomic_json(out/'result.json',result)
        print(json.dumps({'success':verdict['success'],'output':str(out),'checks':verdict['checks']}),flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        import sys
        sys.stderr.flush()
        if getattr(args, 'bridge', None):
            from .trial_bridge import report_failure
            report_failure(args.bridge, error)
        raise
    finally:app.close()


def main():
    from .task import SCENARIOS
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--scenario', choices=SCENARIOS, default='standard')
    p.add_argument('--memory');p.add_argument('--output',required=True);p.add_argument('--gui',action='store_true')
    p.add_argument('--bridge',help='Owned Trial IPC directory; wait for a plan after sensor reset')
    p.add_argument('--runtime-root',help='Owned Kit portable data/cache/log directory')
    args=p.parse_args()
    if bool(args.memory)==bool(args.bridge):p.error('Use exactly one of --memory or --bridge')
    run(args)

if __name__=='__main__':main()
