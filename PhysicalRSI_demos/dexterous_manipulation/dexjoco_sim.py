"""Galbot Astra Direct + unmodified local DexJoCo + PhysicalRSI Core.

This is a separate online-agent development track. Seeds 0, 1 and 2 are sealed.
The policy receives RGB, the first 46 proprioceptive values and robot/camera
calibration. Extra state entries containing object ground truth are excluded.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, Revision, System2Trial
from PhysicalRSI_core.experiments import ExperimentRuntime, Budget
from .common import Rejected, digest, file_hash, implementation_hash, save
from .contracts import vector, pose, object_schema
from .evidence import verify_media
from .policy import AgentTools, GalbotPolicy
from .runtime_profile import RuntimeProfile
from .dexjoco_robot_feedback import robot_geometry, fixed_task_context
from .native_dependencies import native_dependencies

TASKS = ("bimanual_photograph", "bimanual_assembly")
LAYOUT = "right_tcp_xyz_wxyz7,left_tcp_xyz_wxyz7,right_allegro16_rad,left_allegro16_rad"


def validate_response(action, observation, limits):
    if not isinstance(action, dict) or action.get("request_id") != observation["request_id"]:
        raise Rejected("Use the latest request_id")
    if not isinstance(action.get("reason"), str) or not action["reason"].strip():
        raise Rejected("Describe the action purpose and evidence")
    if action.get("mode") in {"finish", "stop"}:
        raise Rejected("Continue until native termination; model finish/stop is unavailable in this development protocol")
    if set(action) != {"mode", "request_id", "reason", "steps", "target"} or action["mode"] != "eef":
        raise Rejected("Expected mode=eef with target, steps, request_id and reason")
    if type(action["steps"]) is not int or not 1 <= action["steps"] <= limits["max_steps"]:
        raise Rejected("steps is outside the configured chunk bound")
    if not isinstance(action["target"], dict) or set(action["target"]) != {"right", "left"}:
        raise Rejected("Supply explicit right and left targets")
    for side in ("right", "left"):
        target = action["target"][side]
        if not isinstance(target, dict) or set(target) != {"position", "quaternion_wxyz", "hand_joints_rad"}:
            raise Rejected("Each side requires position[3], quaternion_wxyz[4], hand_joints_rad[16]")
        p,q = pose(target)
        old_p,old_q = pose(observation["current_eef"][side])
        if np.linalg.norm(p-old_p) > limits["max_translation_m"]+1e-8:
            raise Rejected("Target translation exceeds local motion bound")
        if (Rotation.from_quat(q[[1,2,3,0]])*Rotation.from_quat(old_q[[1,2,3,0]]).inv()).magnitude() > limits["max_rotation_rad"]+1e-8:
            raise Rejected("Target rotation exceeds local motion bound")
        hand = vector(target["hand_joints_rad"],16,"hand_joints_rad")
        bounds = np.asarray(observation["hand_limits_rad"][side])
        if np.any((hand < bounds[:,0]) | (hand > bounds[:,1])):
            raise Rejected("Hand targets exceed robot actuator limits")
    return copy.deepcopy(action)


def sim_tool_specs():
    string = dict(type="string")
    def arr(n): return dict(type="array",items=dict(type="number"),minItems=n,maxItems=n)
    arm = object_schema(dict(position=arr(3),quaternion_wxyz=arr(4),hand_joints_rad=arr(16)))
    action = object_schema(dict(request_id=string,reason=string,mode=dict(type="string",enum=["eef"]),
        steps=dict(type="integer",minimum=1,maximum=30),target=object_schema(dict(right=arm,left=arm))))
    return [dict(type="function",name=name,description=description,inputSchema=object_schema(args))
        for name,description,args in (
            ("dexjoco_start","Read this prepared DexJoCo episode's RGB/proprioception; no second reset.",dict(task=string,output_dir=string)),
            ("dexjoco_act","Execute bounded EEF and 16-joint hand targets through native simulation; return fresh feedback.",dict(observation_path=string,response=action,output_dir=string)))]


class DexJoCoTools(AgentTools):
    validate = staticmethod(validate_response)

    def handlers(self):
        return dict(dexjoco_start=self.start,dexjoco_act=self.act)

    def next_call(self):
        if self.phase == "start":
            return dict(tool="dexjoco_start",arguments=dict(task=self.task,output_dir="observations"))
        return dict(tool="dexjoco_act",observation_path=self.observation_path,request_id=self.request["request_id"])


class DexJoCoProfile(RuntimeProfile):
    policy_name = "galbot-astra-dexjoco-direct"
    tools_class = DexJoCoTools
    allow_same_thread_network_continue = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skill_root = Path(__file__).parent/"skills/dexjoco-astra-direct"

    def tool_specs(self): return sim_tool_specs()

    def prepare_workspace(self, audit):
        audit = Path(audit).resolve()
        agent = audit/"agent"
        agent.mkdir()
        (agent/"scratch").mkdir()
        (agent/"context").mkdir()
        shutil.copytree(self.skill_root, agent/".agents/skills/dexjoco-astra-direct")
        if self.memory:
            shutil.copy2(self.memory,agent/"context/candidate_memory.md")
        if getattr(self,"candidate",None):
            from PhysicalRSI_core.self_harness.artifacts import verify_harness
            from .candidate_programs import read_programs
            candidate=self.candidate
            verify_harness(candidate)
            program_hashes={}
            programs=read_programs(candidate["root"])
            (agent/"programs").mkdir()
            for item in programs:
                source=Path(candidate["root"])/"programs"/item["name"]
                destination=agent/"programs"/item["name"]
                shutil.copy2(source,destination)
                program_hashes[item["name"]]=file_hash(destination)
                if program_hashes[item["name"]]!=candidate["components"]["control"]["programs/"+item["name"]]:
                    raise ValueError("Loaded helper differs from the committed candidate")
            shutil.copy2(Path(candidate["root"])/"programs.json",agent/"context/candidate_programs.json")
            memory_sha=file_hash(agent/"context/candidate_memory.md")
            skill_sha=file_hash(agent/".agents/skills/dexjoco-astra-direct/SKILL.md")
            if (memory_sha!=candidate["components"]["memory_rules"]["memory.md"] or
                    skill_sha!=candidate["components"]["skills"]["skill/SKILL.md"]):
                raise ValueError("Loaded policy artifacts differ from the committed candidate")
            save(audit/"loaded_candidate.json",dict(candidate_id=candidate["id"],
                candidate_sha256=verify_harness(candidate),memory_sha256=memory_sha,
                skill_sha256=skill_sha,program_sha256=program_hashes,workspace=str(agent)))
        save(agent/"workspace.json",dict(robot_profile="dexjoco-panda-allegro",python_executable=sys.executable,
            layout=LAYOUT,observations_path=str(audit.parent/"observations"),notes_path=str(agent/"NOTES.md"),
            history_path=str(audit.parent/"history.json"),
            scratch=str(agent/"scratch"),simulation_paused_between_tools=True))
        (agent/"NOTES.md").write_text("# Episode notes\n\nNo observations yet.\n")
        (agent/"AGENTS.md").write_text(
            "Follow the injected dexjoco-astra-direct skill. Use native image, file and shell tools for analysis.\n"
            "Keep notes and helper code in this workspace. Control only through dexjoco_* tools.\n"
            "Do not read simulator source, object-state arrays, demos, prior final rollouts, or evaluator internals.\n"
            "Do not alter host evidence, create another simulator, teleport objects, or bypass the action tools.\n")
        return agent

    def initial_prompt(self, rollout):
        return ("You are the live Galbot Astra Direct policy for a DexJoCo DEVELOPMENT episode. "
                "Use the real RGB/proprioception feedback and native toolset to pursue the task. "
                "The simulator advances only when dexjoco_act is called; reasoning time does not advance physics. "
                "Read the timing fields: one native control step is 0.02 seconds. Each action holds its fixed target, "
                "and the torque controller needs simulated time to track it. Use 20-30 steps for a clear-space approach, "
                "then check measured error; use short chunks only where contact requires close observation. "
                "Read context/candidate_memory.md if present before choosing actions. "
                "If context/candidate_programs.json exists, read the inherited helper inventory; use programs/ "
                "through your normal shell tool when relevant. Helpers analyze public observations; all motion "
                "must still go through dexjoco_act with fresh measured feedback. "
                "Read task_context for the declared fixed task rules, and robot_geometry for measured palm and "
                "fingertip FK. These supply no scene-object pose. Use them with actual RGB to plan grasp and contact. "
                "Make progress with bounded actions, observe, correct, and continue until native_success or budget end. "
                "Maintain concise operational notes, not private chain of thought. First call: "+json.dumps(rollout.next_call()))


class DexJoCoEnvironment:
    mode = "simulation"

    def __init__(self, source, task, seed, evidence, *, max_physics_steps=1500):
        if seed in {0,1,2} or seed < 0:
            raise ValueError("Use precommitted development seeds outside sealed 0,1,2")
        if task not in TASKS:
            raise ValueError("Unsupported task")
        self.source,self.task,self.seed,self.evidence = Path(source).resolve(),task,seed,Path(evidence).resolve()
        self.dependencies=native_dependencies(str(self.source))
        self.max_physics_steps=max_physics_steps
        sys.path.insert(0,str(self.source/"dexjoco"))
        config_path=self.source/"configs/rand_obj"/(task+".yaml")
        self.task_config=yaml.safe_load(config_path.read_text())
        self.prompt=self.task_config["prompt"]
        self.native=None
        self.latest=None
        self.physics_steps=0
        self.transitions=[]
        self.history=[]
        self.native_success=False
        self.native_done=False
        self.video=None
        self.video_frames=0
        self.video_log=None
        self.limits=dict(max_steps=30,max_translation_m=.05,max_rotation_rad=.35)
        package=self.source/"dexjoco/dexjoco"
        self.source_files=[package/"sim/envs"/("panda_"+task+"_env.py"),package/"sim/mujoco_gym_env.py",
            package/"sim/envs/xmls/panda_allegro_right.xml",package/"sim/envs/xmls/panda_allegro_left.xml",
            package/"tasks"/task/"config.py",
            package/"tasks/policy_wrappers.py",package/"tasks/obs_adapters.py",config_path]

    def identity(self):
        return dict(name="dexjoco-native-simulation",task=self.task,seed=self.seed,
            native_dependencies_sha256=self.dependencies.check(),
            source={str(p.relative_to(self.source)):file_hash(p) for p in self.source_files},
            adapter=implementation_hash(),max_physics_steps=self.max_physics_steps,limits=self.limits)

    def describe(self):
        return Embodiment("dexjoco-panda-allegro",digest(self.identity()),"simulation",
            Contract("dexjoco.rgb3.proprio46/v1",unit=LAYOUT,frame="world",embodiment="panda-allegro"),
            Contract("dexjoco.dual_eef.hand16/v1",unit="meter,wxyz,hand_rad",frame="world",embodiment="panda-allegro"))

    def reset(self,case,context):
        context.check()
        self.dependencies.check(hash_bytes=True)
        save(self.evidence/"native_dependencies.json",self.dependencies.manifest)
        os.environ.setdefault("MUJOCO_GL","egl")
        from dexjoco.tasks.mappings import CONFIG_MAPPING
        self.native=CONFIG_MAPPING[self.task]().get_environment(policy_mode=True,render_mode="rgb_array",
            randomize=False,seed=self.seed,realtime=False,image_obs=True)
        raw,info=self.native.reset(seed=self.seed)
        self._last_raw=raw
        core=self.native.unwrapped
        self.simulation_start_time=float(core.data.time)
        self.control_dt=float(core.control_dt)
        self.physics_dt=float(core.physics_dt)
        self.task_context=fixed_task_context(self.task)
        # Native tasks use fixed horizons in their unchanged step methods.
        horizons=dict(bimanual_photograph=1000,bimanual_assembly=1500,
                      bimanual_hanoi=1500,bimanual_microwave_cook=1100,bimanual_unlock_ipad=1200)
        self.episode_control_limit=min(self.max_physics_steps,horizons[self.task])
        self._record_video(raw)
        self.hand_bounds=np.asarray(core._model.actuator_ctrlrange[core._allegro_ctrl_ids]).reshape(2,16,2)
        self.hand_names={"right":list(core._allegro_joint_right_names),"left":list(core._allegro_joint_left_names)} if hasattr(core,"_allegro_joint_right_names") else {}
        return dict(ready=True,observation=self._record(raw,execution=None))

    def _record_video(self,raw):
        """Save every native control observation, including the initial frame."""
        frames=[np.asarray(raw[self.task_config["camera_mapping"][camera]],np.uint8)
                for camera in ("base","wrist_left","wrist_right")]
        frame=np.concatenate(frames,axis=1)
        if self.video is None:
            self.evidence.mkdir(parents=True,exist_ok=True)
            self.video_log=(self.evidence/"video_encoder.log").open("wb")
            h,w=frame.shape[:2]
            self.video=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-n",
                "-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(1/self.control_dt),
                "-i","pipe:0","-an","-c:v","libx264","-threads","2","-preset","veryfast",
                "-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",
                str(self.evidence/"rollout.mp4")],stdin=subprocess.PIPE,stderr=self.video_log)
        self.video.stdin.write(frame.tobytes())
        self.video_frames+=1

    def _record(self,raw,execution):
        state=np.asarray(raw["state"][:46],float)
        if state.shape != (46,) or not np.isfinite(state).all():
            raise ValueError("Invalid native proprioception")
        request_id=uuid.uuid4().hex
        folder=self.evidence/"observations"/request_id
        folder.mkdir(parents=True)
        images=[]
        for camera in ("base","wrist_left","wrist_right"):
            path=folder/(camera+".png")
            Image.fromarray(np.asarray(raw[self.task_config["camera_mapping"][camera]],np.uint8)).save(path)
            images.append(dict(camera=camera,path=str(path),sha256=file_hash(path)))
        eef={s:dict(position=state[i*7:i*7+3].tolist(),quaternion_wxyz=state[i*7+3:i*7+7].tolist(),
                    hand_joints_rad=state[14+i*16:30+i*16].tolist()) for i,s in enumerate(("right","left"))}
        if execution is not None:
            execution["tracking_error"]={}
            for side in ("right","left"):
                target=execution["target"][side]
                measured=eef[side]
                tq=np.asarray(target["quaternion_wxyz"])
                mq=np.asarray(measured["quaternion_wxyz"])
                execution["tracking_error"][side]=dict(
                    translation_m=float(np.linalg.norm(np.asarray(target["position"])-measured["position"])),
                    rotation_rad=float((Rotation.from_quat(tq[[1,2,3,0]])*
                        Rotation.from_quat(mq[[1,2,3,0]]).inv()).magnitude()),
                    maximum_hand_joint_error_rad=float(np.max(np.abs(np.asarray(target["hand_joints_rad"])-
                        measured["hand_joints_rad"]))))
        core=self.native.unwrapped
        calibration={}
        ids={"base":core._front_camera_id,"wrist_left":core._wrist_left_camera_id,"wrist_right":core._wrist_right_camera_id}
        for camera,camera_id in ids.items():
            calibration[camera]=dict(position_world=core._data.cam_xpos[camera_id].tolist(),
                rotation_camera_to_world=core._data.cam_xmat[camera_id].reshape(3,3).tolist(),
                vertical_fov_degrees=float(core._model.cam_fovy[camera_id]),
                axes="OpenGL: camera looks along -Z, +Y is up")
        packet=dict(request_id=request_id,observation_path=str(folder/"observation.json"),
            task=self.prompt,layout=LAYOUT,state=state.tolist(),current_eef=eef,images=images,
            hand_limits_rad={s:self.hand_bounds[i].tolist() for i,s in enumerate(("right","left"))},
            hand_joint_names=self.hand_names,camera_calibration=calibration,frame="world",limits=self.limits,
            robot_geometry=robot_geometry(core),task_context=self.task_context,
            physics_steps=self.physics_steps,physics_steps_remaining=self.episode_control_limit-self.physics_steps,
            native_control_steps=self.physics_steps,
            native_control_limit=self.episode_control_limit,
            native_control_steps_remaining=self.episode_control_limit-self.physics_steps,
            history_path=str(self.evidence/"history.json"),
            timing=dict(control_dt_s=self.control_dt,physics_dt_s=self.physics_dt,
                integrator_steps_per_control_step=int(core._n_substeps),
                simulation_time_s=float(core.data.time)-self.simulation_start_time,
                target_execution="fixed target held for all requested control steps",
                legacy_physics_steps_unit="native env.step calls, not MuJoCo integration steps"),
            execution=execution,native_success=self.native_success,gateway_mode="simulation",
            rollout_finished=self.native_done or self.physics_steps>=self.max_physics_steps)
        save(packet["observation_path"],packet)
        self.history.append(dict(decision=len(self.history),observation_path=packet["observation_path"],
            native_control_steps=self.physics_steps,current_eef=eef,execution=execution,
            native_success=self.native_success,rollout_finished=packet["rollout_finished"]))
        save(self.evidence/"history.json",self.history)
        self.latest=packet
        return packet

    def step(self,action,context):
        context.check()
        validate_response(action,self.latest,self.limits)
        steps=min(action["steps"],self.max_physics_steps-self.physics_steps)
        acknowledgements=[]
        # Match upstream GPTOnlyTools: keep the requested goal fixed throughout
        # the chunk. DexJoCo performs its own feedback control at every substep.
        poses,hands=[],[]
        for side in ("right","left"):
            target=action["target"][side]
            poses.append(np.r_[target["position"],target["quaternion_wxyz"]])
            hands.append(np.asarray(target["hand_joints_rad"],float))
        native_action=np.concatenate([*poses,*hands])
        for index in range(steps):
            context.check()
            raw,reward,terminated,truncated,info=self.native.step(native_action)
            self._last_raw=raw
            self.physics_steps+=1
            self.native_success=bool(info.get("succeed",False))
            self.native_done=bool(terminated or truncated)
            self._record_video(raw)
            record=dict(physics_step=self.physics_steps,action=native_action.tolist(),reward=float(reward),
                        simulation_time_s=float(self.native.unwrapped.data.time)-self.simulation_start_time,
                        native_success=self.native_success,terminated=bool(terminated),truncated=bool(truncated))
            acknowledgements.append(record)
            self.transitions.append(record)
            if self.native_done:
                break
        save(self.evidence/"native_transitions.json",self.transitions)
        packet=self._record(raw,dict(no_execution=False,physics_steps_executed=len(acknowledgements),
            target=action["target"],reason=action["reason"],native_terminated=self.native_done))
        return dict(observation=packet,terminated=packet["rollout_finished"])

    def stop_only(self):
        return dict(quiescent=True,simulation_paused=True,physics_steps=self.physics_steps)

    def quiesce(self,context=None): return self.stop_only()

    def close(self):
        try:
            if self.video is not None:
                video,self.video=self.video,None
                try:
                    video.stdin.close()
                    result=video.wait(timeout=30)
                    if result!=0:
                        raise RuntimeError("Rollout video encoding failed; see video_encoder.log")
                    path=self.evidence/"rollout.mp4"
                    save(self.evidence/"video_manifest.json",dict(path=str(path),sha256=file_hash(path),
                        frames=self.video_frames,fps=1/self.control_dt,
                        cameras=["base","wrist_left","wrist_right"],
                        source="initial observation and every native control-step observation",
                        native_control_steps=self.physics_steps,continuous_control_frames=True))
                finally:
                    if video.poll() is None:
                        video.kill();video.wait()
                    self.video_log.close()
        finally:
            if self.native is not None:
                self.native.close()
                self.native=None
            self.dependencies.check(hash_bytes=True)


class NativeVerifier:
    def __init__(self,environment): self.environment=environment
    def identity(self):
        return dict(name="dexjoco-native-info-succeed",revision=implementation_hash())
    def verify(self,case,trace):
        media=verify_media(trace)
        success=self.environment.native_success
        if success != bool(trace[-1]["observation"]["native_success"]):
            raise ValueError("Native success disagrees with recorded transition")
        return dict(outcome="success" if success else "failure",
            reason="Unmodified DexJoCo info.succeed at the recorded terminal transition",
            measurements=dict(media,physics_steps=self.environment.physics_steps,native_success=success,
                              agent_track=True,official_native_policy_score=False))


def run_episode(args):
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    env=DexJoCoEnvironment(args.source,args.task,args.seed,root,max_physics_steps=args.physics_steps)
    profile=DexJoCoProfile(auth=args.auth,login_home=args.login_home,memory=args.memory)
    policy=GalbotPolicy(env,profile,root/"galbot",codex=args.codex,timeout=args.agent_timeout)
    runtime=ExperimentRuntime(root/"experiments")
    trial=System2Trial(Revision("physicalrsi-core","3c8cf7f12d4041babe30bab98854cc504a5afe88"),
        args.candidate_id,policy.describe().revision,"development")
    save(root/"precommit.json",dict(task=args.task,seed=args.seed,track="online-agent-development",
         max_actions=args.max_actions,max_physics_steps=args.physics_steps,policy=policy.identity(),
         environment=env.identity(),sealed_final_seeds=[0,1,2]))
    try:
        receipt=runtime.run("episode",task=env.prompt,case=dict(task=args.task,seed=args.seed,split="development"),
            scope="dexjoco-galbot-direct-development",environment=env,policy=policy,verifier=NativeVerifier(env),
            budget=Budget(max_steps=args.max_actions,seconds=args.seconds),system2=trial)
        save(root/"result.json",receipt)
        print(json.dumps(dict(state=receipt["state"],outcome=receipt["outcome"],physics_steps=env.physics_steps,evidence=str(root))),flush=True)
        return receipt
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,help="Path to the compatible native DexJoCo checkout")
    parser.add_argument("--task",choices=TASKS,required=True)
    parser.add_argument("--seed",type=int,default=1103)
    parser.add_argument("--output",required=True)
    parser.add_argument("--auth",choices=["api","chatgpt"],default="chatgpt")
    parser.add_argument("--login-home")
    parser.add_argument("--codex",default="codex")
    parser.add_argument("--memory")
    parser.add_argument("--candidate-id",default="baseline")
    parser.add_argument("--max-actions",type=int,default=1500)
    parser.add_argument("--physics-steps",type=int,default=1500,help="Native control steps; historical option name")
    parser.add_argument("--seconds",type=float,default=7200.)
    parser.add_argument("--agent-timeout",type=float,default=180.)
    run_episode(parser.parse_args())


if __name__ == "__main__": main()
