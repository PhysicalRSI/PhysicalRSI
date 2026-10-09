import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PhysicalRSI_core.contracts import Context

from PhysicalRSI_demos.dexterous_manipulation.common import Rejected
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_sim import DexJoCoEnvironment, validate_response, sim_tool_specs
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_robot_feedback import robot_geometry
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_campaign import ReflectionTools, freeze_candidate, verify_executing_source
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from hybrid_rollout.robodojo.io import InputError
from PhysicalRSI_demos.dexterous_manipulation.evidence_evaluator import trajectory_media
from PhysicalRSI_demos.dexterous_manipulation.common import save


def test_memory_candidate_keeps_foundation_and_changes_freeze(tmp_path):
    parent=freeze_candidate(tmp_path/"parent","parent")
    child=freeze_candidate(tmp_path/"child","child",parent=parent,
        memory="Inspect measured progress before repeating actions.\n",provenance={"episode":"test-fixture"})
    assert child["components"]["foundation"]==parent["components"]["foundation"]
    assert verify_harness(child)!=verify_harness(parent)
    verify_executing_source(child)
    (tmp_path/"child/memory.md").write_text("changed after freezing")
    with pytest.raises(ValueError,match="Changed or missing"):
        verify_harness(child)


def test_reflection_requires_real_evidence_ids():
    tool=ReflectionTools(dict(episodes=[dict(id="actual-development-episode")]))
    tool.read()
    with pytest.raises(InputError,match="actual development"):
        tool.submit("Only use observed feedback. "*10,["invented"])
    tool.submit("Only use observed feedback. "*10,["actual-development-episode"])
    assert tool.phase=="done"


@pytest.mark.parametrize("task",["bimanual_photograph","bimanual_assembly"])
def test_native_simulator_feedback_without_object_truth(task,tmp_path,monkeypatch):
    pytest.importorskip("mujoco")
    configured=os.environ.get("DEXJOCO_SOURCE")
    if not configured or not Path(configured).is_dir():
        pytest.skip("Set DEXJOCO_SOURCE to run the native simulator checks")
    source=Path(configured).resolve()
    monkeypatch.setenv("MUJOCO_GL","egl")
    monkeypatch.setenv("EGL_PLATFORM","surfaceless")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES",str(source/"configs/egl/10_nvidia.json"))
    env=DexJoCoEnvironment(source,task,1109,tmp_path/task,max_physics_steps=2)
    try:
        observation=env.reset({},Context("test"))["observation"]
        assert len(observation["state"])==46
        assert not any(k in observation for k in ("camera_ori_pose","peg_ori_pose","socket_ori_pose","table_delta_height"))
        assert set(observation["robot_geometry"]["hands"])=={"left","right"}
        assert set(observation["robot_geometry"]["hands"]["right"]["fingertips"])=={"index","middle","ring","thumb"}
        with pytest.raises(Rejected,match="native termination"):
            validate_response(dict(mode="finish",request_id=observation["request_id"],reason="Expected failure"),
                              observation,observation["limits"])
        target=copy.deepcopy(observation["current_eef"])
        for side in target:
            bounds=np.asarray(observation["hand_limits_rad"][side])
            target[side]["hand_joints_rad"]=np.clip(target[side]["hand_joints_rad"],bounds[:,0],bounds[:,1]).tolist()
        response=dict(mode="eef",request_id=observation["request_id"],reason="Native hold test",steps=2,target=target)
        result=env.step(response,Context("test"))
        assert env.physics_steps==2
        assert result["terminated"] is True
        assert result["observation"]["request_id"]!=observation["request_id"]
        assert result["observation"]["native_success"] is env.native_success
        assert len(env.transitions[-1]["action"])==46
        assert result["observation"]["timing"]["simulation_time_s"]==pytest.approx(.04)
        assert result["observation"]["timing"]["integrator_steps_per_control_step"]==10
    finally:
        env.close()
    video=json.loads((tmp_path/task/"video_manifest.json").read_text())
    assert video["frames"]==3
    assert video["fps"]==pytest.approx(50)
    assert result["observation"]["native_control_steps_remaining"]==0
    assert "translation_m" in result["observation"]["execution"]["tracking_error"]["left"]
    history=json.loads((tmp_path/task/"history.json").read_text())
    assert history[-1]["observation_path"]==result["observation"]["observation_path"]
    trajectory=tmp_path/"trajectory.json"
    save(trajectory,[dict(observation=observation),result])
    closure=trajectory_media(trajectory,tmp_path)
    assert len([name for name in closure if name.endswith(".png")])==6
    movie=Path(video["path"])
    assert closure[str(movie.relative_to(tmp_path))]==video["sha256"]
    movie.write_bytes(movie.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="video changed"):
        trajectory_media(trajectory,tmp_path)


def test_robot_fk_does_not_reveal_object_state(tmp_path,monkeypatch):
    mujoco=pytest.importorskip("mujoco")
    configured=os.environ.get("DEXJOCO_SOURCE")
    if not configured or not Path(configured).is_dir():
        pytest.skip("Set DEXJOCO_SOURCE to run the native simulator checks")
    source=Path(configured).resolve()
    monkeypatch.setenv("MUJOCO_GL","egl")
    monkeypatch.setenv("EGL_PLATFORM","surfaceless")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES",str(source/"configs/egl/10_nvidia.json"))
    env=DexJoCoEnvironment(source,"bimanual_photograph",1519,tmp_path/"fk-isolation",max_physics_steps=1)
    try:
        env.reset({},Context("fk-isolation"))
        core=env.native.unwrapped
        before=robot_geometry(core)
        free_ids=np.flatnonzero(core.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        assert len(free_ids)>0
        # Perturb only free object poses in this test fixture, preserving all
        # robot joints; the exported robot FK must remain identical.
        for joint in free_ids:
            address=core.model.jnt_qposadr[joint]
            core.data.qpos[address:address+3]+=[.17,.13,.09]
        mujoco.mj_forward(core.model,core.data)
        assert robot_geometry(core)==before
        assert {t["name"] for t in sim_tool_specs()}=={"dexjoco_start","dexjoco_act"}
    finally:
        env.close()


def test_fixed_target_chunk_provides_controller_time(tmp_path,monkeypatch):
    """A 5 cm goal must be held, not repeatedly ramped from the starting pose."""
    pytest.importorskip("mujoco")
    configured=os.environ.get("DEXJOCO_SOURCE")
    if not configured or not Path(configured).is_dir():
        pytest.skip("Set DEXJOCO_SOURCE to run the native simulator checks")
    source=Path(configured).resolve()
    monkeypatch.setenv("MUJOCO_GL","egl")
    monkeypatch.setenv("EGL_PLATFORM","surfaceless")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES",str(source/"configs/egl/10_nvidia.json"))
    env=DexJoCoEnvironment(source,"bimanual_photograph",1509,tmp_path/"tracking",max_physics_steps=30)
    try:
        observation=env.reset({},Context("tracking"))["observation"]
        target=copy.deepcopy(observation["current_eef"])
        start=np.asarray(target["left"]["position"])
        target["left"]["position"]=(start+np.array([.03,0,-.04])).tolist()
        for side in target:
            bounds=np.asarray(observation["hand_limits_rad"][side])
            target[side]["hand_joints_rad"]=np.clip(target[side]["hand_joints_rad"],bounds[:,0],bounds[:,1]).tolist()
        result=env.step(dict(mode="eef",request_id=observation["request_id"],reason="Fixed target response test",
                             steps=30,target=target),Context("tracking"))
        measured=np.asarray(result["observation"]["current_eef"]["left"]["position"])
        assert np.linalg.norm(measured-start)>.035
        assert np.linalg.norm(measured-target["left"]["position"])<.015
        assert result["observation"]["timing"]["simulation_time_s"]==pytest.approx(.6)
    finally:
        env.close()
