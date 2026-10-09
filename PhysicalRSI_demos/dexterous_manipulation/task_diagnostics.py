"""Post-episode replay feedback for System 2, never an acting-policy sensor.

Replay has no model, robot or trial-quota access. Every recorded public-state
checkpoint and native terminal flag must match before diagnostic feedback is
published. Selection continues to use the original native success verdict.
"""
import json
from pathlib import Path
import sys

import numpy as np

from PhysicalRSI_core.infra.storage import digest as core_digest, relative_path
from .common import file_hash, save
from .native_dependencies import native_dependencies


def _read(path):
    return json.loads(Path(path).read_text())


def diagnostic_inputs(trajectory, source):
    trajectory = Path(trajectory).resolve()
    trial = trajectory.parent
    specification = _read(trial / "experiment.json")
    receipt = _read(trial / "receipt.json")
    case = specification["case"]
    if (case.get("split") not in {"development", "validation"}
            or type(case.get("seed")) is not int or case["seed"] <= 2
            or receipt["outcome"] not in {"success", "failure"}):
        raise ValueError("Diagnostics require a completed, non-final policy episode")
    if specification["task"] not in {"bimanual_assembly", "bimanual_photograph"}:
        raise ValueError("Unsupported diagnostic task")
    for name, sha in receipt["evidence"].items():
        path = trial / relative_path(name)
        if not path.resolve().is_relative_to(trial) or file_hash(path) != sha:
            raise ValueError("Completed episode evidence changed")
    trace = _read(trajectory)
    live = Path(trace[0]["observation"]["observation_path"]).parents[2]
    transitions_path = live / "native_transitions.json"
    transitions = _read(transitions_path)
    if not transitions or len(transitions) != trace[-1]["observation"]["native_control_steps"]:
        raise ValueError("Native replay must cover the complete recorded trajectory")
    if bool(transitions[-1]["native_success"]) != (receipt["outcome"] == "success"):
        raise ValueError("Replay source and native verdict disagree")
    if not (live / "video_manifest.json").is_file():
        raise ValueError("Episode has not completed teardown")
    dependency = native_dependencies(str(Path(source).resolve()))
    dependency.check(hash_bytes=True)
    recorded_dependencies = live / "native_dependencies.json"
    if recorded_dependencies.exists() and _read(recorded_dependencies) != dependency.manifest:
        raise ValueError("Replay native source or dependencies differ from the episode")
    inputs = {str(p): file_hash(p) for p in (
        trajectory, trial / "experiment.json", trial / "receipt.json", transitions_path)}
    if recorded_dependencies.exists():
        inputs[str(recorded_dependencies)] = file_hash(recorded_dependencies)
    return specification, trace, transitions, inputs, core_digest(dependency.manifest)


def assembly_sample(core, info, step):
    m, d = core.model, core.data
    peg = m.body("industreal_round_peg_8mm").id
    socket = m.body("industreal_tray_insert_round_peg_8mm").id
    cylinder = m.geom("industreal_round_peg_8mm_collision").id
    box = m.geom("industreal_round_peg_8mm_collision_upper").id
    wall = m.geom("industreal_tray_insert_round_peg_8mm_wall_pos_x").id
    bottom = m.geom("industreal_tray_insert_round_peg_8mm_bottom_contact").id
    rotation = d.xmat[socket].reshape(3, 3)
    tip = d.geom_xpos[cylinder] - d.geom_xmat[cylinder].reshape(3, 3)[:, 2] * m.geom_size[cylinder, 1]
    other = d.geom_xpos[box] + d.geom_xmat[box].reshape(3, 3)[:, 2] * m.geom_size[box, 2]
    tip_local = rotation.T @ (tip - d.xpos[socket])
    other_local = rotation.T @ (other - d.xpos[socket])
    axis = rotation.T @ d.xmat[peg].reshape(3, 3)[:, 2]
    rim = float(m.geom_pos[wall, 2] + m.geom_size[wall, 2])
    bottom_top = float(m.geom_pos[bottom, 2] + m.geom_size[bottom, 2])
    clearance = float(m.geom_pos[wall, 0] - m.geom_size[wall, 0] - m.geom_size[cylinder, 0])
    return dict(step=step, bottom_contact=bool(info["bottom_contact"]),
                stable_count=int(info["success_stable_count"]),
                tip_xy=tip_local[:2].tolist(), tip_z=float(tip_local[2]),
                other_xy=other_local[:2].tolist(), other_z=float(other_local[2]),
                axis_degrees=float(np.degrees(np.arccos(np.clip(axis[2], -1, 1)))),
                rim=rim, bottom=bottom_top, clearance=clearance)


def summarize(task, rows):
    """Report observed bottlenecks, not a universal causal proof or a score."""
    if task == "bimanual_photograph":
        flags = ("region_pass", "angle_pass", "shutter_pressed")
        counts = {k + "_steps": sum(bool(r[k]) for r in rows) for k in flags}
        counts["simultaneous_pass_steps"] = sum(all(r[k] for k in flags) for r in rows)
        counts["position_and_angle_steps"] = sum(r["region_pass"] and r["angle_pass"] for r in rows)
        counts["index_shutter_pressed_steps"] = sum(bool(r["index_shutter_pressed"]) for r in rows)
        labels = ["never_satisfied_" + k for k in flags if not counts[k + "_steps"]]
        if all(counts[k + "_steps"] for k in flags) and not counts["simultaneous_pass_steps"]:
            labels.append("conditions_satisfied_at_different_times")
        return dict(observed_bottlenecks=labels, measurements=counts,
                    interpretation="Position, viewing direction and shutter must coincide; index contact is diagnostic, not an additional success requirement.")
    if task != "bimanual_assembly" or not rows:
        raise ValueError("Unsupported or empty diagnostic trajectory")
    first = rows[0]
    near = lambda xy, z, r: np.linalg.norm(xy) < .06 and r["bottom"] - .02 < z < r["rim"] + .08
    tip_near = [r for r in rows if near(r["tip_xy"], r["tip_z"], r)]
    wrong_near = [r for r in rows if r["axis_degrees"] > 90 and near(r["other_xy"], r["other_z"], r)]
    counts = dict(bottom_contact_steps=sum(r["bottom_contact"] for r in rows),
                  maximum_stable_count=max(r["stable_count"] for r in rows),
                  correct_tip_near_socket_steps=len(tip_near),
                  opposite_end_near_socket_steps=len(wrong_near))
    labels = []
    if not counts["bottom_contact_steps"]:
        labels.append("never_contacted_socket_bottom")
    elif counts["maximum_stable_count"] < 30:
        labels.append("bottom_contact_not_sustained_for_30_steps")
    if wrong_near:
        labels.append("opposite_end_presented_near_socket")
    if not tip_near:
        labels.append("correct_tip_never_reached_socket_neighborhood")
    closest = min(rows, key=lambda r: np.linalg.norm([*r["tip_xy"], r["tip_z"] - r["rim"]]))
    # Errors only; no current world coordinates, pose arrays or target actions.
    alignment = dict(step=closest["step"], lateral_error_norm_mm=float(np.linalg.norm(closest["tip_xy"]) * 1000),
                     maximum_axis_offset_mm=max(abs(x) for x in closest["tip_xy"]) * 1000,
                     axis_error_degrees=closest["axis_degrees"],
                     height_above_rim_mm=(closest["tip_z"] - closest["rim"]) * 1000,
                     height_above_bottom_mm=(closest["tip_z"] - closest["bottom"]) * 1000)
    if tip_near and alignment["maximum_axis_offset_mm"] > first["clearance"] * 1000:
        labels.append("closest_tip_offset_exceeds_nominal_clearance")
    return dict(observed_bottlenecks=labels, measurements=counts,
                closest_correct_tip_to_rim=alignment,
                fixed_geometry=dict(nominal_single_side_clearance_mm=first["clearance"] * 1000,
                                    rim_to_bottom_depth_mm=(first["rim"] - first["bottom"]) * 1000),
                interpretation="Closest approach and broad neighborhood counts diagnose this recorded attempt only. Nominal clearance assumes aligned axes; these errors are not live policy observations or commanded targets.")


def diagnose_trajectory(trajectory, source, output):
    specification, trace, transitions, inputs, dependency_sha = diagnostic_inputs(trajectory, source)
    binding = dict(inputs=inputs, native_dependencies_sha256=dependency_sha,
                   implementation_sha256=file_hash(__file__))
    output = Path(output)
    if output.exists():
        existing = _read(output)
        body={k:v for k,v in existing.items() if k!="report_sha256"}
        if existing.get("report_sha256") != core_digest(body) or existing["binding"] != binding:
            raise ValueError("Frozen diagnostic inputs changed")
        return existing
    checkpoints = {s["observation"]["native_control_steps"]: s["observation"]["state"] for s in trace}
    sys.path.insert(0, str(Path(source).resolve() / "dexjoco"))
    from dexjoco.tasks.mappings import CONFIG_MAPPING
    task, seed = specification["task"], specification["case"]["seed"]
    env = CONFIG_MAPPING[task]().get_environment(policy_mode=True, render_mode="rgb_array",
            randomize=False, seed=seed, realtime=False, image_obs=False)
    errors, rows = [], []
    def check(step, raw):
        if step in checkpoints:
            error = float(np.max(np.abs(np.asarray(raw["state"][:46]) - checkpoints[step])))
            if not np.isfinite(error) or error > 1e-7:
                raise ValueError(f"Diagnostic replay diverged at public checkpoint {step}: {error}")
            errors.append(error)
    try:
        raw, _ = env.reset(seed=seed)
        check(0, raw)
        for index, expected in enumerate(transitions, 1):
            if expected["physics_step"] != index:
                raise ValueError("Noncontiguous recorded controls")
            raw, _, terminated, truncated, info = env.step(np.asarray(expected["action"], dtype=float))
            check(index, raw)
            if (bool(info["succeed"]) != expected["native_success"] or
                    bool(terminated) != expected["terminated"] or bool(truncated) != expected["truncated"]):
                raise ValueError("Diagnostic replay native outcome differs")
            if task == "bimanual_assembly":
                rows.append(assembly_sample(env.unwrapped, info, index))
            else:
                rows.append({k: bool(info[k]) for k in
                             ("region_pass", "angle_pass", "shutter_pressed", "index_shutter_pressed")})
    finally:
        env.close()
    if len(errors) != len(checkpoints):
        raise ValueError("Diagnostic replay missed a public checkpoint")
    report = dict(schema="dexjoco.post-episode-diagnostics/v1", binding=binding,
                  role="simulation-assisted System 2 feedback after a completed episode; not an online sensor",
                  task=task, native_success=bool(transitions[-1]["native_success"]),
                  native_controls=len(transitions), matched_checkpoints=len(errors),
                  maximum_checkpoint_error=max(errors), summary=summarize(task, rows),
                  qualification=None, changes_native_score=False)
    report["report_sha256"]=core_digest(report)
    save(output, report)
    return report
