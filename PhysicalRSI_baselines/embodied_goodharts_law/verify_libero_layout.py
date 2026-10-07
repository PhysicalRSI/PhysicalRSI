"""Verify native layout replay before comparing candidate policies.

Rebuild the pinned task, require an identical compiled model, and restore the
captured state without an XML round trip. Exact public observations are required.
This is a simulator preflight, not a policy trial or benchmark qualification.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from PhysicalRSI_core.infra.artifacts import Artifacts
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest
from .sample_libero_layout import native_environment, public_observation, source_identity


def verify(args):
    import numpy as np

    layout = json.loads(args.layout.read_text())
    if layout.get("schema") != "physicalrsi.egl-layout/v1" or layout.get("state") != "captured":
        raise ValueError("Expected a captured EGL layout")
    identity = layout["identity"]
    if digest(identity) != layout["layout_sha256"]:
        raise ValueError("Layout identity differs from its digest")
    args.benchmark = identity["benchmark"]["family"]
    args.source = Path(identity["benchmark"]["source"]).resolve()
    if source_identity(args.source, args.benchmark) != identity["benchmark"]:
        raise ValueError("Benchmark source identity changed")
    args.suite = identity["suite"]
    args.seed = layout["seed"]
    args.camera_size = identity.get("camera_size", 128)
    args.asset_root = Path(identity["assets_root"]) if "assets_root" in identity else None
    store = Artifacts(args.layout.parent / "artifacts")
    private = layout["private_reset"]
    if (private["state"]["sha256"] != identity["state_sha256"] or
            private["model_xml"]["sha256"] != identity["model_sha256"]):
        raise ValueError("Reset artifacts differ from the declared layout")
    state = store.read_array(private["state"], max_bytes=16 * 1024 * 1024)
    xml = store.read(private["model_xml"], max_bytes=32 * 1024 * 1024)
    expected = {key: store.read_array(value, max_bytes=16 * 1024 * 1024)
                for key, value in layout["observation"].items()}
    geometry = None
    if "geometry_sha256" in identity:
        from .compare_layouts import read_geometry
        _, geometry = read_geometry(args.layout)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("Expected finite flat simulator state")
    with native_environment(args) as (env, task, bddl, assets):
        if task.name != identity["task"] or file_digest(bddl) != identity["bddl_sha256"]:
            raise ValueError("Task definition differs from the captured layout")
        env.reset()
        rebuilt_xml = env.sim.model.get_xml().encode()
        if rebuilt_xml != xml:
            raise ValueError("Rebuilt compiled model differs; state-only replay is not admissible")
        assets_verified = None
        if "asset_inventory_sha256" in identity:
            import robosuite
            from .model_assets import model_asset_inventory
            inventory = layout["private_model_assets"]
            if digest(inventory) != identity["asset_inventory_sha256"]:
                raise ValueError("Asset inventory differs from the layout identity")
            current = model_asset_inventory(rebuilt_xml, allowed_roots=[
                assets, Path(robosuite.__file__).parent / "models/assets"])
            if current != inventory:
                raise ValueError("Referenced mesh or texture content changed")
            assets_verified = True
        # Change physics without resetting: Plus can change model parameters on
        # reset, and a simulator state vector cannot restore those parameters.
        # Every candidate must reconstruct the captured model before replay.
        for _ in range(10):
            env.step(np.zeros(8))
        if np.array_equal(env.get_sim_state(), state):
            raise ValueError("Replay probe did not perturb simulator state")
        if env.sim.model.get_xml().encode() != xml:
            raise ValueError("Physics probe changed the compiled model")
        observation = public_observation(env.set_init_state(state.copy()))
        if not np.array_equal(env.get_sim_state(), state):
            raise ValueError("Restored simulator state differs from the capture")
        if geometry is not None:
            actual_geometry = {name: np.concatenate([env.sim.data.body_xpos[body], env.sim.data.body_xquat[body]])
                               for name, body in env.env.obj_body_id.items()}
            if (set(actual_geometry) != set(geometry) or
                    any(not np.array_equal(actual_geometry[name], geometry[name]) for name in geometry)):
                raise ValueError("Restored object placement differs from the capture")
        if set(observation) != set(expected):
            raise ValueError("Public observation channels changed")
        checks = {}
        for key in expected:
            actual, reference = np.asarray(observation[key]), expected[key]
            exact = (actual.dtype == reference.dtype and actual.shape == reference.shape
                     and np.array_equal(actual, reference))
            checks[key] = bool(exact)
        if not all(checks.values()):
            raise ValueError("Replay observation mismatch: " + ", ".join(k for k, v in checks.items() if not v))
        success = bool(env.check_success())
        if success != layout["initial_official_success"]:
            raise ValueError("Initial official success changed on replay")
        result = {"schema": "physicalrsi.egl-layout-replay/v1", "state": "verified",
                  "scope": "native-layout-replay-preflight", "qualification": None,
                  "benchmark_trial": False, "layout_sha256": layout["layout_sha256"],
                  "layout_file_sha256": file_digest(args.layout),
                  "model_sha256": hashlib.sha256(rebuilt_xml).hexdigest(),
                  "perturbation": {"kind": "zero-action-physics-steps", "steps": 10},
                  "state_exact": True, "observation_exact": checks,
                  "object_geometry_exact": True if geometry is not None else None,
                  "referenced_assets_verified": assets_verified,
                  "initial_official_success": success,
                  "verifier_sha256": file_digest(Path(__file__)),
                  "environment_helper_sha256": file_digest(Path(sys.modules[native_environment.__module__].__file__)),
                  "limitations": ["Full simulator dependency closure is not locked by this check.",
                                  "Exact replay was checked only in this environment; no cross-platform guarantee."]
                                  + (["Referenced external assets were not captured in this legacy layout."]
                                     if assets_verified is None else [])}
    atomic_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.layout, args.output = args.layout.resolve(), args.output.resolve()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.task_id < 0:
        parser.error("Use a finite positive timeout and nonnegative task index")
    if args.worker:
        verify(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"state": "started", "scope": "native-layout-replay-preflight", "qualification": None}
    atomic_json(args.output / "verification.json", record)
    with (args.output / "worker.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-m",
                                    "PhysicalRSI_baselines.embodied_goodharts_law.verify_libero_layout",
                                    *sys.argv[1:], "--worker"], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            atomic_json(args.output / "verification.json", dict(record, state="timed_out"))
            raise SystemExit("Layout replay timed out; inspect worker.log")
    if code or not (args.output / "result.json").exists():
        atomic_json(args.output / "verification.json", dict(record, state="failed", returncode=code))
        raise SystemExit("Layout replay failed; inspect worker.log")
    atomic_json(args.output / "verification.json", dict(record, state="completed", returncode=code))
    print(args.output / "result.json")


if __name__ == "__main__":
    main()
