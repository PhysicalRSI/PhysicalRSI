"""Capture one native development layout; never report a benchmark score.

Run with a simulator-capable interpreter. The trusted worker reads simulator
state for reset/replay infrastructure; these private artifacts are not policy
observations. Each invocation uses a new output directory and a bounded process.
"""
import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

from PhysicalRSI_core.infra.artifacts import Artifacts
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest
from .model_assets import model_asset_inventory

PINS = {
    "libero": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "libero-plus": "4976dc30028e805ff8094b55501d532c48fec182",
}


def source_identity(source, family):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != PINS[family]:
        raise ValueError("Benchmark source differs from the reviewed revision")
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source, text=True)
    if changed.strip():
        raise ValueError("Benchmark has tracked modifications; declare a new protocol")
    return {"family": family, "revision": revision, "source": str(source)}


def public_observation(observation):
    """The admitted camera and robot channels; excludes object state."""
    channels = {"agentview_image", "agentview_depth", "robot0_eye_in_hand_image",
                "robot0_eye_in_hand_depth", "robot0_joint_pos", "robot0_joint_vel",
                "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
    return {key: value for key, value in observation.items() if key in channels}


@contextmanager
def native_environment(args):
    """Construct the pinned native task in this trusted simulator process."""
    import numpy as np
    camera_size = getattr(args, "camera_size", 128)
    if type(camera_size) is not int or not 64 <= camera_size <= 512:
        raise ValueError("Camera size must be an integer between 64 and 512")
    source_identity(args.source, args.benchmark)
    # Force the original/Plus nested package, not ASPIRE's differently packaged fork.
    sys.path.insert(0, str(args.source))
    config = args.output / "config"
    config.mkdir()
    base = args.source / "libero/libero"
    assets = args.asset_root.resolve() if args.asset_root is not None else base / "assets"
    if not assets.is_dir():
        raise ValueError("Benchmark assets are not prepared")
    # Native arena code also derives this path from its source location.
    # A config override alone would silently leave that code on other assets.
    if (base / "assets").resolve() != assets.resolve():
        raise ValueError("The benchmark checkout's assets path must resolve to --asset-root")
    values = {"benchmark_root": base, "bddl_files": base / "bddl_files",
              "init_states": base / "init_files", "assets": assets,
              "datasets": args.output / "unused-datasets"}
    (config / "config.yaml").write_text("".join(k + ": " + json.dumps(str(v)) + "\n" for k, v in values.items()))
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    os.environ.setdefault("MUJOCO_GL", "egl")
    benchmark = importlib.import_module("libero.libero.benchmark")
    envs = importlib.import_module("libero.libero.envs")
    if not Path(benchmark.__file__).resolve().is_relative_to(args.source):
        raise ValueError("Wrong LIBERO package imported")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.get_num_tasks():
        raise ValueError("Task index is outside the selected suite")
    task = suite.get_task(args.task_id)
    bddl = Path(suite.get_task_bddl_file_path(args.task_id)).resolve()
    if not bddl.is_relative_to(base):
        raise ValueError("Task definition is outside the selected benchmark")
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = None
    try:
        env = envs.OffScreenRenderEnv(bddl_file_name=str(bddl), controller="JOINT_POSITION",
                                     camera_heights=camera_size, camera_widths=camera_size, camera_depths=True)
        env.seed(args.seed)
        yield env, task, bddl, assets
    finally:
        if env is not None:
            env.close()


def capture(args):
    import numpy as np
    source = source_identity(args.source, args.benchmark)
    with native_environment(args) as (env, task, bddl, assets):
        obs = env.reset()
        store = Artifacts(args.output / "artifacts")
        state = store.encode(np.asarray(env.get_sim_state()).copy())
        model_xml = env.sim.model.get_xml().encode()
        xml = store.encode(model_xml)
        import robosuite
        inventory = model_asset_inventory(model_xml, allowed_roots=[
            assets, Path(robosuite.__file__).parent / "models/assets"])
        # Trusted geometry evidence distinguishes object-placement changes from
        # changes to a seed label, elapsed simulation time or camera noise.
        geometry = {name: np.concatenate([env.sim.data.body_xpos[body], env.sim.data.body_xquat[body]])
                    for name, body in sorted(env.env.obj_body_id.items())}
        geometry_sha256 = digest({name: pose.tolist() for name, pose in geometry.items()})
        # Only camera channels and explicitly named robot proprioception are public.
        public = public_observation(obs)
        observation = store.encode(public)
        identity = {"benchmark": source, "suite": args.suite, "task": task.name,
                    "camera_size": getattr(args, "camera_size", 128),
                    "assets_root": str(assets),
                    "bddl_sha256": file_digest(bddl), "model_sha256": xml["sha256"],
                    "state_sha256": state["sha256"], "geometry_sha256": geometry_sha256,
                    "asset_inventory_sha256": digest(inventory)}
        record = {"schema": "physicalrsi.egl-layout/v1", "state": "captured", "scope": "simulation-development-layout",
                  "qualification": None, "seed": args.seed, "identity": identity,
                  "layout_sha256": digest(identity), "private_reset": {"state": state, "model_xml": xml,
                                                                         "object_root_poses": store.encode(geometry)},
                  "private_model_assets": inventory,
                  "observation": observation, "task_language": task.language,
                  "initial_official_success": bool(env.check_success()),
                  "generator_sha256": file_digest(Path(__file__)), "novelty": "not-yet-compared"}
    # Publish only after simulator cleanup. A captured layout is not a solved task.
    atomic_json(args.output / "layout.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=PINS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, help="Prepared local assets; defaults to the benchmark source's assets directory")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--camera-size", type=int, default=128,
                        help="Public RGB-D height and width (64–512); recorded in layout identity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 64 <= args.camera_size <= 512:
        parser.error("Camera size must be between 64 and 512")
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.seed < 2**32 or args.task_id < 0:
        parser.error("Use a finite positive timeout, a uint32 seed and a nonnegative task index")
    if args.worker:
        capture(args)
        return
    source_identity(args.source, args.benchmark)
    args.output.mkdir(parents=True, exist_ok=False)
    journal = {"state": "started", "scope": "simulation-development-layout", "qualification": None,
               "benchmark": args.benchmark, "suite": args.suite, "task_id": args.task_id, "seed": args.seed}
    atomic_json(args.output / "generation.json", journal)
    with (args.output / "worker.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-m", "PhysicalRSI_baselines.embodied_goodharts_law.sample_libero_layout",
                                    *sys.argv[1:], "--worker"], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            atomic_json(args.output / "generation.json", dict(journal, state="timed_out"))
            raise SystemExit("Layout generation timed out; preserved worker log")
    if code or not (args.output / "layout.json").exists():
        atomic_json(args.output / "generation.json", dict(journal, state="failed", returncode=code))
        raise SystemExit("Layout generation failed; inspect worker.log")
    atomic_json(args.output / "generation.json", dict(journal, state="completed", returncode=code,
                layout_sha256=file_digest(args.output / "layout.json")))
    print(args.output / "layout.json")


if __name__ == "__main__":
    main()
