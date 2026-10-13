"""Materialize an immutable, pass-through OpenPI seed from installed artifacts.

This creates a candidate for review, not a benchmark admission or a score.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, identifier, locked, relative_path
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from .openpi_backend import OPENPI_REVISION, TOKENIZER
from .engine import manifest as engine_manifest

CONTROL = '''def initialize(task):
    return {}


def step(observation, state, primitive):
    return primitive(observation)
'''


def checkpoint_inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Checkpoint must be an extracted physical directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint inventory must not contain symlinks")
        if path.is_file():
            if path.name.endswith((".incomplete", ".part", ".tmp")):
                raise ValueError("Checkpoint download is incomplete")
            result[str(path.relative_to(root))] = file_digest(path)
    if not result:
        raise ValueError("Checkpoint directory is empty")
    return result


def materialize(output, *, source, checkpoint, task, environment_manifest,
                training_config="pi05_b1k", repo_id=None, action_horizon=16):
    output = Path(output).absolute()
    identifier(output.name)
    identifier(task)
    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon <= 0:
        raise ValueError("Action horizon must be a positive integer")
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30).strip()
    if revision != OPENPI_REVISION:
        raise ValueError("Use the pinned official OpenPI checkout")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"],
                   check=True, timeout=120)
    files = checkpoint_inventory(checkpoint)
    repo_id = repo_id or task
    norm_stats = "assets/" + str(relative_path(repo_id)) + "/norm_stats.json"
    if norm_stats not in files:
        raise ValueError("Checkpoint lacks normalization statistics for the declared repo_id")
    backend = dict(kind="openpi-b1k/v1", source_root=str(source), source_revision=revision,
                   tokenizer=dict(TOKENIZER),
                   adapter_sha256=file_digest(Path(__file__).with_name("openpi_backend.py")),
                   checkpoint_root=str(Path(checkpoint).resolve()), checkpoint_files=files,
                   task=task, training_config=training_config, repo_id=repo_id,
                   control_mode="receding_horizon", action_horizon=action_horizon)
    output.parent.mkdir(parents=True, exist_ok=True)
    with locked(output.parent / f".{output.name}.lock"):
        if output.exists() or output.is_symlink():
            raise FileExistsError("Seed destination already exists; use a new revision directory")
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            (stage / "control.py").write_text(CONTROL)
            atomic_json(stage / "backend.json", backend)
            atomic_json(stage / "engine.json", engine_manifest())
            shutil.copyfile(environment_manifest, stage / "environment.txt")
            shutil.copyfile(source / "uv.lock", stage / "uv.lock")
            atomic_json(stage / "seed-policy.json", {
                "schema": "physicalrsi.behavior-seed/v1", "task": task,
                "skill_selection": "Always call the fixed official action backend",
                "skills": [], "tools": [], "prompts": "Official task registry prompt",
                "memory_rules": "No cross-episode mutable memory; no rollout-time code generation",
                "qualification": None, "admission": "pending independent review and native validation"})
            components = {kind: {"seed-policy.json": file_digest(stage / "seed-policy.json")}
                          for kind in CLOSURE}
            for kind in ("foundation", "weights", "assets", "configuration"):
                components[kind] = {"backend.json": file_digest(stage / "backend.json")}
            components["control"] = {"control.py": file_digest(stage / "control.py")}
            components["dependencies"] = {
                name: file_digest(stage / name) for name in ("uv.lock", "environment.txt", "engine.json")}
            candidate = dict(id=output.name, root=str(stage), components=components)
            revision = verify_harness(candidate)
            candidate["root"] = str(output)
            atomic_json(stage / "candidate.json", candidate)
            atomic_json(stage / "materialization.json", {
                "candidate_sha256": revision, "qualification": None,
                "checkpoint_files_verified": len(files),
                "scope": "Artifact materialization only; no model inference or benchmark admission"})
            stage.rename(output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True,
                        help="Installed policy environment's pip freeze or uv pip freeze output")
    parser.add_argument("--training-config", default="pi05_b1k")
    parser.add_argument("--repo-id")
    parser.add_argument("--action-horizon", type=int, default=16)
    args = vars(parser.parse_args())
    materialize(**args)
    print(args["output"] / "candidate.json")


if __name__ == "__main__":
    main()
