"""Materialize a frozen published champion parent; does not certify its score."""
import argparse
from pathlib import Path
import shutil
import tempfile

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, identifier, locked
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from . import champion_backend, champion_observation, engine
from .openpi_backend import TOKENIZER
from .seed import CONTROL, checkpoint_inventory


def materialize(output, *, source, checkpoint, utility, transport, environment_manifest,
                task="turning_on_radio"):
    output = Path(output).absolute()
    identifier(output.name)
    if task not in champion_backend.CHECKPOINT_2_TASKS:
        raise ValueError("Task is not assigned to the published checkpoint_2")
    task_id = champion_backend.CHECKPOINT_2_TASKS[task]
    source = Path(source).resolve()
    backend = dict(kind=champion_backend.KIND, task=task, task_id=task_id,
        checkpoint_name="checkpoint_2", source_root=str(source),
        source_revision=champion_observation.SOURCE_REVISION,
        checkpoint_root=str(Path(checkpoint).resolve()), checkpoint_files=checkpoint_inventory(checkpoint),
        checkpoint_repository="IliaLarchenko/behavior_submission",
        checkpoint_repository_revision=champion_observation.CHECKPOINT_REVISION,
        adapter_sha256=file_digest(Path(champion_backend.__file__)),
        projection_sha256=file_digest(Path(champion_observation.__file__)),
        legacy_utility=str(Path(utility).resolve()), transport_source=str(Path(transport).resolve()),
        transport_sha256=file_digest(Path(transport)), tokenizer=dict(TOKENIZER))
    champion_backend.verify(backend)
    output.parent.mkdir(parents=True, exist_ok=True)
    with locked(output.parent / ("." + output.name + ".lock")):
        if output.exists() or output.is_symlink():
            raise FileExistsError("Use a fresh champion candidate revision")
        stage = Path(tempfile.mkdtemp(prefix="." + output.name, dir=output.parent))
        try:
            (stage / "control.py").write_text(CONTROL)
            atomic_json(stage / "backend.json", backend)
            atomic_json(stage / "engine.json", engine.manifest())
            shutil.copyfile(environment_manifest, stage / "environment.txt")
            shutil.copyfile(source / "openpi/uv.lock", stage / "uv.lock")
            atomic_json(stage / "seed-policy.json", dict(
                schema="physicalrsi.behavior-seed/v1", task=task,
                declared_tasks=[task],
                skill_selection="Published champion checkpoint_2 with default stage tracking and corrections",
                skills=[], tools=[], prompts=f"Frozen task embedding {task_id}",
                memory_rules="Reset upstream episode state; no cross-episode mutable memory",
                qualification=None, admission="Pending independent review and native validation",
                scope="2025 champion transferred to 2026 onboard interface; not a reproduction score"))
            components = {kind: {"seed-policy.json": file_digest(stage / "seed-policy.json")} for kind in CLOSURE}
            for kind in ("foundation", "weights", "assets", "configuration"):
                components[kind] = {"backend.json": file_digest(stage / "backend.json")}
            components["control"] = {"control.py": file_digest(stage / "control.py")}
            components["dependencies"] = {name: file_digest(stage / name)
                                          for name in ("uv.lock", "environment.txt", "engine.json")}
            candidate = dict(id=output.name, root=str(stage), components=components)
            revision = verify_harness(candidate)
            candidate["root"] = str(output)
            atomic_json(stage / "candidate.json", candidate)
            atomic_json(stage / "materialization.json", dict(candidate_sha256=revision,
                checkpoint_files_verified=len(backend["checkpoint_files"]), qualification=None,
                scope="Artifact materialization only; no native score or admission"))
            stage.rename(output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", choices=sorted(champion_backend.CHECKPOINT_2_TASKS),
                        default="turning_on_radio")
    for name in ("source", "checkpoint", "utility", "transport", "environment-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(materialize(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
