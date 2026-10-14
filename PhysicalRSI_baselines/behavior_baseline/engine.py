"""Bind shared System 1 execution code to a frozen candidate dependency."""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest, read_json


def manifest():
    root = Path(__file__).resolve().parents[2]
    files = ["PhysicalRSI_baselines/behavior_baseline/" + name for name in
             ("engine.py", "program.py", "observations.py", "openpi_backend.py", "serve.py", "diagnostics.py",
              "champion_backend.py", "champion_observation.py", "champion_chunk_restore.py",
              "champion_restore_params.py", "champion_policy_config.py")]
    files += ["PhysicalRSI_core/" + name for name in
              ("contracts.py", "embodiment.py", "infra/storage.py", "self_harness/artifacts.py")]
    return dict(schema="physicalrsi.behavior-system1-engine/v1",
                files={name: file_digest(root / name) for name in files})


def verify(candidate):
    path = Path(candidate["root"]) / "engine.json"
    if (not path.is_file()
            or candidate["components"]["dependencies"].get("engine.json") != file_digest(path)):
        raise ValueError("Bind the System 1 engine in the candidate dependency closure")
    frozen = read_json(path)
    if frozen != manifest():
        raise ValueError("Shared System 1 implementation differs from the frozen candidate")
    return frozen
