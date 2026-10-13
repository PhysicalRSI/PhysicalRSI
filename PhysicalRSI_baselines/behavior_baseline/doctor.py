"""Inspect the external runtime without importing or starting Isaac Sim."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json

BEHAVIOR_REVISION = "a8247a8cc1633fe1ca0cc66aa07243d46c64f155"


def inspect(runtime):
    root = Path(runtime).resolve()
    source = root / "sources/BEHAVIOR-1K"
    python = root / "env/bin/python"
    checks = {}
    try:
        git = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15, check=True)
        checks["source"] = {"ready": git.stdout.strip() == BEHAVIOR_REVISION, "revision": git.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        checks["source"] = {"ready": False, "reason": str(error)}
    try:
        code = ("import importlib.metadata as m,json,sys; "
                "names=['torch','omnigibson','bddl','isaacsim','hydra-core','websockets']; "
                "installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
                "print(json.dumps({'python':list(sys.version_info[:3]),"
                "'packages':{n:installed.get(n) for n in names}}))")
        process = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60, check=True)
        value = json.loads(process.stdout)
        checks["packages"] = dict(value, ready=value["python"][:2] == [3, 11] and all(value["packages"].values()))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        checks["packages"] = {"ready": False, "reason": str(error)}
    paths = ["behavior-1k-assets/metadata/categories.txt", "omnigibson-robot-assets",
             "2026-challenge-task-instances/metadata/B100_task_misc.csv", "omnigibson.key"]
    checks["assets"] = {"ready": all((root / "data" / p).exists() for p in paths),
                        "missing": [p for p in paths if not (root / "data" / p).exists()]}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=True)
        checks["gpu"] = {"ready": bool(gpu.stdout.strip()), "devices": gpu.stdout.strip().splitlines()}
    except (OSError, subprocess.SubprocessError) as error:
        checks["gpu"] = {"ready": False, "reason": str(error)}
    free = shutil.disk_usage(root).free
    return {"schema": "physicalrsi.behavior-preflight/v1", "runtime": str(root), "checks": checks,
            "free_bytes": free, "prerequisites_present": all(v["ready"] for v in checks.values()),
            "simulator_verified": False, "benchmark_verified": False, "qualification": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.runtime)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
