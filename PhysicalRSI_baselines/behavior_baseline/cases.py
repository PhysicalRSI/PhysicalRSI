"""Preregister all Challenge tasks using filenames and public task metadata only."""
import argparse
import csv
import io
from pathlib import Path
import re
import zipfile

from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI_core.self_harness.protocol import CaseProtocol

TASK_ARCHIVE_SHA256 = "d4ac3d72dd585178e85d542f28d1b48233b886c80e196e2f9d3aa5993ba21f81"


def build_splits(archive):
    archive = Path(archive)
    if file_digest(archive) != TASK_ARCHIVE_SHA256:
        raise ValueError("Task archive differs from the inspected official release")
    with zipfile.ZipFile(archive) as zipped:
        tasks = list(csv.DictReader(io.StringIO(zipped.read("metadata/B100_task_misc.csv").decode())))
        if len(tasks) != 100 or len({r["Task"] for r in tasks}) != 100:
            raise ValueError("Expected the complete 100-task official Challenge")
        names = zipped.namelist()
        splits = {split: {} for split in ("development", "validation", "test")}
        for row in tasks:
            task = row["Task"]
            pattern = re.compile(r"_task_" + re.escape(task) + r"_0_(\d+)_template-tro_state\.json$")
            ids = {}
            for mode, prefix in (("train", "scenes/"), ("public_test", "scene_test/public/")):
                values = [int(match[1]) for name in names if name.startswith(prefix) and (match := pattern.search(name))]
                if len(set(values)) != len(values):
                    raise ValueError(f"Ambiguous reset identity: {task}/{mode}")
                ids[mode] = set(values)
            if ids["train"] not in (set(range(300)), set(range(1, 301))) or ids["public_test"] != set(range(301, 321)):
                raise ValueError(f"Incomplete or changed official reset coverage: {task}")
            # The official archive uses 0..299 for 50 tasks and 1..300 for
            # 50 others. Split actual identities, never invent instance zero.
            training = sorted(ids["train"])
            for split, mode, instances in (("development", "train", training[:200]),
                                           ("validation", "train", training[200:]),
                                           ("test", "public_test", range(301, 321))):
                splits[split][task] = [dict(instance_id=i, mode=mode,
                                           instance_index=i if mode == "train" else i - 301)
                                      for i in instances]
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    splits = build_splits(args.archive)
    protocol = CaseProtocol(args.output / "cases", splits=splits, identity_keys=("instance_id",))
    report = {"schema": "physicalrsi.behavior-case-inventory/v1", "archive_sha256": TASK_ARCHIVE_SHA256,
              "case_protocol": protocol.identity(), "task_count": len(splits["development"]),
              "case_counts": {split: sum(map(len, cases.values())) for split, cases in splits.items()},
              "note": "Validation is held out from RSI development, not necessarily from baseline pretraining; final public tests are report-only."}
    atomic_json(args.output / "case_inventory.json", report)
    print(report)


if __name__ == "__main__":
    main()
