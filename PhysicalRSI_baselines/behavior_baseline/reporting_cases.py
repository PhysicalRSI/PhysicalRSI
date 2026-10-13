"""Freeze the prescribed 100-by-10 Challenge report cohort from public inventory."""
import argparse
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, read_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol


def register_reporting(source, output, *, expected_revision):
    source, output = Path(source), Path(output)
    manifest = read_json(source)
    if digest(manifest) != expected_revision:
        raise ValueError("Admitted source inventory changed")
    if manifest.get("identity_keys") != ["instance_id"] or manifest.get("test_role") != "report_only":
        raise ValueError("Expected admitted report-only instance inventory")
    splits = deepcopy(manifest["splits"])
    if set(splits) != {"development", "validation", "test"}:
        raise ValueError("Expected three admitted splits")
    tasks = set(splits["development"])
    if len(tasks) != 100 or any(set(rows) != tasks for rows in splits.values()):
        raise ValueError("Reporting requires all 100 tasks")
    excluded = {}
    for task in sorted(tasks):
        expected = [dict(instance_id=i + 301, instance_index=i, mode="public_test") for i in range(20)]
        if sorted(splits["test"][task], key=lambda c: c["instance_index"]) != expected:
            raise ValueError("Expected exactly the 20 prescribed public instances per task")
        splits["test"][task] = expected[:10]
        excluded[task] = expected[10:]
    # The source pool may already belong to an active engineering round.
    # Create a distinct commitment; never change that pool or its claims.
    output.mkdir(parents=True, exist_ok=False)
    protocol = CaseProtocol(output / "cases", splits=splits, identity_keys=("instance_id",))
    record = dict(schema="physicalrsi.behavior-reporting-cohort/v1",
                  source_inventory_sha256=expected_revision, case_protocol=protocol.identity(),
                  task_count=100, report_instances_per_task=10, report_rollouts_per_instance=1,
                  report_episode_count=1000, report_instance_indices=list(range(10)),
                  excluded_public_cases=excluded,
                  official_rule="https://behavior.stanford.edu/challenge/submission.html",
                  qualification=None,
                  note="Prescribed reporting identities only; no scenes read or rollouts executed. "
                       "Extra public instances are excluded from the reported mean. "
                       "Development and validation remain unchanged.")
    atomic_json(output / "reporting-cohort.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    record = register_reporting(args.source, args.output, expected_revision=args.expected_revision)
    print({k: v for k, v in record.items() if k != "excluded_public_cases"})


if __name__ == "__main__":
    main()
