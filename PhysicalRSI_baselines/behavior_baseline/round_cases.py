"""Deterministic, disjoint round allocation from an admitted Challenge case pool.

This allocates reset identities, not experiments or benchmark qualifications.
The caller supplies the official pool's independently recorded revision and
registers each returned protocol with core research lineage before execution.
"""
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, locked, read_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol


class RoundCases:
    def __init__(self, root, pool, *, expected_pool_revision, max_rounds,
                 development_per_task, validation_per_task):
        if not isinstance(pool, CaseProtocol) or pool.identity()["revision"] != expected_pool_revision:
            raise ValueError("Case pool differs from the admitted revision")
        for count in (max_rounds, development_per_task, validation_per_task):
            if type(count) is not int or count < 1:
                raise ValueError("Round and case counts must be positive integers")
        # Inspect the frozen manifest internally; test cases are never returned
        # through development/validation APIs or used for allocation decisions.
        source = read_json(pool.root / "protocol.json")
        if digest(source) != expected_pool_revision:
            raise ValueError("Case pool changed while allocating rounds")
        splits = source["splits"]
        for split, count in (("development", development_per_task), ("validation", validation_per_task)):
            if any(len(values) < count * max_rounds for values in splits[split].values()):
                raise ValueError("Insufficient fresh cases for the complete round schedule")
        canonical = {}
        for split, tasks in splits.items():
            canonical[split] = {}
            for task, values in tasks.items():
                rows = []
                for value in values:
                    mode = "public_test" if split == "test" else "train"
                    instance = value["instance_id"]
                    index = instance - 301 if mode == "public_test" else instance
                    if (type(instance) is not int or index < 0 or value.get("mode") != mode
                            or value.get("instance_index") != index
                            or (mode == "public_test" and index >= 20)):
                        raise ValueError("Case does not match native Challenge reset semantics")
                    rows.append(dict(task=task, mode=mode, instance_id=instance,
                                     instance_index=index, rollout_id=0))
                canonical[split][task] = rows
        self.root = Path(root).resolve()
        self.pool = pool
        self.manifest = dict(schema="physicalrsi.behavior-round-cases/v1",
            pool_revision=expected_pool_revision, max_rounds=max_rounds,
            development_per_task=development_per_task, validation_per_task=validation_per_task,
            allocation="contiguous_preregistered_pool_order", cases=canonical,
            qualification=None, test_role="report_only")
        self.revision = digest(self.manifest)
        with locked(self.root / ".lock"):
            path = self.root / "schedule.json"
            if path.exists() and read_json(path) != self.manifest:
                raise ValueError("Frozen round schedule changed")
            if not path.exists():
                atomic_json(path, self.manifest)

    def protocol(self, round_index):
        if type(round_index) is not int or not 0 <= round_index < self.manifest["max_rounds"]:
            raise ValueError("Round index outside the frozen schedule")
        if (self.pool.identity()["revision"] != self.manifest["pool_revision"]
                or read_json(self.root / "schedule.json") != self.manifest):
            raise ValueError("Frozen round allocation changed")
        splits = {}
        for split, tasks in self.manifest["cases"].items():
            if split == "test":
                splits[split] = tasks
            else:
                count = self.manifest[split + "_per_task"]
                splits[split] = {task: values[round_index * count:(round_index + 1) * count]
                                 for task, values in tasks.items()}
        return CaseProtocol(self.root / f"round-{round_index:03d}", splits=splits,
                            identity_keys=("instance_id",))
