"""Reserve a complete native trial plan before any simulator execution.

Policy services, benchmark admission and fresh cohort registration remain the
orchestrator's responsibilities. This transport uses the core trial quota and
preserves unresolved native evidence instead of silently rerunning an episode.
"""
from dataclasses import asdict
from pathlib import Path

from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from . import native


class NativeTrials:
    def __init__(self, runtime, output, *, trials, quota):
        self.runtime = Path(runtime).resolve()
        self.output = Path(output).resolve()
        self.quota = quota
        if quota.root.is_relative_to(self.output):
            raise ValueError("Keep mutable trial quota outside experiment evidence")
        if not isinstance(trials, list) or not trials:
            raise ValueError("Declare a complete nonempty trial plan")
        rows, seen = [], set()
        for inputs in trials:
            inputs = dict(inputs)
            origin = inputs.get("system2")
            if not isinstance(origin, System2Trial) or inputs.get("max_steps") is not None:
                raise ValueError("Declare System 2 attribution and use native full episode limits")
            inputs["policy"] = "websocket"
            inputs["policy_revision"] = origin.candidate_sha256
            identity = dict(inputs, system2=asdict(origin))
            key = digest(identity)
            folder = self.output / "trials" / key
            spec = native.specification(self.runtime, folder, **inputs)
            semantic = digest(dict(origin=asdict(origin), case=spec["case"]))
            if semantic in seen:
                raise ValueError("Repeated trial in native plan")
            seen.add(semantic)
            rows.append(dict(id=key, inputs=identity, specification=spec))
        self.plan = dict(schema="physicalrsi.behavior-native-plan/v1", runtime=str(self.runtime),
                         output=str(self.output), quota=quota.identity(), trials=rows,
                         implementation_sha256=file_digest(Path(__file__)),
                         qualification=None)
        with locked(self.output / ".plan.lock"):
            path = self.output / "plan.json"
            if path.exists() and read_json(path) != self.plan:
                raise ValueError("Frozen native trial plan changed")
            # Atomic core reservation admits all candidates/cases together.
            receipt = quota.reserve(self._requests())
            if not path.exists():
                atomic_json(path, self.plan)
            quota_path = self.output / "quota.json"
            if quota_path.exists() and read_json(quota_path) != receipt:
                raise ValueError("Native plan quota receipt changed")
            atomic_json(quota_path, receipt)

    def _requests(self):
        plan_sha = digest(self.plan)
        return {digest(dict(output=str(self.output), trial_id=row["id"])):
                dict(plan_sha256=plan_sha, trial_id=row["id"])
                for row in self.plan["trials"]}

    def _reconcile(self):
        """Check the complete reserved plan while the caller holds its lock."""
        if file_digest(Path(__file__)) != self.plan["implementation_sha256"]:
            raise ValueError("Native plan executor implementation changed")
        if read_json(self.output / "plan.json") != self.plan:
            raise ValueError("Frozen native trial plan changed")
        if self.quota.identity() != self.plan["quota"]:
            raise ValueError("Trial quota identity changed")
        if self.quota.reserve(self._requests()) != read_json(self.output / "quota.json"):
            raise ValueError("Native plan quota receipt changed")
        # Reconcile the whole plan before starting any missing episode.
        for row in self.plan["trials"]:
            folder = self.output / "trials" / row["id"]
            if (folder / "receipt.json").exists():
                native.read_receipt(folder)
                if read_json(folder / "specification.json") != row["specification"]:
                    raise ValueError("Existing trial differs from reserved plan")

    def reconcile(self):
        """Reconcile all evidence before an orchestrator starts policy services."""
        with locked(self.output / ".execution.lock"):
            self._reconcile()

    def run(self, index):
        if type(index) is not int or not 0 <= index < len(self.plan["trials"]):
            raise ValueError("Trial index outside frozen plan")
        with locked(self.output / ".execution.lock"):
            self._reconcile()
            row = self.plan["trials"][index]
            inputs = dict(row["inputs"])
            origin = inputs["system2"]
            inputs["system2"] = System2Trial(**dict(origin, system2=Revision(**origin["system2"])))
            return native.run(self.runtime, self.output / "trials" / row["id"],
                              expected_specification=row["specification"], **inputs)
