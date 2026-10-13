"""Concurrent candidate evaluation with verified, durable completion.

Suite factories must create independent adapters and permit concurrent calls.
Device ownership remains the ExperimentRuntime/DeviceRegistry responsibility;
queue capacity is a separate, explicitly configured scheduling allowance.
"""
from copy import deepcopy
from pathlib import Path
import math

from ..infra.scheduler import JobScheduler
from ..infra.storage import file_digest, read_json
from .artifacts import verify_harness
from .evaluation import ExperimentEvaluator


class ScheduledEvaluator(ExperimentEvaluator):
    def __init__(self, *, scheduler, resources, workers=2, wait_seconds=3600, **kwargs):
        if not isinstance(scheduler, JobScheduler):
            raise ValueError("Declare a durable JobScheduler")
        if (not resources or any(k not in scheduler.capacities or type(v) is not int
                                or not 1 <= v <= scheduler.capacities[k] for k, v in resources.items())
                or type(workers) is not int or workers < 1
                or isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float))
                or not math.isfinite(wait_seconds) or wait_seconds <= 0):
            raise ValueError("Declare a feasible candidate resource request and worker count")
        self.scheduler, self.resources = scheduler, deepcopy(resources)
        self.workers, self.wait_seconds = workers, wait_seconds
        super().__init__(**kwargs)

    def identity(self):
        return dict(super().identity(), scheduling=dict(scheduler=self.scheduler.identity(),
            resources=self.resources, workers=self.workers, wait_seconds=self.wait_seconds,
            implementation=file_digest(Path(__file__))))

    def _output(self, output):
        output = super()._output(output)
        if self.scheduler.root.is_relative_to(output) or output.is_relative_to(self.scheduler.root):
            raise ValueError("Keep the scheduler outside comparison evidence")
        return output

    def _arguments(self, payload):
        if payload.get("schema") != "physicalrsi.scheduled-evaluation/v1":
            raise ValueError("Unknown scheduled evaluation payload")
        output = self._output(payload["output"])
        if read_json(output / "parent.json")["revision"] != payload["parent_revision"]:
            raise ValueError("Scheduled evaluation parent changed")
        return payload["candidate"], payload["comparison"], payload["cohort"], output

    def execute(self, payload):
        return self.evaluate(*self._arguments(payload))

    def collect(self, payload):
        return self.read_evaluation(*self._arguments(payload))

    def replacement_plan(self, payload):
        """Verify explicit Trial replacements without executing any adapters."""
        candidate, comparison, cohort, output = self._arguments(payload)
        _, _, directory, plan = self._evaluation_plan(candidate, comparison, cohort, output)
        return self._replacement_plan(directory, plan)

    def resume_replacements(self, key, *, operator, reason, evidence):
        """Release an inactive queue reservation for verified replacement work."""
        self._stable()
        return self.scheduler.resume_replacements(key, self, operator=operator, reason=reason, evidence=evidence)

    def evaluate_pool(self, candidates, comparison, cohort, output):
        output = self._output(output)
        if {h["id"]: verify_harness(h) for h in candidates} != comparison["candidates"]:
            raise ValueError("Scheduled candidate pool differs from comparison")
        payloads = []
        for candidate in candidates:
            _, _, directory, plan = self._evaluation_plan(candidate, comparison, cohort, output)
            if read_json(directory / "plan.json") != plan:
                raise ValueError("Scheduled trial plan changed")
            self.quota.verify(self._reservations(directory, plan))
            payloads.append(dict(schema="physicalrsi.scheduled-evaluation/v1", candidate=deepcopy(candidate),
                comparison=deepcopy(comparison), cohort=deepcopy(cohort), output=str(output),
                parent_revision=read_json(output / "parent.json")["revision"]))
        ids = self.scheduler.submit(payloads, handler=self, resources=self.resources)
        return self.scheduler.run_until(ids, self, workers=self.workers, wait_seconds=self.wait_seconds)

    resume_evaluation_pool = evaluate_pool
