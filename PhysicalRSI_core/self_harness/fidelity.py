"""Development screening followed by independent, full paired validation.

The cheap protocol can change adapters/budgets but uses the preregistered
development cases. Its measurements only decide which children receive full
validation. They are never substituted for promotion scores or final tests.
"""
from copy import deepcopy
import math
from pathlib import Path

from ..contracts import ReconciliationRequired
from ..infra.storage import file_digest
from .artifacts import verify_harness
from .evaluation import ExperimentEvaluator, freeze_json
from .scheduled import ScheduledEvaluator


class _ScreeningJobs:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def identity(self):
        return dict(kind="development-screening-jobs/v1", evaluator=self.evaluator.identity(),
                    implementation=file_digest(Path(__file__)))

    def execute(self, payload):
        return self.evaluator.development(payload["candidate"], Path(payload["output"]))

    def collect(self, payload):
        return self.evaluator.read_development(payload["candidate"], Path(payload["output"]))


class MultiFidelityEvaluator(ScheduledEvaluator):
    def __init__(self, *, screening, screening_resources, minimum_success_rate,
                 screen_incumbent=False, **kwargs):
        if (not isinstance(screening, ExperimentEvaluator)
                or type(screen_incumbent) is not bool
                or isinstance(minimum_success_rate, bool) or not isinstance(minimum_success_rate, (int, float))
                or not math.isfinite(minimum_success_rate) or not 0 <= minimum_success_rate <= 1):
            raise ValueError("Declare a screening evaluator and fixed success threshold")
        self.screening, self.screening_resources = screening, deepcopy(screening_resources)
        self.minimum_success_rate = minimum_success_rate
        self.screen_incumbent = screen_incumbent
        super().__init__(**kwargs)
        if (not screening_resources or any(k not in self.scheduler.capacities or type(v) is not int
                or not 1 <= v <= self.scheduler.capacities[k] for k, v in screening_resources.items())):
            raise ValueError("Screening resource request exceeds capacity")
        if (screening.scope != self.scope or screening.quota.identity() != self.quota.identity()
                or screening.suite.protocol.identity() != self.suite.protocol.identity()):
            raise ValueError("Fidelity stages must share their scope, quota and preregistered case protocol")
        if screening.system2 == self.system2:
            raise ValueError("Declare distinct System 2 revisions for screening and full validation")

    def identity(self):
        return dict(super().identity(), fidelity=dict(screening=self.screening.identity(),
            resources=self.screening_resources, minimum_success_rate=self.minimum_success_rate,
            screen_incumbent=self.screen_incumbent,
            implementation=file_digest(Path(__file__))))

    def policy_identity(self):
        value = super().policy_identity()
        value["fidelity"]["screening"] = self.screening.policy_identity()
        return value

    def screen_pool(self, parent, candidates, output):
        self._stable()
        output = self._output(output)
        candidates = deepcopy(candidates)
        hashes = {h["id"]: verify_harness(h) for h in candidates}
        if (len(candidates) < 2 or len(hashes) != len(candidates)
                or len(set(hashes.values())) != len(candidates)
                or hashes.get(parent["id"]) != verify_harness(parent)):
            raise ValueError("Screening requires the exact incumbent and distinct changed candidates")
        measured = [h for h in candidates if self.screen_incumbent or h["id"] != parent["id"]]
        directory = output / "experiments/screening/development"
        low = self.screening
        cases = low._cases(low.suite.development_cases(deepcopy(parent)))
        self._stable()
        if {h["id"]: verify_harness(h) for h in candidates} != hashes:
            raise ValueError("Candidate changed before screening")
        plans = []
        for candidate in measured:
            root = low._directory(directory, "development", candidate["id"])
            plan = low._plan(candidate["id"], verify_harness(candidate), cases, split="development")
            freeze_json(root / "plan.json", plan)
            plans.append((root, plan))
        low._reserve(plans)  # Every planned screening trial is charged before effects.
        handler = _ScreeningJobs(low)
        payloads = [dict(candidate=deepcopy(candidate), output=str(directory)) for candidate in measured]
        ids = self.scheduler.submit(payloads, handler=handler, resources=self.screening_resources, priority=1)
        results = self.scheduler.run_until(ids, handler, workers=self.workers, wait_seconds=self.wait_seconds)
        admissions, evidence = {}, {}
        if not self.screen_incumbent:
            admissions[parent["id"]] = dict(accepted=True, freeze_sha256=hashes[parent["id"]],
                measured=False, per_task=None, reason="incumbent retained for full comparison without screening")
        for candidate, result in zip(measured, results):
            if any(row["outcome"] not in {"success", "failure"} for row in result["episodes"]):
                raise ReconciliationRequired("Screening has no definite outcome; full validation was not dispatched")
            per_task = {task: sum(row["outcome"] == "success" for row in result["episodes"] if row["task"] == task)
                        / len(values) for task, values in cases.items()}
            passed = all(score >= self.minimum_success_rate for score in per_task.values())
            incumbent = candidate["id"] == parent["id"]
            admissions[candidate["id"]] = dict(accepted=incumbent or passed,
                freeze_sha256=verify_harness(candidate), measured=True, per_task=per_task,
                reason="incumbent retained for full comparison" if incumbent else
                       "development screening passed" if passed else "development screening below threshold")
            evidence.update({str((directory / name).relative_to(output)): sha for name, sha in result["evidence"].items()})
        self._stable()
        if {h["id"]: verify_harness(h) for h in candidates} != hashes:
            raise ValueError("Candidate changed during screening")
        return dict(schema="physicalrsi.fidelity-screening/v1", evaluator_revision=low.revision,
                    policy=self.identity()["fidelity"], candidates=admissions, evidence=evidence,
                    role="screening_only", qualification=None)

    resume_screening = screen_pool
