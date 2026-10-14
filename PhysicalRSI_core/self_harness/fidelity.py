"""Development screening followed by independent, full paired validation.

The cheap protocol can change adapters/budgets but uses the preregistered
development cases. Its measurements only decide which children receive full
validation. They are never substituted for promotion scores or final tests.
"""
from copy import deepcopy
import math
from pathlib import Path
from statistics import mean

from ..contracts import ReconciliationRequired
from ..infra.storage import file_digest, read_json
from .artifacts import verify_harness
from .evaluation import ExperimentEvaluator, freeze_json
from .experiments import experiment_score
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

    def replacement_plan(self, payload):
        low = self.evaluator
        low._stable()
        candidate = payload["candidate"]
        output = low._output(payload["output"])
        directory = low._directory(output, "development", candidate["id"])
        saved = read_json(directory / "plan.json")
        plan = low._plan(candidate["id"], verify_harness(candidate),
                         low._cases(saved["cases"]), split="development")
        return low._replacement_plan(directory, plan)


class MultiFidelityEvaluator(ScheduledEvaluator):
    def __init__(self, *, screening, screening_resources, minimum_success_rate,
                 screen_incumbent=False, survivor_limit=None, ranking_profile=None, **kwargs):
        if (not isinstance(screening, ExperimentEvaluator)
                or type(screen_incumbent) is not bool
                or isinstance(minimum_success_rate, bool) or not isinstance(minimum_success_rate, (int, float))
                or not math.isfinite(minimum_success_rate) or not 0 <= minimum_success_rate <= 1):
            raise ValueError("Declare a screening evaluator and fixed success threshold")
        self.screening, self.screening_resources = screening, deepcopy(screening_resources)
        self.minimum_success_rate = minimum_success_rate
        self.screen_incumbent = screen_incumbent
        if survivor_limit is None:
            if ranking_profile is not None:
                raise ValueError("A screening ranking profile requires a survivor limit")
        else:
            if type(survivor_limit) is not int or survivor_limit < 1:
                raise ValueError("Screening survivor limit must be a positive integer")
            if not isinstance(ranking_profile, dict) or set(ranking_profile) != set(screening.budgets):
                raise ValueError("Screening ranking profile must cover every screening task")
            for spec in ranking_profile.values():
                if not isinstance(spec, dict) or set(spec) != {"weight", "score_range"}:
                    raise ValueError("Declare each screening task's weight and score range")
                bounds = spec["score_range"]
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError("Screening score range requires two bounds")
                values = [spec["weight"], *bounds]
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                    raise ValueError("Screening weights and bounds must be finite numbers")
                low, high = bounds
                if spec["weight"] <= 0 or low >= high or not math.isfinite(high - low):
                    raise ValueError("Screening weights and score ranges must be positive")
        self.survivor_limit = survivor_limit
        self.ranking_profile = ({task: dict(weight=spec["weight"], score_range=list(spec["score_range"]))
                                 for task, spec in ranking_profile.items()} if ranking_profile is not None else None)
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
            survivor_limit=self.survivor_limit, ranking_profile=self.ranking_profile,
            implementation=file_digest(Path(__file__))))

    def policy_identity(self):
        value = super().policy_identity()
        value["fidelity"]["screening"] = self.screening.policy_identity()
        return value

    def authorize_screening_replacement(self, candidate, *, output, run_id,
                                       failed_run_id=None, operator, reason, evidence, quiescent):
        """Authorize a new cheap Trial under the frozen screening allowance."""
        self._stable()
        directory = self._output(output) / "experiments/screening/development"
        result = self.screening.authorize_replacement(candidate, output=directory,
            split="development", run_id=run_id, failed_run_id=failed_run_id,
            operator=operator, reason=reason, evidence=evidence, quiescent=quiescent)
        self._stable()
        return result

    def resume_screening_replacements(self, key, *, operator, reason, evidence):
        """Resume only an inactive screening job with an authorized new Trial."""
        self._stable()
        return self.scheduler.resume_replacements(key, _ScreeningJobs(self.screening),
            operator=operator, reason=reason, evidence=evidence)

    def _ranking_scores(self, episodes):
        """Normalize each task before weighting; case counts do not set weights."""
        per_task = {}
        for task, spec in self.ranking_profile.items():
            low, high = spec["score_range"]
            scores = [experiment_score(dict(outcome=row["outcome"],
                verdict=dict(measurements=row["measurements"])),
                score_measurement=self.screening.score_measurement, score_range=(low, high))
                for row in episodes if row["task"] == task]
            if not scores:
                raise ValueError("Screening ranking requires a complete task panel")
            per_task[task] = float(mean((score - low) / (high - low) for score in scores))
        largest = max(spec["weight"] for spec in self.ranking_profile.values())
        weights = {task: spec["weight"] / largest for task, spec in self.ranking_profile.items()}
        total = math.fsum(weights.values())
        score = math.fsum(per_task[task] * (weight / total) for task, weight in weights.items())
        return dict(per_task=per_task, score=score)

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
                        / sum(row["task"] == task for row in result["episodes"]) for task in cases}
            passed = all(score >= self.minimum_success_rate for score in per_task.values())
            incumbent = candidate["id"] == parent["id"]
            admissions[candidate["id"]] = dict(accepted=incumbent or passed,
                freeze_sha256=verify_harness(candidate), measured=True, per_task=per_task,
                reason="incumbent retained for full comparison" if incumbent else
                       "development screening passed" if passed else "development screening below threshold")
            if self.survivor_limit is not None:
                admissions[candidate["id"]]["ranking"] = self._ranking_scores(result["episodes"])
            evidence.update({str((directory / name).relative_to(output)): sha for name, sha in result["evidence"].items()})
        ranking = None
        if self.survivor_limit is not None:
            # The incumbent never competes for a slot. Content identity breaks
            # exact ties independently of names, input order or completion order.
            ranking = sorted((name for name, row in admissions.items()
                              if name != parent["id"] and row["accepted"]),
                             key=lambda name: (-admissions[name]["ranking"]["score"], hashes[name]))
            for index, name in enumerate(ranking):
                admissions[name]["ranking"]["rank"] = index + 1
                if index >= self.survivor_limit:
                    admissions[name].update(accepted=False, reason="development screening survivor limit")
        self._stable()
        if {h["id"]: verify_harness(h) for h in candidates} != hashes:
            raise ValueError("Candidate changed during screening")
        return dict(schema="physicalrsi.fidelity-screening/v1", evaluator_revision=low.revision,
                    policy=self.identity()["fidelity"], candidates=admissions, evidence=evidence,
                    ranking=ranking,
                    role="screening_only", qualification=None)

    resume_screening = screen_pool
