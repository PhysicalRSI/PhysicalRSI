"""Bounded experiment-backed development and paired Self-Harness evaluation.

Suites own task sampling, admission and adapter construction. Their factories
and sampling methods must not reset or actuate a device. Effects start only in
ExperimentRuntime after the complete plan has a durable quota reservation.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .. import experiments
from ..contracts import ReconciliationRequired
from ..embodiment import Revision, System2Trial
from ..experiments import Budget, ExperimentRuntime
from ..infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json, relative_path
from . import now
from .artifacts import verify_harness
from .experiments import experiment_result, experiment_score
from .selection import comparison_identity


class ExperimentSuite(Protocol):
    def identity(self) -> dict: ...
    def admit(self, candidate: dict) -> dict: ...
    def development_cases(self, parent: dict) -> dict[str, list[dict]]: ...
    def validation_cases(self, comparison: dict) -> dict[str, list[dict]]: ...
    def environment(self, task: str): ...
    def policy(self, candidate: dict, task: str): ...
    def verifier(self, task: str): ...


def freeze_json(path, value):
    """Reuse an identical artifact; the caller owns serialization of writes."""
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Frozen evaluation artifact changed: " + str(path))
    else:
        atomic_json(path, value)


class ExperimentEvaluator:
    def __init__(self, *, suite: ExperimentSuite, budgets: dict[str, Budget],
                 scope: str, system2: Revision, quota, device_registry=None,
                 score_measurement=None):
        if (not budgets or any(not isinstance(t, str) or not t for t in budgets)
                or any(not isinstance(b, Budget) for b in budgets.values())):
            raise ValueError("Declare an experiment budget for each task")
        if not scope or not isinstance(system2, Revision):
            raise ValueError("Declare evaluation scope and System 2 identity")
        if score_measurement is not None and (not isinstance(score_measurement, str) or not score_measurement):
            raise ValueError("A score measurement must be a nonempty name")
        self.suite, self.budgets, self.scope = suite, deepcopy(budgets), scope
        self.system2, self.quota, self.device_registry = system2, quota, device_registry
        self.score_measurement = score_measurement
        self._identity = deepcopy(self.identity())
        self.revision = digest(self._identity)

    def identity(self):
        return dict(kind="experiment-evaluator/v1", suite=self.suite.identity(),
                    budgets={task: asdict(budget) for task, budget in self.budgets.items()},
                    scope=self.scope, system2=asdict(self.system2), quota=self.quota.identity(),
                    device_registry=self.device_registry.identity() if self.device_registry else None,
                    score_measurement=self.score_measurement,
                    implementation={name: file_digest(path) for name, path in {
                        "evaluator": Path(__file__), "runtime": Path(experiments.__file__),
                        "bridge": Path(__file__).with_name("experiments.py"),
                        "scoring": Path(__file__).with_name("selection.py"),
                    }.items()})

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError("Experiment evaluator identity changed")

    def admit(self, candidate, output):
        self._stable()
        sha = verify_harness(candidate)
        result = deepcopy(self.suite.admit(deepcopy(candidate)))
        self._stable()
        if verify_harness(candidate) != sha:
            raise ValueError("Candidate changed during admission")
        return dict(result, freeze_sha256=sha)

    def _cases(self, cases, profile=None):
        if not isinstance(cases, dict) or set(cases) != set(self.budgets):
            raise ValueError("Case tasks differ from the declared experiment budgets")
        if profile is not None and set(profile["tasks"]) != set(cases):
            raise ValueError("Case tasks differ from the comparison profile")
        seen = set()
        for task, values in cases.items():
            if not isinstance(values, list) or not values or any(not isinstance(c, dict) for c in values):
                raise ValueError("Each task needs nonempty public JSON cases")
            if profile is not None and len(values) != profile["tasks"][task]["episodes"]:
                raise ValueError("Case count differs from the comparison profile")
            for case in values:
                sha = digest(case)
                if sha in seen:
                    raise ValueError("Repeated case in experiment plan")
                seen.add(sha)
        return deepcopy(cases)

    def _directory(self, output, split, candidate_id):
        return Path(output).resolve() / "experiments" / split / identifier(candidate_id)

    def _output(self, output):
        output = Path(output).resolve()
        for store in (self.quota, self.device_registry):
            if store is not None and store.root.is_relative_to(output):
                raise ValueError("Keep mutable infrastructure outside round evidence")
        return output

    def _plan(self, candidate_id, sha, cases, *, split, comparison=None, cohort=None):
        return dict(schema="physicalrsi.evaluation-plan/v1", split=split,
                    candidate_id=candidate_id, candidate_sha256=sha,
                    evaluator_revision=self.revision, scope=self.scope,
                    comparison_sha256=digest(comparison) if comparison is not None else None,
                    cohort_sha256=digest(cohort) if cohort is not None else None,
                    cases=cases, trials=[
                        dict(task=task, case=case, budget=asdict(self.budgets[task]),
                             run_id="trial-" + digest(dict(task=task, case=case)))
                        for task, values in sorted(cases.items()) for case in values])

    def _reservations(self, directory, plan):
        return {digest(dict(directory=str(directory), run_id=row["run_id"])):
                dict(plan_sha256=digest(plan), run_id=row["run_id"])
                for row in plan["trials"]}

    def _reserve(self, plans):
        # Reserve the entire candidate pool before evaluating its first member.
        requests = {}
        for directory, plan in plans:
            requests.update(self._reservations(directory, plan))
        self.quota.reserve(requests)
        for directory, plan in plans:
            receipt = self.quota.reserve(self._reservations(directory, plan))
            freeze_json(directory / "quota.json", receipt)

    def _run_plan(self, candidate, directory, plan, *, comparison=None):
        if verify_harness(candidate) != plan["candidate_sha256"]:
            raise ValueError("Candidate differs from the frozen trial plan")
        self._stable()
        if plan["split"] == "validation" and (comparison is None or digest(comparison) != plan["comparison_sha256"]):
            raise ValueError("Validation scoring requires the frozen comparison")
        runtime = ExperimentRuntime(directory / "trials", device_registry=self.device_registry)
        origin = System2Trial(self.system2, candidate["id"], plan["candidate_sha256"],
                              plan["split"], plan["comparison_sha256"], plan["cohort_sha256"])

        def require_validation_outcome(receipt, task):
            if plan["split"] == "validation" and receipt["outcome"] not in {"success", "failure"}:
                raise ReconciliationRequired(
                    "Validation trial has no definite task outcome: "
                    + str(runtime.root / receipt["id"]) + "; preserve its evidence before further evaluation")
            if plan["split"] == "validation":
                try:
                    experiment_score(receipt, score_measurement=self.score_measurement,
                                     score_range=comparison["profile"]["tasks"][task]["score_range"])
                except (ValueError, OverflowError) as error:
                    raise ReconciliationRequired(
                        "Validation trial has an invalid score measurement: "
                        + str(runtime.root / receipt["id"])) from error

        # Discover every unresolved trial before executing any missing one.
        for row in plan["trials"]:
            folder = runtime.root / row["run_id"]
            if (folder / "receipt.json").exists():
                receipt = runtime.read(row["run_id"])
                specification = read_json(folder / "experiment.json")
                expected = dict(task=row["task"], case=row["case"], budget=row["budget"],
                                scope=self.scope, system2=asdict(origin))
                if any(specification.get(key) != value for key, value in expected.items()):
                    raise ValueError("Completed trial differs from the frozen plan")
                require_validation_outcome(receipt, row["task"])
        receipts = []
        for row in plan["trials"]:
            self._stable()
            if verify_harness(candidate) != plan["candidate_sha256"]:
                raise ValueError("Candidate changed between trials")
            task = row["task"]
            receipt = runtime.run(
                row["run_id"], task=task, case=deepcopy(row["case"]), scope=self.scope,
                environment=self.suite.environment(task), policy=self.suite.policy(deepcopy(candidate), task),
                verifier=self.suite.verifier(task), budget=self.budgets[task], system2=origin)
            require_validation_outcome(receipt, task)
            receipts.append(receipt)
        self._stable()
        if verify_harness(candidate) != plan["candidate_sha256"]:
            raise ValueError("Candidate changed during trial execution")
        return runtime, receipts

    def development(self, parent, output):
        """Run or resume incumbent development; preserve unscored outcomes."""
        output = self._output(output)
        self._stable()
        sha = verify_harness(parent)
        directory = self._directory(output, "development", parent["id"])
        with locked(directory / ".plan.lock"):
            path = directory / "plan.json"
            cases = (read_json(path)["cases"] if path.exists()
                     else self.suite.development_cases(deepcopy(parent)))
            plan = self._plan(parent["id"], sha, self._cases(cases), split="development")
            freeze_json(path, plan)
            self._reserve([(directory, plan)])
            _, receipts = self._run_plan(parent, directory, plan)
            rows = [dict(task=row["task"], case=row["case"], run_id=row["run_id"],
                         outcome=receipt["outcome"], measurements=receipt["verdict"]["measurements"])
                    for row, receipt in zip(plan["trials"], receipts)]
            evidence = {str(p.relative_to(output)): file_digest(p)
                        for p in directory.rglob("*.json")}
            return dict(split="evolve", evaluator_revision=self.revision, episodes=rows,
                        evidence=evidence, costs=dict(reserved_trials=len(rows)))

    def _comparison(self, comparison):
        comparison = deepcopy(comparison)
        comparison_identity(comparison)
        if comparison["scope"] != self.scope:
            raise ValueError("Comparison scope differs from evaluator scope")
        if set(comparison["profile"]["tasks"]) != set(self.budgets):
            raise ValueError("Comparison tasks differ from the declared experiment budgets")
        return comparison

    def validation(self, comparison, output):
        """Freeze fresh cases and reserve all candidates before their evaluation."""
        output = self._output(output)
        self._stable()
        comparison = self._comparison(comparison)
        with locked(output / ".validation-plan.lock"):
            path = output / "validation-cases.json"
            cases = (read_json(path) if path.exists()
                     else self.suite.validation_cases(deepcopy(comparison)))
            cases = self._cases(cases, comparison["profile"])
            parent_id = comparison["parent_id"]
            development_path = self._directory(output, "development", parent_id) / "plan.json"
            development = read_json(development_path)
            if (development["candidate_sha256"] != comparison["candidates"][parent_id]
                    or development["evaluator_revision"] != self.revision):
                raise ValueError("Development plan differs from the frozen incumbent")
            seen = {digest(c) for values in development["cases"].values() for c in values}
            layouts = {task: [digest(c) for c in values] for task, values in cases.items()}
            if seen.intersection(sha for values in layouts.values() for sha in values):
                raise ValueError("Validation cases overlap development")
            freeze_json(path, cases)
            cohort_path = output / "validation-cohort.json"
            generated_at = read_json(cohort_path)["generated_at"] if cohort_path.exists() else now()
            cohort = dict(split="validation", comparison_sha256=digest(comparison),
                          generated_at=generated_at, layouts=layouts,
                          admission_evidence={str(p.relative_to(output)): file_digest(p)
                                              for p in (path, development_path)})
            freeze_json(cohort_path, cohort)
            plans = []
            for candidate_id, sha in comparison["candidates"].items():
                directory = self._directory(output, "validation", candidate_id)
                plan = self._plan(candidate_id, sha, cases, split="validation",
                                  comparison=comparison, cohort=cohort)
                freeze_json(directory / "plan.json", plan)
                plans.append((directory, plan))
            self._reserve(plans)
            self._stable()
            return cohort

    def resume_validation(self, comparison, output):
        # Sampling has no external effects; saved cases are never redrawn.
        return self.validation(comparison, output)

    def evaluate(self, candidate, comparison, cohort, output):
        output = self._output(output)
        self._stable()
        comparison = self._comparison(comparison)
        sha = verify_harness(candidate)
        if comparison["candidates"].get(candidate["id"]) != sha:
            raise ValueError("Candidate is absent from the frozen pool")
        if (read_json(output / "validation-cohort.json") != cohort
                or cohort["comparison_sha256"] != digest(comparison)):
            raise ValueError("Validation cohort changed")
        for name, expected in cohort["admission_evidence"].items():
            if file_digest(output / relative_path(name)) != expected:
                raise ValueError("Validation case admission evidence changed")
        cases = self._cases(read_json(output / "validation-cases.json"), comparison["profile"])
        if {task: [digest(c) for c in values] for task, values in cases.items()} != cohort["layouts"]:
            raise ValueError("Validation cases differ from cohort layouts")
        directory = self._directory(output, "validation", candidate["id"])
        with locked(directory / ".plan.lock"):
            plan = self._plan(candidate["id"], sha, cases, split="validation",
                              comparison=comparison, cohort=cohort)
            freeze_json(directory / "plan.json", plan)
            self._reserve([(directory, plan)])
            runtime, _ = self._run_plan(candidate, directory, plan, comparison=comparison)
            result = experiment_result(
                runtime, [row["run_id"] for row in plan["trials"]], candidate_id=candidate["id"],
                comparison=comparison, cohort=cohort, evaluator_revision=self.revision,
                evidence_root=output, score_measurement=self.score_measurement)
            for row in result["episodes"]:
                row["evidence_sha256"].update({str(p.relative_to(output)): file_digest(p)
                    for p in (directory / "plan.json", directory / "quota.json")})
            return result

    def resume_evaluation(self, candidate, comparison, cohort, output):
        """Verify finished trials, block unknown trials, then run missing trials."""
        return self.evaluate(candidate, comparison, cohort, output)
