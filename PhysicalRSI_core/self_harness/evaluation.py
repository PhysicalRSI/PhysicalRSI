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
from ..experiments import Budget, ExperimentRuntime, trial_adapters, trial_factory_kind
from ..trials import Evaluation, Trial, task_case_identity
from ..infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json, relative_path
from ..infra.trial_recovery import TrialRecovery, before_execution, inactive
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
    # Stateful/concurrent suites may instead provide trial(candidate, task)
    # returning one independent TrialAdapters bundle, without execution effects.


def freeze_json(path, value):
    """Reuse an identical JSON artifact; the caller serializes concurrent writes."""
    if path.exists():
        if digest(read_json(path)) != digest(value):
            raise ValueError("Frozen evaluation artifact changed: " + str(path))
    else:
        atomic_json(path, value)


class ExperimentEvaluator:
    def __init__(self, *, suite: ExperimentSuite, budgets: dict[str, Budget],
                 scope: str, system2: Revision, quota, device_registry=None,
                 score_measurement=None, max_replacements=0):
        if (not budgets or any(not isinstance(t, str) or not t for t in budgets)
                or any(not isinstance(b, Budget) for b in budgets.values())):
            raise ValueError("Declare an experiment budget for each task")
        if not scope or not isinstance(system2, Revision):
            raise ValueError("Declare evaluation scope and System 2 identity")
        if score_measurement is not None and (not isinstance(score_measurement, str) or not score_measurement):
            raise ValueError("A score measurement must be a nonempty name")
        if type(max_replacements) is not int or not 0 <= max_replacements <= 10:
            raise ValueError('Declare zero to ten replacements per planned Trial')
        self.suite, self.budgets, self.scope = suite, deepcopy(budgets), scope
        self.system2, self.quota, self.device_registry = system2, quota, device_registry
        self.score_measurement = score_measurement
        self.max_replacements = max_replacements
        self._identity = deepcopy(self.identity())
        self.revision = digest(self._identity)

    def identity(self):
        return dict(kind="experiment-evaluator/v1", suite=self.suite.identity(),
                    trial_factory=trial_factory_kind(self.suite),
                    case_identity="task-case/v1",
                    budgets={task: asdict(budget) for task, budget in self.budgets.items()},
                    scope=self.scope, system2=asdict(self.system2), quota=self.quota.identity(),
                    device_registry=self.device_registry.identity() if self.device_registry else None,
                    score_measurement=self.score_measurement,
                    max_replacements=self.max_replacements,
                    implementation=dict(experiments.runtime_implementation(), **{
                        name: file_digest(path) for name, path in {
                        "evaluator": Path(__file__),
                        "bridge": Path(__file__).with_name("experiments.py"),
                        "scoring": Path(__file__).with_name("selection.py"),
                        "rollout": Path(__file__).with_name("rollout.py"),
                        "harness_artifacts": Path(__file__).with_name("artifacts.py"),
                    }.items()}))

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError("Experiment evaluator identity changed")

    def policy_identity(self):
        from .bindings import policy_identity
        value = self.identity()
        value["suite"] = policy_identity(self.suite)
        return value

    def register_round(self, state, round_identity, *, context=None):
        method = getattr(self.suite, "register_round", None)
        if not callable(method):
            raise ValueError("Research evaluation requires preregistered split registration")
        return method(state, round_identity, context=context)

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
                sha = task_case_identity(task, case)
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
                             run_id="trial-" + task_case_identity(task, case))
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

    def _recovery(self, directory, plan):
        return TrialRecovery(directory, plan, quota=self.quota, device_registry=self.device_registry,
                             maximum=self.max_replacements)

    def _execution_rows(self, directory, plan):
        recovery = self._recovery(directory, plan)
        return [dict(row, logical_run_id=row['run_id'], **{
            k: v for k, v in recovery.resolve(row['run_id']).items()}) for row in plan['trials']]

    def _replacement_plan(self, directory, plan):
        """Audit an inactive plan before releasing its queue reservation."""
        self._stable()
        with inactive(directory / '.plan.lock'):
            if read_json(directory / 'plan.json') != plan:
                raise ValueError('Replacement plan changed')
            self.quota.verify(self._reservations(directory, plan))
            runtime = ExperimentRuntime(directory / 'trials', device_registry=self.device_registry)
            authorizations, pending = [], 0
            for row in self._execution_rows(directory, plan):
                folder = runtime.root / row['run_id']
                if row['replacement'] is not None:
                    authorizations.append(row['replacement']['authorization_sha256'])
                if (folder / 'receipt.json').exists():
                    audited = runtime.read_evidence(row['run_id'])
                    if audited['specification'].get('replacement') != row['replacement']:
                        raise ValueError('Completed replacement differs from the frozen plan')
                else:
                    runtime.require_unstarted(row['run_id'], replacement=row['replacement'])
                    pending += 1
                    if row['replacement'] is not None:
                        previous = read_json(folder / 'recovery/source/experiment.json')
                        before_execution(folder, row['run_id'], dict(previous, replacement=row['replacement']), self.device_registry)
            self._stable()
            return dict(schema='physicalrsi.scheduled-replacements/v1', authorizations=sorted(authorizations), pending=pending)

    def authorize_replacement(self, candidate, *, output, split, run_id, failed_run_id=None,
                              operator, reason, evidence, quiescent):
        """Authorize one new execution without dispatching it or changing its plan.

        The caller independently establishes quiescence. Reset/action-complete
        Trials cannot be replaced, and the configured allowance defaults to zero.
        Scheduled evaluations additionally need explicit queue reconciliation.
        """
        if split not in {'development', 'validation', 'test'}:
            raise ValueError('Unknown recovery split')
        self._stable()
        output = self._output(output)
        directory = self._directory(output, split, candidate['id'])
        with inactive(directory / '.plan.lock'):
            plan = read_json(directory / 'plan.json')
            if (plan['evaluator_revision'] != self.revision or plan['candidate_id'] != candidate['id']
                    or plan['candidate_sha256'] != verify_harness(candidate) or plan['split'] != split
                    or plan['scope'] != self.scope):
                raise ValueError('Recovery differs from the frozen candidate/evaluator plan')
            if read_json(directory / 'quota.json') != self.quota.verify(self._reservations(directory, plan)):
                raise ValueError('Recovery plan lacks its original quota reservation')
            replacement_id = self._recovery(directory, plan).authorize(run_id, failed_run_id or run_id,
                operator=operator, reason=reason, evidence=evidence, quiescent=quiescent)
            self._stable()
            return dict(logical_run_id=run_id, failed_run_id=failed_run_id or run_id, run_id=replacement_id)

    def _run_plan(self, candidate, directory, plan, *, comparison=None):
        if verify_harness(candidate) != plan["candidate_sha256"]:
            raise ValueError("Candidate differs from the frozen trial plan")
        self._stable()
        if plan["split"] == "validation" and (comparison is None or digest(comparison) != plan["comparison_sha256"]):
            raise ValueError("Validation scoring requires the frozen comparison")
        runtime = ExperimentRuntime(directory / "trials", device_registry=self.device_registry)
        execution_rows = self._execution_rows(directory, plan)
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
        for row in execution_rows:
            folder = runtime.root / row["run_id"]
            if (folder / "receipt.json").exists():
                audited = runtime.read_evidence(row["run_id"])
                receipt, specification = audited['receipt'], audited['specification']
                expected = dict(task=row["task"], case=row["case"], budget=row["budget"],
                                scope=self.scope, system2=asdict(origin), replacement=row['replacement'])
                if any(specification.get(key) != value for key, value in expected.items()):
                    raise ValueError("Completed trial differs from the frozen plan")
                require_validation_outcome(receipt, row["task"])
            else:
                runtime.require_unstarted(row['run_id'], replacement=row['replacement'])
                if row['replacement'] is not None:
                    from ..infra.trial_recovery import before_execution
                    previous = read_json(folder / 'recovery/source/experiment.json')
                    before_execution(folder, row['run_id'], dict(previous, replacement=row['replacement']), self.device_registry)
        receipts = []
        for row in execution_rows:
            self._stable()
            if verify_harness(candidate) != plan["candidate_sha256"]:
                raise ValueError("Candidate changed between trials")
            task = row["task"]
            trial = Trial(row["run_id"], task=task, case=row["case"], scope=self.scope,
                          budget=self.budgets[task], system2=origin, replacement=row['replacement'])
            adapters = trial_adapters(self.suite, candidate, task)
            self._stable()
            if verify_harness(candidate) != plan["candidate_sha256"]:
                raise ValueError("Candidate changed during trial adapter construction")
            receipt = runtime.run_trial(trial, environment=adapters.environment,
                                        policy=adapters.policy, verifier=adapters.verifier)
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
            self._run_plan(parent, directory, plan)
            return self.read_development(parent, output)

    def read_development(self, parent, output):
        """Verify already completed development; never fill missing trials."""
        output = self._output(output)
        self._stable()
        directory = self._directory(output, "development", parent["id"])
        saved = read_json(directory / "plan.json")
        plan = self._plan(parent["id"], verify_harness(parent), self._cases(saved["cases"]), split="development")
        if saved != plan or read_json(directory / "quota.json") != self.quota.verify(self._reservations(directory, plan)):
            raise ValueError("Development plan or quota changed")
        runtime = ExperimentRuntime(directory / "trials", device_registry=self.device_registry)
        execution_rows = self._execution_rows(directory, plan)
        values = [Evaluation.read(runtime, row["run_id"]).to_dict() for row in execution_rows]
        if any(value['trial'].get('replacement') != row['replacement']
               for row, value in zip(execution_rows, values)):
            raise ValueError('Development replacement differs from the frozen plan')
        rows = [dict(task=row["task"], case=row["case"], run_id=row["run_id"],
                     outcome=value["outcome"], measurements=value["verdict"]["measurements"])
                for row, value in zip(execution_rows, values)]
        evidence = {str(p.relative_to(output)): file_digest(p) for p in directory.rglob("*.json")}
        for value in values:
            for name, sha in value["evidence"].items():
                evidence[str(Path(name).relative_to(output))] = sha
        result = dict(split="evolve", evaluator_revision=self.revision, episodes=rows,
                      evaluations=values, evidence=evidence,
                      costs=dict(reserved_trials=len(rows) + sum(row['attempts'] for row in execution_rows)))
        from .rollout import development_evaluations
        development_evaluations(parent, result, output)
        return result

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
            seen = {task_case_identity(task, c) for task, values in development["cases"].items() for c in values}
            layouts = {task: [task_case_identity(task, c) for c in values] for task, values in cases.items()}
            if seen.intersection(sha for values in layouts.values() for sha in values):
                raise ValueError("Validation cases overlap development")
            freeze_json(path, cases)
            cohort_path = output / "validation-cohort.json"
            generated_at = read_json(cohort_path)["generated_at"] if cohort_path.exists() else now()
            cohort = dict(split="validation", comparison_sha256=digest(comparison), case_identity="task-case/v1",
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

    def _evaluation_plan(self, candidate, comparison, cohort, output):
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
        if (cohort.get("case_identity") != "task-case/v1"
                or {task: [task_case_identity(task, c) for c in values] for task, values in cases.items()} != cohort["layouts"]):
            raise ValueError("Validation cases differ from cohort layouts")
        directory = self._directory(output, "validation", candidate["id"])
        plan = self._plan(candidate["id"], sha, cases, split="validation",
                          comparison=comparison, cohort=cohort)
        return output, comparison, directory, plan

    def evaluate(self, candidate, comparison, cohort, output):
        output, comparison, directory, plan = self._evaluation_plan(candidate, comparison, cohort, output)
        with locked(directory / ".plan.lock"):
            freeze_json(directory / "plan.json", plan)
            self._reserve([(directory, plan)])
            self._run_plan(candidate, directory, plan, comparison=comparison)
            return self.read_evaluation(candidate, comparison, cohort, output)

    def read_evaluation(self, candidate, comparison, cohort, output):
        """Rebuild a result from complete native receipts, without running trials."""
        output, comparison, directory, plan = self._evaluation_plan(candidate, comparison, cohort, output)
        if read_json(directory / "plan.json") != plan:
            raise ValueError("Frozen evaluation plan changed")
        receipt = self.quota.verify(self._reservations(directory, plan))
        if read_json(directory / "quota.json") != receipt:
            raise ValueError("Evaluation quota receipt changed")
        runtime = ExperimentRuntime(directory / "trials", device_registry=self.device_registry)
        execution_rows = self._execution_rows(directory, plan)
        result = experiment_result(
            runtime, [row["run_id"] for row in execution_rows], candidate_id=candidate["id"],
            comparison=comparison, cohort=cohort, evaluator_revision=self.revision,
            evidence_root=output, score_measurement=self.score_measurement,
            replacements={row['run_id']: row['replacement'] for row in execution_rows})
        for row in result["episodes"]:
            row["evidence_sha256"].update({str(p.relative_to(output)): file_digest(p)
                for p in (directory / "plan.json", directory / "quota.json")})
            if (directory / 'replacements.json').exists():
                path = directory / 'replacements.json'
                row['evidence_sha256'][str(path.relative_to(output))] = file_digest(path)
        self._stable()
        return result

    def resume_evaluation(self, candidate, comparison, cohort, output):
        """Verify finished trials, block unknown trials, then run missing trials."""
        return self.evaluate(candidate, comparison, cohort, output)

    def report_test(self, candidate, *, selection_sha256, output):
        """Report a single final candidate on preregistered held-out cases.

        This never calls the selector or updates memory. Native uncertain/invalid
        outcomes remain visible; all planned cases stay in the denominator.
        The caller owns the final-selection claim and a single report workspace.
        """
        output = self._output(output)
        self._stable()
        sha = verify_harness(candidate)
        with locked(output / ".report.lock"):
            cases = self._cases(self.suite.claim_test(candidate_sha256=sha,
                selection_sha256=selection_sha256, claim_root=output / "claim"))
            comparison = dict(candidate_sha256=sha, selection_sha256=selection_sha256)
            cohort = dict(split="test", cases=cases)
            directory = self._directory(output, "test", candidate["id"])
            plan = self._plan(candidate["id"], sha, cases, split="test", comparison=comparison, cohort=cohort)
            freeze_json(directory / "plan.json", plan)
            self._reserve([(directory, plan)])
            _, receipts = self._run_plan(candidate, directory, plan)
            rows = [dict(task=row["task"], run_id=receipt['id'], case_sha256=digest(row["case"]),
                         outcome=receipt["outcome"], measurements=receipt["verdict"]["measurements"])
                    for row, receipt in zip(plan["trials"], receipts)]
            metrics = {}
            for task, values in cases.items():
                outcomes = [row["outcome"] for row in rows if row["task"] == task]
                metrics[task] = dict(planned=len(values), successes=outcomes.count("success"),
                    failures=outcomes.count("failure"), uncertain=outcomes.count("uncertain"),
                    invalid=outcomes.count("invalid"), success_rate=outcomes.count("success") / len(values))
            report = dict(schema="physicalrsi.final-test/v1", candidate_sha256=sha,
                          selection_sha256=selection_sha256, evaluator_revision=self.revision,
                          episodes=rows, metrics=metrics, scope=self.scope, qualification=None,
                          evidence={str(p.relative_to(output)): file_digest(p)
                                    for p in output.rglob("*") if p.is_file()
                                    and p.name != "report.json" and p.suffix != ".lock"})
            freeze_json(output / "report.json", report)
            return report
