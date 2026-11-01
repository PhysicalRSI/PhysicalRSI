"""Bounded experiment-backed development and paired Self-Harness evaluation.

Suites own task sampling, admission and adapter construction. Their factories
and sampling methods must not reset or actuate a device. Effects start only in
ExperimentRuntime after the complete plan has a durable quota reservation.
"""

from copy import deepcopy
from dataclasses import asdict
import math
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
from .experiments import experiment_result, experiment_score, repetition_count
from .selection import comparison_identity, cohort_order, betting_evidence


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
                 score_measurement=None, max_replacements=0, repetitions=1,
                 infrastructure_retries=2):
        if (not budgets or any(not isinstance(t, str) or not t for t in budgets)
                or any(not isinstance(b, Budget) for b in budgets.values())):
            raise ValueError("Declare an experiment budget for each task")
        if not scope or not isinstance(system2, Revision):
            raise ValueError("Declare evaluation scope and System 2 identity")
        if score_measurement is not None and (not isinstance(score_measurement, str) or not score_measurement):
            raise ValueError("A score measurement must be a nonempty name")
        if type(max_replacements) is not int or not 0 <= max_replacements <= 10:
            raise ValueError('Declare zero to ten replacements per planned Trial')
        if type(infrastructure_retries) is not int or not 0 <= infrastructure_retries <= 10:
            raise ValueError('Declare zero to ten infrastructure retries per planned Trial')
        self.suite, self.budgets, self.scope = suite, deepcopy(budgets), scope
        self.system2, self.quota, self.device_registry = system2, quota, device_registry
        self.score_measurement = score_measurement
        self.max_replacements = max_replacements
        self.infrastructure_retries = infrastructure_retries
        self.repetitions = repetition_count(repetitions)
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
                    infrastructure_retries=self.infrastructure_retries,
                    repetitions=self.repetitions,
                    implementation=dict(experiments.runtime_implementation(), **{
                        name: file_digest(path) for name, path in {
                        "evaluator": Path(__file__),
                        "bridge": Path(__file__).with_name("experiments.py"),
                        "scoring": Path(__file__).with_name("selection.py"),
                        "rollout": Path(__file__).with_name("rollout.py"),
                        "harness_artifacts": Path(__file__).with_name("artifacts.py"),
                    }.items()}))

    def _stable(self):
        if digest(self.identity()) != self.revision:
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

    def _plan(self, candidate_id, sha, cases, *, split, comparison=None, cohort=None,
              case_order=None):
        ordered = ([(task, case) for task, case in case_order]
                   if case_order is not None else
                   [(task, case) for task, values in sorted(cases.items()) for case in values])
        return dict(schema="physicalrsi.evaluation-plan/v1", split=split,
                    candidate_id=candidate_id, candidate_sha256=sha,
                    evaluator_revision=self.revision, scope=self.scope,
                    comparison_sha256=digest(comparison) if comparison is not None else None,
                    cohort_sha256=digest(cohort) if cohort is not None else None,
                    infrastructure_retries=self.infrastructure_retries,
                    cases=cases, trials=[
                        dict(task=task, case=case, budget=asdict(self.budgets[task]),
                             run_id="trial-" + task_case_identity(task, case)
                             + (f"-repeat-{repeat:03d}" if self.repetitions > 1 else ""))
                        for task, case in ordered
                        for repeat in range(self.repetitions)])

    def _infrastructure_retry_id(self, directory, logical_run_id, index):
        return "infrastructure-retry-" + digest(dict(directory=str(directory),
                                                       run_id=logical_run_id, index=index))

    def _infrastructure_record(self, directory, plan):
        path = Path(directory) / "infrastructure.json"
        if not path.exists():
            return None
        record = read_json(path)
        if (record.get("schema") != "physicalrsi.infrastructure-retries/v1"
                or record.get("plan_sha256") != digest(plan)
                or not isinstance(record.get("trials"), dict)):
            raise ValueError("Infrastructure retry record differs from the frozen plan")
        logical_ids = {row["run_id"] for row in plan["trials"]}
        if set(record["trials"]) - logical_ids:
            raise ValueError("Infrastructure retry record contains an unknown trial")
        for logical, entry in record["trials"].items():
            if (not isinstance(entry, dict) or not isinstance(entry.get("attempts"), list)
                    or not entry["attempts"] or entry["attempts"][0] != logical
                    or len(entry["attempts"]) > self.infrastructure_retries + 1
                    or len(set(entry["attempts"])) != len(entry["attempts"])):
                raise ValueError("Infrastructure retry attempts are invalid")
            for index, attempt in enumerate(entry["attempts"][1:], 1):
                if attempt != self._infrastructure_retry_id(directory, logical, index):
                    raise ValueError("Infrastructure retry identity changed")
            selected, unavailable = entry.get("selected"), entry.get("unavailable", False)
            if type(unavailable) is not bool or (unavailable and selected is not None):
                raise ValueError("Infrastructure retry state is invalid")
            if selected is not None and selected not in entry["attempts"]:
                raise ValueError("Infrastructure retry selection is invalid")
        return record

    def _reservations(self, directory, plan):
        reservations = {digest(dict(directory=str(directory), run_id=row["run_id"])):
                        dict(plan_sha256=digest(plan), run_id=row["run_id"])
                        for row in plan["trials"]}
        record = self._infrastructure_record(directory, plan)
        if record is not None:
            for entry in record.get("trials", {}).values():
                for run_id in entry.get("attempts", [])[1:]:
                    reservations[digest(dict(directory=str(directory), run_id=run_id))] = dict(
                        plan_sha256=digest(plan), run_id=run_id)
        return reservations

    def _reserve(self, plans):
        # Reserve the entire candidate pool before evaluating its first member.
        requests = {}
        for directory, plan in plans:
            requests.update(self._reservations(directory, plan))
        self.quota.reserve(requests)
        for directory, plan in plans:
            receipt = self.quota.reserve(self._reservations(directory, plan))
            quota_path = directory / "quota.json"
            if quota_path.exists():
                previous = read_json(quota_path)
                if not set(previous.get("trials", {})) <= set(receipt["trials"]):
                    raise ValueError("Infrastructure retry reservation removed an earlier trial")
                atomic_json(quota_path, receipt)
            else:
                freeze_json(quota_path, receipt)

    def _recovery(self, directory, plan):
        return TrialRecovery(directory, plan, quota=self.quota, device_registry=self.device_registry,
                             maximum=self.max_replacements)

    def _execution_rows(self, directory, plan):
        recovery = self._recovery(directory, plan)
        metadata = self._infrastructure_record(directory, plan)
        result = []
        for row in plan['trials']:
            value = dict(row, logical_run_id=row['run_id'], **{
                k: v for k, v in recovery.resolve(row['run_id']).items()})
            entry = (metadata or {}).get("trials", {}).get(row["run_id"])
            if entry is not None:
                selected = entry.get("selected")
                if entry.get("unavailable"):
                    value["run_id"] = None
                    value["infrastructure_unavailable"] = True
                elif selected:
                    value["run_id"] = selected
                value["infrastructure_attempts"] = list(entry.get("attempts", []))
            result.append(value)
        return result

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
            if plan["split"] == "validation" and receipt["outcome"] not in {"success", "failure", "deadline"}:
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

        expected_spec = lambda row: dict(task=row["task"], case=row["case"], budget=row["budget"],
                                         scope=self.scope, system2=asdict(origin), replacement=row['replacement'])

        # Discover every existing attempt before executing any missing one.
        # A completed infrastructure receipt is retryable; all other completed
        # receipts are audited against the frozen logical slot.
        for row in execution_rows:
            if row.get("run_id") is None:
                continue
            folder = runtime.root / row["run_id"]
            if (folder / "receipt.json").exists():
                audited = runtime.read_evidence(row["run_id"])
                receipt, specification = audited['receipt'], audited['specification']
                if receipt.get("outcome") == "infrastructure-fault":
                    if any(specification.get(key) != value
                           for key, value in expected_spec(row).items()):
                        raise ValueError("Faulted trial differs from the frozen plan")
                    continue
                if any(specification.get(key) != value for key, value in expected_spec(row).items()):
                    raise ValueError("Completed trial differs from the frozen plan")
                require_validation_outcome(receipt, row["task"])
            else:
                runtime.require_unstarted(row['run_id'], replacement=row['replacement'])
                if row['replacement'] is not None:
                    from ..infra.trial_recovery import before_execution
                    previous = read_json(folder / 'recovery/source/experiment.json')
                    before_execution(folder, row['run_id'], dict(previous, replacement=row['replacement']), self.device_registry)
        receipts = []
        metadata_path = directory / "infrastructure.json"
        metadata = self._infrastructure_record(directory, plan)

        def save_metadata():
            nonlocal metadata
            if metadata is not None:
                atomic_json(metadata_path, metadata)

        def reserve_retry(run_id):
            request = {digest(dict(directory=str(directory), run_id=run_id)):
                       dict(plan_sha256=digest(plan), run_id=run_id)}
            self.quota.reserve(request)
            receipt = self.quota.reserve(self._reservations(directory, plan))
            atomic_json(directory / "quota.json", receipt)

        for row in execution_rows:
            entry = (metadata or {}).get("trials", {}).get(row["logical_run_id"])
            if entry and (entry.get("unavailable") or entry.get("selected")):
                if entry.get("unavailable"):
                    continue
                selected = entry["selected"]
                receipt = runtime.read(selected)
                require_validation_outcome(receipt, row["task"])
                receipts.append(receipt)
                continue
            self._stable()
            if verify_harness(candidate) != plan["candidate_sha256"]:
                raise ValueError("Candidate changed between trials")
            task = row["task"]
            attempts = list(entry.get("attempts", [])) if entry else [row["logical_run_id"]]
            selected_receipt = None
            while attempts and len(attempts) <= self.infrastructure_retries + 1:
                attempt_id = attempts[-1]
                folder = runtime.root / attempt_id
                if (folder / "receipt.json").exists():
                    receipt = runtime.read(attempt_id)
                else:
                    trial = Trial(attempt_id, task=task, case=row["case"], scope=self.scope,
                                  budget=self.budgets[task], system2=origin, replacement=row['replacement'])
                    adapters = trial_adapters(self.suite, candidate, task)
                    self._stable()
                    if verify_harness(candidate) != plan["candidate_sha256"]:
                        raise ValueError("Candidate changed during trial adapter construction")
                    receipt = runtime.run_trial(trial, environment=adapters.environment,
                                                policy=adapters.policy, verifier=adapters.verifier)
                if receipt.get("outcome") != "infrastructure-fault":
                    expected = expected_spec(row)
                    audited = runtime.read_evidence(attempt_id)
                    if any(audited['specification'].get(key) != value for key, value in expected.items()):
                        raise ValueError("Completed trial differs from the frozen plan")
                    require_validation_outcome(receipt, task)
                    selected_receipt = receipt
                    if entry is not None:
                        entry.update(selected=attempt_id, attempts=attempts, unavailable=False)
                        save_metadata()
                    break
                fault_audit = runtime.read_evidence(attempt_id)
                if any(fault_audit['specification'].get(key) != value
                       for key, value in expected_spec(row).items()):
                    raise ValueError("Faulted retry differs from the frozen plan")
                # The faulted attempt is already closed and auditable before
                # this durable metadata record permits a retry.
                if metadata is None:
                    metadata = dict(schema="physicalrsi.infrastructure-retries/v1",
                                    plan_sha256=digest(plan), trials={})
                entry = metadata["trials"].setdefault(row["logical_run_id"],
                    dict(attempts=list(attempts), selected=None, unavailable=False))
                entry["attempts"] = list(attempts)
                save_metadata()
                if len(attempts) > self.infrastructure_retries:
                    entry["unavailable"] = True
                    entry["selected"] = None
                    save_metadata()
                    break
                next_id = self._infrastructure_retry_id(directory, row["logical_run_id"], len(attempts))
                attempts.append(next_id)
                entry["attempts"] = list(attempts)
                save_metadata()
                reserve_retry(next_id)
            if selected_receipt is not None:
                receipts.append(selected_receipt)
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
        planned_counts = {}
        available_counts = {}
        for row in execution_rows:
            key = (row["task"], task_case_identity(row["task"], row["case"]))
            planned_counts[key] = planned_counts.get(key, 0) + 1
            if row.get("run_id") is not None:
                available_counts[key] = available_counts.get(key, 0) + 1
        complete_cases = {key for key, count in planned_counts.items()
                          if available_counts.get(key, 0) == count}
        available_rows = [row for row in execution_rows
                          if row.get("run_id") is not None
                          and (row["task"], task_case_identity(row["task"], row["case"]))
                          in complete_cases]
        values = [Evaluation.read(runtime, row["run_id"]).to_dict() for row in available_rows]
        if any(value['trial'].get('replacement') != row['replacement']
               for row, value in zip(available_rows, values)):
            raise ValueError('Development replacement differs from the frozen plan')
        rows = [dict(task=row["task"], case=row["case"], run_id=row["run_id"],
                     outcome=value["outcome"], measurements=value["verdict"]["measurements"])
                for row, value in zip(available_rows, values)]
        unavailable = [dict(task=row["task"], case=row["case"],
                            run_id=row["logical_run_id"],
                            attempts=row.get("infrastructure_attempts", []))
                       for row in execution_rows if row.get("run_id") is None]
        evidence = {str(p.relative_to(output)): file_digest(p) for p in directory.rglob("*.json")}
        for value in values:
            for name, sha in value["evidence"].items():
                evidence[str(Path(name).relative_to(output))] = sha
        result = dict(split="evolve", evaluator_revision=self.revision, episodes=rows,
                      evaluations=values, evidence=evidence,
                      infrastructure_unavailable=unavailable,
                      costs=dict(reserved_trials=len(plan["trials"]) +
                                 sum(row['attempts'] for row in execution_rows)))
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
        if comparison["profile"].get("infrastructure_retries", 2) != self.infrastructure_retries:
            raise ValueError("Comparison infrastructure retry allowance differs from the evaluator")
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
                          infrastructure_retries=self.infrastructure_retries,
                          admission_evidence={str(p.relative_to(output)): file_digest(p)
                                              for p in (path, development_path)})
            if self.repetitions > 1:
                cohort["repetitions"] = self.repetitions
            freeze_json(cohort_path, cohort)
            plans = []
            for candidate_id, sha in comparison["candidates"].items():
                directory = self._directory(output, "validation", candidate_id)
                plan = self._plan(candidate_id, sha, cases, split="validation",
                                  comparison=comparison, cohort=cohort)
                freeze_json(directory / "plan.json", plan)
                plans.append((directory, plan))
            if not getattr(self, "_sequential_mode", False):
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
                or cohort["comparison_sha256"] != digest(comparison)
                or repetition_count(cohort.get("repetitions", 1)) != self.repetitions):
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
        available_rows = [row for row in execution_rows if row.get("run_id") is not None]
        if not available_rows:
            raise ReconciliationRequired("No paired validation cases remain after infrastructure faults")
        result = experiment_result(
            runtime, [row["run_id"] for row in available_rows], candidate_id=candidate["id"],
            comparison=comparison, cohort=cohort, evaluator_revision=self.revision,
            evidence_root=output, score_measurement=self.score_measurement,
            replacements={row['run_id']: row['replacement'] for row in available_rows},
            case_pairs={(row["task"], task_case_identity(row["task"], row["case"]))
                        for row in available_rows})
        unavailable_pairs = sorted({(row["task"], task_case_identity(row["task"], row["case"]))
                                    for row in execution_rows if row.get("run_id") is None})
        if unavailable_pairs:
            result["infrastructure_unavailable"] = unavailable_pairs
        for row in result["episodes"]:
            row["evidence_sha256"].update({str(p.relative_to(output)): file_digest(p)
                for p in (directory / "plan.json", directory / "quota.json")})
            if (directory / "infrastructure.json").exists():
                path = directory / "infrastructure.json"
                row["evidence_sha256"][str(path.relative_to(output))] = file_digest(path)
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
                metrics[task] = dict(planned=len(outcomes), successes=outcomes.count("success"),
                    failures=outcomes.count("failure"), uncertain=outcomes.count("uncertain"),
                    invalid=outcomes.count("invalid"), success_rate=outcomes.count("success") / len(outcomes))
                if self.repetitions > 1:
                    metrics[task].update(cases=len(values), repetitions=self.repetitions,
                        case_success_rate=sum(all(row["outcome"] == "success" for row in rows
                            if row["task"] == task and row["case_sha256"] == digest(case))
                            for case in values) / len(values))
            report = dict(schema="physicalrsi.final-test/v1", candidate_sha256=sha,
                          selection_sha256=selection_sha256, evaluator_revision=self.revision,
                          episodes=rows, metrics=metrics, scope=self.scope, qualification=None,
                          evidence={str(p.relative_to(output)): file_digest(p)
                                    for p in output.rglob("*") if p.is_file()
                                    and p.name != "report.json" and p.suffix != ".lock"})
            freeze_json(output / "report.json", report)
            return report


class SequentialEvaluator:
    """Block evaluator for the opt-in paired-betting selection method.

    The wrapped :class:`ExperimentEvaluator` remains responsible for trial
    construction, recovery, quotas and evidence. This adapter only changes the
    validation execution schedule and records the durable stopping prefix.
    """

    def __init__(self, evaluator, *, block_cases, selector_profile):
        required = ("identity", "validation", "_plan", "_run_plan", "_reserve")
        if any(not callable(getattr(evaluator, name, None)) for name in required):
            raise ValueError("SequentialEvaluator requires an experiment evaluator port")
        if type(block_cases) is not int or block_cases < 1:
            raise ValueError("block_cases must be a positive integer")
        method = selector_profile.get("statistics", {}).get("method")
        if method != "paired-betting":
            raise ValueError("SequentialEvaluator requires paired-betting statistics")
        self.evaluator = evaluator
        self.block_cases = block_cases
        self.selector_profile = deepcopy(selector_profile)
        self.evaluator._sequential_mode = True

    def __getattr__(self, name):
        return getattr(self.evaluator, name)

    def identity(self):
        return self.evaluator.identity()

    def admit(self, candidate, output):
        return self.evaluator.admit(candidate, output)

    def validation(self, comparison, output):
        return self.evaluator.validation(comparison, output)

    def register_round(self, *args, **kwargs):
        return self.evaluator.register_round(*args, **kwargs)

    def _block_cases(self, comparison, cohort, cases, pairs):
        by_sha = {task_case_identity(task, case): case
                  for task, values in cases.items() for case in values}
        result = {task: [] for task in cases}
        for task, layout in pairs:
            result[task].append(by_sha[layout])
        return result

    def _ordered_case_values(self, cases, pairs):
        by_sha = {task_case_identity(task, case): case
                  for task, values in cases.items() for case in values}
        return [(task, by_sha[layout]) for task, layout in pairs]

    def _block_result(self, candidate, comparison, cohort, output, directory, plan, pairs):
        runtime = ExperimentRuntime(directory / "trials", device_registry=self.evaluator.device_registry)
        execution_rows = self.evaluator._execution_rows(directory, plan)
        available = [row for row in execution_rows if row.get("run_id") is not None]
        if self.evaluator.repetitions > 1:
            planned, counts = {}, {}
            for row in execution_rows:
                key = (row["task"], task_case_identity(row["task"], row["case"]))
                planned[key] = planned.get(key, 0) + 1
                if row.get("run_id") is not None:
                    counts[key] = counts.get(key, 0) + 1
            complete = {key for key, count in planned.items() if counts.get(key, 0) == count}
            available = [row for row in available
                         if (row["task"], task_case_identity(row["task"], row["case"])) in complete]
        rows = [row["run_id"] for row in available]
        result = experiment_result(
            runtime, rows, candidate_id=candidate["id"], comparison=comparison,
            cohort=cohort, evaluator_revision=self.evaluator.revision,
            evidence_root=output, score_measurement=self.evaluator.score_measurement,
            replacements={row["run_id"]: row["replacement"] for row in available},
            case_pairs={(row["task"], task_case_identity(row["task"], row["case"]))
                        for row in available},
        )
        unavailable = sorted({(row["task"], task_case_identity(row["task"], row["case"]))
                             for row in execution_rows if row.get("run_id") is None})
        if unavailable:
            result["infrastructure_unavailable"] = unavailable
        return result

    def _combine_results(self, candidate, comparison, cohort, output, block_records):
        combined = None
        for record in block_records:
            partial = self._block_result(candidate, comparison, cohort, output,
                                          Path(record["directory"]), record["plan"],
                                          record["pairs"])
            if combined is None:
                combined = partial
            else:
                combined["episodes"].extend(partial["episodes"])
        if combined is None:
            raise ValueError("Sequential evaluation has no completed blocks")
        order = {pair: index for index, pair in enumerate(cohort_order(comparison, cohort))}
        combined["episodes"] = sorted(
            combined["episodes"],
            key=lambda row: order[(row["task"], row["layout_sha256"])],
        )
        return combined

    def _states(self, comparison, cohort, results, evaluated_pairs):
        profile = comparison["profile"]
        tasks = profile["tasks"]
        largest = max(spec["weight"] for spec in tasks.values())
        weights = {task: spec["weight"] / largest for task, spec in tasks.items()}
        total_weight = math.fsum(weights.values())
        by_candidate = {
            result["candidate_id"]: {(row["task"], row["layout_sha256"]): row
                                     for row in result["episodes"]}
            for result in results
        }
        parent = comparison["parent_id"]
        rule = profile["statistics"]
        alpha_c = rule["alpha"] / rule["max_comparisons"]
        threshold = profile["minimum_gain"] + profile["tie_tolerance"]
        common = set.intersection(*(set(rows) for rows in by_candidate.values()))
        evaluated_pairs = [pair for pair in evaluated_pairs if pair in common]
        counts = {task: sum(pair[0] == task for pair in evaluated_pairs) for task in tasks}
        total_cases = len(evaluated_pairs)
        factors = {task: (weights[task] / total_weight) * total_cases / counts[task]
                   for task in tasks if counts[task]}
        states = {}
        for child in comparison["candidates"]:
            if child == parent:
                continue
            values = []
            for pair in evaluated_pairs:
                task, _ = pair
                low, high = tasks[task]["score_range"]
                parent_score = (by_candidate[parent][pair]["score"] - low) / (high - low)
                child_score = (by_candidate[child][pair]["score"] - low) / (high - low)
                values.append(factors[task] * (child_score - parent_score))
            states[child] = betting_evidence(
                values, threshold, alpha_c, min_cases=rule.get("min_cases", 8),
                bet_clip=rule.get("bet_clip", 0.5))
        return states

    def evaluate_pool(self, accepted, comparison, cohort, output):
        output = self.evaluator._output(output)
        cases = self.evaluator._cases(read_json(output / "validation-cases.json"), comparison["profile"])
        order = cohort_order(comparison, cohort)
        state_path = output / "sequential.json"
        if state_path.exists():
            state = read_json(state_path)
            if state.get("comparison_sha256") != digest(comparison) or state.get("cohort_sha256") != digest(cohort):
                raise ValueError("Sequential evaluation state differs from the frozen round")
            if state.get("block_cases") != self.block_cases:
                raise ValueError("Sequential block size changed during resume")
            blocks_run = state["blocks_run"]
        else:
            state = dict(schema="physicalrsi.sequential-evaluation/v1",
                         comparison_sha256=digest(comparison), cohort_sha256=digest(cohort),
                         block_cases=self.block_cases, blocks=[], blocks_run=0,
                         evaluated_cases=0)
            blocks_run = 0
        block_records = {candidate["id"]: [] for candidate in accepted}
        for block_index in range(blocks_run):
            pairs = order[block_index * self.block_cases:(block_index + 1) * self.block_cases]
            for candidate in accepted:
                directory = (output / "experiments" / "validation" / candidate["id"] /
                             "sequential" / f"block-{block_index:04d}")
                plan = self.evaluator._plan(
                    candidate["id"], comparison["candidates"][candidate["id"]],
                    self._block_cases(comparison, cohort, cases, pairs), split="validation",
                    comparison=comparison, cohort=cohort,
                    case_order=self._ordered_case_values(cases, pairs))
                block_records[candidate["id"]].append(dict(directory=str(directory), plan=plan, pairs=pairs))
        if not state_path.exists():
            state["blocks_run"] = 0
        while blocks_run * self.block_cases < len(order):
            pairs = order[blocks_run * self.block_cases:(blocks_run + 1) * self.block_cases]
            for candidate in accepted:
                directory = (output / "experiments" / "validation" / candidate["id"] /
                             "sequential" / f"block-{blocks_run:04d}")
                block_cases = self._block_cases(comparison, cohort, cases, pairs)
                plan = self.evaluator._plan(candidate["id"], comparison["candidates"][candidate["id"]],
                                            block_cases, split="validation", comparison=comparison, cohort=cohort,
                                            case_order=self._ordered_case_values(cases, pairs))
                with locked(directory / ".plan.lock"):
                    freeze_json(directory / "plan.json", plan)
                    self.evaluator._reserve([(directory, plan)])
                    self.evaluator._run_plan(candidate, directory, plan, comparison=comparison)
                block_records[candidate["id"]].append(dict(directory=str(directory), plan=plan, pairs=pairs))
            blocks_run += 1
            prefix = order[:blocks_run * self.block_cases]
            partial_results = [self._combine_results(candidate, comparison, cohort, output,
                                                       block_records[candidate["id"]])
                               for candidate in accepted]
            states = self._states(comparison, cohort, partial_results, prefix)
            state["blocks"].append(dict(block=blocks_run - 1, cases_evaluated=len(prefix),
                                        decisions={name: value["decision"] for name, value in states.items()},
                                        states=states))
            state.update(blocks_run=blocks_run, evaluated_cases=len(prefix))
            atomic_json(state_path, state)
            if states and all(value["decision"] in {"accept", "reject"} for value in states.values()):
                break
        return [self._combine_results(candidate, comparison, cohort, output,
                                      block_records[candidate["id"]])
                for candidate in accepted]

    def resume_evaluation_pool(self, accepted, comparison, cohort, output):
        return self.evaluate_pool(accepted, comparison, cohort, output)
