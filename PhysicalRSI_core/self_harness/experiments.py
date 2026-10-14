"""Adapt completed experiments into one candidate's paired evaluation result.

Task evaluators still own admission, fresh case generation and the scoring
protocol. This bridge verifies candidate/comparison attribution and raw receipts
before exposing scores to the existing Self-Harness selector.
"""

from copy import deepcopy
import math
from pathlib import Path
from statistics import mean

from ..infra.storage import digest, file_digest, read_json
from ..trials import Evaluation, task_case_identity
from .selection import comparison_identity


def repetition_count(value):
    """Bound independent executions per case; replacements are separate."""
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("Declare one to one hundred repetitions per case")
    return value


def _aggregate_repetitions(rows, tasks):
    """One paired observation per case, retaining every execution's evidence.

    Repeats estimate a case's mean score, not additional independent layouts.
    A case meets success gates only when all its executions succeed. Declared
    cost measurements retain their existing mean-per-execution interpretation.
    """
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["layout_sha256"]), []).append(row)
    result = []
    for (task, layout), repeats in groups.items():
        measurements, evidence = {}, {}
        for key in tasks[task].get("cost_limits", {}):
            values = [row["measurements"].get(key) for row in repeats]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Repeated measured cost must be finite and nonnegative")
            measurements[key] = float(mean(values))
        for row in repeats:
            for path, sha in row["evidence_sha256"].items():
                if path in evidence and evidence[path] != sha:
                    raise ValueError("Conflicting repeated evaluation evidence")
                evidence[path] = sha
        result.append(dict(task=task, layout_sha256=layout, state="completed",
            score=float(mean(row["score"] for row in repeats)),
            success=all(row["success"] for row in repeats),
            success_fraction=sum(row["success"] for row in repeats) / len(repeats),
            measurements=measurements, evidence_sha256=evidence,
            repetitions=len(repeats), replicates=repeats))
    return result


def experiment_score(receipt, *, score_measurement, score_range):
    """Validate the same measured score before more trials and at selection."""
    if receipt["outcome"] not in {"success", "failure"}:
        raise ValueError("Experiment has no definite task outcome; preserve it without scoring")
    score = (int(receipt["outcome"] == "success") if score_measurement is None else
             receipt["verdict"]["measurements"].get(score_measurement))
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Experiment score measurement must be a finite scalar")
    low, high = score_range
    if not low <= score <= high:
        raise ValueError("Experiment score lies outside the declared native range")
    return score


def experiment_result(runtime, run_ids, *, candidate_id, comparison, cohort,
                      evaluator_revision, evidence_root, score_measurement=None, replacements=None):
    """Build a selection result from exact paired cases and definite outcomes.

    Cases must hash to the cohort's layout identities. By default the native
    score is binary success. An explicit measurement name uses a finite scalar
    from the independent verdict instead. Keep that mapping fixed in the
    evaluator protocol. Invalid, uncertain and interrupted trials never become
    zero scores or disappear from the cohort.
    """
    comparison_sha = comparison_identity(comparison)
    cohort_sha = digest(cohort)
    if cohort.get("comparison_sha256") != comparison_sha or cohort.get("split") != "validation":
        raise ValueError("Experiment cohort differs from the frozen comparison")
    identity_scheme = cohort.get("case_identity", "case/v1")
    repetitions = repetition_count(cohort.get("repetitions", 1))
    if identity_scheme not in {"case/v1", "task-case/v1"}:
        raise ValueError("Unsupported experiment case identity scheme")
    if candidate_id not in comparison["candidates"] or not evaluator_revision:
        raise ValueError("Declare an admitted candidate and evaluator revision")
    run_ids = tuple(run_ids)
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("Experiment IDs must be nonempty and unique")
    if replacements is not None:
        replacements = deepcopy(replacements)
        if set(replacements) != set(run_ids):
            raise ValueError('Replacement references must cover exactly the planned experiments')
    root = Path(evidence_root).resolve()
    tasks = comparison["profile"]["tasks"]
    layouts = cohort["layouts"]
    if set(layouts) != set(tasks) or any(
        len(layouts[task]) != tasks[task]["episodes"] for task in tasks
    ):
        raise ValueError("Experiment cohort coverage differs from the declared task profile")
    all_cases = [case for cases in layouts.values() for case in cases]
    if len(set(all_cases)) != len(all_cases):
        raise ValueError("Experiment cohort contains repeated cases")
    expected = {(task, case) for task, cases in layouts.items() for case in cases}
    used, rows, protocols, system2_revisions = {}, [], [], set()
    for run_id in run_ids:
        evaluation = Evaluation.read(runtime, run_id)
        value = evaluation.to_dict()
        if replacements is not None and value['trial'].get('replacement') != replacements[run_id]:
            raise ValueError('Evaluation replacement differs from the frozen plan')
        receipt = value["receipt"]
        folder = runtime.root / run_id
        specification = read_json(folder / "experiment.json")
        origin = specification.get("system2")
        if not origin or any(origin.get(key) != value for key, value in dict(
            candidate_id=candidate_id,
            candidate_sha256=comparison["candidates"][candidate_id],
            comparison_sha256=comparison_sha,
            cohort_sha256=cohort_sha,
            split="validation",
        ).items()):
            raise ValueError("Experiment does not belong to this candidate/comparison/cohort")
        binding = specification.get("binding")
        if (not binding or binding["system1"]["revision"] != origin["candidate_sha256"]
                or specification["scope"] != comparison["scope"]):
            raise ValueError("Experiment execution revision or scope differs")
        system2_revisions.add(digest(origin["system2"]))
        task, case = specification["task"], specification["case"]
        pair = (task, task_case_identity(task, case) if identity_scheme == "task-case/v1" else digest(case))
        if pair not in expected or used.get(pair, 0) >= repetitions:
            raise ValueError("Experiment cases are not paired with the frozen cohort")
        success = receipt["outcome"] == "success"
        score = experiment_score(receipt, score_measurement=score_measurement,
                                 score_range=tasks[pair[0]]["score_range"])
        evidence = {}
        for name, sha in value["evidence"].items():
            path = Path(name)
            if not path.is_relative_to(root):
                raise ValueError("Experiment evidence lies outside the comparison root")
            if file_digest(path) != sha:
                raise ValueError("Experiment evidence changed while building evaluation")
            evidence[str(path.relative_to(root))] = sha
        used[pair] = used.get(pair, 0) + 1
        rows.append(dict(task=pair[0], layout_sha256=pair[1], state="completed",
                         score=score, success=success, evidence_sha256=evidence,
                         trial_sha256=value["trial_sha256"], evaluation_sha256=evaluation.revision,
                         measurements=receipt["verdict"]["measurements"]))
        # Policies may differ; execution conditions and independent scoring must
        # agree for each paired case across the whole candidate pool.
        protocols.append(dict(
            task=pair[0], case_sha256=pair[1], embodiment=binding["embodiment"],
            budget=specification["budget"], scope=specification["scope"],
            system2=origin["system2"], score_measurement=score_measurement,
            adapters={key: value for key, value in specification["adapters"].items()
                      if key != "policy"}))
    if set(used) != expected or any(count != repetitions for count in used.values()) or len(system2_revisions) != 1:
        raise ValueError("Incomplete experiment cohort or mixed System 2 revisions")
    if repetitions > 1:
        rows = _aggregate_repetitions(rows, tasks)
    return dict(candidate_id=candidate_id,
                kind=comparison.get("evaluation_kind", "policy_evaluation"),
                native_exit_code=0, comparison_sha256=comparison_sha, cohort_sha256=cohort_sha,
                freeze_sha256=comparison["candidates"][candidate_id],
                evaluator_revision=evaluator_revision,
                experiment_protocol_sha256=digest(sorted(
                    protocols, key=lambda row: (row["task"], row["case_sha256"]))),
                episodes=sorted(rows, key=lambda row: (row["task"], row["layout_sha256"])))
