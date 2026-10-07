"""Adapt completed experiments into one candidate's paired evaluation result.

Task evaluators still own admission, fresh case generation and the scoring
protocol. This bridge verifies candidate/comparison attribution and raw receipts
before exposing scores to the existing Self-Harness selector.
"""

import math
from pathlib import Path

from ..infra.storage import digest, file_digest, read_json
from .selection import comparison_identity


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
                      evaluator_revision, evidence_root, score_measurement=None):
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
    if candidate_id not in comparison["candidates"] or not evaluator_revision:
        raise ValueError("Declare an admitted candidate and evaluator revision")
    run_ids = tuple(run_ids)
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("Experiment IDs must be nonempty and unique")
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
    used, rows, protocols, system2_revisions = set(), [], [], set()
    for run_id in run_ids:
        receipt = runtime.read(run_id)
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
        pair = (specification["task"], digest(specification["case"]))
        if pair not in expected or pair in used:
            raise ValueError("Experiment cases are not paired with the frozen cohort")
        success = receipt["outcome"] == "success"
        score = experiment_score(receipt, score_measurement=score_measurement,
                                 score_range=tasks[pair[0]]["score_range"])
        evidence = {}
        for name in ["receipt.json", *receipt["evidence"]]:
            path = (folder / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Experiment evidence lies outside the comparison root")
            sha = file_digest(path)
            if name != "receipt.json" and sha != receipt["evidence"][name]:
                raise ValueError("Experiment evidence changed while building evaluation")
            if name == "receipt.json" and read_json(path) != receipt:
                raise ValueError("Experiment receipt changed while building evaluation")
            evidence[str(path.relative_to(root))] = sha
        used.add(pair)
        rows.append(dict(task=pair[0], layout_sha256=pair[1], state="completed",
                         score=score, success=success, evidence_sha256=evidence))
        # Policies may differ; execution conditions and independent scoring must
        # agree for each paired case across the whole candidate pool.
        protocols.append(dict(
            task=pair[0], case_sha256=pair[1], embodiment=binding["embodiment"],
            budget=specification["budget"], scope=specification["scope"],
            system2=origin["system2"], score_measurement=score_measurement,
            adapters={key: value for key, value in specification["adapters"].items()
                      if key != "policy"}))
    if used != expected or len(system2_revisions) != 1:
        raise ValueError("Incomplete experiment cohort or mixed System 2 revisions")
    return dict(candidate_id=candidate_id,
                kind=comparison.get("evaluation_kind", "policy_evaluation"),
                native_exit_code=0, comparison_sha256=comparison_sha, cohort_sha256=cohort_sha,
                freeze_sha256=comparison["candidates"][candidate_id],
                evaluator_revision=evaluator_revision,
                experiment_protocol_sha256=digest(sorted(
                    protocols, key=lambda row: (row["task"], row["case_sha256"]))),
                episodes=sorted(rows, key=lambda row: (row["task"], row["layout_sha256"])))
