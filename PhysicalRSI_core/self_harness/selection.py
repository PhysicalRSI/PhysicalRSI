"""Paired survivor selection for a previously frozen comparison.

This checks supplied identities and local evidence, not the truthfulness of a
producer, policy isolation, layout novelty, or completeness of its dependency
closure. Admission, full candidate freezing and layout registration remain
separate prerequisites. No benchmark-specific names or score scales live here.
"""

import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    identifier,
    locked,
    relative_path,
)


def _number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite numeric value")
    return value


def _sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Expected a SHA256 identity")
    return value


def _time(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None or value > datetime.now(timezone.utc):
        raise ValueError("Invalid event timestamp")
    return value


def _sample_sd(values):
    """Return the sample standard deviation, or None for fewer than two values."""
    if len(values) < 2:
        return None
    center = math.fsum(values) / len(values)
    return math.sqrt(max(0.0, math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)))


def _beta_continued_fraction(a, b, x):
    """Evaluate the incomplete-beta continued fraction using stdlib math only."""
    # Numerical Recipes' Lentz recurrence is stable for the parameter ranges
    # used by the fixed-n Student t test.
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    value = d
    for m in range(1, 10000):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        value *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return value


def _regularized_beta(a, b, x):
    """Regularized incomplete beta I_x(a,b), with no third-party dependency."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return beta_term * _beta_continued_fraction(a, b, x) / a
    return 1.0 - beta_term * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_survival(t_statistic, degrees_of_freedom):
    """One-sided Student t survival function for a finite statistic."""
    if not math.isfinite(t_statistic) or degrees_of_freedom <= 0:
        return 0.0 if t_statistic > 0 else 1.0
    if t_statistic == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic * t_statistic)
    tail = 0.5 * _regularized_beta(degrees_of_freedom / 2.0, 0.5, x)
    return tail if t_statistic > 0 else 1.0 - tail


# Acklam's rational approximation, accurate to better than 1e-9 over the
# ordinary probability range and substantially better than the 1e-6 contract.
_NORMAL_A = (-3.969683028665376e1, 2.209460984245205e2,
             -2.759285104469687e2, 1.383577518672690e2,
             -3.066479806614716e1, 2.506628277459239e0)
_NORMAL_B = (-5.447609879822406e1, 1.615858368580409e2,
             -1.556989798598866e2, 6.680131188771972e1,
             -1.328068155288572e1)
_NORMAL_C = (-7.784894002430293e-3, -3.223964580411365e-1,
             -2.400758277161838e0, -2.549732539343734e0,
             4.374664141464968e0, 2.938163982698783e0)
_NORMAL_D = (7.784695709041462e-3, 3.224671290700398e-1,
             2.445134137142996e0, 3.754408661907416e0)


def _normal_quantile(probability):
    """Inverse standard normal CDF using a rational approximation."""
    if probability <= 0.0:
        return -math.inf
    if probability >= 1.0:
        return math.inf
    low = 0.02425
    high = 1.0 - low
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        numerator = (((((_NORMAL_C[0] * q + _NORMAL_C[1]) * q + _NORMAL_C[2]) * q
                       + _NORMAL_C[3]) * q + _NORMAL_C[4]) * q + _NORMAL_C[5])
        denominator = (((( _NORMAL_D[0] * q + _NORMAL_D[1]) * q + _NORMAL_D[2]) * q
                        + _NORMAL_D[3]) * q + 1.0)
        return numerator / denominator
    if probability > high:
        q = math.sqrt(-2.0 * math.log1p(-probability))
        numerator = (((((_NORMAL_C[0] * q + _NORMAL_C[1]) * q + _NORMAL_C[2]) * q
                       + _NORMAL_C[3]) * q + _NORMAL_C[4]) * q + _NORMAL_C[5])
        denominator = (((( _NORMAL_D[0] * q + _NORMAL_D[1]) * q + _NORMAL_D[2]) * q
                        + _NORMAL_D[3]) * q + 1.0)
        return -numerator / denominator
    q = probability - 0.5
    r = q * q
    numerator = (((((_NORMAL_A[0] * r + _NORMAL_A[1]) * r + _NORMAL_A[2]) * r
                   + _NORMAL_A[3]) * r + _NORMAL_A[4]) * r + _NORMAL_A[5]) * q
    denominator = (((((_NORMAL_B[0] * r + _NORMAL_B[1]) * r + _NORMAL_B[2]) * r
                     + _NORMAL_B[3]) * r + _NORMAL_B[4]) * r + 1.0)
    return numerator / denominator


def _normal_upper_quantile(tail_probability):
    """Return z with P(Z >= z)=tail_probability without 1-tail rounding."""
    if tail_probability <= 0.0:
        return math.inf
    if tail_probability >= 1.0:
        return -math.inf
    if tail_probability > 0.5:
        return -_normal_quantile(tail_probability)
    q = math.sqrt(-2.0 * math.log(tail_probability))
    numerator = (((((_NORMAL_C[0] * q + _NORMAL_C[1]) * q + _NORMAL_C[2]) * q
                   + _NORMAL_C[3]) * q + _NORMAL_C[4]) * q + _NORMAL_C[5])
    denominator = (((( _NORMAL_D[0] * q + _NORMAL_D[1]) * q + _NORMAL_D[2]) * q
                    + _NORMAL_D[3]) * q + 1.0)
    return -numerator / denominator


def comparison_identity(comparison):
    """Validate the predeclared pool and metric profile; return its identity."""
    if comparison.get("schema_version") != 1:
        raise ValueError("Unsupported comparison schema")
    identifier(comparison["round_id"])
    _time(comparison["frozen_at"])
    candidates = comparison["candidates"]
    parent = comparison["parent_id"]
    if len(candidates) < 2 or parent not in candidates:
        raise ValueError("Comparison requires the parent and at least one child")
    for name, freeze in candidates.items():
        identifier(name)
        _sha(freeze)
    validate_profile(comparison["profile"])
    return digest(comparison)


def validate_profile(profile):
    """Reject an invalid scoring protocol before any development or trials."""
    retries = profile.get("infrastructure_retries", 2)
    if type(retries) is not int or not 0 <= retries <= 10:
        raise ValueError("Declare zero to ten infrastructure retries")
    tasks = profile["tasks"]
    if not tasks:
        raise ValueError("Empty task scope")
    for task, spec in tasks.items():
        identifier(task)
        if _number(spec["weight"]) <= 0:
            raise ValueError("Task weights must be positive")
        if type(spec["episodes"]) is not int or spec["episodes"] < 1:
            raise ValueError("Invalid cohort size")
        low, high = map(_number, spec["score_range"])
        if low >= high or not math.isfinite(high - low):
            raise ValueError("Invalid native score range")
        if not 0 <= _number(spec["maximum_regression"]) <= 1:
            raise ValueError("Invalid normalized regression limit")
        for key in ("minimum_success_rate", "maximum_success_regression"):
            if key in spec and not 0 <= _number(spec[key]) <= 1:
                raise ValueError("Invalid success constraint")
        cost_limits = spec.get("cost_limits", {})
        if not isinstance(cost_limits, dict):
            raise ValueError("Cost limits must map measurement names to bounds")
        for measurement, limit in cost_limits.items():
            identifier(measurement)
            if _number(limit) < 0:
                raise ValueError("Cost limits must be nonnegative")
    if not 0 <= _number(profile["minimum_gain"]) <= 1:
        raise ValueError("Invalid normalized improvement margin")
    if not 0 <= _number(profile["tie_tolerance"]) < 1:
        raise ValueError("Invalid tie tolerance")
    if "cost_reduction" in profile or "capability_qualification" in profile:
        raise ValueError(
            "Optional cost/qualification policies are not admitted in the minimal core"
        )
    if "statistics" in profile:
        rule = profile["statistics"]
        methods = {"paired-hoeffding", "paired-permutation", "paired-t", "paired-betting"}
        if not isinstance(rule, dict):
            raise ValueError(
                "Declare a statistics method from paired-hoeffding, paired-permutation, paired-t, paired-betting"
            )
        method = rule.get("method")
        if method not in methods:
            raise ValueError(
                "Unknown statistics method; choose paired-hoeffding, paired-permutation, paired-t, or paired-betting"
            )
        allowed = {"method", "alpha", "max_comparisons"}
        if method == "paired-permutation":
            allowed.add("resamples")
        if method == "paired-betting":
            allowed.update({"min_cases", "bet_clip"})
        if set(rule) - allowed:
            raise ValueError("Statistics block contains an unsupported key")
        if (
            not 0 < _number(rule.get("alpha")) < 1
            or type(rule.get("max_comparisons")) is not int
            or rule["max_comparisons"] < 1
        ):
            raise ValueError("Declare alpha and a bounded comparison budget")
        if "resamples" in rule and (
            method != "paired-permutation"
            or type(rule["resamples"]) is not int
            or not 200 <= rule["resamples"] <= 100000
        ):
            raise ValueError("Permutation resamples must be an integer from 200 through 100000")
        if method == "paired-betting":
            if type(rule.get("min_cases", 8)) is not int or rule.get("min_cases", 8) < 4:
                raise ValueError("Betting min_cases must be an integer of at least 4")
            if not 0 < _number(rule.get("bet_clip", 0.5)) <= 1:
                raise ValueError("Betting bet_clip must be in (0, 1]")


def cohort_order(comparison, cohort):
    """Return frozen validation pairs in balanced round-robin task order.

    Tasks are visited in lexical order at each layout index, so prefixes contain
    nearly equal numbers of cases from every task.
    """
    tasks = sorted(comparison["profile"]["tasks"])
    layouts = cohort["layouts"]
    if set(layouts) != set(tasks):
        raise ValueError("Cohort task coverage differs")
    return [(task, layouts[task][index])
            for index in range(max((len(layouts[task]) for task in tasks), default=0))
            for task in tasks if index < len(layouts[task])]


def betting_evidence(x_values, threshold, alpha_c, *, min_cases=8, bet_clip=0.5):
    """Anytime-valid paired betting evidence for ``mean(x) > threshold``.

    Contributions are clipped to [-1, 1] before mapping to ``y`` in [0, 1].
    The predictable plug-in bet uses the variance of observations available
    before the current case, with initial variance 1/4.
    """
    if type(min_cases) is not int or min_cases < 4:
        raise ValueError("min_cases must be an integer of at least 4")
    if not 0 < _number(alpha_c) < 1 or not 0 < _number(bet_clip) <= 1:
        raise ValueError("Invalid betting parameters")
    values = [max(-1.0, min(1.0, _number(value))) for value in x_values]
    log_plus = log_minus = 0.0
    max_plus = max_minus = 0.0
    history = []
    stopping = None
    decision = "undecided"
    target = math.log(1.0 / alpha_c)
    for index, value in enumerate(values, 1):
        y = max(0.0, min(1.0, (value - threshold + 1.0) / 2.0))
        previous_count = len(history)
        if previous_count < 2:
            variance = 0.25
        else:
            center = math.fsum(history) / previous_count
            variance = math.fsum((item - center) ** 2 for item in history) / (previous_count - 1)
            variance = max(variance, 1e-12)
        lam = math.sqrt(2.0 * math.log(2.0 / alpha_c) /
                        (variance * index * math.log(index + 1.0)))
        lam = max(-bet_clip, min(bet_clip, lam))
        centered = 2.0 * y - 1.0
        log_plus += math.log1p(lam * centered)
        log_minus += math.log1p(-lam * centered)
        max_plus = max(max_plus, log_plus)
        max_minus = max(max_minus, log_minus)
        history.append(y)
        if index >= min_cases and stopping is None:
            if log_plus >= target:
                decision, stopping = "accept", index
            elif log_minus >= target:
                decision, stopping = "reject", index
    return dict(decision=decision, stopping_index=stopping,
                log_capital_plus=log_plus, log_capital_minus=log_minus,
                log_K_plus=log_plus, log_K_minus=log_minus,
                n_used=len(values), max_log_capital_plus=max_plus,
                max_log_capital_minus=max_minus,
                max_log_K_plus=max_plus, max_log_K_minus=max_minus)


def validate_admission_binding(
    admission, *, candidate_id, freeze_sha256, comparison, alias, current_revision
):
    """Validate a producer receipt bound to the current release interaction.

    Paired score selection intentionally stays agnostic to how an external
    evaluator was run.  A release operation that opts into this stricter gate
    must nevertheless prove that each CODE_POLICY admission was produced for this
    candidate, comparison, protocol/profile snapshot and current parent.  The
    check is opt-in so old synthetic/core fixtures remain valid and cannot be
    mistaken for the stronger external-admission contract.
    """
    if (
        not isinstance(admission, dict)
        or admission.get("status", "completed") != "completed"
    ):
        raise ValueError("Admission binding requires a completed producer receipt")
    if (
        admission.get("candidate_id") != candidate_id
        or admission.get("freeze_sha256") != freeze_sha256
    ):
        raise ValueError("Admission binding candidate identity mismatch")
    comparison_sha = digest(comparison)
    if admission.get("comparison_sha256") != comparison_sha:
        raise ValueError("Admission binding comparison identity mismatch")
    protocol = admission.get("protocol_identity")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("Admission binding protocol identity required")
    profile_sha = admission.get("profile_sha256")
    if not _sha(profile_sha) or profile_sha != digest(comparison.get("profile")):
        raise ValueError("Admission binding profile identity required")
    resource_profile_sha = admission.get("resource_profile_sha256")
    if not _sha(resource_profile_sha):
        raise ValueError("Admission binding resource profile identity required")
    if (
        comparison.get("resource_plan_sha256") is not None
        and resource_profile_sha != comparison["resource_plan_sha256"]
    ):
        raise ValueError("Admission binding resource profile mismatch")
    producer = admission.get("producer")
    if (
        not isinstance(producer, dict)
        or not isinstance(producer.get("id"), str)
        or not producer["id"].strip()
        or not _sha(producer.get("receipt_sha256"))
    ):
        raise ValueError("Admission binding producer receipt required")
    if (
        comparison.get("protocol_sha256") is not None
        and protocol != comparison["protocol_sha256"]
    ):
        raise ValueError("Admission binding protocol identity mismatch")
    binding = admission.get("current_binding")
    if not isinstance(binding, dict):
        raise ValueError("Admission binding current release identity required")
    expected = dict(
        alias=alias,
        revision=current_revision,
        parent_id=comparison.get("parent_id"),
        parent_freeze_sha256=comparison.get("candidates", {}).get(
            comparison.get("parent_id")
        ),
    )
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Admission binding differs from current release parent")
    skill_choice_sha = admission.get("skill_choice_policy_sha256")
    if not _sha(skill_choice_sha):
        raise ValueError("Admission binding skill_choice policy identity required")
    return True


def select_survivor(
    comparison, cohort, results, *, evidence_root, qualification_repository=None
):
    """Require complete paired results; retain parent unless a child improves.

    Scores are normalized by predeclared task ranges, then task-weighted. Every
    result must use the exact same task/layout pairs and evaluator identity.
    Missing episodes, runtime errors, changed evidence and duplicate layouts
    invalidate selection rather than being silently excluded or scored as zero.
    """
    comparison_sha = comparison_identity(comparison)
    if cohort.get("comparison_sha256") != comparison_sha:
        raise ValueError("Cohort belongs to another frozen comparison")
    if _time(cohort["generated_at"]) <= _time(comparison["frozen_at"]):
        raise ValueError(
            "Validation must be generated after the entire pool/profile freezes"
        )
    # JSON object order is not part of the frozen comparison identity. Use one
    # order for arithmetic and rejection lists before and after persistence.
    tasks = {name: comparison["profile"]["tasks"][name]
             for name in sorted(comparison["profile"]["tasks"])}
    largest = max(spec["weight"] for spec in tasks.values())
    weights = {task: spec["weight"] / largest for task, spec in tasks.items()}
    total_weight = math.fsum(weights.values())
    layouts = cohort["layouts"]
    statistics = comparison["profile"].get("statistics")
    betting = bool(statistics and statistics.get("method") == "paired-betting")
    if set(layouts) != set(tasks):
        raise ValueError("Cohort task coverage differs")
    seen = set()
    expected = set()
    for task, spec in tasks.items():
        if len(layouts[task]) != spec["episodes"]:
            raise ValueError("Incomplete validation cohort")
        for sha in layouts[task]:
            _sha(sha)
            if sha in seen:
                raise ValueError("Duplicate geometry across task cohorts")
            seen.add(sha)
            expected.add((task, sha))
    frozen_order = cohort_order(comparison, cohort)
    expected = set(frozen_order)
    cohort_cases = len(frozen_order)
    cohort_sha = digest(cohort)
    pool = comparison["candidates"]
    if len(results) != len(pool) or {r["candidate_id"] for r in results} != set(pool):
        raise ValueError("Missing, duplicate or unexpected candidate result")
    if any("experiment_protocol_sha256" in result for result in results):
        protocols = {_sha(result.get("experiment_protocol_sha256")) for result in results}
        if len(protocols) != 1:
            raise ValueError("Paired experiment execution or scoring protocols differ")
    # Infrastructure exhaustion removes a logical pair from the comparison for
    # every arm.  Results may therefore contain different subsets of the
    # frozen cohort; retain only the intersection so an arm that completed a
    # case cannot turn the other arm's unavailable case into a policy failure.
    available_sets = []
    for result in results:
        pairs = {(row.get("task"), row.get("layout_sha256")) for row in result.get("episodes", [])}
        if any(task not in tasks or layout not in set(layouts[task]) for task, layout in pairs):
            raise ValueError("Result contains a case outside the frozen cohort")
        available_sets.append(pairs)
    if (betting and len({frozenset(pairs) for pairs in available_sets}) != 1
            and not any(result.get("infrastructure_unavailable") for result in results)):
        raise ValueError("Sequential candidate evaluation is not the frozen prefix")
    if available_sets:
        expected = set.intersection(*available_sets)
        if not expected:
            raise ValueError("No paired cases remain after infrastructure faults")
        if not betting:
            cohort_cases = len(expected)
        if not betting and any(not any(pair[0] == task for pair in expected) for task in tasks):
            raise ValueError("Infrastructure faults removed every case for a task")
    evaluation_order = [pair for pair in frozen_order if pair in expected]
    if betting:
        filtered_sets = [frozenset(pair for pair in pairs if pair in expected)
                         for pairs in available_sets]
        if (len(set(filtered_sets)) != 1
                and not any(result.get("infrastructure_unavailable") for result in results)):
            raise ValueError("Sequential candidates must cover the identical prefix")
    root = Path(evidence_root).resolve()
    metrics = {}
    paired_scores = {}
    revisions = set()
    verified_evidence = {}
    for result in results:
        name = result["candidate_id"]
        if (
            result.get("kind") != comparison.get("evaluation_kind", "policy_evaluation")
            or type(result.get("native_exit_code")) is not int
            or result["native_exit_code"] != 0
        ):
            raise ValueError(
                "Expected a successfully completed native policy evaluation"
            )
        if (
            result.get("comparison_sha256") != comparison_sha
            or result.get("cohort_sha256") != cohort_sha
            or result.get("freeze_sha256") != pool[name]
        ):
            raise ValueError("Result/candidate identity mismatch")
        revision = result.get("evaluator_revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("Missing evaluator identity")
        revisions.add(revision)
        rows = [row for row in result["episodes"]
                if (row.get("task"), row.get("layout_sha256")) in expected]
        if betting:
            min_cases = statistics.get("min_cases", 8)
            if len(rows) < min_cases or len(rows) > cohort_cases:
                raise ValueError("Invalid sequential candidate evaluation prefix")
            pairs = [(row["task"], row["layout_sha256"]) for row in rows]
            if (len(set(pairs)) != len(pairs)
                    or set(pairs) != set(evaluation_order[:len(pairs)])):
                raise ValueError("Sequential candidate evaluation is not the frozen prefix")
        elif len(rows) != len(expected):
            raise ValueError("Incomplete candidate evaluation")
        used = set()
        scores = {task: [] for task in tasks}
        success = {task: [] for task in tasks}
        costs = {task: {key: [] for key in spec.get("cost_limits", {})} for task, spec in tasks.items()}
        paired_scores[name] = {}
        for row in rows:
            pair = (row["task"], row["layout_sha256"])
            if pair not in expected or pair in used:
                raise ValueError("Unpaired or repeated episode")
            used.add(pair)
            if row.get("state") != "completed" or type(row.get("success")) is not bool:
                raise ValueError("Runtime failure or missing native outcome")
            score = _number(row["score"])
            low, high = tasks[pair[0]]["score_range"]
            if not low <= score <= high:
                raise ValueError("Native score outside declared range")
            evidence = row.get("evidence_sha256")
            if not isinstance(evidence, dict) or not evidence:
                raise ValueError("Missing native evidence")
            for relative, sha in evidence.items():
                path = (root / relative_path(relative)).resolve()
                if (
                    not path.is_relative_to(root)
                    or not path.is_file()
                    or (verified_evidence[path] if path in verified_evidence else file_digest(path)) != _sha(sha)
                ):
                    raise ValueError(
                        "Native evidence missing, changed or outside the evidence root"
                    )
                verified_evidence[path] = sha
            scores[pair[0]].append((score - low) / (high - low))
            paired_scores[name][pair] = (score - low) / (high - low)
            success[pair[0]].append(row["success"])
            for key, values in costs[pair[0]].items():
                value = _number(row.get("measurements", {}).get(key))
                if value < 0:
                    raise ValueError("Measured cost must be nonnegative")
                values.append(value)
        per_task = {task: float(mean(values)) for task, values in scores.items()}
        metrics[name] = dict(
            normalized_score=math.fsum(per_task[t] * (weights[t] / total_weight)
                                       for t in tasks),
            per_task=per_task,
            success_rate={t: sum(v) / len(v) for t, v in success.items()},
        )
        if any(costs.values()):
            metrics[name]["mean_cost"] = {task: {key: float(mean(values))
                                               for key, values in items.items()} for task, items in costs.items()}
    if len(revisions) != 1:
        raise ValueError("Mixed evaluator revisions")
    evaluated_cases = len(results[0]["episodes"])
    if betting and any(len(result["episodes"]) != evaluated_cases for result in results):
        raise ValueError("Sequential candidates must cover the identical prefix")
    parent = comparison["parent_id"]
    parent_metrics = metrics[parent]
    profile = comparison["profile"]
    eligible = []
    rejection = {}
    statistics = profile.get("statistics")
    radius = None
    if statistics:
        if len(pool) - 1 > statistics["max_comparisons"]:
            raise ValueError("Candidate pool exceeds statistical comparison budget")
        # Independent paired case differences lie in [-1, 1]. Weighted
        # Hoeffding plus a union bound covers the preregistered total number of
        # candidate comparisons, including adaptive rounds with fresh cases.
        if statistics["method"] == "paired-hoeffding":
            counts = {task: sum(pair[0] == task for pair in expected) for task in tasks}
            variance_bound = math.fsum((weights[task] / total_weight) ** 2 / counts[task]
                                       for task in tasks if counts[task])
            # Form the ratio in log space: a valid small alpha can overflow the
            # quotient even though the confidence radius is finite.
            log_correction = math.log(statistics["max_comparisons"]) - math.log(statistics["alpha"])
            radius = math.sqrt(2 * log_correction * variance_bound)

    available_is_prefix = expected == set(frozen_order[:len(expected)])
    n_total = (cohort_cases if betting and available_is_prefix else len(expected))
    # For a case in task t, x_i = (w_t / W) * (n_total / n_t) * d_i.
    # Therefore mean(x) = sum_t (1/n_total) * (w_t/W) * (n_total/n_t)
    # * sum_i d_i = sum_t (w_t/W) * mean(d_t), exactly the normalized gain.
    available_counts = {task: sum(pair[0] == task for pair in expected) for task in tasks}
    contribution_factor = {
        task: (weights[task] / total_weight) * n_total / available_counts[task]
        for task in tasks if available_counts[task]
    }
    contributions = {
        name: [
            contribution_factor[pair[0]] * (paired_scores[name][pair] - paired_scores[parent][pair])
            for pair in (evaluation_order[:evaluated_cases] if betting else sorted(expected))
        ]
        for name in metrics
        if name != parent
    }
    pooled_contributions = [value for values in contributions.values() for value in values]
    pooled_sd = _sample_sd(pooled_contributions)
    alpha_per_comparison = (
        statistics["alpha"] / statistics["max_comparisons"] if statistics else None
    )
    cohort_power = dict(
        n_cases=evaluated_cases if betting else n_total,
        pooled_sd=pooled_sd,
        mde_80_alpha=(
            pooled_sd
            * (_normal_upper_quantile(alpha_per_comparison) + _normal_quantile(0.8))
            / math.sqrt(evaluated_cases if betting else n_total)
            if pooled_sd is not None and alpha_per_comparison is not None
            else None
        ),
        alpha_per_comparison=alpha_per_comparison,
    )
    # Report whether this cohort can ever certify an improvement. Permutation
    # p-values cannot fall below 2 / (1 + 2**n) (exact) or 1 / (1 + resamples),
    # so a small cohort with a corrected alpha can be unable to accept any child.
    if statistics and statistics["method"] == "paired-permutation":
        cases = n_total
        floor = (2.0 / (1 + 2 ** cases)) if cases <= 12 else 1.0 / (1 + statistics.get("resamples", 2000))
        cohort_power["min_attainable_p"] = floor
        cohort_power["attainable"] = floor <= alpha_per_comparison
    for name, metric in metrics.items():
        if name == parent:
            continue
        regressed = [
            t
            for t in tasks
            if parent_metrics["per_task"][t] - metric["per_task"][t]
            > tasks[t]["maximum_regression"] + profile["tie_tolerance"]
        ]
        gain = metric["normalized_score"] - parent_metrics["normalized_score"]
        success_regressed = [t for t, spec in tasks.items()
            if (metric["success_rate"][t] + profile["tie_tolerance"] < spec.get("minimum_success_rate", 0)
                or parent_metrics["success_rate"][t] - metric["success_rate"][t]
                > spec.get("maximum_success_regression", 1) + profile["tie_tolerance"])]
        over_cost = [t + ":" + key for t, spec in tasks.items()
                     for key in sorted(spec.get("cost_limits", {}))
                     if metric["mean_cost"][t][key] > spec["cost_limits"][key]]
        statistical_accept = True
        statistical_p = None
        if statistics:
            differences = [paired_scores[name][pair] - paired_scores[parent][pair]
                           for pair in (evaluation_order[:evaluated_cases] if betting else sorted(expected))]
            if statistics["method"] == "paired-hoeffding":
                metric["improvement_evidence"] = dict(
                    method=statistics["method"], normalized_gain=gain, lower_bound=gain-radius,
                    alpha_per_comparison=alpha_per_comparison,
                    paired_wins=sum(d > 0 for d in differences), paired_losses=sum(d < 0 for d in differences),
                    paired_ties=sum(d == 0 for d in differences), independent_cases_required=True)
            elif statistics["method"] == "paired-betting":
                values = contributions[name]
                threshold = profile["minimum_gain"] + profile["tie_tolerance"]
                betting_state = betting_evidence(
                    values, threshold, alpha_per_comparison,
                    min_cases=statistics.get("min_cases", 8),
                    bet_clip=statistics.get("bet_clip", 0.5),
                )
                statistical_accept = betting_state["decision"] == "accept"
                metric["improvement_evidence"] = dict(
                    method="paired-betting", normalized_gain=gain,
                    alpha_per_comparison=alpha_per_comparison,
                    threshold=threshold, evaluated_cases=evaluated_cases,
                    cohort_cases=cohort_cases, **betting_state,
                )
            else:
                values = contributions[name]
                threshold = profile["minimum_gain"] + profile["tie_tolerance"]
                centered = [value - threshold for value in values]
                observed = math.fsum(centered) / len(centered)
                sample_sd = _sample_sd(values)
                if statistics["method"] == "paired-t":
                    if sample_sd is None or sample_sd == 0.0:
                        # A zero-variance cohort is accepted only for a strict
                        # positive gain with at least two paired cases.
                        standard_error = 0.0 if sample_sd == 0.0 and len(values) >= 2 else None
                        t_statistic = math.inf if observed > 0.0 and standard_error == 0.0 else (
                            -math.inf if observed < 0.0 and standard_error == 0.0 else None
                        )
                        statistical_p = 0.0 if observed > 0.0 and len(values) >= 2 else 1.0
                        statistical_accept = observed > 0.0 and len(values) >= 2
                    else:
                        standard_error = sample_sd / math.sqrt(len(values))
                        t_statistic = observed / standard_error
                        statistical_p = _student_t_survival(t_statistic, len(values) - 1)
                        statistical_accept = statistical_p <= alpha_per_comparison
                    metric["improvement_evidence"] = dict(
                        method="paired-t",
                        normalized_gain=gain,
                        p_value=statistical_p,
                        alpha_per_comparison=alpha_per_comparison,
                        n_cases=len(values),
                        paired_wins=sum(d > 0 for d in differences),
                        paired_losses=sum(d < 0 for d in differences),
                        paired_ties=sum(d == 0 for d in differences),
                        t_statistic=t_statistic,
                        standard_error=standard_error,
                    )
                else:
                    resamples = statistics.get("resamples", 2000)
                    exact = len(centered) <= 12
                    draws = 1 << len(centered) if exact else resamples
                    count = 0
                    if exact:
                        for mask in range(draws):
                            total = math.fsum(
                                value if (mask >> index) & 1 else -value
                                for index, value in enumerate(centered)
                            )
                            if total / len(centered) >= observed:
                                count += 1
                    else:
                        generator = random.Random(int(comparison_sha[:16], 16))
                        for _ in range(draws):
                            total = math.fsum(
                                value if generator.getrandbits(1) else -value
                                for value in centered
                            )
                            if total / len(centered) >= observed:
                                count += 1
                    statistical_p = (1 + count) / (1 + draws)
                    statistical_accept = statistical_p <= alpha_per_comparison
                    metric["improvement_evidence"] = dict(
                        method="paired-permutation",
                        normalized_gain=gain,
                        p_value=statistical_p,
                        alpha_per_comparison=alpha_per_comparison,
                        n_cases=len(values),
                        paired_wins=sum(d > 0 for d in differences),
                        paired_losses=sum(d < 0 for d in differences),
                        paired_ties=sum(d == 0 for d in differences),
                        resamples=resamples,
                        exact=exact,
                    )
        if regressed:
            rejection[name] = dict(reason="task_regression", tasks=regressed)
        elif success_regressed:
            rejection[name] = dict(reason="success_constraint", tasks=success_regressed)
        elif over_cost:
            rejection[name] = dict(reason="cost_constraint", measurements=over_cost)
        elif statistics and statistics["method"] == "paired-betting" and not statistical_accept:
            evidence = metric["improvement_evidence"]
            rejection[name] = dict(reason="insufficient_evidence", normalized_gain=gain,
                                    betting_state=evidence)
        elif gain <= profile["minimum_gain"] + profile["tie_tolerance"]:
            rejection[name] = dict(
                reason="no_sufficient_improvement", normalized_gain=gain
            )
        elif statistics and statistics["method"] == "paired-hoeffding" and gain - radius <= profile["minimum_gain"] + profile["tie_tolerance"]:
            rejection[name] = dict(reason="insufficient_evidence", lower_bound=gain-radius, normalized_gain=gain)
        elif statistics and statistics["method"] in {"paired-t", "paired-permutation"} and not statistical_accept:
            rejection[name] = dict(reason="insufficient_evidence", normalized_gain=gain,
                                    p_value=statistical_p)
        else:
            eligible.append(name)
    winner = parent
    if eligible:
        best = max(metrics[n]["normalized_score"] for n in eligible)
        winner = min(
            n
            for n in eligible
            if best - metrics[n]["normalized_score"] <= profile["tie_tolerance"]
        )
    return dict(
        schema_version=1,
        comparison_sha256=comparison_sha,
        cohort_sha256=cohort_sha,
        parent_id=parent,
        survivor_id=winner,
        survivor_freeze_sha256=pool[winner],
        decision="retain_parent" if winner == parent else "inherit_child",
        metrics=metrics,
        rejected=rejection,
        result_sha256={r["candidate_id"]: digest(r) for r in results},
        cohort_power=cohort_power,
        evaluated_cases=evaluated_cases,
        cohort_cases=cohort_cases,
        scope=comparison.get(
            "scope",
            "paired evidence integrity only; no automatic physical qualification",
        ),
    )


def write_selection(path, selection):
    """Persist one decision without overwriting an earlier survivor record."""
    path = Path(path)
    with locked(path.with_suffix(".lock")):
        if path.exists():
            raise FileExistsError("Selection already exists")
        atomic_json(path, selection)
