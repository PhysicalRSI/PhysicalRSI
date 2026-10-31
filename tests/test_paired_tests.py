import random

import pytest

from PhysicalRSI_core.infra.storage import digest, file_digest
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.selection import (
    _normal_quantile,
    _regularized_beta,
    _student_t_survival,
    select_survivor,
    validate_profile,
)


def _comparison(tmp_path, parent, child, *, weights=None, statistics=None):
    source = tmp_path / "evidence.json"
    source.write_text("{}")
    weights = weights or {"task": 1.0}
    tasks = {
        task: dict(weight=weight, episodes=len(parent[task]), score_range=[0, 1], maximum_regression=0)
        for task, weight in weights.items()
    }
    profile = dict(tasks=tasks, minimum_gain=0, tie_tolerance=0)
    if statistics is not None:
        profile["statistics"] = statistics
    comparison = dict(
        schema_version=1,
        round_id="paired-tests",
        frozen_at=now(),
        parent_id="parent",
        candidates={"parent": "1" * 64, "child": "2" * 64},
        profile=profile,
    )
    cohort = dict(
        comparison_sha256=digest(comparison),
        generated_at=now(),
        layouts={task: [digest([task, index]) for index in range(len(values))] for task, values in parent.items()},
    )
    results = []
    for name, values in (("parent", parent), ("child", child)):
        rows = []
        for task, scores in values.items():
            for index, score in enumerate(scores):
                rows.append(dict(
                    task=task,
                    layout_sha256=cohort["layouts"][task][index],
                    state="completed",
                    score=score,
                    success=True,
                    measurements={},
                    evidence_sha256={"evidence.json": file_digest(source)},
                ))
        results.append(dict(
            candidate_id=name,
            kind="policy_evaluation",
            native_exit_code=0,
            comparison_sha256=digest(comparison),
            cohort_sha256=digest(cohort),
            freeze_sha256=comparison["candidates"][name],
            evaluator_revision="paired-tests",
            episodes=rows,
        ))
    return comparison, cohort, results


def test_incomplete_beta_and_t_survival_known_value():
    assert _regularized_beta(5, 0.5, 10 / 14) == pytest.approx(0.07338803477, rel=1e-8)
    assert _student_t_survival(2.0, 10) == pytest.approx(0.0367, abs=1e-4)


def test_normal_quantile_accuracy():
    assert _normal_quantile(0.975) == pytest.approx(1.9599639845, abs=1e-6)
    assert _normal_quantile(0.001) == pytest.approx(-3.090232306, abs=1e-6)


def test_weighted_contribution_mean_identity(tmp_path):
    parent = {"a": [0.2, 0.3], "b": [0.4], "c": [0.5, 0.6, 0.7]}
    child = {"a": [0.4, 0.2], "b": [0.8], "c": [0.7, 0.8, 0.6]}
    comparison, cohort, results = _comparison(tmp_path, parent, child, weights={"a": 1, "b": 2, "c": 3})
    selected = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    total = sum(map(len, parent.values()))
    largest = 3
    total_weight = 1 / 3 + 2 / 3 + 1
    x = []
    for task in parent:
        factor = (comparison["profile"]["tasks"][task]["weight"] / largest) / total_weight
        factor *= total / len(parent[task])
        x.extend(factor * (c - p) for p, c in zip(parent[task], child[task]))
    gain = selected["metrics"]["child"]["normalized_score"] - selected["metrics"]["parent"]["normalized_score"]
    assert sum(x) / len(x) == pytest.approx(gain)


def test_paired_t_accepts_clear_gain_and_rejects_equal_child(tmp_path):
    stats = {"method": "paired-t", "alpha": 0.05, "max_comparisons": 1}
    parent = {"task": [0.5] * 20}
    child = {"task": [0.7 + (index % 3) * 0.02 for index in range(20)]}
    comparison, cohort, results = _comparison(tmp_path, parent, child, statistics=stats)
    selected = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    assert selected["survivor_id"] == "child"
    evidence = selected["metrics"]["child"]["improvement_evidence"]
    assert evidence["method"] == "paired-t" and evidence["p_value"] <= 0.05

    equal = {"task": [0.5] * 20}
    comparison, cohort, results = _comparison(tmp_path, parent, equal, statistics=stats)
    selected = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    assert selected["survivor_id"] == "parent"
    assert selected["metrics"]["child"]["improvement_evidence"]["p_value"] == 1.0


def test_paired_permutation_exact_matches_monte_carlo(tmp_path):
    stats = {"method": "paired-permutation", "alpha": 0.05, "max_comparisons": 1, "resamples": 5000}
    parent = {"task": [0.5] * 10}
    child = {"task": [0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4]}
    comparison, cohort, results = _comparison(tmp_path, parent, child, statistics=stats)
    selected = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    exact = selected["metrics"]["child"]["improvement_evidence"]["p_value"]
    values = [score - 0.5 for score in child["task"]]
    observed = sum(values) / len(values)
    generator = random.Random(int(selected["comparison_sha256"][:16], 16))
    count = 0
    for _ in range(5000):
        if sum(value if generator.getrandbits(1) else -value for value in values) / len(values) >= observed:
            count += 1
    monte_carlo = (1 + count) / 5001
    assert abs(exact - monte_carlo) <= 0.02


def test_permutation_null_false_acceptance_is_controlled():
    rng = random.Random(1234)
    accepted = 0
    for _ in range(2000):
        values = [rng.choice((-1.0, 1.0)) for _ in range(13)]
        observed = sum(values) / len(values)
        flips = random.Random(rng.randrange(1 << 30))
        count = sum(
            sum(value if flips.getrandbits(1) else -value for value in values) / len(values) >= observed
            for _ in range(200)
        )
        if (1 + count) / 201 <= 0.05:
            accepted += 1
    assert accepted / 2000 <= 0.08


def test_cohort_power_present_and_mde_decreases_with_n(tmp_path):
    stats = {"method": "paired-t", "alpha": 0.05, "max_comparisons": 1}
    parent = {"task": [0.5 + (index % 2) * 0.1 for index in range(8)]}
    child = {"task": [value + 0.1 + (index % 3) * 0.01 for index, value in enumerate(parent["task"])]}
    comparison, cohort, results = _comparison(tmp_path, parent, child, statistics=stats)
    small = select_survivor(comparison, cohort, results, evidence_root=tmp_path)["cohort_power"]
    parent = {"task": [0.5 + (index % 2) * 0.1 for index in range(32)]}
    child = {"task": [value + 0.1 + (index % 3) * 0.01 for index, value in enumerate(parent["task"])]}
    comparison, cohort, results = _comparison(tmp_path, parent, child, statistics=stats)
    large = select_survivor(comparison, cohort, results, evidence_root=tmp_path)["cohort_power"]
    assert small["n_cases"] == 8 and large["n_cases"] == 32
    assert small["mde_80_alpha"] > large["mde_80_alpha"]


@pytest.mark.parametrize("method", ["paired-hoeffding", "paired-t", "paired-permutation"])
def test_profile_accepts_all_fixed_n_methods(method):
    rule = {"method": method, "alpha": 0.05, "max_comparisons": 2}
    if method == "paired-permutation":
        rule["resamples"] = 200
    validate_profile(dict(
        tasks={"task": dict(weight=1, episodes=2, score_range=[0, 1], maximum_regression=0)},
        minimum_gain=0,
        tie_tolerance=0,
        statistics=rule,
    ))
