import pytest

from PhysicalRSI_core.infra.storage import digest, file_digest
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.selection import select_survivor, validate_profile


def comparison(tmp_path, *, episodes=40, parent_score=0, child_score=1, child_success=True,
               statistics=True, constraints=None):
    source = tmp_path / "native.json"
    source.write_text("{}")
    profile = dict(tasks={"task": dict(weight=1, episodes=episodes, score_range=[0, 1],
                       maximum_regression=0, **(constraints or {}))}, minimum_gain=0, tie_tolerance=1e-10)
    if statistics:
        profile["statistics"] = dict(method="paired-hoeffding", alpha=.05, max_comparisons=10)
    frozen = dict(schema_version=1, round_id="comparison", frozen_at=now(), parent_id="parent",
                  candidates={"parent": "1" * 64, "child": "2" * 64}, profile=profile)
    cohort = dict(comparison_sha256=digest(frozen), generated_at=now(),
                  layouts={"task": [digest(index) for index in range(episodes)]})
    results = []
    for name, score, success in (("parent", parent_score, True), ("child", child_score, child_success)):
        rows = [dict(task="task", layout_sha256=case, state="completed", score=score, success=success,
                     measurements=dict(calls=10 if name == "parent" else 20),
                     evidence_sha256={"native.json": file_digest(source)}) for case in cohort["layouts"]["task"]]
        results.append(dict(candidate_id=name, kind="policy_evaluation", native_exit_code=0,
                            comparison_sha256=digest(frozen), cohort_sha256=digest(cohort),
                            freeze_sha256=frozen["candidates"][name], evaluator_revision="frozen", episodes=rows))
    return frozen, cohort, results


@pytest.mark.parametrize("episodes,winner,reason", [(2, "parent", "insufficient_evidence"), (40, "child", None)])
def test_bounded_paired_confidence_distinguishes_small_and_large_samples(tmp_path, episodes, winner, reason):
    frozen, cohort, results = comparison(tmp_path, episodes=episodes)
    results[1]["episodes"].reverse()
    value = select_survivor(frozen, cohort, results, evidence_root=tmp_path)
    assert value["survivor_id"] == winner
    assert value["rejected"].get("child", {}).get("reason") == reason
    evidence = value["metrics"]["child"]["improvement_evidence"]
    assert evidence["paired_wins"] == episodes and evidence["paired_losses"] == 0
    assert evidence["alpha_per_comparison"] == .005


def test_continuous_score_cannot_override_declared_success_constraint(tmp_path):
    frozen, cohort, results = comparison(tmp_path, statistics=False, parent_score=.5, child_score=.8,
                                        child_success=False, constraints=dict(maximum_success_regression=0))
    value = select_survivor(frozen, cohort, results, evidence_root=tmp_path)
    assert value["survivor_id"] == "parent" and value["rejected"]["child"]["reason"] == "success_constraint"


def test_measured_cost_limit_is_enforced_and_missing_cost_is_invalid(tmp_path):
    frozen, cohort, results = comparison(tmp_path, constraints=dict(cost_limits={"calls": 15}))
    value = select_survivor(frozen, cohort, results, evidence_root=tmp_path)
    assert value["survivor_id"] == "parent" and value["rejected"]["child"]["reason"] == "cost_constraint"
    del results[1]["episodes"][0]["measurements"]["calls"]
    with pytest.raises(ValueError, match="finite numeric"):
        select_survivor(frozen, cohort, results, evidence_root=tmp_path)


@pytest.mark.parametrize("field,value", [("alpha", 0), ("alpha", True), ("method", "unknown"),
                                       ("max_comparisons", 0), ("max_comparisons", True)])
def test_invalid_statistics_fail_at_profile_preflight(tmp_path, field, value):
    profile = comparison(tmp_path)[0]["profile"]
    profile["statistics"][field] = value
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_multiple_comparison_correction_changes_evidence_threshold(tmp_path):
    frozen, cohort, results = comparison(tmp_path, episodes=40, child_score=.5)
    value = select_survivor(frozen, cohort, results, evidence_root=tmp_path)
    assert value["survivor_id"] == "parent"
    frozen["profile"]["statistics"]["max_comparisons"] = 1
    cohort["comparison_sha256"] = digest(frozen)
    for result in results:
        result.update(comparison_sha256=digest(frozen), cohort_sha256=digest(cohort))
    value = select_survivor(frozen, cohort, results, evidence_root=tmp_path)
    assert value["survivor_id"] == "child"


def test_relative_weight_overflow_preserves_scores_and_confidence(tmp_path):
    from copy import deepcopy

    frozen, cohort, results = comparison(tmp_path)
    frozen['profile']['tasks']['second'] = deepcopy(frozen['profile']['tasks']['task'])
    cohort['layouts']['second'] = [digest(['second', case]) for case in cohort['layouts']['task']]
    for result in results:
        result['episodes'] += [dict(row, task='second', layout_sha256=digest(['second', row['layout_sha256']]))
                               for row in result['episodes']]
    decisions = []
    for weight in (1., 1e308):
        for spec in frozen['profile']['tasks'].values():
            spec['weight'] = weight
        cohort['comparison_sha256'] = digest(frozen)
        for result in results:
            result.update(comparison_sha256=digest(frozen), cohort_sha256=digest(cohort))
        value = select_survivor(frozen, cohort, results, evidence_root=tmp_path)
        assert value['survivor_id'] == 'child'
        decisions.append(value)
    assert decisions[0]['metrics'] == decisions[1]['metrics']
    assert decisions[0]['rejected'] == decisions[1]['rejected']


@pytest.mark.parametrize('alpha', [1e-308, 1e-320])
def test_strict_confidence_decision_stays_finite_and_persistable(tmp_path, alpha):
    from decimal import Decimal, localcontext
    import math
    from PhysicalRSI_core.infra.storage import read_json
    from PhysicalRSI_core.self_harness.selection import write_selection

    frozen, cohort, results = comparison(tmp_path)
    frozen['profile']['statistics']['alpha'] = alpha
    cohort['comparison_sha256'] = digest(frozen)
    for result in results:
        result.update(comparison_sha256=digest(frozen), cohort_sha256=digest(cohort))
    value = select_survivor(frozen, cohort, results, evidence_root=tmp_path)
    evidence = value['metrics']['child']['improvement_evidence']
    assert value['decision'] == 'retain_parent'
    assert value['rejected']['child']['reason'] == 'insufficient_evidence'
    assert math.isfinite(evidence['lower_bound'])
    with localcontext() as context:
        context.prec = 50
        radius = (2 * (Decimal(10) / Decimal.from_float(alpha)).ln() / 40).sqrt()
    assert evidence['lower_bound'] == pytest.approx(1 - float(radius))
    assert evidence['alpha_per_comparison'] > 0
    path = tmp_path / 'selection.json'
    write_selection(path, value)
    assert read_json(path) == value
    with pytest.raises(FileExistsError):
        write_selection(path, value)
    assert read_json(path) == value
