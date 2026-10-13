from copy import deepcopy

import pytest

from PhysicalRSI_baselines.behavior_baseline.protocol import freeze_protocol, summarize


def test_comparison_freezes_dependencies_and_rejects_reset_leakage(tmp_path):
    conditions = dict(benchmark_revision="upstream-sha", asset_revision="assets-sha", robot="R1Pro",
                      observation_contract="rgb-depth-proprio", action_contract="joint-controls",
                      episode_budget={"steps": "official-human-normalized"}, baselines={"pi05": "checkpoint-sha"},
                      metric="macro_mean_final_q_score")
    splits = {split: {"task": [{"instance_id": i}]} for i, split in enumerate(("development", "validation", "test"))}
    cases, sha = freeze_protocol(tmp_path, conditions=conditions, splits=splits)
    assert cases.cases("development") == splits["development"]
    assert freeze_protocol(tmp_path, conditions=conditions, splits=splits)[1] == sha
    with pytest.raises(ValueError, match="final-candidate"):
        cases.cases("test")
    changed = dict(conditions, robot="other-robot")
    with pytest.raises(ValueError, match="Frozen comparison"):
        freeze_protocol(tmp_path, conditions=changed, splits=splits)
    leaked = deepcopy(splits)
    leaked["test"]["task"][0]["instance_id"] = 0
    with pytest.raises(ValueError, match="Reset identity reused"):
        freeze_protocol(tmp_path / "leaked", conditions=conditions, splits=leaked)


def cohort():
    return [dict(task=task, instance_id=i, rollout_id=0, comparison_sha256="comparison",
                 outcome="success" if score == 1 else "failure", q_score=score, receipt_sha256="a" * 64)
            for task, i, score in [("a", 0, 1), ("a", 1, 0), ("b", 0, 0)]]


def test_mean_weights_tasks_equally_and_includes_policy_failures():
    rows = cohort()
    result = summarize(rows, expected_cases=rows, comparison_sha256="comparison")
    assert result["macro_mean_final_q_score"] == .25
    assert result["per_task"]["a"]["episodes"] == 2


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "conditions", "invalid", "nan", "out_of_range"])
def test_bad_or_incomplete_evidence_cannot_produce_mean(mutation):
    rows, expected = cohort(), cohort()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "conditions":
        rows[0]["comparison_sha256"] = "different"
    elif mutation == "invalid":
        rows[0]["outcome"] = "invalid"
    elif mutation == "nan":
        rows[0]["q_score"] = float("nan")
    else:
        rows[0]["q_score"] = 1.1
    with pytest.raises(ValueError):
        summarize(rows, expected_cases=expected, comparison_sha256="comparison")


def test_aggregation_reads_bound_native_evidence_and_rejects_other_policy(tmp_path):
    from PhysicalRSI_baselines.behavior_baseline.doctor import BEHAVIOR_REVISION
    from PhysicalRSI_baselines.behavior_baseline.protocol import verified_native_row
    from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest

    # Synthetic files exercise attribution; they are not simulator evidence.
    spec = dict(purpose="policy_evaluation", source_revision=BEHAVIOR_REVISION,
                policy_revision="a" * 64, argv=["fixture"])
    result = dict(task="radio", instance_id=1, rollout_id=0, success=False, q_score={"final": .25})
    metadata = dict(schema="physicalrsi.behavior-policy-service/v1", task="radio",
                    candidate_sha256="a" * 64, policy_identity={"fixture": True},
                    policy_identity_sha256=digest({"fixture": True}))
    for name, data in [("specification.json", spec), ("result.json", result),
                       ("policy_service.json", metadata)]:
        atomic_json(tmp_path / name, data)
    receipt = dict(state="completed", specification_sha256=digest(spec), outcome="failure",
                   native_result=result, native_result_file="result.json", evidence={
                       name: file_digest(tmp_path / name) for name in
                       ("specification.json", "result.json", "policy_service.json")})
    atomic_json(tmp_path / "receipt.json", receipt)
    kwargs = dict(expected_specification=spec, comparison_sha256="comparison", task="radio", instance_id=1)
    row = verified_native_row(tmp_path, **kwargs)
    assert row["q_score"] == .25 and row["outcome"] == "failure"
    with pytest.raises(ValueError, match="preregistered"):
        verified_native_row(tmp_path, **dict(kwargs, expected_specification=dict(spec, policy_revision="b" * 64)))
    with pytest.raises(ValueError, match="another episode"):
        verified_native_row(tmp_path, **dict(kwargs, instance_id=2))
    atomic_json(tmp_path / "result.json", dict(result, q_score={"final": 1.}))
    with pytest.raises(ValueError, match="evidence changed"):
        verified_native_row(tmp_path, **kwargs)
