from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from PhysicalRSI_baselines.behavior_baseline.bridge import selection_result
from PhysicalRSI_baselines.behavior_baseline.doctor import BEHAVIOR_REVISION
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest
from PhysicalRSI_core.self_harness.selection import select_survivor


def fixture(tmp_path):
    # Synthetic receipts test core selection wiring, never physical performance.
    now = datetime.now(timezone.utc)
    comparison = dict(schema_version=1, round_id="fixture", frozen_at=(now-timedelta(seconds=2)).isoformat(),
                      parent_id="parent", candidates={"parent": "a"*64, "child": "b"*64},
                      profile=dict(tasks={t: dict(weight=1, episodes=1, score_range=[0, 1], maximum_regression=0)
                                          for t in ("task_a", "task_b")}, minimum_gain=.01, tie_tolerance=0))
    cases = [dict(task=t, mode="train", instance_index=200, instance_id=200, rollout_id=0)
             for t in comparison["profile"]["tasks"]]
    cohort = dict(comparison_sha256=digest(comparison), split="validation",
                  generated_at=(now-timedelta(seconds=1)).isoformat(),
                  layouts={c["task"]: [digest(c)] for c in cases})
    plans = {}
    for candidate, revision in comparison["candidates"].items():
        plans[candidate] = []
        for case in cases:
            folder = tmp_path / candidate / case["task"]
            origin = dict(system2=dict(name="fixture", revision="1"), candidate_id=candidate,
                          candidate_sha256=revision, split="validation",
                          comparison_sha256=digest(comparison), cohort_sha256=digest(cohort))
            spec = dict(purpose="policy_evaluation", source_revision=BEHAVIOR_REVISION,
                        implementation_sha256="c"*64, policy_revision=revision, system2=origin,
                        case=case, seconds=3600, argv=["fixture", "--output-dir", str(folder)])
            result = dict(task=case["task"], instance_id=200, rollout_id=0, success=False,
                          q_score={"final": .25 if candidate == "parent" else .5})
            metadata = dict(schema="physicalrsi.behavior-policy-service/v1", task=case["task"],
                            candidate_sha256=revision, policy_identity={"fixture": candidate},
                            policy_identity_sha256=digest({"fixture": candidate}))
            for name, value in [("specification.json", spec), ("result.json", result),
                                ("policy_service.json", metadata)]:
                atomic_json(folder / name, value)
            receipt = dict(state="completed", specification_sha256=digest(spec), outcome="failure",
                           native_result=result, native_result_file="result.json", evidence={
                               n: file_digest(folder/n) for n in ("specification.json", "result.json", "policy_service.json")})
            atomic_json(folder / "receipt.json", receipt)
            plans[candidate].append(dict(folder=folder, specification=spec))
    return comparison, cohort, plans


def test_native_evidence_enters_existing_core_paired_selection(tmp_path):
    comparison, cohort, plans = fixture(tmp_path)
    kwargs = dict(comparison=comparison, cohort=cohort, evaluator_revision="fixture", evidence_root=tmp_path)
    results = [selection_result(plan, candidate_id=name, **kwargs) for name, plan in plans.items()]
    result = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    assert result["survivor_id"] == "child"
    assert result["decision"] == "inherit_child"
    # Core selection rechecks actual files, not just the row's score.
    atomic_json(plans["child"][0]["folder"] / "result.json", {"tampered": True})
    with pytest.raises(ValueError, match="evidence"):
        select_survivor(comparison, cohort, results, evidence_root=tmp_path)


def test_bridge_rejects_missing_cases_and_false_attribution(tmp_path):
    comparison, cohort, plans = fixture(tmp_path)
    kwargs = dict(candidate_id="parent", comparison=comparison, cohort=cohort,
                  evaluator_revision="fixture", evidence_root=tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        selection_result(plans["parent"][:1], **kwargs)
    wrong = deepcopy(plans["parent"])
    wrong[0]["specification"]["system2"]["candidate_id"] = "child"
    with pytest.raises(ValueError, match="attribution"):
        selection_result(wrong, **kwargs)
