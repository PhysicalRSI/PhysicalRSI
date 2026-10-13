"""Freeze comparison conditions and reject incomplete or incomparable results.

This module does not run a simulator or certify policy observation compliance.
Native evaluator outputs remain the authority for individual episode scores.
"""
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol


def verified_native_row(folder, *, expected_specification, comparison_sha256, task, instance_id):
    """Read one native trial against its independently frozen launch specification.

    Callers must bind that specification to the comparison before execution.
    This checks receipt attribution and integrity, not policy admission or assets.
    Local zero-action smoke and shortened episodes cannot enter official means.
    """
    from .doctor import BEHAVIOR_REVISION
    from .native import parse_result, read_receipt

    folder = Path(folder)
    receipt = read_receipt(folder)
    specification = read_json(folder / "specification.json")
    if specification != expected_specification:
        raise ValueError("Trial differs from its preregistered launch specification")
    if (specification.get("purpose") != "policy_evaluation"
            or specification.get("source_revision") != BEHAVIOR_REVISION
            or not specification.get("policy_revision")
            or "--max-steps" in specification.get("argv", [])):
        raise ValueError("Only full-budget pinned native policy evaluations can be scored")
    if not {"specification.json", "policy_service.json"}.issubset(receipt["evidence"]):
        raise ValueError("Trial must preserve its launch and policy-service evidence")
    metadata = read_json(folder / "policy_service.json")
    if (metadata.get("schema") != "physicalrsi.behavior-policy-service/v1"
            or metadata.get("candidate_sha256") != specification["policy_revision"]
            or metadata.get("task") != task
            or metadata.get("policy_identity_sha256") != digest(metadata.get("policy_identity"))):
        raise ValueError("Trial policy service differs from the declared candidate/task")
    result = parse_result(folder / receipt["native_result_file"], task=task, instance_id=instance_id)
    row = dict(task=task, instance_id=instance_id, rollout_id=result["rollout_id"],
               comparison_sha256=comparison_sha256, outcome=receipt["outcome"],
               q_score=result["q_score"]["final"], receipt_sha256=file_digest(folder / "receipt.json"))
    # Reject a concurrently replaced receipt rather than mixing generations.
    if read_json(folder / "receipt.json") != receipt:
        raise ValueError("Native receipt changed during aggregation")
    return row


def freeze_protocol(root, *, conditions, splits):
    """Bind core case splits to the same evaluator, assets and baseline contracts."""
    required = {"benchmark_revision", "asset_revision", "robot", "observation_contract",
                "action_contract", "episode_budget", "baselines", "metric"}
    if set(conditions) != required or any(not value for value in conditions.values()):
        raise ValueError("Declare all comparison conditions")
    if conditions["metric"] != "macro_mean_final_q_score":
        raise ValueError("Primary metric must be native final Q-score, averaged equally across tasks")
    baselines = conditions["baselines"]
    if (not isinstance(baselines, dict) or not baselines
            or any(not name or not isinstance(revision, str) or not revision for name, revision in baselines.items())):
        raise ValueError("Pin each baseline implementation/checkpoint identity")
    # Validate without persisting before acquiring the comparison lock.
    digest(conditions)
    root = Path(root)
    with locked(root / ".comparison.lock"):
        path = root / "comparison.json"
        binding = {"schema": "physicalrsi.behavior-comparison/v1", "conditions": deepcopy(conditions),
                   "splits_sha256": digest(splits), "qualification": None}
        if path.exists() and read_json(path) != binding:
            raise ValueError("Frozen comparison changed; use a new research workspace")
        cases = CaseProtocol(root / "cases", splits=splits, identity_keys=("instance_id",))
        if not path.exists():
            atomic_json(path, binding)
    return cases, digest(binding)


def summarize(rows, *, expected_cases, comparison_sha256):
    """Macro mean over a complete paired cohort, never a successful-only subset.

    Infrastructure errors and missing trials must be reconciled, not silently
    dropped or converted into policy failures. Multiple rollouts must have
    distinct rollout_id values in expected_cases and rows.
    """
    expected = [(c["task"], c["instance_id"], c["rollout_id"]) for c in expected_cases]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected cases must be nonempty and unique")
    expected = set(expected)
    seen, scores, successes = set(), defaultdict(list), defaultdict(list)
    evidence = []
    for row in rows:
        key = (row["task"], row["instance_id"], row["rollout_id"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate trial")
        if row["comparison_sha256"] != comparison_sha256:
            raise ValueError("Incomparable evaluation conditions")
        if row["outcome"] not in {"success", "failure"}:
            raise ValueError("Unresolved trial; no benchmark mean is available")
        score = row["q_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Native Q-score must be finite and in [0, 1]")
        receipt = row["receipt_sha256"]
        if not isinstance(receipt, str) or len(receipt) != 64 or any(c not in "0123456789abcdef" for c in receipt):
            raise ValueError("Each result must reference its verified trial receipt")
        seen.add(key)
        scores[key[0]].append(score)
        successes[key[0]].append(row["outcome"] == "success")
        evidence.append(receipt)
    if seen != expected:
        raise ValueError("Incomplete cohort; missing trials cannot be excluded from the mean")
    per_task = {task: {"mean_q_score": sum(values) / len(values),
                       "success_rate": sum(successes[task]) / len(values), "episodes": len(values)}
                for task, values in sorted(scores.items())}
    return {"comparison_sha256": comparison_sha256, "per_task": per_task,
            "macro_mean_final_q_score": sum(v["mean_q_score"] for v in per_task.values()) / len(per_task),
            "macro_success_rate": sum(v["success_rate"] for v in per_task.values()) / len(per_task),
            "receipt_sha256s": sorted(evidence), "qualification": None}
