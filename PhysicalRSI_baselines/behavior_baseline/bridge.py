"""Convert complete native validation evidence into core Self-Harness results.

Admission, fresh split registration, trial reservations and process orchestration
remain caller responsibilities. This bridge never invents scores or runs trials.
"""
from pathlib import Path

from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import digest, file_digest, read_json
from PhysicalRSI_core.self_harness.selection import comparison_identity
from .native import read_receipt
from .protocol import verified_native_row


def selection_result(trials, *, candidate_id, comparison, cohort, evaluator_revision, evidence_root):
    """Each trial supplies folder and an independently preregistered specification."""
    comparison_sha = comparison_identity(comparison)
    if (cohort.get("comparison_sha256") != comparison_sha or cohort.get("split") != "validation"
            or candidate_id not in comparison["candidates"] or not evaluator_revision):
        raise ValueError("Declare a matching validation cohort and candidate")
    tasks = comparison["profile"]["tasks"]
    if len({spec["weight"] for spec in tasks.values()}) != 1 or any(
        list(spec["score_range"]) != [0, 1] for spec in tasks.values()
    ):
        raise ValueError("BEHAVIOR selection requires equal task weights and native Q-score range")
    if set(cohort["layouts"]) != set(tasks) or any(
        len(cohort["layouts"][task]) != spec["episodes"] for task, spec in tasks.items()
    ):
        raise ValueError("Validation cohort does not cover the declared task profile")
    expected = {(task, sha) for task, values in cohort["layouts"].items() for sha in values}
    if len({sha for _, sha in expected}) != sum(spec["episodes"] for spec in tasks.values()):
        raise ValueError("Case identities must include task identity and be unique")
    root = Path(evidence_root).resolve()
    seen, rows, protocols, system2_revisions = set(), [], [], set()
    for trial in trials:
        folder = Path(trial["folder"]).resolve()
        spec = trial["specification"]
        if not folder.is_relative_to(root):
            raise ValueError("Native evidence must be inside the comparison evidence root")
        origin = spec.get("system2") or {}
        binding = dict(candidate_id=candidate_id, candidate_sha256=comparison["candidates"][candidate_id],
                       comparison_sha256=comparison_sha, cohort_sha256=digest(cohort), split="validation")
        if any(origin.get(k) != v for k, v in binding.items()) or not origin.get("system2"):
            raise ValueError("Native trial attribution differs from the frozen comparison")
        System2Trial(**dict(origin, system2=Revision(**origin["system2"])))
        system2_revisions.add(digest(origin["system2"]))
        if spec.get("policy_revision") != binding["candidate_sha256"]:
            raise ValueError("Served policy revision differs from the admitted candidate")
        case = spec["case"]
        pair = (case["task"], digest(case))
        if pair not in expected or pair in seen or case["mode"] != "train" or case["rollout_id"] != 0:
            raise ValueError("Unexpected, repeated or non-validation native case")
        row = verified_native_row(folder, expected_specification=spec, comparison_sha256=comparison_sha,
                                  task=case["task"], instance_id=case["instance_id"])
        receipt = read_receipt(folder)
        evidence = {str((folder / name).relative_to(root)): sha for name, sha in receipt["evidence"].items()}
        receipt_path = folder / "receipt.json"
        if file_digest(receipt_path) != row["receipt_sha256"]:
            raise ValueError("Native receipt changed while binding selection evidence")
        evidence[str(receipt_path.relative_to(root))] = row["receipt_sha256"]
        rows.append(dict(task=pair[0], layout_sha256=pair[1], state="completed", score=row["q_score"],
                         success=row["outcome"] == "success", evidence_sha256=evidence,
                         measurements={"final_q_score": row["q_score"]}))
        # Endpoint and output location may differ; native execution flags may not.
        argv = list(spec["argv"])
        for flag in ("--output-dir", "--host", "--port"):
            if flag in argv:
                argv[argv.index(flag) + 1] = "<transport>"
        protocols.append(dict(case=case, argv=argv, seconds=spec["seconds"],
                              source_revision=spec["source_revision"],
                              implementation_sha256=spec["implementation_sha256"],
                              graphics_manifests=spec.get("graphics_manifests", {}),
                              temporary_directory=spec.get("temporary_directory"),
                              system2=origin["system2"], metric="native_final_q_score"))
        seen.add(pair)
    if seen != expected:
        raise ValueError("Incomplete native validation cohort")
    if len(system2_revisions) != 1:
        raise ValueError("Mixed System 2 revisions within a validation cohort")
    return dict(candidate_id=candidate_id, kind=comparison.get("evaluation_kind", "policy_evaluation"),
                native_exit_code=0, comparison_sha256=comparison_sha, cohort_sha256=digest(cohort),
                freeze_sha256=comparison["candidates"][candidate_id], evaluator_revision=evaluator_revision,
                experiment_protocol_sha256=digest(sorted(protocols, key=lambda p: digest(p["case"]))),
                episodes=sorted(rows, key=lambda r: (r["task"], r["layout_sha256"])))
