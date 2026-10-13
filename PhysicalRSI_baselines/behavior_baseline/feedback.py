"""Export development scores and onboard video for System 2 proposal input.

Raw native logs and metric dictionaries remain evaluator evidence. This is an
explicit data projection, not a filesystem sandbox for arbitrary proposal code.
"""
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile

from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from .native import read_receipt
from .protocol import verified_native_row
from .diagnostics import verify_trace


def export_development(trials, *, parent_id, parent_revision, expected_cases,
                       evaluator_revision, output):
    """Export the complete preregistered incumbent development cohort only.

    Each trial supplies its folder and independently frozen specification. The
    caller owns case admission and experiment reservations. Only explicit score
    fields and the pinned native evaluator's head/wrist RGB video are exposed.
    """
    if not evaluator_revision or not expected_cases:
        raise ValueError("Declare development cases and evaluator identity")
    fields = {"task", "mode", "instance_index", "instance_id", "rollout_id"}
    if any(set(c) != fields or c["mode"] != "train" or c["rollout_id"] != 0 for c in expected_cases):
        raise ValueError("Use canonical development reset identities")
    expected = {digest(c) for c in expected_cases}
    if len(expected) != len(expected_cases):
        raise ValueError("Repeated development reset identity")
    output = Path(output).resolve()
    bundle = output / "development-feedback"
    rows, videos, seen, origins = [], [], set(), set()
    for trial in trials:
        folder = Path(trial["folder"]).resolve()
        spec = trial["specification"]
        origin = spec.get("system2") or {}
        binding = dict(parent_id=origin.get("candidate_id"), parent_revision=origin.get("candidate_sha256"))
        if (binding != dict(parent_id=parent_id, parent_revision=parent_revision)
                or origin.get("split") != "development"
                or spec.get("policy_revision") != parent_revision):
            raise ValueError("Only the declared incumbent's development evidence can guide proposals")
        attribution = System2Trial(**dict(origin, system2=Revision(**origin["system2"])))
        origins.add(digest(asdict(attribution.system2)))
        case = spec["case"]
        case_sha = digest(case)
        if case_sha not in expected or case_sha in seen:
            raise ValueError("Unexpected or repeated development episode")
        row = verified_native_row(folder, expected_specification=spec, comparison_sha256=None,
                                  task=case["task"], instance_id=case["instance_id"])
        receipt = read_receipt(folder)
        if file_digest(folder / "receipt.json") != row["receipt_sha256"]:
            raise ValueError("Development receipt changed during export")
        video_name = f"native/videos/{case['task']}_{case['instance_id']}_0.mp4"
        if "--write-video" not in spec["argv"] or video_name not in receipt["evidence"]:
            raise ValueError("Development feedback requires bound native onboard video")
        destination = "development-feedback/videos/" + case_sha + ".mp4"
        videos.append((folder / video_name, output / destination, receipt["evidence"][video_name]))
        episode = dict(task=case["task"], case=case, case_sha256=case_sha,
                         outcome=row["outcome"], measurements={"final_q_score": row["q_score"]},
                         onboard_video=destination, native_receipt_sha256=row["receipt_sha256"])
        metadata = read_json(folder / "policy_service.json")
        diagnostics = metadata["policy_identity"].get("diagnostics")
        if diagnostics is not None:
            if (diagnostics.get("candidate_sha256") != parent_revision
                    or diagnostics.get("task") != case["task"]
                    or diagnostics.get("case_id") != digest([case["task"], {"instance_id": case["instance_id"]}])
                    or "development-trace.jsonl" not in receipt["evidence"]):
                raise ValueError("Development diagnostic evidence lacks matching receipt attribution")
            trace = folder / "development-trace.jsonl"
            verified = verify_trace(trace, expected_identity=diagnostics,
                                    expected_steps=receipt["native_result"].get("steps"))
            if verified["sha256"] != receipt["evidence"][trace.name]:
                raise ValueError("Development diagnostic evidence changed")
            trace_destination = "development-feedback/traces/" + case_sha + ".jsonl"
            videos.append((trace, output / trace_destination, verified["sha256"]))
            episode["onboard_diagnostics"] = trace_destination
        rows.append(episode)
        seen.add(case_sha)
    if seen != expected or len(origins) != 1:
        raise ValueError("Incomplete development cohort or mixed System 2 revisions")
    summary = dict(schema="physicalrsi.behavior-development-feedback/v1", parent_id=parent_id,
                   parent_revision=parent_revision, evaluator_revision=evaluator_revision,
                   episodes=sorted(rows, key=lambda row: row["case_sha256"]), qualification=None)
    with locked(bundle / ".lock"):
        summary_path = bundle / "episodes.json"
        if summary_path.exists() and read_json(summary_path) != summary:
            raise ValueError("Frozen development feedback changed")
        evidence = {}
        for source, target, sha in videos:
            if target.exists():
                if file_digest(target) != sha:
                    raise ValueError("Exported onboard video changed")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    shutil.copyfile(source, temporary)
                    if file_digest(temporary) != sha:
                        raise ValueError("Onboard video changed during export")
                    temporary.rename(target)
                finally:
                    temporary.unlink(missing_ok=True)
            evidence[str(target.relative_to(output))] = sha
        if not summary_path.exists():
            atomic_json(summary_path, summary)
        evidence[str(summary_path.relative_to(output))] = file_digest(summary_path)
        return dict(split="evolve", evaluator_revision=evaluator_revision,
                    episodes=summary["episodes"], evidence=evidence,
                    scope=("Complete development cohort: native final Q-score, onboard RGB video and public action/proprio diagnostics"
                           if any("onboard_diagnostics" in row for row in rows) else
                           "Complete development cohort: native final Q-score and onboard RGB video only"))
