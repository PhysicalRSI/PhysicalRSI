from dataclasses import asdict

import pytest

from PhysicalRSI_baselines.behavior_baseline import native
from PhysicalRSI_baselines.behavior_baseline.feedback import export_development
from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest


def fixture_trial(tmp_path):
    # Synthetic receipt and video bytes test projection/integrity, not physics.
    folder = tmp_path / "native-trial"
    origin = System2Trial(Revision("fixture", "1"), "parent", "a" * 64, "development")
    spec = native.specification(tmp_path, folder, task="radio", instance_index=200,
                                policy="websocket", policy_revision="a" * 64, system2=origin)
    result = dict(task="radio", instance_id=200, rollout_id=0, success=False,
                  q_score={"final": .25}, privileged_object_state="must_not_enter_feedback")
    metadata = dict(schema="physicalrsi.behavior-policy-service/v1", task="radio",
                    candidate_sha256="a" * 64, policy_identity={}, policy_identity_sha256=digest({}))
    for name, data in [("specification.json", spec), ("result.json", result), ("policy_service.json", metadata)]:
        atomic_json(folder / name, data)
    video = folder / "native/videos/radio_200_0.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture video bytes, not simulator evidence")
    evidence = {str(path.relative_to(folder)): file_digest(path) for path in folder.rglob("*") if path.is_file()}
    atomic_json(folder / "receipt.json", dict(state="completed", specification_sha256=digest(spec),
        evidence=evidence, native_result_file="result.json", native_result=result, outcome="failure"))
    return dict(folder=folder, specification=spec)


def test_feedback_exposes_only_complete_development_scores_and_onboard_video(tmp_path):
    trial = fixture_trial(tmp_path)
    kwargs = dict(parent_id="parent", parent_revision="a" * 64,
                  expected_cases=[trial["specification"]["case"]], evaluator_revision="fixture",
                  output=tmp_path / "round")
    feedback = export_development([trial], **kwargs)
    assert feedback["split"] == "evolve"
    assert feedback["episodes"][0]["measurements"] == {"final_q_score": .25}
    assert "must_not_enter_feedback" not in repr(feedback)
    assert all(name.startswith("development-feedback/") for name in feedback["evidence"])
    assert not any("result.json" in name for name in feedback["evidence"])
    assert export_development([trial], **kwargs) == feedback
    with pytest.raises(ValueError, match="Incomplete"):
        export_development([], **dict(kwargs, output=tmp_path / "missing"))
    changed = dict(trial, specification=dict(trial["specification"], system2=asdict(
        System2Trial(Revision("fixture", "1"), "parent", "a" * 64, "validation", "b" * 64, "c" * 64))))
    with pytest.raises(ValueError, match="development evidence"):
        export_development([changed], **dict(kwargs, output=tmp_path / "leaked"))
    video = next((kwargs["output"] / "development-feedback/videos").glob("*.mp4"))
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="video changed"):
        export_development([trial], **kwargs)


def test_feedback_admits_only_receipt_bound_public_diagnostics(tmp_path):
    import numpy as np
    from PhysicalRSI_baselines.behavior_baseline.diagnostics import DevelopmentTrace, seal_trace
    from PhysicalRSI_baselines.behavior_baseline.observations import PROPRIO
    from PhysicalRSI_core.infra.storage import read_json
    trial = fixture_trial(tmp_path)
    folder = trial["folder"]
    source = tmp_path / "live.jsonl"
    with DevelopmentTrace(source, split="development", candidate_sha256="a" * 64,
                          task="radio", case_id=digest(["radio", {"instance_id": 200}])) as writer:
        writer.append(dict(task="radio", proprio={k: np.zeros((1, len(v))) for k, v in PROPRIO.items()}),
                      primitive_actions=[], issued_action=np.zeros((1, 23)))
    identity = writer.identity()
    seal_trace(source, folder / "development-trace.jsonl", expected_identity=identity, expected_steps=1)
    metadata = read_json(folder / "policy_service.json")
    metadata["policy_identity"] = dict(diagnostics=identity)
    metadata["policy_identity_sha256"] = digest(metadata["policy_identity"])
    atomic_json(folder / "policy_service.json", metadata)
    result = read_json(folder / "result.json")
    result["steps"] = 1
    atomic_json(folder / "result.json", result)
    receipt = read_json(folder / "receipt.json")
    receipt["native_result"] = result
    receipt["evidence"] = {str(p.relative_to(folder)): file_digest(p) for p in folder.rglob("*")
                           if p.is_file() and p.name != "receipt.json"}
    atomic_json(folder / "receipt.json", receipt)
    kwargs = dict(parent_id="parent", parent_revision="a" * 64,
                  expected_cases=[trial["specification"]["case"]], evaluator_revision="fixture",
                  output=tmp_path / "round")
    feedback = export_development([trial], **kwargs)
    artifact = feedback["episodes"][0]["onboard_diagnostics"]
    assert feedback["evidence"][artifact] == file_digest(folder / "development-trace.jsonl")
    assert file_digest(kwargs["output"] / artifact) == feedback["evidence"][artifact]
    assert export_development([trial], **kwargs) == feedback
    (folder / "development-trace.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="evidence changed"):
        export_development([trial], **kwargs)
