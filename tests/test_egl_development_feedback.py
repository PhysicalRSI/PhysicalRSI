from copy import deepcopy

from PhysicalRSI_core.infra.storage import canonical, digest
from PhysicalRSI_baselines.embodied_goodharts_law.development_feedback import compact_feedback


def test_large_native_evidence_stays_linked_without_exceeding_proposal_size():
    measurements = {"official_success": False, "native_physics_steps": 1000,
        "native_actions": [{"step": i, "action": [0.123456789] * 8} for i in range(1000)],
        "pose_trace": [{"state": "failed", "target": [0.] * 7,
                        "position_error_m": .02, "start_physics_step": i * 20,
                        "end_physics_step": (i + 1) * 20} for i in range(40)]}
    feedback = {"split": "evolve", "evidence": {"receipt.json": "a" * 64},
        "episodes": [{"task": "native", "case": {"seed": 1}, "run_id": "trial",
                      "outcome": "failure", "measurements": measurements}]}
    original = deepcopy(feedback)
    result = compact_feedback(feedback)
    summary = result["episodes"][0]["measurements"]
    assert feedback == original
    assert result["evidence"] == original["evidence"]
    assert result["episodes"][0]["outcome"] == "failure"
    assert summary["full_measurements_sha256"] == digest(measurements)
    assert summary["failed_pose_count"] == 40 and len(summary["last_failed_poses"]) == 8
    assert len(canonical(result)) < 8192
