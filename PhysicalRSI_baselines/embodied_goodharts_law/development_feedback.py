"""Keep native evidence complete while bounding the System 2 proposal summary."""
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator


def compact_feedback(feedback):
    episodes = []
    for episode in feedback["episodes"]:
        measurements = episode["measurements"]
        summary = {key: measurements[key] for key in (
            "official_success", "native_physics_steps", "native_action_count",
            "termination_reason", "task", "reset_geometry_sha256",
        ) if key in measurements}
        summary["full_measurements_sha256"] = digest(measurements)
        summary["recorded_native_actions"] = len(measurements.get("native_actions", []))
        failures = [row for row in measurements.get("pose_trace", []) if row.get("state") == "failed"]
        summary["failed_pose_count"] = len(failures)
        summary["last_failed_poses"] = [{key: row[key] for key in (
            "target", "actual_pose", "position_error_m", "orientation_error_rad",
            "start_physics_step", "end_physics_step", "error",
        ) if key in row} for row in failures[-8:]]
        episodes.append({**episode, "measurements": summary})
    # Evidence references still identify the complete original receipts and
    # trajectories. No recorded trial or scoring input is rewritten.
    return {**feedback, "episodes": episodes}


class CompactDevelopmentEvaluator(ExperimentEvaluator):
    def identity(self):
        return {**super().identity(), "development_summary_implementation": file_digest(Path(__file__))}

    def development(self, parent, output):
        return compact_feedback(super().development(parent, output))
