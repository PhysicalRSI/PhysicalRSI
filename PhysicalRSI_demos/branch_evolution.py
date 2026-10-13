"""CPU demonstration: recover a losing specialization for a later proposal.

Round one deliberately searches one-task repairs. Round two composes an
archived specialization with the incumbent's measured successful controls.
This exercises branch reuse, not a claim of an optimal search algorithm.
"""
from pathlib import Path
import shutil

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import EvolutionSpec, verify_harness
from .recursive_research import manifest
from .rollout_evolution import fit_coefficients, setup as calibration_setup


class BranchEvolver:
    def __init__(self, archive):
        self.archive = archive

    def identity(self):
        return dict(kind="archived-calibration-composition/v1", source=file_digest(Path(__file__)))

    def propose_from_archive(self, parent, evaluations, context, search, output):
        controls = read_json(Path(parent["root"]) / "control.json")
        evidence = {p: sha for value in evaluations for p, sha in value["evidence"].items()}
        provenance = dict(parent_sha256=verify_harness(parent), method="single-task-response-calibration",
            environment="software-calibration", evidence=evidence, costs=dict(proposals=1), used_lessons=[],
            hypothesis="Calibrate one action contract from its observed response",
            expected_effect="Improve that task without reducing other task scores",
            required_checks=["paired-position-success"])
        if context["index"] == 0:
            coefficients = fit_coefficients(evaluations)
            if coefficients is None:
                return []
            children = []
            for task, coefficient in sorted(coefficients.items()):
                root = output / ("specialist-" + task)
                shutil.copytree(parent["root"], root)
                updated = dict(controls, **{task: coefficient})
                atomic_json(root / "control.json", updated)
                children.append(manifest(root, "specialist-" + task, changes=dict(control=updated), **provenance))
            return children
        failed = {value["trial"]["task"] for value in evaluations if value["outcome"] == "failure"}
        if not failed:
            return []
        # Only scores from the frozen historical view inform branch retrieval.
        useful = [row for row in search["candidates"]
                  if row["disposition"] == "evaluated_not_selected" and row["metrics"]
                  and all(row["metrics"]["per_task"][task] == 1 for task in failed)]
        if not useful:
            return []
        source = min(useful, key=lambda row: (row["archive_revision"], row["candidate_id"]))
        child = self.archive.fork(search["view"], source["archive_revision"], source["candidate_id"],
            parent=parent, output=output / "composed-candidate", name="composed",
            evolution=EvolutionSpec(("control",)))
        updated = read_json(Path(child["root"]) / "control.json")
        for task, coefficient in controls.items():
            if task not in failed:
                updated[task] = coefficient
        atomic_json(Path(child["root"]) / "control.json", updated)
        child["components"]["control"]["control.json"] = file_digest(Path(child["root"]) / "control.json")
        child.update(changes=dict(control=updated), hypothesis="Compose the archived specialist with successful incumbent controls",
            expected_effect="Recover the missing task without losing the incumbent's capability",
            required_checks=["paired-position-success"])
        child["evidence"].update(evidence)
        return [child]


def setup(root):
    return calibration_setup(root, evolver_factory=BranchEvolver, max_candidates=2,
                             task_weights={"axis": 1, "vector": 2}, max_trials=26)


def run(root):
    campaign, initial, suite, quota = setup(root)
    result = campaign.run(initial)
    result.update(final_test=campaign.report_test(initial), quota=quota.status())
    return result
