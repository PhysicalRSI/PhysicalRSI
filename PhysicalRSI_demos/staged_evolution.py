"""Measured CPU calibration search with cheap screening and concurrent validation.

The proposal pool deliberately includes overcorrections to exercise rejection.
This is a small infrastructure acceptance example, not a general search result.
"""
from pathlib import Path
import shutil

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from .recursive_research import manifest
from .rollout_evolution import LeastSquaresEvolver, setup as calibration_setup


class CalibrationSearch(LeastSquaresEvolver):
    def __init__(self, archive):
        pass

    def identity(self):
        return dict(kind="calibration-with-overcorrections/v1", implementation=file_digest(Path(__file__)),
                    fitting=super().identity())

    def propose(self, parent, evaluations, context, output):
        children = super().propose(parent, evaluations, context, output)
        if not children:
            return []
        fitted = children[0]
        controls = read_json(Path(fitted["root"]) / "control.json")
        for scale in (2, 3, 4):
            root = output / ("overcorrection-" + str(scale))
            shutil.copytree(fitted["root"], root)
            changed = {task: value * scale for task, value in controls.items()}
            atomic_json(root / "control.json", changed)
            provenance = {key: value for key, value in fitted.items() if key not in {"id", "root", "components", "changes"}}
            children.append(manifest(root, "overcorrection-" + str(scale), changes=dict(control=changed), **provenance))
        return children


def setup(root, *, screened=True, screen_incumbent=False, max_trials=None):
    limit = (24 if screen_incumbent else 22) if screened else 26
    return calibration_setup(root, evolver_factory=CalibrationSearch, max_candidates=4,
        max_trials=max_trials if max_trials is not None else limit,
        scheduled_workers=2, screening_threshold=1.0 if screened else None,
        screen_incumbent=screen_incumbent)


def run(root, *, screened=True, screen_incumbent=False):
    campaign, initial, _, quota = setup(root, screened=screened, screen_incumbent=screen_incumbent)
    result = campaign.run(initial)
    return dict(result, final_test=campaign.report_test(initial), quota=quota.status())
