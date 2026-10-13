"""Task-specific clearance hypothesis from independent checks and public trace."""
import math
from pathlib import Path
from statistics import median

from PhysicalRSI_core.infra.storage import file_digest
from .task import TASK, SCOPE


class ClearanceFeedback:
    """Grasp-like aperture followed by obstacle disturbance suggests low carry.

    The thresholds and monotonicity hypothesis are authored. This classifier
    does not establish causality or label every failed transfer as low clearance.
    Fresh screening and paired validation remain necessary.
    """
    def identity(self):
        return dict(kind='grasp-and-obstacle-clearance-hypothesis/v1',
                    implementation=file_digest(Path(__file__)),
                    task_implementation=file_digest(Path(__file__).with_name('task.py')),
                    grasp_phase=3, aperture_interval_m=[.01, .065])

    def classify(self, value):
        if value['trial']['task'] != TASK or value['trial']['scope'] != SCOPE:
            raise ValueError('Clearance feedback requires the declared Isaac task')
        if value['outcome'] == 'success':
            return 'satisfied'
        if (value['outcome'] != 'failure' or
                value['verdict']['measurements'].get('checks', {}).get('obstacle_undisturbed') is not False):
            return 'unresolved'
        apertures = []
        for row in value['trajectory'][-1].get('native_rollout', []):
            if row.get('phase') != 3:
                continue
            joints = row.get('observation', {}).get('joint_positions')
            if (not isinstance(joints, list) or len(joints) != 9 or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in joints)):
                return 'unresolved'
            apertures.append(sum(joints[-2:]))
        return 'insufficient' if apertures and .01 < median(apertures) < .065 else 'unresolved'
