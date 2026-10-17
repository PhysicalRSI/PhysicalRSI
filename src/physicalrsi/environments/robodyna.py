"""Contract fixture only. No RoboDyna SDK or native benchmark is bundled."""
import math
from pathlib import Path
from PhysicalRSI_core.infra.storage import file_digest
from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment


class RobodynaFixture:
    """Observe a moving point twice; command its future interception position."""
    supports_video = False

    def identity(self):
        return dict(kind='constant-velocity-fixture', source=file_digest(Path(__file__)))

    def describe(self):
        return Embodiment('point-fixture', self.identity()['source'], 'software',
                          Contract('observation'), Contract('action'))

    def reset(self, case, context):
        context.check()
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
               for v in case.values()):
            raise ValueError('Finite fixture parameters required')
        self.case = dict(case)
        self.expected = case['position'] + case['velocity'] * case['horizon']
        return dict(ready=True, observation=dict(position=case['position'],
            previous_position=case['position']-case['velocity'], sample_dt=1.0,
            horizon=case['horizon']))

    def step(self, action, context):
        context.check()
        position = action['target_position']
        if not math.isfinite(position):
            raise ValueError('Finite target required')
        return dict(terminated=True, observation=dict(robot_position=position,
            target_position=self.expected))
