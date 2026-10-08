from pathlib import Path

from PhysicalRSI_core.infra.storage import digest
from PhysicalRSI_baselines.robotworld.native_evaluator import RoboCasaNativeEvaluator


class FakeTrials:
    def identity(self):
        return {'kind': 'fixture-native-trials'}


def test_evaluator_freezes_paired_cases_and_native_protocol(tmp_path):
    evaluator = RoboCasaNativeEvaluator(trials=FakeTrials(), fixed_components={},
                                        task_indices={'CloseDrawer': 5},
                                        budgets={'CloseDrawer': 450})
    comparison = {'profile': {'tasks': {'CloseDrawer': {'episodes': 2}}}}
    cases = evaluator._cases(comparison)
    assert cases['CloseDrawer'][0]['reset_seed'] == 12
    assert cases['CloseDrawer'][1]['reset_seed'] == 13
    assert len({digest(case) for case in cases['CloseDrawer']}) == 2
    assert evaluator.identity()['kind'].endswith('evaluator/v1')
