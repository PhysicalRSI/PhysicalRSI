import json
from copy import deepcopy

import pytest

from PhysicalRSI_Autoresearch.study import DEFAULT_PROPOSALS, run_study
from PhysicalRSI_baselines.opentrons_liquid_handling.policy import Case, plan
from PhysicalRSI_baselines.opentrons_liquid_handling.audit import audit


class Simulator:
    identity = {'kind': 'test-double-not-opentrons'}

    def __init__(self):
        self.calls = 0

    def __call__(self, protocol, output, timeout):
        self.calls += 1
        compile(protocol.read_text(), str(protocol), 'exec')
        (output / 'simulator.log').write_text('TEST DOUBLE: syntax only')
        return {'passed': True}


def test_resume_preserves_completed_rounds_and_validation_is_last(tmp_path):
    sim = Simulator(); root = tmp_path / 'study'
    partial = run_study(root, sim, max_rounds=2)
    assert partial['status'] == 'running' and partial['validation_passed'] is None
    assert sim.calls == 4 and not (root / 'validation').exists()
    assert [r['status'] for r in partial['rounds']] == ['keep', 'discard']
    result = run_study(root, sim, resume=True)
    assert result['selected'] == 'reuse-mix-tip' and result['validation_passed']
    assert sim.calls == 10
    assert run_study(root, sim, resume=True) == result and sim.calls == 10
    assert (root / 'report.md').exists() and (root / 'results.tsv').exists()
    assert (root / 'sources/PhysicalRSI_Autoresearch/study.py').is_file()


def test_ties_keep_incumbent_and_validation_does_not_select(tmp_path):
    proposals = deepcopy(DEFAULT_PROPOSALS[:1])
    proposals.append(dict(proposals[0], name='equivalent'))
    sim = Simulator()
    def evaluate(protocol, output, timeout):
        if 'validation' in protocol.parts:
            return {'passed': False, 'error': 'validation-only failure'}
        return sim(protocol, output, timeout)
    evaluate.identity = sim.identity
    result = run_study(tmp_path / 'study', evaluate, proposals=proposals)
    assert result['selected'] == 'mixed' and not result['validation_passed']
    assert result['rounds'][-1]['status'] == 'discard'


def test_simulator_error_is_recorded_without_killing_remaining_search(tmp_path):
    sim = Simulator()
    def evaluate(protocol, output, timeout):
        if '0000' in protocol.parts:
            raise TimeoutError('fixture timeout')
        return sim(protocol, output, timeout)
    evaluate.identity = sim.identity
    result = run_study(tmp_path / 'study', evaluate, max_rounds=2)
    assert result['rounds'][0]['status'] == 'error'
    assert result['selected'] is None


def test_corrupted_evidence_blocks_resume(tmp_path):
    root = tmp_path / 'study'; sim = Simulator()
    result = run_study(root, sim, max_rounds=1)
    receipt = root / result['rounds'][0]['receipt']
    (receipt.parent / '0/actions.json').write_text('[]')
    with pytest.raises(ValueError, match='artifact changed'):
        run_study(root, sim, resume=True)
    assert sim.calls == 2


def test_changed_proposal_or_budget_cannot_reuse_old_results(tmp_path):
    root = tmp_path / 'study'; sim = Simulator()
    run_study(root, sim, max_rounds=1)
    with pytest.raises(ValueError, match='configuration or source changed'):
        run_study(root, sim, resume=True, trial_seconds=30)


def test_interrupted_trial_is_preserved_and_retried(tmp_path):
    sim = Simulator(); root = tmp_path / 'study'
    def interrupted(*args):
        raise KeyboardInterrupt
    interrupted.identity = sim.identity
    with pytest.raises(KeyboardInterrupt):
        run_study(root, interrupted)
    result = run_study(root, sim, resume=True, max_rounds=1)
    assert result['rounds'][0]['receipt'].endswith('attempt-0001/receipt.json')
    assert (root / 'development/0000/attempt-0000/0/actions.json').exists()


@pytest.mark.parametrize('case', [Case(2,30), Case(6,100), Case(12,150)])
def test_same_well_tip_reuse_reduces_tips_without_changing_ideal_concentrations(case):
    old = audit(case, plan(case, 'batched'))
    new = audit(case, plan(case, DEFAULT_PROPOSALS[-1]['settings']))
    assert new['passed'] and new['wells'] == old['wells']
    assert new['tips'] == old['tips'] - case.wells
