from copy import deepcopy
import pytest
from PhysicalRSI_baselines.opentrons_liquid_handling.policy import Case, plan, protocol_source
from PhysicalRSI_baselines.opentrons_liquid_handling.audit import audit


def test_mixing_is_required_even_when_commands_are_legal():
    result=audit(Case(),plan(Case(),'unmixed'))
    assert not result['passed']
    assert result['errors']==['Aspiration from unmixed dilution']


@pytest.mark.parametrize('case',[Case(4,100),Case(6,100),Case(8,80),Case(12,120)])
def test_batching_preserves_mass_balance_and_reduces_tips(case):
    baseline=audit(case,plan(case,'mixed'));optimized=audit(case,plan(case,'batched'))
    assert baseline['passed'] and optimized['passed']
    assert baseline['wells']==optimized['wells']
    assert optimized['tips']<baseline['tips']
    compile(protocol_source(plan(case,'batched')),'protocol.py','exec')


def test_wrong_volume_and_missing_mix_are_rejected():
    case=Case();actions=plan(case,'batched')
    broken=deepcopy(actions)
    next(x for x in broken if x['op']=='dispense')['volume']+=10
    assert not audit(case,broken)['passed']
    broken=[x for x in actions if x['op']!='mix']
    assert not audit(case,broken)['passed']
