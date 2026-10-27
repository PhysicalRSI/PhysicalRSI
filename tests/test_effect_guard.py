import pytest

from physicalrsi.skills.effect_guard import effect_decision, effect_guard
from PhysicalRSI_core.contracts import Context


def test_observed_effect_requires_safe_finish_not_repeat_or_task_success():
    skill = effect_guard()
    assert skill.execute(dict(effect_observed=True, execution_ok=True), Context('effect'))['decision'] == 'finish_safe'
    assert effect_decision(effect_observed=False, execution_ok=True)['decision'] == 'continue'
    assert effect_decision(effect_observed=True, execution_ok=False)['decision'] == 'abort'


@pytest.mark.parametrize('value', [None, 1, 'yes'])
def test_unknown_or_ambiguous_effect_is_rejected(value):
    with pytest.raises(ValueError):
        effect_decision(effect_observed=value, execution_ok=True)
