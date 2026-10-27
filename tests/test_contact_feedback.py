import copy

import pytest

from physicalrsi.skills.contact_feedback import contact_progress


def attempt(after, before=(0, 0, 0), target=(1, 0, 0), planned=True):
    return dict(before_m=list(before), after_m=list(after), target_m=list(target), plan_success=planned)


def test_sideways_slip_and_planner_success_do_not_prove_progress():
    value = dict(attempts=[attempt((0, .02, 0)), attempt((-.01, 0, 0))], recoveries_used=0)
    original = copy.deepcopy(value)
    skill = contact_progress()
    assert skill.execute(value)['decision'] == 'recontact'
    assert value == original
    value['recoveries_used'] = 2
    assert skill.execute(value)['decision'] == 'abort'


def test_contact_recovery_is_not_triggered_by_steady_progress():
    value = dict(attempts=[attempt((0, 0, 0)), attempt((.01, 0, 0))], recoveries_used=0)
    assert contact_progress().execute(value)['decision'] == 'continue'
    value['attempts'] = [attempt((.99, 0, 0))]
    assert contact_progress().execute(value)['decision'] == 'stop'
    value['attempts'] = [attempt((0, 0, 0), planned=False)]
    assert contact_progress().execute(value)['decision'] == 'replan'


def test_rotation_and_translation_do_not_change_contact_decision():
    value = dict(attempts=[attempt((.0001, .02, 0))]*2, recoveries_used=0)
    transformed = copy.deepcopy(value)
    for item in transformed['attempts']:
        for key in ('before_m', 'after_m', 'target_m'):
            x,y,z = item[key]
            item[key] = [5-y, 2+x, z-1]
    first, second = (contact_progress().execute(v) for v in (value, transformed))
    assert first['decision'] == second['decision'] == 'recontact'
    assert first['projected_progress_m'] == pytest.approx(second['projected_progress_m'])


def test_no_history_and_invalid_measurements():
    assert contact_progress().execute(dict(attempts=[], recoveries_used=0))['decision'] == 'observe'
    with pytest.raises(ValueError):
        contact_progress().execute(dict(attempts=[attempt((float('nan'),0,0))], recoveries_used=0))
    with pytest.raises(ValueError):
        contact_progress(consecutive_stalls=True)


def test_glancing_progress_can_trigger_early_recontact_without_changing_default():
    value = dict(attempts=[attempt((.01, .03, 0))]*2, recoveries_used=0)
    assert contact_progress().execute(value)['decision'] == 'continue'
    result = contact_progress(minimum_direction_cosine=.7).execute(value)
    assert result['decision'] == 'recontact'
    assert result['direction_cosine'] == pytest.approx(1 / (10**.5))
    value['attempts'] = [attempt((.02, .001, 0))]*2
    assert contact_progress(minimum_direction_cosine=.7).execute(value)['decision'] == 'continue'
    for invalid in (True, -0.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            contact_progress(minimum_direction_cosine=invalid)
