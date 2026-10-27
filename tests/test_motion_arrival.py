import pytest
from PhysicalRSI_core.contracts import Context
from physicalrsi.skills.motion_arrival import arrival_decision, motion_arrival


def test_successful_plan_with_no_actual_motion_retries_then_aborts():
    for attempts, expected in [(1, 'retry'), (2, 'retry'), (3, 'abort')]:
        result = arrival_decision([.1, 0., 0.], [0., 0., 0.], plan_success=True, attempts=attempts)
        assert result['decision'] == expected
        assert result['position_error_m'] == .1


def test_observed_arrival_and_failed_planner_remain_distinct():
    assert arrival_decision([0., 0., 0.], [.01, 0., 0.], plan_success=True, attempts=1)['decision'] == 'complete'
    assert arrival_decision([0., 0., 0.], [0., 0., 0.], plan_success=False, attempts=1)['decision'] == 'abort'


@pytest.mark.parametrize('options', [{'max_attempts': 0}, {'max_attempts': True},
    {'position_tolerance_m': float('nan')}, {'position_tolerance_m': 0}])
def test_invalid_policy_parameters_fail_closed(options):
    with pytest.raises(ValueError):
        motion_arrival(**options)


def test_skill_revision_covers_parameters_and_executes_generic_coordinates():
    a, b = motion_arrival(), motion_arrival(max_attempts=2)
    assert a.revision != b.revision
    result = a.execute(dict(target_m=[1., 2., 3.], observed_m=[1., 2., 3.],
        plan_success=True, attempts=1), Context('generic-arrival'))
    assert result['decision'] == 'complete'
