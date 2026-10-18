import copy
import math

import pytest

from physicalrsi.skills.dynamic_interception import dynamic_interception, intercept_prediction, motion_estimation


def observation():
    return dict(phase='flight', samples=[dict(timestamp_s=t, position_m=[t, 0, 1]) for t in (0., .1, .2)],
        geometry=dict(capture_height_m=0., gravity_m_s2=10.), robot=dict(target_position_m=[0, 0, 0],
            workspace_min_m=[-5, -5, -5], workspace_max_m=[5, 5, 5], max_speed_m_s=10.))


def test_composed_ballistic_prediction_and_pure_replay():
    value = observation()
    original = copy.deepcopy(value)
    skill = dynamic_interception()
    result = skill.execute(value)
    assert result['feasible']
    assert result['target_position_m'] == pytest.approx([.2+math.sqrt(.2), 0, 0])
    assert result['interception_time_s'] == pytest.approx(.2+math.sqrt(.2))
    assert result['residual_rms_m'] < 1e-12
    assert value == original
    assert skill.execute(value) == result
    assert len(skill.composition.nodes) == 3


def test_shared_skill_rotation_transfers_guide_prediction():
    value = observation()
    value['phase'] = 'rolling'
    value['samples'] = [dict(timestamp_s=t, position_m=[t, 0, 1-t]) for t in (0., .1, .2)]
    value['geometry']['path_m'] = [[0, 0, 1], [.5, 0, .5]]
    skill = dynamic_interception(rolling_energy_factor=0.)
    first = skill.execute(value)
    for sample in value['samples']:
        x, y, z = sample['position_m']
        sample['position_m'] = [y, x, z]
    value['geometry']['path_m'] = [[0, 0, 1], [0, .5, .5]]
    second = skill.execute(value)
    assert first['feasible'] and second['feasible']
    assert second['target_position_m'] == pytest.approx([first['target_position_m'][1], first['target_position_m'][0], 0])


def test_evolution_changes_prediction_revision_only():
    a = dynamic_interception(rolling_energy_factor=0.)
    b = dynamic_interception(rolling_energy_factor=5/7)
    assert a.revision != b.revision
    for index in (0, 2):
        assert a.composition.nodes[index].revision == b.composition.nodes[index].revision
    assert a.composition.nodes[1].revision != b.composition.nodes[1].revision


def test_hold_insufficient_history_and_no_hidden_state():
    value = observation()
    value['samples'] = value['samples'][:1]
    result = dynamic_interception().execute(value)
    assert not result['feasible'] and result['target_position_m'] is None
    value = observation()
    value['phase'] = 'dropping'
    assert dynamic_interception().execute(value)['target_position_m'] is None


def test_unreachable_target_not_silently_clipped():
    value = observation()
    value['robot']['workspace_max_m'][0] = .1
    result = dynamic_interception().execute(value)
    assert result['reason'] == 'outside_workspace'
    assert result['target_position_m'][0] > .1


@pytest.mark.parametrize('field,value', [('velocity_gain', float('nan')), ('rolling_energy_factor', 1.1), ('max_horizon_s', 0.)])
def test_bounded_parameters(field, value):
    with pytest.raises(ValueError):
        intercept_prediction(**{field: value})


def test_timestamp_failure_and_missing_guide():
    value = observation()
    value['samples'][1]['timestamp_s'] = 0.
    with pytest.raises(ValueError, match='increasing'):
        motion_estimation().execute(value)
    value = observation()
    value['phase'] = 'rolling'
    with pytest.raises(ValueError, match='guide'):
        dynamic_interception().execute(value)


def test_uphill_barrier_and_deadline_are_explicit_failures():
    value = observation()
    value['phase'] = 'rolling'
    value['geometry']['path_m'] = [[0, 0, 1], [1, 0, 3], [2, 0, .5]]
    result = dynamic_interception(rolling_energy_factor=1.).execute(value)
    assert not result['feasible']
    assert result['reason'] == 'insufficient_observed_forward_motion'
    value = observation()
    value['robot']['max_speed_m_s'] = .001
    assert dynamic_interception().execute(value)['reason'] == 'insufficient_time'


def test_acceleration_hypothesis_changes_predicted_distance():
    value = observation()
    value['phase'] = 'rolling'
    value['samples'] = [dict(timestamp_s=t, position_m=[t, 0, 1-t]) for t in (0., .1, .2)]
    value['geometry']['path_m'] = [[0, 0, 1], [.5, 0, .5]]
    constant = dynamic_interception(rolling_energy_factor=0.).execute(value)
    rolling = dynamic_interception(rolling_energy_factor=5/7).execute(value)
    assert rolling['target_position_m'][0] > constant['target_position_m'][0]
    assert rolling['time_to_intercept_s'] < constant['time_to_intercept_s']
