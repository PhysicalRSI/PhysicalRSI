import copy
import pytest
from physicalrsi.skills.exit_transition import exit_transition


def observations(offset=.055, speed=.1, gap=.04):
    return dict(last_guided_sample=dict(timestamp_s=1., position_m=[0,0,1]),
        flight_samples=[dict(timestamp_s=1.+t, position_m=[offset+speed*t,0,1-5*t*t])
                        for t in [gap, gap+.04, gap+.08, gap+.12]],
        geometry=dict(exit_position_m=[0,0,1], exit_tangent=[1,0,0], gravity_m_s2=10.))


def test_recovers_timing_range_without_claiming_exact_launch_time():
    result=exit_transition().execute(observations())
    assert result['valid']
    assert result['position_offset_range_m']==pytest.approx([.055,.059])
    assert result['transition_time_range_s']==[1.,1.04]
    assert result['available_after_s']==1.16


def test_continuous_fast_exit_is_not_a_position_jump():
    result=exit_transition().execute(observations(offset=0,speed=1.))
    assert result['position_offset_range_m']==pytest.approx([0.,.04])


def test_rotation_translation_and_tangent_scale_invariance():
    a=observations();b=copy.deepcopy(a)
    b['geometry'].update(exit_position_m=[3,2,6],exit_tangent=[0,7,0])
    for sample in [b['last_guided_sample']]+b['flight_samples']:
        x,y,z=sample['position_m'];sample['position_m']=[3-y,2+x,5+z]
    assert exit_transition().execute(a)['position_offset_range_m']==pytest.approx(
        exit_transition().execute(b)['position_offset_range_m'])


def test_nonballistic_samples_are_rejected_without_offset():
    x=observations();x['flight_samples'][2]['position_m'][0]+=.03
    result=exit_transition().execute(x)
    assert result['valid'] is False
    assert 'position_offset_range_m' not in result


def test_insufficient_history_is_not_imputed():
    x=observations();x['flight_samples']=x['flight_samples'][:2]
    assert exit_transition().execute(x)['reason']=='insufficient_transition_samples'


@pytest.mark.parametrize('kind',['time','gravity','axis','nan'])
def test_invalid_observations_rejected(kind):
    x=observations()
    if kind=='time':x['flight_samples'][1]['timestamp_s']=1.
    if kind=='gravity':x['geometry']['gravity_m_s2']=-10
    if kind=='axis':x['geometry']['exit_tangent']=[0,0,0]
    if kind=='nan':x['flight_samples'][0]['position_m'][0]=float('nan')
    with pytest.raises(ValueError):exit_transition().execute(x)


def test_parameter_changes_are_versioned():
    assert exit_transition().revision!=exit_transition(max_residual_m=.002).revision


def test_range_includes_interior_extremum_for_vertical_exit():
    x=observations();x['geometry']['exit_tangent']=[0,0,1]
    for sample in x['flight_samples']:
        t=sample['timestamp_s']-1.04
        sample['position_m']=[0,0,1-.1*t-5*t*t]
    result=exit_transition().execute(x)
    assert result['position_offset_range_m']==pytest.approx([-.004,.0005])
