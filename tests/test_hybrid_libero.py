import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.hybrid_libero import (
    action_contract, learned_provider, validate_actions,
)
from PhysicalRSI_core.contracts import Context


@pytest.mark.parametrize('actions', [[[0] * 8], [[float('nan')] * 7], [[2] * 7], [[True] * 7]])
def test_joint_actions_and_malformed_controls_cannot_enter_osc(actions):
    with pytest.raises(ValueError): validate_actions(actions, controller='OSC_POSE')


def test_action_spaces_are_distinct():
    assert action_contract('OSC_POSE') != action_contract('JOINT_POSITION')
    validate_actions([[0] * 7], controller='OSC_POSE')
    validate_actions([[0] * 8], controller='JOINT_POSITION')


def test_backend_retains_goal_and_requires_positive_queue_reset():
    received = []
    pins = {k: 'a' * 64 for k in ['checkpoint', 'implementation', 'preprocessing', 'normalization']}
    def predict(request, context):
        received.append(request['goal'])
        return [[0.] * 7]
    provider, release = learned_provider('learned-fixture', controller='OSC_POSE',
        period_seconds=.05, pins=pins, predict=predict, reset=lambda context: None)
    packet = {'schema': 'physicalrsi.observation/v1', 'episode': 'test',
              'clock_domain': 'local', 'sequence': 1, 'payload': {}}
    chunk = provider({'goal': 'bowl on cabinet, not the other bowl', 'observation': packet}, Context('test'))
    assert received == ['bowl on cabinet, not the other bowl']
    assert chunk['actions'] == [[0.] * 7]
    assert release({}, Context('test')) == {'stopped': False}
    with pytest.raises(ValueError, match='Pin'):
        learned_provider('unpinned', controller='OSC_POSE', period_seconds=.05,
                         pins={}, predict=predict, reset=lambda context: True)
