from time import monotonic

import pytest

from PhysicalRSI_core.contracts import Contract, Context, Operation, ReconciliationRequired
from PhysicalRSI_core.skill_execution import CHECK, RELEASE, REQUEST, STOPPED, SkillExecutor, SkillStage
from PhysicalRSI_core.timing import ControlTiming, action_chunk

ACTION = Contract('test-action', unit='normalized', frame='robot')


def observation(sequence, **payload):
    return dict(schema='physicalrsi.observation/v1', episode='test', clock_domain='local',
                sequence=sequence, captured_at=monotonic(), payload=payload)


def stage(name, events, *, provider=None, monitor=None, release=None, **budgets):
    def predict(request, context):
        events.append(('predict', name, request['goal']))
        return action_chunk(request['observation'], [[1], [2]], period_seconds=.05)

    def check(request, context):
        state = request['observation']['payload'].get(name, 'running')
        return dict(status=state, evidence={'public_observation': state})

    def stop(request, context):
        context.check()
        events.append(('stop', name))
        return {'stopped': True}

    return SkillStage(name, 'grounded goal ' + name,
        Operation(name, 'provider-v1', REQUEST, ACTION, provider or predict),
        Operation(name + '-check', 'monitor-v1', REQUEST, CHECK, monitor or check),
        Operation(name + '-stop', 'stop-v1', RELEASE, STOPPED, release or stop),
        **(dict(max_actions=8, max_observations=10, seconds=10) | budgets))


def executor(tmp_path, stages, **kwargs):
    return SkillExecutor(stages, action_contract=ACTION,
                       instruction='Put the black bowl on the cabinet onto the plate',
                       timing=ControlTiming(.05, 2, 10),
                       validate_actions=lambda actions: None,
                       validator_revision='test-validator-v1',
                       workspace=tmp_path / 'executor', **kwargs)


def test_handoff_requires_ack_and_new_observation_and_releases_old_provider(tmp_path):
    events = []
    r = executor(tmp_path, [stage('code', events), stage('learned-fixture', events)])
    c = Context('test')
    first = r.advance(observation(0), c)
    with pytest.raises(ReconciliationRequired):
        r.advance(observation(1, code='succeeded'), c)
    with pytest.raises(ReconciliationRequired):
        r.acknowledge('wrong')
    r.acknowledge(first['token'])
    assert r.advance(observation(1, code='succeeded'), c)['status'] == 'handoff'
    assert events == [('stop', 'code'), ('predict', 'code', 'grounded goal code'), ('stop', 'code')]
    second = r.advance(observation(2), c)
    r.acknowledge(second['token'])
    assert r.advance(observation(3, **{'learned-fixture': 'succeeded'}), c)['status'] == 'completed'
    r.close()
    assert events[-1] == ('stop', 'learned-fixture')
    assert (r.root / 'plan.json').is_file()
    assert len(list((r.root / 'events').glob('*.json'))) >= 10


def test_unknown_never_runs_next_provider_and_clears_current_state(tmp_path):
    events = []
    r = executor(tmp_path, [stage('pick', events), stage('place', events)])
    c = Context('test')
    result = r.advance(observation(0), c)
    r.acknowledge(result['token'])
    assert r.advance(observation(1, pick='unknown'), c)['status'] == 'needs_observation'
    assert events[-1] == ('stop', 'pick') and r.index == 0 and not r.owner
    result = r.advance(observation(2), c)
    assert result['status'] == 'action'
    r.acknowledge(result['token'])
    r.close()


def test_predictions_cannot_assert_completion(tmp_path):
    def liar(request, context):
        return {'success': True, 'predicted_value': 1.0}
    events = []
    r = executor(tmp_path, [stage('pick', events, provider=liar)])
    with pytest.raises(ValueError, match='invalid'):
        r.advance(observation(0), Context('test'))
    assert r.state == 'failed' and events == [('stop', 'pick'), ('stop', 'pick')]


@pytest.mark.parametrize('change', ['old-sequence', 'episode', 'clock', 'stale'])
def test_observation_must_be_fresh_and_from_same_episode(tmp_path, change):
    events = []
    r = executor(tmp_path, [stage('pick', events)])
    c = Context('test')
    result = r.advance(observation(0), c)
    r.acknowledge(result['token'])
    obs = observation(1)
    if change == 'old-sequence': obs['sequence'] = 0
    if change == 'episode': obs['episode'] = 'another'
    if change == 'clock': obs['clock_domain'] = 'another'
    if change == 'stale': obs['captured_at'] -= 11
    with pytest.raises(ValueError):
        r.advance(obs, c)
    assert events[-1] == ('stop', 'pick')


def test_uncertain_release_blocks_handoff(tmp_path):
    releases = iter([True, False])
    r = executor(tmp_path, [stage('a', [], release=lambda req, ctx: {'stopped': next(releases)}), stage('b', [])])
    c = Context('test')
    result = r.advance(observation(0), c)
    r.acknowledge(result['token'])
    with pytest.raises(ReconciliationRequired):
        r.advance(observation(1, a='succeeded'), c)
    assert r.index == 0 and r.state == 'needs_reconciliation'
    with pytest.raises(ReconciliationRequired): r.close()


def test_cancellation_still_releases_inference_state(tmp_path):
    events = []
    r = executor(tmp_path, [stage('a', events)])
    c = Context('test')
    result = r.advance(observation(0), c)
    r.acknowledge(result['token'])
    c.cancelled.set()
    with pytest.raises(RuntimeError): r.advance(observation(1), c)
    assert events[-1] == ('stop', 'a')


@pytest.mark.parametrize('budget', [{'max_actions': 2}, {'max_observations': 1}])
def test_stage_budget_cannot_be_bypassed_by_more_observations(tmp_path, budget):
    r = executor(tmp_path, [stage('a', [], **budget)])
    c = Context('test')
    result = r.advance(observation(0), c)
    r.acknowledge(result['token'])
    assert r.advance(observation(1), c)['status'] == 'failed'


def test_controller_contract_mismatch_and_workspace_reuse_rejected(tmp_path):
    s = stage('a', [])
    with pytest.raises(ValueError, match='action contract'):
        SkillExecutor([s], action_contract=Contract('joint-8d'), timing=ControlTiming(.05, 2, 1),
                    instruction='Task', validator_revision='test-validator-v1',
                    validate_actions=lambda _: None, workspace=tmp_path / 'bad')
    executor(tmp_path, [s])
    with pytest.raises(FileExistsError): executor(tmp_path, [s])


def test_real_isolated_code_provider_and_host_provider_handoff(tmp_path, isolation_runtime):
    from PhysicalRSI_baselines.robodojo.code_execution import execute_policy

    def isolated(request, context):
        actions = execute_policy('def policy(robot, memory):\n    return [[memory["delta"]]]\n',
            {'delta': .2}, handlers={}, runtime=isolation_runtime,
            output=tmp_path / 'isolated-code', timeout_s=10, deadline=context.deadline)
        return action_chunk(request['observation'], actions, period_seconds=.05)

    events = []
    r = executor(tmp_path, [stage('code', events, provider=isolated), stage('host-fixture', events)])
    c = Context('test', deadline=monotonic() + 30)
    first = r.advance(observation(0), c)
    assert first['command']['actions'] == [[.2]]
    r.acknowledge(first['token'])
    assert r.advance(observation(1, code='succeeded'), c)['status'] == 'handoff'
    second = r.advance(observation(2), c)
    assert second['command']['actions'] == [[1], [2]]
    r.acknowledge(second['token'])
    r.close()


def test_full_instruction_survives_stage_goal_and_identity_copies(tmp_path):
    seen = []
    def predict(request, context):
        seen.append((request['instruction'], request['goal']))
        return action_chunk(request['observation'], [[0]], period_seconds=.05)
    r = executor(tmp_path, [stage('grasp', [], provider=predict)])
    r.identity['instruction'] = 'Discard the original relation'
    result = r.advance(observation(0), Context('test'))
    assert seen == [('Put the black bowl on the cabinet onto the plate', 'grounded goal grasp')]
    r.acknowledge(result['token'])
    r.close()


def test_initial_state_must_be_released_before_any_inference(tmp_path):
    events = []
    r = executor(tmp_path, [stage('a', events, release=lambda req, ctx: {'stopped': False})])
    with pytest.raises(ReconciliationRequired):
        r.advance(observation(0), Context('test'))
    assert events == [] and r.state == 'needs_reconciliation'


def test_reentrant_provider_call_cannot_create_second_action(tmp_path):
    def predict(request, context):
        with pytest.raises(RuntimeError, match='reentrant'):
            r.advance(observation(1), context)
        return action_chunk(request['observation'], [[0]], period_seconds=.05)
    r = executor(tmp_path, [stage('a', [], provider=predict)])
    result = r.advance(observation(0), Context('test'))
    r.acknowledge(result['token'])
    r.close()


def test_uncertain_execution_cannot_be_closed_and_reused(tmp_path):
    r = executor(tmp_path, [stage('a', [])])
    result = r.advance(observation(0), Context('test'))
    with pytest.raises(ReconciliationRequired): r.close()
    assert r.pending == result['token']


def test_monitor_failure_never_starts_a_provider(tmp_path):
    events = []
    r = executor(tmp_path, [stage('a', events)])
    assert r.advance(observation(0, a='failed'), Context('test'))['status'] == 'failed'
    assert events == []


def test_expired_stage_stops_before_more_inference(tmp_path):
    events = []
    r = executor(tmp_path, [stage('a', events)])
    result = r.advance(observation(0), Context('test'))
    r.acknowledge(result['token'])
    r.started -= 11
    assert r.advance(observation(1), Context('test'))['status'] == 'failed'
    assert sum(event[0] == 'predict' for event in events) == 1
