"""Protocol tests; native simulation and real isolation are separate preflights."""
from pathlib import Path
from threading import get_ident

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law import recorded_cap
from PhysicalRSI_core.contracts import Contract, Context
from PhysicalRSI_core.embodiment import Embodiment, System1
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from test_egl_candidate_program import fixture_candidate


def setup(tmp_path):
    candidate = fixture_candidate(tmp_path / 'candidate')
    observation, action = Contract('test-cap-reply'), Contract('test-cap-request')
    embodiment = Embodiment('test-cap', '1', 'simulation', observation, action)
    system1 = System1('test-code', verify_harness(candidate), observation, action)
    runtime = tmp_path / 'runtime'
    atomic_json(runtime / 'manifest.json', {'scope': 'protocol-test-stub'})
    return candidate, embodiment, system1, runtime


def test_requests_are_recorded_before_effect_and_private_score_is_not_returned(tmp_path, monkeypatch):
    candidate, embodiment, system1, runtime = setup(tmp_path)
    owner = get_ident()
    effects, replies = [], []
    root = tmp_path / 'experiments'

    class API:
        @property
        def handlers(self):
            def move(args, kwargs, *, deadline):
                assert get_ident() == owner  # Native effects stay on the experiment thread.
                pending = read_json(root / 'trial/pending-action.json')
                assert pending['state'] == 'dispatched'
                assert pending['action']['method'] == 'move'
                effects.append(args[0])
                return {'position': args[0]}
            return {'move': move}

    def executor(source, memory, *, handlers, **limits):
        assert get_ident() != owner
        replies.append(handlers['move']([2], {}, deadline=limits['deadline']))
        return {'success': True, 'official_success': True}  # Untrusted self-report.

    monkeypatch.setattr(recorded_cap, 'execute_policy', executor)
    environment = recorded_cap.RecordedPrimitiveEnvironment(specification=embodiment,
        identity={'fixture': 1}, factory=lambda case: (API(), lambda: {'official_success': False, 'private': 'withheld'}, lambda: None))
    policy = recorded_cap.RecordedCapPolicy(candidate, specification=system1, capabilities=['move'], runtime=runtime, output=tmp_path / 'policy')
    try:
        result = ExperimentRuntime(root).run('trial', task='test', case={}, scope='software-protocol-test',
            environment=environment, policy=policy, verifier=recorded_cap.OfficialProxyVerifier(), budget=Budget(3, 10))
    finally:
        environment.close()
    assert effects == [2]
    assert replies == [{'position': 2}]
    assert result['outcome'] == 'failure'
    assert result['steps'] == 2  # One request and a separate program completion.
    assert read_json(root / 'trial/policy.json')['stopped']
    assert ExperimentRuntime(root).read('trial') == result


def test_unknown_and_out_of_sequence_requests_never_actuate(tmp_path):
    _, embodiment, _, _ = setup(tmp_path)
    effects = []
    class API:
        handlers = {'move': lambda args, kwargs, deadline: effects.append(args)}
    env = recorded_cap.RecordedPrimitiveEnvironment(specification=embodiment, identity={'fixture': 1},
        factory=lambda case: (API(), lambda: {'official_success': False}, lambda: None))
    context = Context('trial')
    env.reset({}, context)
    for method, index in [('get_sim_state', 0), ('move', 1)]:
        with pytest.raises(ValueError, match='Invalid recorded'):
            env.step(dict(kind='call', id=index, method=method, args=[], kwargs={}), context)
    assert effects == []
    env.close()


def test_native_failure_keeps_dispatched_evidence_and_stops_waiting_policy(tmp_path, monkeypatch):
    candidate, embodiment, system1, runtime = setup(tmp_path)
    effects = []
    class API:
        @property
        def handlers(self):
            def move(args, kwargs, *, deadline):
                effects.append('may-have-actuated')
                raise RuntimeError('native result unavailable')
            return {'move': move}
    def executor(source, memory, *, handlers, **limits):
        return handlers['move']([], {}, deadline=limits['deadline'])
    monkeypatch.setattr(recorded_cap, 'execute_policy', executor)
    env = recorded_cap.RecordedPrimitiveEnvironment(specification=embodiment, identity={'fixture': 1},
        factory=lambda case: (API(), lambda: {'official_success': False}, lambda: None))
    policy = recorded_cap.RecordedCapPolicy(candidate, specification=system1, capabilities=['move'], runtime=runtime, output=tmp_path / 'policy')
    root = tmp_path / 'experiments'
    with pytest.raises(RuntimeError, match='native result unavailable'):
        ExperimentRuntime(root).run('trial', task='test', case={}, scope='software-protocol-test',
            environment=env, policy=policy, verifier=recorded_cap.OfficialProxyVerifier(), budget=Budget(3, 10))
    env.close()
    assert effects == ['may-have-actuated']
    assert read_json(root / 'trial/receipt.json')['state'] == 'needs_reconciliation'
    assert read_json(root / 'trial/pending-action.json')['state'] == 'dispatched'
    assert read_json(root / 'trial/policy.json')['stopped'] is True


def test_official_termination_stops_policy_waiting_for_reply(tmp_path, monkeypatch):
    candidate, embodiment, system1, runtime = setup(tmp_path)
    effects = []
    class API:
        handlers = {'move': lambda args, kwargs, deadline: effects.append('moved')}
    def executor(source, memory, *, handlers, **limits):
        return handlers['move']([], {}, deadline=limits['deadline'])
    monkeypatch.setattr(recorded_cap, 'execute_policy', executor)
    env = recorded_cap.RecordedPrimitiveEnvironment(specification=embodiment, identity={'fixture': 1},
        factory=lambda case: (API(), lambda: {'official_success': bool(effects)}, lambda: None))
    policy = recorded_cap.RecordedCapPolicy(candidate, specification=system1, capabilities=['move'], runtime=runtime, output=tmp_path / 'policy')
    root = tmp_path / 'experiments'
    result = ExperimentRuntime(root).run('trial', task='test', case={}, scope='software-protocol-test',
        environment=env, policy=policy, verifier=recorded_cap.OfficialProxyVerifier(), budget=Budget(3, 10))
    env.close()
    assert result['outcome'] == 'success'
    assert result['steps'] == 1
    assert read_json(root / 'trial/policy.json') == {'stopped': True, 'program_returned': False}
