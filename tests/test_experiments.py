import json

import pytest

from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.self_harness import ReconciliationRequired
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


def run(runtime, **changes):
    options = dict(task='counter', case={'initial':0,'target':3}, scope='software',
                   environment=CounterEnvironment(), policy=CounterPolicy(),
                   verifier=CounterVerifier(), budget=Budget(max_steps=5,seconds=5))
    options.update(changes)
    return runtime.run('trial', **options)


def test_completed_receipt_is_read_without_reexecution(tmp_path):
    class Counting(CounterEnvironment):
        resets = 0
        def reset(self, case, context):
            self.resets += 1
            return super().reset(case, context)
    runtime=ExperimentRuntime(tmp_path)
    env=Counting()
    first=run(runtime,environment=env)
    assert first['outcome']=='success'
    assert run(runtime,environment=env)==first
    assert env.resets==1
    assert first['steps']==3
    with pytest.raises(ValueError,match='different experiment'):
        run(runtime,budget=Budget(max_steps=10,seconds=5))


def test_termination_does_not_override_independent_verification(tmp_path):
    result=run(ExperimentRuntime(tmp_path),policy=CounterPolicy(2))
    assert result['state']=='completed'
    assert result['outcome']=='failure'
    assert result['verdict']['measurements']['position']==4
    assert result['qualification'] is None


def test_invalid_reset_is_not_policy_failure(tmp_path):
    class Unready(CounterEnvironment):
        def reset(self,case,context):return {'ready':False}
        def step(self,*args):raise AssertionError('Must not execute')
    result=run(ExperimentRuntime(tmp_path),environment=Unready())
    assert result['outcome']=='invalid' and result['steps']==0


def test_unknown_verdict_is_preserved(tmp_path):
    class Unknown(CounterVerifier):
        def verify(self,case,trace):
            return {'outcome':'uncertain','reason':'Independent sensor unavailable','measurements':{}}
    assert run(ExperimentRuntime(tmp_path),verifier=Unknown())['outcome']=='uncertain'


def test_interrupted_effect_is_never_retried(tmp_path):
    class Interrupted(CounterEnvironment):
        calls=0
        def step(self,action,context):
            self.calls+=1
            raise RuntimeError('Connection lost after dispatch')
    env=Interrupted();runtime=ExperimentRuntime(tmp_path)
    with pytest.raises(RuntimeError,match='Connection lost'):
        run(runtime,environment=env)
    with pytest.raises(ReconciliationRequired):
        run(runtime,environment=env)
    assert env.calls==1
    receipt=json.loads((tmp_path/'trial/receipt.json').read_text())
    assert receipt['state']=='needs_reconciliation'
    assert (tmp_path/'trial/pending-action.json').is_file()


def test_modified_evidence_cannot_be_reused(tmp_path):
    runtime=ExperimentRuntime(tmp_path);run(runtime)
    (tmp_path/'trial/trajectory.json').write_text('[]')
    with pytest.raises(ValueError,match='evidence changed'):
        run(runtime)


@pytest.mark.parametrize('failure', [None, 'reset', 'step', 'verification'])
def test_phase_timings_separate_execution_verification_evidence_and_interrupted_cleanup(tmp_path, monkeypatch, failure):
    from PhysicalRSI_core import contracts, experiments

    clock = [1000.]
    monkeypatch.setattr(experiments, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(contracts, 'monotonic', lambda: clock[0])
    original_digest = experiments.file_digest

    def measured_digest(path):
        if path.name == 'verdict.json':
            clock[0] += 11  # Final evidence hashing belongs to finalization.
        return original_digest(path)

    monkeypatch.setattr(experiments, 'file_digest', measured_digest)

    class TimedEnvironment(CounterEnvironment):
        def reset(self, case, context):
            clock[0] += 3
            if failure == 'reset':
                raise RuntimeError('reset interrupted')
            return super().reset(case, context)

        def step(self, action, context):
            clock[0] += 5
            if failure == 'step':
                raise RuntimeError('step interrupted')
            return super().step(action, context)

    class TimedPolicy(CounterPolicy):
        def act(self, observation, context):
            clock[0] += 2
            return super().act(observation, context)

        def end_episode(self, context):
            clock[0] += 13
            return dict(stopped=True)

    class TimedVerifier(CounterVerifier):
        def verify(self, case, trace):
            clock[0] += 7
            if failure == 'verification':
                raise RuntimeError('verification interrupted')
            return super().verify(case, trace)

    runtime = ExperimentRuntime(tmp_path)
    options = dict(environment=TimedEnvironment(), policy=TimedPolicy(), verifier=TimedVerifier(),
                   case=dict(initial=0, target=1), budget=Budget(1, 100))
    if failure is None:
        receipt = run(runtime, **options)
        assert run(runtime, **options) == receipt
        expected = dict(preparation=0, reset=3, rollout=20, verification=7, finalization=11, cleanup=0)
    else:
        with pytest.raises(RuntimeError, match='interrupted'):
            run(runtime, **options)
        receipt = json.loads((tmp_path / 'trial/receipt.json').read_text())
        expected = dict(preparation=0, reset=3, rollout=0 if failure == 'reset' else 7 if failure == 'step' else 20,
                        verification=7 if failure == 'verification' else 0, finalization=0,
                        cleanup=13 if failure == 'step' else 0)
        assert receipt['state'] == 'needs_reconciliation'
        with pytest.raises(ReconciliationRequired):
            run(runtime, **options)
    assert receipt['timing'] == dict(schema='physicalrsi.trial-timing/v1', phase_seconds=expected)
    assert receipt['elapsed_seconds'] == sum(expected.values())


def test_execution_identity_survives_reopening_and_copying_evidence(tmp_path):
    import shutil
    from PhysicalRSI_core.trials import Evaluation
    runtime = ExperimentRuntime(tmp_path / 'original')
    receipt = run(runtime)
    token = receipt['execution_id']
    assert len(token) == 32 and all(c in '0123456789abcdef' for c in token)
    assert run(runtime) == receipt
    shutil.copytree(runtime.root, tmp_path / 'archived')
    copied = ExperimentRuntime(tmp_path / 'archived')
    assert Evaluation.read(copied, 'trial').to_dict()['receipt'] == receipt
    # Historical receipts remain readable and receive no invented identity.
    legacy = dict(receipt)
    legacy.pop('execution_id')
    from PhysicalRSI_core.infra.storage import atomic_json
    atomic_json(copied.root / 'trial/receipt.json', legacy)
    assert copied.read('trial') == legacy


def test_execution_identity_is_durable_before_interrupted_reset(tmp_path):
    seen = []

    class Interrupted(CounterEnvironment):
        def reset(self, case, context):
            seen.append(json.loads((tmp_path / 'trial/receipt.json').read_text())['execution_id'])
            raise RuntimeError('reset interrupted')

    runtime = ExperimentRuntime(tmp_path)
    with pytest.raises(RuntimeError, match='reset interrupted'):
        run(runtime, environment=Interrupted())
    receipt = json.loads((tmp_path / 'trial/receipt.json').read_text())
    assert seen == [receipt['execution_id']]
    with pytest.raises(ReconciliationRequired):
        run(runtime, environment=Interrupted())
    assert json.loads((tmp_path / 'trial/receipt.json').read_text()) == receipt
    assert len(seen) == 1


@pytest.mark.parametrize('token', [None, 1, '', 'a'*31, 'A'*32, 'g'*32])
def test_malformed_execution_identity_is_rejected(tmp_path, token):
    from PhysicalRSI_core.infra.storage import atomic_json
    runtime = ExperimentRuntime(tmp_path)
    receipt = run(runtime)
    receipt['execution_id'] = token
    atomic_json(tmp_path / 'trial/receipt.json', receipt)
    with pytest.raises(ValueError, match='execution identity'):
        runtime.read('trial')
