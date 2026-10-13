import pytest

from PhysicalRSI_core import contracts, experiments
from PhysicalRSI_core.diagnostics import profile_trials
from PhysicalRSI_core.experiments import ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


@pytest.fixture
def measured(tmp_path, monkeypatch):
    clock, resets = [0.], []
    monkeypatch.setattr(experiments, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(contracts, 'monotonic', lambda: clock[0])

    class Environment(CounterEnvironment):
        def reset(self, case, context):
            resets.append(case)
            clock[0] += 2
            return super().reset(case, context)

        def step(self, action, context):
            clock[0] += 3
            return super().step(action, context)

    runtime = ExperimentRuntime(tmp_path / 'trials')
    for name, target in [('two', 2), ('three', 3)]:
        runtime.run(name, task='counter', case=dict(initial=0, target=target), scope='software',
                    environment=Environment(), policy=CounterPolicy(1), verifier=CounterVerifier())
    return runtime, resets


def test_profile_audits_completed_trials_without_reexecution_or_writes(measured):
    runtime, resets = measured
    before = {str(p): file_digest(p) for p in runtime.root.rglob('*') if p.is_file()}
    report = profile_trials([(runtime, 'two'), (runtime, 'three')])
    summary = report['summary']
    assert summary['elapsed_seconds'] == dict(total=19., mean=9.5, maximum=11.)
    assert summary['phase_seconds']['reset'] == dict(observed_trials=2, total=4.)
    assert summary['phase_seconds']['rollout'] == dict(observed_trials=2, total=15.)
    assert summary['phase_timed_trials'] == 2 and summary['missing_phase_trials'] == 0
    assert report['by_task']['counter'] == summary
    assert len({row['execution_id'] for row in report['trials']}) == 2
    assert len(resets) == 2
    assert {str(p): file_digest(p) for p in runtime.root.rglob('*') if p.is_file()} == before


def test_legacy_timing_coverage_is_explicit(measured):
    runtime, _ = measured
    path = runtime.root / 'three/receipt.json'
    receipt = read_json(path)
    del receipt['timing']
    atomic_json(path, receipt)
    report = profile_trials([(runtime, 'two'), (runtime, 'three')])
    assert report['summary']['elapsed_seconds']['total'] == 19.
    assert report['summary']['missing_phase_trials'] == 1
    assert report['summary']['phase_seconds']['reset'] == dict(observed_trials=1, total=2.)
    assert report['trials'][1]['phase_seconds'] is None


def test_profile_rejects_duplicates_and_changed_rollout_evidence(measured):
    runtime, _ = measured
    with pytest.raises(ValueError, match='distinct Trial'):
        profile_trials([(runtime, 'two'), (runtime, 'two')])
    (runtime.root / 'two/trajectory.json').write_text('[]')
    with pytest.raises(ValueError):
        profile_trials([(runtime, 'two')])
    with pytest.raises(ValueError, match='requires completed Trials'):
        profile_trials([])


@pytest.mark.parametrize('damage', ['null', 'schema', 'missing-phase', 'negative', 'inconsistent-total'])
def test_profile_rejects_malformed_or_inconsistent_timing(measured, damage):
    runtime, resets = measured
    path = runtime.root / 'two/receipt.json'
    receipt = read_json(path)
    if damage == 'null':
        receipt['timing'] = None
    elif damage == 'schema':
        receipt['timing']['schema'] = 'unknown'
    elif damage == 'missing-phase':
        del receipt['timing']['phase_seconds']['reset']
    elif damage == 'negative':
        receipt['timing']['phase_seconds']['reset'] = -1.
    else:
        receipt['elapsed_seconds'] += 10.
    atomic_json(path, receipt)
    with pytest.raises(ValueError):
        profile_trials([(runtime, 'two')])
    assert len(resets) == 2
