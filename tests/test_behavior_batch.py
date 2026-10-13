from contextlib import contextmanager

import pytest

from PhysicalRSI_baselines.behavior_baseline import batch, native, service
from PhysicalRSI_baselines.behavior_baseline.trials import NativeTrials
from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'candidate'
    atomic_json(root / 'backend.json', dict(task='radio', kind='openpi-b1k/v1'))
    candidate = dict(id='parent', root=str(root), components={
        kind: {'backend.json': file_digest(root / 'backend.json')} for kind in CLOSURE})
    atomic_json(root / 'candidate.json', candidate)
    revision = verify_harness(candidate)
    origin = System2Trial(Revision('fixture', '1'), 'parent', revision, 'development')
    quota = TrialQuota(tmp_path / 'quota', max_trials=2)
    trials = NativeTrials(tmp_path, tmp_path / 'run', quota=quota,
                         trials=[dict(task='radio', instance_index=i, system2=origin) for i in (0, 1)])
    events = []

    @contextmanager
    def served(*args, **kwargs):
        events.append('start')
        try:
            yield dict(host='127.0.0.1', port=8000, policy_revision=revision)
        finally:
            events.append('stop')

    def run(runtime, folder, *, expected_specification, **kwargs):
        events.append(kwargs['instance_index'])
        atomic_json(folder / 'specification.json', expected_specification)
        result = dict(success=False)
        atomic_json(folder / 'native.json', result)
        receipt = dict(state='completed', specification_sha256=digest(expected_specification),
                       evidence={'native.json': file_digest(folder / 'native.json')},
                       native_result_file='native.json', native_result=result, outcome='failure')
        atomic_json(folder / 'receipt.json', receipt)
        return receipt

    monkeypatch.setattr(service, 'policy_service', served)
    monkeypatch.setattr(native, 'run', run)
    kwargs = dict(candidates={'parent': root / 'candidate.json'}, python=tmp_path / 'python')
    return trials, kwargs, events


def test_reserved_batch_reuses_service_and_completed_trials(tmp_path, monkeypatch):
    # Software fixtures verify supervision and evidence reuse, not native scores.
    trials, kwargs, events = fixture(tmp_path, monkeypatch)
    receipts = batch.execute(trials, **kwargs)
    assert len(receipts) == 2
    assert events == ['start', 0, 1, 'stop']
    assert trials.quota.status()['reserved_trials'] == 2
    assert batch.execute(trials, **kwargs) == receipts
    assert events == ['start', 0, 1, 'stop']


def test_unresolved_plan_rejected_before_service_start(tmp_path, monkeypatch):
    trials, kwargs, events = fixture(tmp_path, monkeypatch)
    folder = trials.output / 'trials' / trials.plan['trials'][1]['id']
    atomic_json(folder / 'receipt.json', dict(state='started'))
    with pytest.raises(ReconciliationRequired):
        batch.execute(trials, **kwargs)
    assert events == []
    with pytest.raises(ValueError, match='registry'):
        batch.execute(trials, **dict(kwargs, candidates={}))


def test_episode_error_exits_service_and_preserves_unresolved_evidence(tmp_path, monkeypatch):
    trials, kwargs, events = fixture(tmp_path, monkeypatch)

    def fail(runtime, folder, **unused):
        atomic_json(folder / 'receipt.json', dict(state='unresolved'))
        raise RuntimeError('Fixture infrastructure failure')

    monkeypatch.setattr(native, 'run', fail)
    with pytest.raises(RuntimeError, match='infrastructure'):
        batch.execute(trials, **kwargs)
    assert events == ['start', 'stop']
    with pytest.raises(ReconciliationRequired):
        batch.execute(trials, **kwargs)
    assert events == ['start', 'stop']
