from copy import deepcopy
import shutil

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory, ResearchMemoryGate, MemoryBoundSuite
from test_improvement_campaign import setup, manifest


def fixture(tmp_path):
    campaign, parent, suite, quota = setup(tmp_path, max_trials=8, max_rounds=1)
    root = tmp_path / 'child'
    shutil.copytree(parent['root'], root)
    atomic_json(root / 'control.json', {'increment': 1})
    child = manifest(root, 'child')
    evidence = tmp_path / 'diagnostic.json'
    atomic_json(evidence, {'observed': 'retained failure recovered; fresh execution untested'})
    files = {str(evidence): file_digest(evidence)}
    lesson = dict(conditions={'task': 'counter'}, hypothesis='A new primitive recovers bounded failures',
                  required_checks=['activation'], counterexample_checks=['old-success'], evidence=files)
    memory = ResearchMemory(tmp_path / 'memory')
    revision = memory.write({'recovery': lesson})
    report = dict(schema='physicalrsi.memory-checks/v1', memory_revision=revision,
                  parent_sha256=verify_harness(parent), candidate_sha256=verify_harness(child),
                  context={'task': 'counter'}, producer='reviewed-checker',
                  lessons={'recovery': {k: {'passed': True, 'evidence': files}
                                       for k in ['activation', 'old-success']}})
    return parent, child, suite, memory, revision, report, evidence


def gate_for(tmp_path, data, report=None):
    parent, child, suite, memory, revision, original, evidence = data
    p = tmp_path / 'checks.json'
    atomic_json(p, report if report is not None else original)
    return ResearchMemoryGate(memory=memory, revision=revision, context={'task': 'counter'},
                              parent_sha256=verify_harness(parent), producer='reviewed-checker',
                              reports={p: file_digest(p)})


def test_memory_is_immutable_scoped_and_evidence_bound(tmp_path):
    data = fixture(tmp_path)
    _, _, _, memory, revision, _, evidence = data
    assert memory.retrieve(revision, {'task': 'other'}) == {}
    value = memory.read(revision)
    value['lessons']['recovery']['hypothesis'] = 'Revised hypothesis'
    updated = memory.write(value['lessons'], parent=revision)
    assert updated != revision and memory.read(updated)['parent'] == revision
    assert memory.read(revision)['lessons']['recovery']['hypothesis'] != 'Revised hypothesis'
    atomic_json(evidence, {'changed': True})
    with pytest.raises(ValueError, match='evidence changed'):
        memory.read(revision)


def test_gate_composes_with_existing_suite_without_running_trials(tmp_path):
    data = fixture(tmp_path)
    parent, child, suite, *_ = data
    wrapped = MemoryBoundSuite(suite, gate_for(tmp_path, data))
    assert wrapped.admit(parent)['accepted']
    result = wrapped.admit(child)
    assert result['accepted'] and result['evidence']['research_memory']['qualification'] is None
    assert suite.resets == []
    with pytest.raises(ValueError, match='current parent'):
        wrapped.development_cases(child)
    with pytest.raises(ValueError, match='comparison parent'):
        wrapped.validation_cases({'parent_id': 'x', 'candidates': {'x': verify_harness(child)}})


@pytest.mark.parametrize('fault', ['missing-check', 'counterexample', 'parent', 'context', 'producer', 'lesson', 'candidate'])
def test_bad_memory_claim_never_reaches_underlying_admission(tmp_path, fault):
    data = fixture(tmp_path)
    parent, child, suite, memory, revision, original, _ = data
    report = deepcopy(original)
    if fault == 'missing-check': del report['lessons']['recovery']['activation']
    elif fault == 'counterexample': report['lessons']['recovery']['old-success']['passed'] = False
    elif fault == 'parent': report['parent_sha256'] = '0' * 64
    elif fault == 'context': report['context']['task'] = 'other'
    elif fault == 'producer': report['producer'] = 'unknown'
    elif fault == 'lesson': report['lessons'] = {}
    elif fault == 'candidate': report['candidate_sha256'] = '0' * 64
    def forbidden(candidate): raise AssertionError('Rejected candidate reached suite')
    suite.admit = forbidden
    wrapped = MemoryBoundSuite(suite, gate_for(tmp_path, data, report))
    assert not wrapped.admit(child)['accepted']
    assert suite.resets == []


def test_missing_or_modified_reports_fail_closed(tmp_path):
    data = fixture(tmp_path)
    gate = gate_for(tmp_path, data)
    p = tmp_path / 'checks.json'
    saved = p.read_bytes()
    p.unlink()
    assert not gate.check(data[1])['accepted']
    p.write_bytes(saved)
    assert gate.check(data[1])['accepted']
    atomic_json(p, {})
    with pytest.raises(ValueError, match='report changed'):
        gate.check(data[1])


def test_snapshot_tampering_rejected(tmp_path):
    data = fixture(tmp_path)
    memory, revision = data[3:5]
    atomic_json(memory.root / (revision + '.json'), {})
    with pytest.raises(ValueError, match='snapshot changed'):
        memory.read(revision)


def test_counterexample_suspends_and_refinement_retains_regression(tmp_path):
    data = fixture(tmp_path)
    memory, original = data[3:5]
    path = tmp_path / 'negative.json'
    atomic_json(path, {'development_observation': 'ambiguous component'})
    evidence = {str(path): file_digest(path)}
    suspended = memory.counterexample(original, 'recovery', check='ambiguous-component',
                                     evidence=evidence, explanation='Dominant component assumption fails')
    assert memory.retrieve(suspended, {'task': 'counter'}) == {}
    assert 'recovery' in memory.retrieve(original, {'task': 'counter'})
    with pytest.raises(ValueError, match='scope'):
        memory.refine(suspended, 'recovery', conditions={'sensor': 'rgbd'},
                      hypothesis='Broadening is unsupported', evidence=evidence)
    refined = memory.refine(suspended, 'recovery', conditions={'task': 'counter', 'component': 'dominant'},
                            hypothesis='Apply only after public component dominance check', evidence=evidence)
    assert memory.read(refined)['parent'] == suspended
    assert memory.retrieve(refined, {'task': 'counter'}) == {}
    lesson = memory.retrieve(refined, {'task': 'counter', 'component': 'dominant'})['recovery']
    assert 'ambiguous-component' in lesson['counterexample_checks']
    assert lesson['counterexamples'][0]['evidence'] == evidence
    atomic_json(path, {'changed': True})
    with pytest.raises(ValueError, match='evidence changed'):
        memory.read(refined)


def test_shared_evidence_is_hashed_once_per_memory_read_and_rechecked_on_next_read(tmp_path, monkeypatch):
    from PhysicalRSI_core.self_harness import research_memory as module
    shared = tmp_path / 'shared-rollout.bin'
    shared.write_bytes(b'rollout evidence\n' * 32768)
    separate = tmp_path / 'second-rollout.bin'
    separate.write_bytes(b'another independent execution')
    lesson = dict(conditions={'task': 'counter'}, hypothesis='Shared development observation',
                  required_checks=['activation'], counterexample_checks=[],
                  evidence={str(shared): file_digest(shared), str(separate): file_digest(separate)})
    memory = ResearchMemory(tmp_path / 'memory')
    revision = memory.write({'lesson-' + str(i): deepcopy(lesson) for i in range(32)})
    calls = []
    original = module.file_digest

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(module, 'file_digest', counted)
    first = memory.read(revision)
    assert len(first['lessons']) == 32
    assert sorted(calls) == sorted([str(shared), str(separate)])
    calls.clear()
    assert memory.read(revision) == first
    assert sorted(calls) == sorted([str(shared), str(separate)])
    shared.write_bytes(b'changed rollout evidence')
    with pytest.raises(ValueError, match='evidence changed'):
        memory.read(revision)


def test_shared_evidence_conflicting_digests_cannot_reuse_a_previous_check(tmp_path):
    from PhysicalRSI_core.infra.storage import digest, read_json
    _, _, _, memory, revision, _, evidence = fixture(tmp_path)
    value = read_json(memory.root / (revision + '.json'))
    value['lessons']['z-conflicting'] = deepcopy(value['lessons']['recovery'])
    value['lessons']['z-conflicting']['evidence'][str(evidence)] = '0' * 64
    changed_revision = digest(value)
    atomic_json(memory.root / (changed_revision + '.json'), value)
    with pytest.raises(ValueError, match='evidence changed'):
        memory.read(changed_revision)


@pytest.mark.parametrize('damage', ['changed', 'missing', 'conflicting'])
def test_memory_successor_checks_shared_evidence_once_and_never_caches_across_writes(tmp_path, monkeypatch, damage):
    from PhysicalRSI_core.self_harness import research_memory as module
    _, _, _, memory, parent, _, evidence = fixture(tmp_path)
    lesson = memory.read(parent)['lessons']['recovery']
    lessons = {'lesson-' + str(i): deepcopy(lesson) for i in range(32)}
    original = module.file_digest
    calls = []

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(module, 'file_digest', counted)
    revision = memory.write(lessons, parent=parent)
    # Parent verification and successor validation are separate observations.
    assert calls == [str(evidence)] * 2
    calls.clear()
    assert memory.write(lessons, parent=parent) == revision
    assert calls == [str(evidence)] * 2
    saved = {p.name: p.read_bytes() for p in memory.root.glob('*.json')}
    if damage == 'changed':
        evidence.write_text('changed evidence')
    elif damage == 'missing':
        evidence.unlink()
    else:
        lessons['lesson-31']['evidence'][str(evidence)] = '0' * 64
    with pytest.raises(ValueError, match='evidence changed'):
        memory.write(lessons, parent=parent)
    assert {p.name: p.read_bytes() for p in memory.root.glob('*.json')} == saved
