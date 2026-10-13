"""Small physical-file fixtures for admission binding; not native qualification."""
import subprocess
from pathlib import Path

import pytest

from PhysicalRSI_baselines.behavior_baseline import admission, engine, openpi_backend
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness


def setup(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.py').write_text('# software source fixture\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True, stderr=subprocess.DEVNULL).strip()
    git('init')
    git('add', 'model.py')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
    source_revision = git('rev-parse', 'HEAD')
    monkeypatch.setattr(openpi_backend, 'OPENPI_REVISION', source_revision)
    tokenizer = tmp_path / 'tokenizer'
    tokenizer.write_bytes(b'fixture tokenizer')
    monkeypatch.setattr(openpi_backend, 'TOKENIZER', dict(uri='fixture', sha256=file_digest(tokenizer)))
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'weights').write_bytes(b'fixture weights')
    root = tmp_path / 'parent'
    atomic_json(root / 'backend.json', dict(kind='openpi-b1k/v1', task='radio',
        source_root=str(source), source_revision=source_revision,
        adapter_sha256=file_digest(Path(openpi_backend.__file__)), tokenizer=openpi_backend.TOKENIZER,
        checkpoint_root=str(checkpoint), checkpoint_files={'weights': file_digest(checkpoint / 'weights')}))
    atomic_json(root / 'engine.json', engine.manifest())
    (root / 'control.py').write_text('def initialize(task): return {}\ndef step(obs, state, primitive): return primitive(obs)\n')
    candidate = dict(id='parent', root=str(root), components={
        key: {'backend.json': file_digest(root / 'backend.json')} for key in CLOSURE})
    candidate['components']['dependencies'] = {'engine.json': file_digest(root / 'engine.json')}
    candidate['components']['control'] = {'control.py': file_digest(root / 'control.py')}
    review_path = tmp_path / 'reviews' / (verify_harness(candidate) + '.json')
    atomic_json(review_path, dict(schema='physicalrsi.behavior-code-review/v1',
        candidate_sha256=verify_harness(candidate), control_sha256=file_digest(root / 'control.py'),
        reviewer='software-fixture', accepted=True, reason='Synthetic review for artifact testing only',
        checks={key: True for key in admission.CHECKS}))
    prerequisite = tmp_path / 'preflight.json'
    atomic_json(prerequisite, {'scope': 'software fixture'})
    gate = admission.ReviewedAdmission(reference=candidate, reviews=review_path.parent,
        evidence={prerequisite: file_digest(prerequisite)}, tokenizer=tokenizer)
    return gate, candidate, review_path, checkpoint, prerequisite


def test_reviewed_gate_verifies_and_copies_bound_evidence(tmp_path, monkeypatch):
    gate, candidate, review, checkpoint, prerequisite = setup(tmp_path, monkeypatch)
    output = tmp_path / 'admission'
    receipt = gate.check(candidate, output)
    assert receipt['accepted'] is True
    assert read_json(output / 'checks.json')['qualification'] is None
    for name, sha in receipt['evidence'].items():
        assert file_digest(output / name) == sha
    assert gate.check(candidate, output) == receipt
    (checkpoint / 'weights').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Checkpoint bytes'):
        gate.check(candidate, tmp_path / 'tampered')


def test_explicit_review_rejection_is_not_promoted_to_acceptance(tmp_path, monkeypatch):
    gate, candidate, path, checkpoint, prerequisite = setup(tmp_path, monkeypatch)
    review = read_json(path)
    review['checks']['no_privileged_state_access'] = False
    atomic_json(path, review)
    receipt = gate.check(candidate, tmp_path / 'rejected')
    assert receipt['accepted'] is False
    assert read_json(tmp_path / 'rejected/checks.json')['checked_checkpoint_inventory'] is False


def test_wrong_review_and_changed_prerequisites_fail_before_acceptance(tmp_path, monkeypatch):
    gate, candidate, path, checkpoint, prerequisite = setup(tmp_path, monkeypatch)
    review = read_json(path)
    atomic_json(path, dict(review, control_sha256='0' * 64))
    with pytest.raises(ValueError, match='exact program'):
        gate.check(candidate, tmp_path / 'wrong-review')
    atomic_json(path, review)
    atomic_json(prerequisite, {'changed': True})
    with pytest.raises(ValueError, match='prerequisite evidence changed'):
        gate.check(candidate, tmp_path / 'wrong-environment')
