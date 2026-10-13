from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import os
import shutil

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, canonical, digest, file_digest, read_json
from PhysicalRSI_core.self_harness import archive as module
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import EvolutionSpec, verify_harness
from test_core_versions import seed


def archive_fixture(root, *, depth=4):
    # Minimal owned archive records for read semantics, not evaluation evidence.
    archive = CandidateArchive(root / 'archive')
    harness = archive.snapshots.capture(seed(root / 'source'))
    policy = {'fixture': 'read-audit'}
    revisions = []
    for index in range(depth):
        view = dict(schema='physicalrsi.archive-view/v1', archive=archive.identity(), scope='software',
                    policy_sha256=digest(policy), revisions=sorted(revisions))
        payloads = {'archive-view.json': canonical(view) + b'\n', 'nested/trace.bin': b'owned fixture bytes'}
        body = dict(schema='physicalrsi.archive-round/v1', scope='software', policy=policy,
            policy_sha256=digest(policy), source_round=str(root / ('fixture-' + str(index))),
            parent_revision='0' * 64, candidates={harness['id']: dict(harness=harness,
                freeze_sha256=verify_harness(harness), disposition='selected', admission=None,
                metrics=None, screening=None, rejection=None)},
            files={name: sha256(payload).hexdigest() for name, payload in payloads.items()}, qualification=None)
        revision = digest(body)
        folder = archive.root / 'entries' / revision
        for name, payload in payloads.items():
            path = folder / 'evidence' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        atomic_json(folder / 'record.json', body)
        revisions.append(revision)
    return archive, revisions, harness


def test_archive_shares_snapshot_checks_but_hashes_each_owned_evidence_file(tmp_path, monkeypatch):
    archive, revisions, harness = archive_fixture(tmp_path)
    counts = Counter()
    original = Path.open
    def opened(path, mode='r', *args, **kwargs):
        if 'r' in mode:
            counts[str(path)] += 1
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened)
    record = archive.read(revisions[-1])
    assert record['revision'] == revisions[-1]
    assert counts[str(Path(harness['root']) / 'control.json')] == 1
    for revision in revisions:
        folder = archive.root / 'entries' / revision
        assert counts[str(folder / 'evidence/nested/trace.bin')] == 1
        assert counts[str(folder / 'evidence/archive-view.json')] == 1
        assert counts[str(folder / 'record.json')] == 2


@pytest.mark.parametrize('kind', ['harness', 'evidence'])
def test_archive_new_read_detects_same_size_same_timestamp_corruption(tmp_path, kind):
    archive, revisions, harness = archive_fixture(tmp_path)
    archive.read(revisions[-1])
    path = (Path(harness['root']) / 'control.json' if kind == 'harness' else
            archive.root / 'entries' / revisions[0] / 'evidence/nested/trace.bin')
    before = path.stat()
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        archive.read(revisions[-1])


@pytest.mark.parametrize('kind', ['fifo', 'symlink_directory', 'extra_file'])
def test_archive_rejects_undeclared_or_nonphysical_entries(tmp_path, kind):
    archive, revisions, _ = archive_fixture(tmp_path)
    evidence = archive.root / 'entries' / revisions[0] / 'evidence'
    if kind == 'fifo':
        os.mkfifo(evidence / 'unexpected.pipe')
    elif kind == 'symlink_directory':
        outside = tmp_path / 'empty';outside.mkdir()
        (evidence / 'alias').symlink_to(outside, target_is_directory=True)
    else:
        (evidence / 'extra.bin').write_bytes(b'undeclared')
    with pytest.raises(ValueError):
        archive.read(revisions[-1])


def test_archive_never_uses_an_unhashed_view_to_skip_damaged_ancestry(tmp_path, monkeypatch):
    archive, revisions, _ = archive_fixture(tmp_path)
    folder = archive.root / 'entries' / revisions[-1]
    view_path = folder / 'evidence/archive-view.json'
    ancestor = archive.root / 'entries' / revisions[0] / 'evidence/nested/trace.bin'
    forged = read_json(view_path)
    forged['revisions'] = []
    opened = Path.open
    changed = []
    @contextmanager
    def checked_stream(stream):
        try:
            yield stream
        finally:
            stream.close()
            if not changed:
                changed.append(True)
                atomic_json(view_path, forged)
                ancestor.write_bytes(b'corrupted ancestor after the view was read')
    def swapped(path, mode='r', *args, **kwargs):
        stream = opened(path, mode, *args, **kwargs)
        return checked_stream(stream) if path == view_path and mode == 'rb' and not changed else stream
    monkeypatch.setattr(Path, 'open', swapped)
    with pytest.raises(ValueError, match='evidence (directory )?changed'):
        archive.read(revisions[-1])
    assert changed


def test_record_marker_is_rechecked_after_its_evidence(tmp_path, monkeypatch):
    archive, revisions, _ = archive_fixture(tmp_path, depth=1)
    folder = archive.root / 'entries' / revisions[0]
    marker = folder / 'record.json'
    real_digest = module.file_digest
    def replaced(path):
        value = real_digest(path)
        if path == folder / 'evidence/nested/trace.bin':
            marker.write_bytes(marker.read_bytes() + b'\n')
        return value
    monkeypatch.setattr(module, 'file_digest', replaced)
    with pytest.raises(ValueError, match='record changed during audit'):
        archive.read(revisions[0])


@pytest.mark.parametrize('dependency', ['artifacts', 'owned_files'])
def test_archive_identity_binds_its_reading_dependencies(tmp_path, monkeypatch, dependency):
    archive, revisions, _ = archive_fixture(tmp_path)
    view = read_json(archive.root / 'entries' / revisions[-1] / 'evidence/archive-view.json')
    source = Path(getattr(module, dependency).__file__)
    real_digest = module.file_digest
    monkeypatch.setattr(module, 'file_digest', lambda path: '0' * 64 if path == source else real_digest(path))
    with pytest.raises(ValueError, match='Invalid or changed archive view'):
        archive.summaries(view)


def test_fork_audits_each_entry_once_without_reusing_checks_in_later_calls(tmp_path, monkeypatch):
    archive, revisions, harness = archive_fixture(tmp_path)
    entry = archive.read(revisions[-1])
    view = archive.view(scope=entry['scope'], policy_sha256=entry['policy_sha256'])
    parent = seed(tmp_path / 'parent')
    atomic_json(Path(parent['root']) / 'control.json', {'component': 'control', 'value': 2})
    parent['components']['control']['control.json'] = file_digest(Path(parent['root']) / 'control.json')
    counts = Counter();read_one = archive._read_one
    def counted(revision, *args, **kwargs):
        counts[revision] += 1
        return read_one(revision, *args, **kwargs)
    monkeypatch.setattr(archive, '_read_one', counted)
    candidate = archive.fork(view, revisions[-1], harness['id'], parent=parent, output=tmp_path / 'proposal',
                             name='proposal', evolution=EvolutionSpec(('control',)))
    assert verify_harness(candidate) == verify_harness(harness)
    assert counts == {revision: 1 for revision in revisions}
    (archive.root / 'entries' / revisions[0] / 'evidence/nested/trace.bin').write_bytes(b'changed')
    with pytest.raises(ValueError):
        archive.fork(view, revisions[-1], harness['id'], parent=parent, output=tmp_path / 'new-proposal',
                     name='new-proposal', evolution=EvolutionSpec(('control',)))
    assert not (tmp_path / 'new-proposal').exists()


def test_mutating_a_returned_record_cannot_change_a_later_audit(tmp_path):
    archive, revisions, _ = archive_fixture(tmp_path)
    expected = archive.read(revisions[-1])
    changed = archive.read(revisions[-1])
    changed['candidates'].clear()
    changed['files'].clear()
    assert archive.read(revisions[-1]) == expected


def write_receipts(archive, revisions):
    directories = []
    for revision in revisions:
        record = read_json(archive.root / 'entries' / revision / 'record.json')
        directory = Path(record['source_round'])
        atomic_json(directory / 'archive.json', dict(archive=archive.identity(), revision=revision))
        directories.append(directory)
    return directories


def test_receipt_batch_checks_shared_ancestors_once_and_starts_fresh(tmp_path, monkeypatch):
    groups = [archive_fixture(tmp_path / name, depth=depth) for name, depth in [('one', 5), ('two', 3)]]
    directories = [d for archive, revisions, _ in groups for d in write_receipts(archive, revisions)]
    calls = Counter()
    original = CandidateArchive._read_one
    def counted(self, revision, *args):
        calls[(self.root, revision)] += 1
        return original(self, revision, *args)
    monkeypatch.setattr(CandidateArchive, '_read_one', counted)
    # Repeated receipt directories still require their source binding, but do
    # not rehash the same retained ancestor or evidence within this audit.
    module.verify_archive_receipts([*directories, directories[0]])
    assert calls == {(a.root, r): 1 for a, revisions, _ in groups for r in revisions}
    archive, revisions, _ = groups[0]
    path = archive.root / 'entries' / revisions[0] / 'evidence/nested/trace.bin'
    saved = path.stat()
    path.write_bytes(b'X' * saved.st_size)
    os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    with pytest.raises(ValueError, match='evidence changed'):
        module.verify_archive_receipts(directories)


@pytest.mark.parametrize('damage', ['identity', 'source_round'])
def test_batch_validates_every_receipt_even_when_its_entry_is_shared(tmp_path, damage):
    archive, revisions, _ = archive_fixture(tmp_path)
    directories = write_receipts(archive, revisions)
    path = directories[0] / 'archive.json'
    value = read_json(path)
    if damage == 'identity':
        value['archive']['implementation'] = '0' * 64
    else:
        value['revision'] = revisions[-1]
    atomic_json(path, value)
    with pytest.raises(ValueError, match='implementation changed|another round'):
        module.verify_archive_receipts(reversed(directories))


@pytest.mark.parametrize('damage', ['symlink', 'replacement', 'extra'])
def test_directory_changes_during_entry_read_are_rejected(tmp_path, monkeypatch, damage):
    archive, revisions, _ = archive_fixture(tmp_path, depth=1)
    directory = archive.root / 'entries' / revisions[0] / 'evidence/nested'
    original = module.file_digest
    def changed(path):
        value = original(path)
        if path == directory / 'trace.bin':
            if damage == 'extra':
                (directory / 'unlisted.bin').write_bytes(b'unlisted')
            else:
                saved = directory.with_name('original-nested')
                directory.rename(saved)
                if damage == 'symlink':
                    directory.symlink_to(saved, target_is_directory=True)
                else:
                    shutil.copytree(saved, directory)
        return value
    monkeypatch.setattr(module, 'file_digest', changed)
    with pytest.raises(ValueError, match='physical directories|directory changed'):
        archive.read(revisions[0])


def test_entry_audit_reuses_directory_stats_but_rechecks_before_return(tmp_path, monkeypatch):
    archive, revisions, _ = archive_fixture(tmp_path, depth=1)
    folder = archive.root / 'entries' / revisions[0]
    counts = Counter()
    original = Path.lstat
    def counted(path, *args, **kwargs):
        counts[path] += 1
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'lstat', counted)
    archive.read(revisions[0])
    # Root, evidence and nested directories each get an initial and final
    # identity check, regardless of how many files share their path prefix.
    # Closure enumeration independently checks the evidence root once.
    assert {p: counts[p] for p in (folder, folder / 'evidence', folder / 'evidence/nested')} == {
        folder: 2, folder / 'evidence': 3, folder / 'evidence/nested': 2}
