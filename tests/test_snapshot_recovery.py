from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState, StateConflict
from PhysicalRSI_core.self_harness import artifacts
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.retention import RetentionSelector
from test_core_versions import seed


def fixture(root):
    source = seed(root / 'source')
    state = HarnessState(root / 'state')
    current = state.initialize(source, policy={}, scope='software')
    damaged = Path(current['harness']['root']) / 'control.json'
    damaged.write_text('damaged original bytes')
    return state, current, source, damaged


def restore(state, current, source):
    return state.restore_snapshot(current['revision'], revision=current['revision'],
                                  source=source, reason='verified independent replica')


def repair_folder(state):
    folders = list((state.root / 'snapshot-repairs').iterdir())
    assert len(folders) == 1
    return folders[0]


def test_explicit_restore_preserves_head_hash_and_damaged_bytes(tmp_path):
    state, current, source, damaged = fixture(tmp_path)
    with pytest.raises(ValueError, match='Changed or missing'):
        state.snapshots.capture(source)
    history = {str(p): file_digest(p) for p in (state.root / 'history').glob('*.json')}
    receipt = restore(state, current, source)
    assert receipt['restored'] and receipt['snapshot_sha256'] == current['freeze_sha256']
    assert receipt['qualification'] is None and state.resolve() == current
    assert (Path(receipt['quarantine']) / 'control.json').read_text() == 'damaged original bytes'
    assert damaged.stat().st_ino != (Path(source['root']) / 'control.json').stat().st_ino
    assert {str(p): file_digest(p) for p in (state.root / 'history').glob('*.json')} == history
    shutil.rmtree(source['root'])
    assert restore(HarnessState(state.root), current, source) == receipt
    assert state.snapshots.capture(current['harness']) == current['harness']


@pytest.mark.parametrize('boundary', ['copy', 'staged', 'prepared', 'quarantined', 'published', 'completed'])
def test_restore_recovers_publication_interruptions(tmp_path, monkeypatch, boundary):
    state, current, source, damaged = fixture(tmp_path)
    real_copy, real_rename, real_write = artifacts.shutil.copyfile, artifacts.os.rename, artifacts.atomic_json
    hit = False

    def interrupt():
        nonlocal hit
        hit = True
        raise RuntimeError('snapshot recovery interrupted')

    def copy(*args, **kwargs):
        result = real_copy(*args, **kwargs)
        if boundary == 'copy' and not hit:
            interrupt()
        return result

    def rename(src, dst):
        result = real_rename(src, dst)
        expected = {'staged': 'prepared', 'quarantined': 'quarantine', 'published': current['freeze_sha256']}
        if not hit and Path(dst).name == expected.get(boundary):
            interrupt()
        return result

    def write(path, value):
        result = real_write(path, value)
        expected = {'prepared': 'prepared.json', 'completed': 'completed.json'}
        if not hit and Path(path).name == expected.get(boundary):
            interrupt()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.shutil, 'copyfile', copy)
        patch.setattr(artifacts.os, 'rename', rename)
        patch.setattr(artifacts, 'atomic_json', write)
        with pytest.raises(RuntimeError, match='recovery interrupted'):
            restore(state, current, source)
    assert hit and read_json(state.root / 'current.json')['revision'] == current['revision']
    if boundary in {'prepared', 'quarantined', 'published', 'completed'}:
        shutil.rmtree(source['root'])  # The durable prepared copy owns the retry.
    receipt = restore(HarnessState(state.root), current, source)
    assert state.resolve() == current and receipt['restored']
    assert (Path(receipt['quarantine']) / 'control.json').read_text() == 'damaged original bytes'
    assert verify_harness(current['harness']) == current['freeze_sha256']
    assert not list(repair_folder(state).glob('.restore-*'))
    assert restore(state, current, source) == receipt


@pytest.mark.parametrize('invalid', ['bytes', 'declaration', 'overlap', 'stale', 'nonancestor', 'reason'])
def test_restore_rejects_invalid_inputs_without_changing_snapshot(tmp_path, invalid):
    state, current, source, damaged = fixture(tmp_path)
    expected, target, reason = current['revision'], current['revision'], 'reviewed restore'
    source = deepcopy(source)
    if invalid == 'bytes':
        (Path(source['root']) / 'control.json').write_text('wrong source')
    elif invalid == 'declaration':
        source['components']['control']['control.json'] = 'a' * 64
    elif invalid == 'overlap':
        source['root'] = str(state.snapshots.root)
    elif invalid == 'stale':
        expected = 'a' * 64
    elif invalid == 'nonancestor':
        target = 'a' * 64
    else:
        reason = '   '
    with pytest.raises(StateConflict if invalid == 'stale' else ValueError):
        state.restore_snapshot(expected, revision=target, source=source, reason=reason)
    assert damaged.read_text() == 'damaged original bytes'
    assert read_json(state.root / 'current.json')['revision'] == current['revision']
    assert not list((state.root / 'snapshot-repairs').glob('*/quarantine'))


@pytest.mark.parametrize('missing', [False, True])
def test_restore_missing_snapshot_or_quarantined_root_symlink(tmp_path, missing):
    state, current, source, _ = fixture(tmp_path)
    target = Path(current['harness']['root'])
    shutil.rmtree(target)
    if not missing:
        target.symlink_to(source['root'], target_is_directory=True)
    receipt = restore(state, current, source)
    assert state.resolve() == current and not target.is_symlink()
    assert verify_harness(source) == current['freeze_sha256']
    if missing:
        assert receipt['quarantine'] is None and receipt['quarantine_sha256'] is None
    else:
        assert Path(receipt['quarantine']).is_symlink()
        assert Path(receipt['quarantine']).readlink() == Path(source['root'])


def test_healthy_snapshot_restore_is_recorded_noop(tmp_path):
    source = seed(tmp_path / 'source')
    state = HarnessState(tmp_path / 'state')
    current = state.initialize(source, policy={}, scope='software')
    inode = (Path(current['harness']['root']) / 'control.json').stat().st_ino
    shutil.rmtree(source['root'])
    receipt = restore(state, current, source)
    assert not receipt['restored'] and receipt['quarantine'] is None
    assert (Path(current['harness']['root']) / 'control.json').stat().st_ino == inode
    assert restore(state, current, source) == receipt and state.resolve() == current


@pytest.mark.parametrize('damage', ['snapshot', 'quarantine', 'request', 'plan', 'completed'])
def test_completed_repair_checks_its_evidence_before_retry(tmp_path, damage):
    state, current, source, damaged = fixture(tmp_path)
    receipt = restore(state, current, source)
    folder = repair_folder(state)
    if damage == 'snapshot':
        damaged.write_text('second damage requires a new explicit operation')
    elif damage == 'quarantine':
        (Path(receipt['quarantine']) / 'control.json').write_text('lost forensic bytes')
    else:
        name = {'request': 'request', 'plan': 'prepared', 'completed': 'completed'}[damage]
        value = read_json(folder / (name + '.json'))
        value['unexpected'] = True
        atomic_json(folder / (name + '.json'), value)
    with pytest.raises(ValueError):
        restore(state, current, source)
    assert read_json(state.root / 'current.json')['revision'] == current['revision']


def test_repair_rejects_target_changes_after_preparation(tmp_path, monkeypatch):
    state, current, source, damaged = fixture(tmp_path)
    real_write = artifacts.atomic_json
    def write(path, value):
        real_write(path, value)
        if Path(path).name == 'prepared.json':
            raise RuntimeError('prepared')
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, 'atomic_json', write)
        with pytest.raises(RuntimeError):
            restore(state, current, source)
    damaged.write_text('changed again before quarantine')
    with pytest.raises(ValueError, match='changed before quarantine'):
        restore(state, current, source)
    assert damaged.read_text() == 'changed again before quarantine'
    assert not (repair_folder(state) / 'quarantine').exists()


def test_repair_rejects_external_history_root(tmp_path):
    source = seed(tmp_path / 'source')
    state = HarnessState(tmp_path / 'state')
    current = state.initialize(source, policy={}, scope='software')
    # Legacy metadata fixture, not an evaluated selection.
    body = {key: value for key, value in current.items() if key != 'revision'}
    external = state._publish(dict(body, harness=source, previous=current['revision']))
    with pytest.raises(ValueError, match='owned snapshot'):
        restore(state, external, source)
    assert not (state.root / 'snapshot-repairs').exists()


def test_restoration_does_not_excuse_damaged_lineage_metadata(tmp_path):
    state, current, source, damaged = fixture(tmp_path)
    path = state.root / 'history' / (current['revision'] + '.json')
    value = read_json(path)
    value['scope'] = 'different'
    atomic_json(path, value)
    with pytest.raises(ValueError, match='Lineage record changed'):
        restore(state, current, source)
    assert damaged.read_text() == 'damaged original bytes'
    assert not (state.root / 'snapshot-repairs').exists()


def test_restored_ancestor_allows_new_retention_round_but_not_invalid_evidence(tmp_path):
    from PhysicalRSI_core.self_harness.archive import CandidateArchive
    from PhysicalRSI_demos.harness_tree import HarnessTree, TASKS, setup
    campaign, initial, suite, quota = setup(tmp_path, max_trials=80)
    result = campaign.run(initial)
    selected = result['current']
    archive = CandidateArchive(tmp_path / 'archive')
    entries = [archive.read(p.parent.name) for p in (archive.root / 'entries').glob('*/record.json')]
    replica = next(row['harness'] for entry in entries for row in entry['candidates'].values()
                   if row['freeze_sha256'] == selected['freeze_sha256'])
    logical = next(p for p in selected['harness']['components']['control'] if p.startswith('objects/'))
    (Path(selected['harness']['root']) / logical).write_text('corruption')
    restored = campaign.state.rollback(selected['revision'], reason='damaged snapshot')
    with pytest.raises(ValueError, match='Changed or missing'):
        RetentionSelector(state=campaign.state, parent_revision=restored['revision'],
                          maximum_regression={task: 0. for task in TASKS})
    receipt = campaign.state.restore_snapshot(restored['revision'], revision=selected['revision'],
        source=replica, reason='independent archive replica')
    assert receipt['restored'] and campaign.state.resolve() == restored
    selector = RetentionSelector(state=campaign.state, parent_revision=restored['revision'],
                                 maximum_regression={task: 0. for task in TASKS})
    assert set(selector.view['high_water'].values()) == {1.}
    context = dict(schema='physicalrsi.round-context/v1', index=3, parent_revision=restored['revision'],
        memory_revision=result['memory_revision'], previous_round=result['revision'], scope=deepcopy(campaign.context),
        lessons=campaign.memory.retrieve(result['memory_revision'], campaign.context),
        budget=dict(max_rounds=4, remaining_rounds=1))
    loop = campaign.build_loop(tmp_path / 'recovery-round', context)
    recovered = loop.run()
    assert recovered['freeze_sha256'] == selected['freeze_sha256'] and recovered['revision'] != selected['revision']
    assert len(HarnessTree(recovered['harness']).body['nodes']) == 3
    assert len(suite.resets) == quota.status()['reserved_trials'] == 80
    assert campaign.build_loop(tmp_path / 'recovery-round', context).run() == recovered
    assert len(suite.resets) == 80
    trajectory = next(Path(p) for p in selected['evidence'] if p.endswith('trajectory.json'))
    trajectory.write_text('damaged evidence is never recovered from a harness snapshot')
    with pytest.raises(ValueError, match='Committed evidence changed'):
        RetentionSelector(state=campaign.state, parent_revision=recovered['revision'],
                          maximum_regression={task: 0. for task in TASKS})
