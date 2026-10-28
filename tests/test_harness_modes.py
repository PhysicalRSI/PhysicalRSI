"""Mode-only drift must not alias a checked runnable artifact revision."""
from copy import deepcopy
from pathlib import Path
import shutil
import stat

import pytest

from PhysicalRSI_core.harness_tree import HarnessTree
from PhysicalRSI_core.infra.owned_files import OwnedFiles
from PhysicalRSI_core.infra.storage import digest, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import artifacts
from PhysicalRSI_core.self_harness.artifacts import (
    EvolutionSpec, HarnessSnapshots, bind_harness_modes, harness_identity, verify_harness,
)
from test_core_versions import seed


def mode(path):
    return stat.S_IMODE(Path(path).stat().st_mode)


def executable(root):
    value = seed(root)
    script = root / 'tool.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    script.chmod(0o755)
    value['components']['tools'][script.name] = file_digest(script)
    return bind_harness_modes(value)


def changed(parent, root, name='tool.sh', permission=0o644):
    shutil.copytree(parent['root'], root)
    value = dict(deepcopy(parent), root=str(root), id=root.name)
    (root / name).chmod(permission)
    value['file_modes'][name] = permission
    return value


def test_mode_changes_have_distinct_revisions_and_cannot_rebind_drift(tmp_path):
    parent = executable(tmp_path / 'parent')
    child = changed(parent, tmp_path / 'child')
    assert parent['components'] == child['components']
    assert verify_harness(parent) != verify_harness(child)
    assert EvolutionSpec(('tools',)).compare(parent, child) == {'tools': ['tool.sh']}
    (Path(parent['root']) / 'tool.sh').chmod(0o644)
    with pytest.raises(ValueError, match='file mode'):
        verify_harness(parent)
    with pytest.raises(ValueError, match='file mode'):
        bind_harness_modes(parent)


@pytest.mark.parametrize('component', ['foundation', 'weights', 'control'])
def test_mode_mutations_obey_the_component_boundary(tmp_path, component):
    parent = executable(tmp_path / 'parent')
    child = changed(parent, tmp_path / 'child', component + '.json', 0o755)
    with pytest.raises(ValueError, match='fixed evolution'):
        EvolutionSpec(('tools',)).compare(parent, child)


def test_legacy_identities_stay_readable_without_silent_upgrade(tmp_path):
    legacy = seed(tmp_path / 'legacy')
    expected = digest(dict(components=legacy['components']))
    assert verify_harness(legacy) == expected
    bound = bind_harness_modes(legacy)
    assert verify_harness(bound) != expected
    with pytest.raises(ValueError, match='add or drop'):
        EvolutionSpec().compare(legacy, bound)
    with pytest.raises(ValueError, match='add or drop'):
        EvolutionSpec().compare(bound, legacy)


def test_owned_capture_requires_the_previously_checked_permission_map(tmp_path):
    source = executable(tmp_path / 'source')
    script = Path(source['root']) / 'tool.sh'
    store = OwnedFiles(tmp_path / 'store')
    files = {'tool.sh': dict(path=str(script), sha256=file_digest(script))}
    script.chmod(0o644)
    with pytest.raises(ValueError, match='checked declaration'):
        store.capture({'test': 1}, files, expected_modes={'tool.sh': 0o755})
    assert not store.root.exists()
    for invalid in ({}, {'tool.sh': True}, {'tool.sh': 0o4755}, {'tool.sh': -1}, {'wrong': 0o644}):
        with pytest.raises(ValueError, match='coverage'):
            store.capture({'test': 1}, files, expected_modes=invalid)
    with pytest.raises(ValueError, match='coverage'):
        store.capture({'test': 1}, files, bind_modes=False, expected_modes={'tool.sh': 0o644})
    script.chmod(0o755)
    revision = store.capture({'test': 1}, files, expected_modes={'tool.sh': 0o755})
    assert store.read(revision)['file_modes'] == {'tool.sh': 0o755}


@pytest.mark.parametrize('invalid', ['missing', 'extra', 'bool', 'negative', 'special', 'null'])
def test_mode_declaration_requires_exact_regular_permission_coverage(tmp_path, invalid):
    value = executable(tmp_path / 'parent')
    if invalid == 'missing':del value['file_modes']['tool.sh']
    elif invalid == 'extra':value['file_modes']['foreign'] = 0o644
    elif invalid == 'null':value['file_modes'] = None
    else:value['file_modes']['tool.sh'] = {'bool':True, 'negative':-1, 'special':0o4755}[invalid]
    with pytest.raises(ValueError, match='file_modes'):
        verify_harness(value)


def test_mode_only_snapshot_repair_preserves_lineage_and_quarantines_drift(tmp_path):
    source = executable(tmp_path / 'source')
    state = HarnessState(tmp_path / 'state')
    head = state.initialize(source, policy={}, scope='software')
    script = Path(head['harness']['root']) / 'tool.sh'
    script.chmod(0o644)
    with pytest.raises(ValueError, match='file mode'):
        state.resolve()
    with pytest.raises(ValueError, match='file mode'):
        state.snapshots.capture(source)
    receipt = state.restore_snapshot(head['revision'], revision=head['revision'], source=source, reason='restore executable mode')
    assert receipt['restored'] and mode(script) == 0o755
    assert mode(Path(receipt['quarantine']) / 'tool.sh') == 0o644
    assert state.resolve() == head
    shutil.rmtree(source['root'])
    assert state.restore_snapshot(head['revision'], revision=head['revision'], source=source, reason='restore executable mode') == receipt


@pytest.mark.parametrize('boundary', ['quarantine', 'publish'])
def test_mode_repair_recovers_after_either_directory_rename(tmp_path, monkeypatch, boundary):
    source = executable(tmp_path / 'source')
    snapshots = HarnessSnapshots(tmp_path / 'snapshots')
    captured = snapshots.capture(source)
    script = Path(captured['root']) / 'tool.sh';script.chmod(0o644)
    real = artifacts.os.rename
    def interrupted(src, dst):
        real(src, dst)
        if Path(dst).name == ('quarantine' if boundary == 'quarantine' else Path(captured['root']).name):
            raise RuntimeError('after durable mode repair rename')
    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, 'rename', interrupted)
        with pytest.raises(RuntimeError, match='durable mode'):
            snapshots.restore(captured, source, operation={'mode-repair':1})
    shutil.rmtree(source['root'])
    receipt = snapshots.restore(captured, source, operation={'mode-repair':1})
    assert receipt['restored'] and mode(script) == 0o755
    assert snapshots.restore(captured, source, operation={'mode-repair':1}) == receipt


def test_mode_repair_rejects_a_byte_identical_but_wrong_mode_source(tmp_path):
    source = executable(tmp_path / 'source')
    snapshots = HarnessSnapshots(tmp_path / 'snapshots')
    captured = snapshots.capture(source)
    (Path(captured['root']) / 'tool.sh').chmod(0o644)
    wrong = changed(source, tmp_path / 'wrong')
    with pytest.raises(ValueError, match='exact declared closure'):
        snapshots.restore(captured, wrong, operation={'mode-repair':1})


def test_owned_files_pin_preserve_and_detect_modes_with_legacy_reads(tmp_path):
    source = executable(tmp_path / 'source')
    path = Path(source['root']) / 'tool.sh'
    files = {'tool.sh':dict(path=str(path), sha256=file_digest(path))}
    store = OwnedFiles(tmp_path / 'owned')
    revision = store.capture({'test':1}, files)
    record = store.read(revision)
    assert record['schema'] == 'physicalrsi.owned-files/v2'
    assert record['file_modes'] == {'tool.sh':0o755}
    stored = store.folder(revision) / 'files/tool.sh'
    assert mode(stored) == 0o755
    stored.chmod(0o644)
    with pytest.raises(ValueError, match='file mode'):
        store.read(revision)
    path.chmod(0o644)
    assert store.capture({'test':1}, files) != revision
    legacy = store.capture({'test':1}, files, bind_modes=False)
    assert store.read(legacy)['schema'] == 'physicalrsi.owned-files/v1'


def test_owned_capture_rejects_source_permission_change_during_copy(tmp_path, monkeypatch):
    source = executable(tmp_path / 'source')
    path = Path(source['root']) / 'tool.sh'
    store = OwnedFiles(tmp_path / 'owned')
    original = shutil.copyfileobj
    def copy(*args, **kwargs):
        original(*args, **kwargs);path.chmod(0o644)
    monkeypatch.setattr(shutil, 'copyfileobj', copy)
    with pytest.raises(ValueError, match='mode or bytes changed'):
        store.capture({}, {'tool.sh':dict(path=str(path), sha256=file_digest(path))})
    assert not list(store.root.glob('*/record.json'))


def test_tree_keeps_same_bytes_with_different_modes_separate(tmp_path):
    parent = executable(tmp_path / 'parent')
    child = changed(parent, tmp_path / 'child')
    candidate = HarnessTree.create({'fallback':dict(parent=None, when={}, harness=parent),
        'restricted':dict(parent='fallback', when={'task':'restricted'}, harness=child)}, keys=['task'],
        output=tmp_path / 'tree', name='mode-tree', evolution=EvolutionSpec(('tools', 'skill_selection')))
    tree = HarnessTree(candidate)
    assert tree.body['schema'] == 'physicalrsi.harness-tree/v2'
    assert len(list((tree.root / 'objects/tools').glob(file_digest(Path(parent['root']) / 'tool.sh') + '-*'))) == 2
    for node, expected in [('fallback', 0o755), ('restricted', 0o644)]:
        leaf = tree.materialize(node, tmp_path / ('materialized-' + node))
        assert mode(Path(leaf['root']) / 'tool.sh') == expected
        assert verify_harness(leaf) == digest(tree.leaf_identity(node))
    stored = tree.materialize('fallback', tmp_path / 'again')
    (Path(stored['root']) / 'tool.sh').chmod(0o644)
    with pytest.raises(ValueError, match='file mode'):
        tree.materialize('fallback', tmp_path / 'again')
