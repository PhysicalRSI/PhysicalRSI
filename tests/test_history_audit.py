from collections import Counter
from copy import deepcopy
from pathlib import Path
import os

import pytest

from PhysicalRSI_core.infra.storage import file_digest
from PhysicalRSI_core.lineage import HarnessState
from test_core_versions import seed


def history_fixture(root, *, defect=None):
    # Metadata-only persistence fixture; these are not evaluated selections.
    state = HarnessState(root / 'state')
    first = state.initialize(seed(root / 'source'), policy={}, scope='software')
    evidence = root / 'observation.bin'
    evidence.write_bytes(b'checked historical observation')
    body = {k: v for k, v in first.items() if k != 'revision'}
    previous = first
    for index in range(3):
        value = deepcopy(body)
        value.update(previous=previous['revision'], action='assessment', evidence={str(evidence): file_digest(evidence)})
        value['harness']['id'] = 'alias-' + str(index)
        if index == 1:
            if defect == 'evidence':
                value['evidence'][str(evidence)] = '0' * 64
            elif defect == 'harness':
                value['harness']['components']['control']['control.json'] = '0' * 64
            elif defect == 'identifier':
                value['harness']['id'] = '../invalid'
        previous = state._publish(value)
    return state, previous, evidence


def test_history_audits_shared_files_once_and_preserves_newest_first_order(tmp_path, monkeypatch):
    state, current, evidence = history_fixture(tmp_path)
    counts = Counter()
    original = Path.open
    def opened(path, mode='r', *args, **kwargs):
        if 'r' in mode:
            counts[str(path)] += 1
        return original(path, mode, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', opened)
        records = state.history(current['revision'])
    assert len(records) == 4 and records[0] == current and records[-1]['action'] == 'bootstrap'
    assert [r['previous'] for r in records[:-1]] == [r['revision'] for r in records[1:]]
    assert counts[str(evidence)] == 1
    for entries in current['harness']['components'].values():
        for name in entries:
            assert counts[str(Path(current['harness']['root']) / name)] == 1
    records[0]['harness']['components'].clear()
    assert state.history(current['revision'])[0] == current


@pytest.mark.parametrize('defect', ['evidence', 'harness', 'identifier'])
def test_shared_file_does_not_excuse_conflicting_historical_declarations(tmp_path, defect):
    state, current, _ = history_fixture(tmp_path, defect=defect)
    assert state.read(current['revision']) == current
    with pytest.raises(ValueError):
        state.history(current['revision'])


@pytest.mark.parametrize('kind', ['evidence', 'harness', 'metadata'])
def test_each_history_call_detects_same_size_same_timestamp_damage(tmp_path, kind):
    state, current, evidence = history_fixture(tmp_path)
    rows = state.history(current['revision'])
    path = (evidence if kind == 'evidence' else Path(current['harness']['root']) / 'control.json'
            if kind == 'harness' else state.root / 'history' / (rows[1]['revision'] + '.json'))
    before = path.stat()
    data = bytearray(path.read_bytes())
    # Keep JSON parseable for a metadata file so the content-address check is exercised.
    index = data.find(b'alias-1') if kind == 'metadata' else 0
    data[index] = ord('b') if kind == 'metadata' else data[index] ^ 1
    path.write_bytes(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError):
        state.history(current['revision'])


def test_history_requires_a_starting_revision(tmp_path):
    state, _, _ = history_fixture(tmp_path)
    with pytest.raises(ValueError, match='content-addressed'):
        state.history(None)
