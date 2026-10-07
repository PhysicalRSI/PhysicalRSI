import base64
import hashlib
import importlib.metadata
from pathlib import Path
from PhysicalRSI_baselines.embodied_goodharts_law import environment_inventory as inventory


def test_inventory_reports_corruption_missing_files_and_editable_sources(tmp_path, monkeypatch):
    metadata = tmp_path / 'fixture-1.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Name: fixture\nVersion: 1.0\n')
    original = b'original module'
    expected = base64.urlsafe_b64encode(hashlib.sha256(original).digest()).rstrip(b'=').decode()
    (tmp_path / 'module.py').write_bytes(b'changed module')
    (metadata / 'direct_url.json').write_text('{"url":"file:///source/fixture","dir_info":{"editable":true}}')
    (metadata / 'RECORD').write_text(
        f'module.py,sha256={expected},{len(original)}\n'
        'missing.py,,\n'
        'cache.pyc,,\n'
        'fixture-1.0.dist-info/direct_url.json,,\n')
    distribution = importlib.metadata.Distribution.at(metadata)
    monkeypatch.setattr(inventory.importlib.metadata, 'distributions', lambda: [distribution])
    result = inventory.capture()
    assert result['complete_execution_closure_attested'] is False
    problems = {x['problem'] for x in result['issues']}
    assert 'RECORD-hash-mismatch' in problems
    assert 'RECORD-size-mismatch' in problems
    assert 'FileNotFoundError' in problems
    assert not any(x.get('path', '').endswith('cache.pyc') for x in result['issues'])
    files = result['distributions'][0]['files']
    module = next(x for x in files if x['distribution_path'] == 'module.py')
    assert module['sha256'] == hashlib.sha256(b'changed module').hexdigest()
    assert result['editable_sources_requiring_separate_inventory'] == [
        {'distribution': 'fixture', 'source_url': 'file:///source/fixture'}]
