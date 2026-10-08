import json
import pytest
from PhysicalRSI_baselines.robotworld import catalog


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    tasks = [dict(benchmark='robocasa', task='CloseDrawer', default_steps=450,
                  code_control_default=False)]
    def git(args):
        assert args[:3] == ['git', '-C', str(tmp_path)]
        assert args[3] == 'show'
        revision, name = args[4].split(':', 1)
        assert revision == catalog.REVISION
        return json.dumps(tasks if name == catalog.FILES[0] else {}).encode()
    monkeypatch.setattr(catalog.subprocess, 'check_output', git)
    return tmp_path, tasks


def test_inventory_is_idempotent_and_detects_tampering(upstream):
    root, _ = upstream
    workspace = root / 'run'
    first = catalog.freeze(root, workspace)
    assert first == catalog.freeze(root, workspace)
    assert first['body']['qualification'] is None
    path = workspace / 'catalog.json'
    altered = json.loads(path.read_text())
    altered['body']['tasks'][0]['default_steps'] = 1
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='Inventory changed'):
        catalog.freeze(root, workspace)


@pytest.mark.parametrize('field,value', [('default_steps', True), ('default_steps', 0),
                                         ('code_control_default', True)])
def test_protocol_changes_require_review(upstream, field, value):
    root, tasks = upstream
    tasks[0][field] = value
    with pytest.raises(ValueError):
        catalog.snapshot(root)


def test_duplicate_task_rejected(upstream):
    root, tasks = upstream
    tasks.append(dict(tasks[0]))
    with pytest.raises(ValueError, match='unique'):
        catalog.snapshot(root)
