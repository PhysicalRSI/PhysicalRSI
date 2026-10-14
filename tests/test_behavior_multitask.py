"""Routing fixtures establish software coverage, never model task competence."""
import dataclasses
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from PhysicalRSI_baselines.behavior_baseline import openpi_backend as backend, seed, service
from PhysicalRSI_core.infra.storage import file_digest, read_json


def specification():
    return dict(kind='openpi-b1k-multitask/v1', task_prompts={'radio': 'turn on radio', 'drawer': 'open drawer'},
                asset_id='full100', checkpoint_files={'assets/full100/norm_stats.json': 'fixture'})


def test_explicit_task_and_complete_norm_binding_required():
    spec = specification()
    assert backend.supported_tasks(spec) == {'radio', 'drawer'}
    assert backend.select_task(spec, 'drawer') == 'drawer'
    for task in (None, 'unknown'):
        with pytest.raises(ValueError, match='served task'):
            backend.select_task(spec, task)
    for changes in ({'task': 'radio'}, {'task_prompts': {}}, {'task_prompts': {'radio': ''}},
                    {'asset_id': '../escape'}, {'checkpoint_files': {}}):
        with pytest.raises(ValueError):
            backend.supported_tasks(dict(spec, **changes))
    assert backend.select_task({'kind': 'openpi-b1k/v1', 'task': 'radio'}) == 'radio'
    argv = service.command('/python', '/candidate', port=8023, episode_seconds=10, task='drawer')
    assert argv[argv.index('--task') + 1] == 'drawer'


def test_multitask_seed_freezes_prompts_and_shared_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(seed.subprocess, 'check_output', lambda *a, **k: seed.OPENPI_REVISION)
    monkeypatch.setattr(seed.subprocess, 'run', lambda *a, **k: None)
    source = tmp_path / 'source'; source.mkdir(); (source / 'uv.lock').write_text('fixture')
    checkpoint = tmp_path / 'checkpoint'; stats = checkpoint / 'assets/full100/norm_stats.json'
    stats.parent.mkdir(parents=True); stats.write_text('{}')
    (checkpoint / 'params').write_text('fixture parameters')
    environment = tmp_path / 'environment'; environment.write_text('fixture')
    prompts = {'radio': 'turn on radio', 'drawer': 'open drawer'}
    candidate = seed.materialize(tmp_path / 'candidate', source=source, checkpoint=checkpoint,
        environment_manifest=environment, task_prompts=prompts, asset_id='full100', repo_id='training/demos')
    spec = read_json(Path(candidate['root']) / 'backend.json')
    assert spec['task_prompts'] == prompts and 'task' not in spec
    assert spec['repo_id'] == 'training/demos' and spec['asset_id'] == 'full100'
    prompts['radio'] = 'changed externally'
    assert read_json(Path(candidate['root']) / 'backend.json')['task_prompts']['radio'] == 'turn on radio'


def test_backend_uses_selected_training_prompt_and_shared_norm(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    checkpoint = tmp_path / 'checkpoint'; stats = checkpoint / 'assets/full100/norm_stats.json'
    stats.parent.mkdir(parents=True); stats.write_text('{}')
    spec = dict(specification(), source_root=str(source), source_revision=backend.OPENPI_REVISION,
        checkpoint_root=str(checkpoint), checkpoint_files={'assets/full100/norm_stats.json': file_digest(stats)},
        adapter_sha256=file_digest(Path(backend.__file__)), training_config='pi05_b1k',
        repo_id='training/demos', control_mode='receding_horizon', action_horizon=16)
    monkeypatch.setattr(backend, 'verify_tokenizer', lambda _: None)
    monkeypatch.setattr(backend.subprocess, 'check_output', lambda *a, **k: backend.OPENPI_REVISION)
    monkeypatch.setattr(backend.subprocess, 'run', lambda *a, **k: None)
    @dataclasses.dataclass
    class Assets:
        asset_id: str
    @dataclasses.dataclass
    class Data:
        repo_id: str = 'old'
        robot_config_name: str = 'old'
        assets: object = None
    @dataclasses.dataclass
    class Config:
        data: object
        model: object
    seen = {}
    def policy(cfg, path, default_prompt):
        seen.update(config=cfg, path=path, prompt=default_prompt)
        return 'model'
    class Wrapper:
        def __init__(self, model, robot, prompt, mode, horizon, max_len):
            seen.update(wrapper_prompt=prompt, horizon=horizon, max_len=max_len)
        def reset(self): seen['reset'] = True
    modules = {
        'openpi': SimpleNamespace(__file__=str(source / '__init__.py')),
        'openpi.configs': SimpleNamespace(),
        'openpi.configs.tasks': SimpleNamespace(TASK_REGISTRY={'b1k': {}}),
        'openpi.policies': SimpleNamespace(policy_config=SimpleNamespace(create_trained_policy=policy)),
        'openpi.shared': SimpleNamespace(),
        'openpi.shared.eval_b1k_wrapper': SimpleNamespace(B1KPolicyWrapper=Wrapper),
        'openpi.training': SimpleNamespace(config=SimpleNamespace(
            get_config=lambda _: Config(Data(), SimpleNamespace(action_horizon=32)), AssetsConfig=Assets)),
    }
    for name, value in modules.items(): monkeypatch.setitem(sys.modules, name, value)
    actual = backend.OpenPIBackend(spec, task='drawer')
    assert seen['prompt'] == seen['wrapper_prompt'] == 'open drawer'
    assert seen['config'].data.assets.asset_id == 'full100'
    assert seen['config'].data.repo_id == 'training/demos'
    assert actual.identity() == spec
    actual.reset('drawer'); assert seen['reset']
    with pytest.raises(ValueError): actual.reset('radio')
