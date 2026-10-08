import json
import sys
from types import ModuleType

import pytest

from PhysicalRSI_baselines.robotworld import robocasa_episode


@pytest.fixture
def native_entry(tmp_path, monkeypatch):
    modules = {}
    for name in ('robocasa', 'robocasa.utils', 'robocasa.utils.dataset_registry',
                 'robocasa.utils.dataset_registry_utils', 'environment',
                 'environment.benchmarks', 'environment.benchmarks.robocasa',
                 'environment.benchmarks.robocasa.policy', 'environment.integrations',
                 'environment.integrations.robocasa_eval'):
        module = modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(modules[parent], child, module)
    modules['robocasa.utils.dataset_registry'].TASK_SET_REGISTRY = {'all_tasks': ['Other', 'CloseDrawer']}
    modules['robocasa.utils.dataset_registry_utils'].get_task_horizon = lambda task: 450
    policy = modules['environment.benchmarks.robocasa.policy']
    policy.BASE = 'Original public observation policy.'
    calls = []
    modules['environment.integrations.robocasa_eval'].main = lambda: calls.append(sys.argv[:])
    case = dict(task='CloseDrawer', task_index=1, task_set='all_tasks', split='pretrain',
                launcher_seed=7, reset_seed=8, horizon=450)
    request = tmp_path / 'request.json'
    policy_root = tmp_path / 'policy'
    policy_root.mkdir()
    (policy_root / 'context.json').write_text(json.dumps(dict(candidate_sha256='frozen', guidance='Reobserve after contact.')))
    def prepare(**changes):
        case.update(changes)
        request.write_text(json.dumps(dict(case=case, candidate_sha256='frozen')))
        monkeypatch.setattr(sys, 'argv', ['entry', '--request', str(request), '--output', str(tmp_path / 'out'),
                                       '--manifest', str(tmp_path / 'build.json'), '--policy-inputs', str(policy_root)])
    prepare()
    return prepare, calls, policy, policy_root


def test_original_evaluator_receives_frozen_reset_without_horizon_override(native_entry):
    _, calls, policy, _ = native_entry
    robocasa_episode.main()
    args = calls[0]
    assert args[args.index('--seed') + 1] == '7'
    assert args[args.index('--num-trials') + 1] == '1'
    assert '--horizon' not in args and '--probe-only' not in args
    assert policy.BASE.startswith('Original public observation policy.')
    assert 'Reobserve after contact.' in policy.BASE


@pytest.mark.parametrize('changes', [dict(horizon=20), dict(reset_seed=7), dict(task_index=0)])
def test_mismatched_native_contract_never_launches(native_entry, changes):
    prepare, calls, _, _ = native_entry
    prepare(**changes)
    with pytest.raises(ValueError, match='frozen request'):
        robocasa_episode.main()
    assert not calls


def test_wrong_candidate_mount_never_launches(native_entry):
    _, calls, _, policy_root = native_entry
    (policy_root / 'context.json').write_text(json.dumps(dict(candidate_sha256='other')))
    with pytest.raises(ValueError, match='candidate'):
        robocasa_episode.main()
    assert not calls
