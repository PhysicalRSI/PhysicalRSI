from copy import deepcopy
from pathlib import Path
import pytest

from PhysicalRSI_baselines.robotworld.policy_bundle import fixed_components, materialize, read_bundle
from PhysicalRSI.Embodied_Harness.memory.store import Snapshot
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import CLOSURE


@pytest.fixture
def seed(tmp_path):
    root = tmp_path / 'seed'
    components = {}
    for kind in CLOSURE:
        p = root / (kind + '.json')
        atomic_json(p, {'fixed': kind})
        components[kind] = {p.name: file_digest(p)}
    candidate = dict(id='fixture', root=str(root), components=components)
    def change(component, name, value):
        atomic_json(root / name, value)
        components[component][name] = file_digest(root / name)
    change('prompts', 'prompts.json', {'guidance': 'Use public observations.'})
    change('skills', 'skills.json', {'files': {'helper.py': 'raise RuntimeError("never execute on host")\n'}})
    change('memory_rules', 'memory.json', {'lessons': ['Inspect feedback after each action.']})
    return candidate, fixed_components(candidate), change


def test_skills_are_data_and_memory_has_a_verified_revision(seed, tmp_path):
    candidate, fixed, _ = seed
    destination = tmp_path / 'policy'
    report = materialize(candidate, fixed=fixed, destination=destination)
    assert (destination / 'skills/helper.py').read_text().startswith('raise RuntimeError')
    assert Snapshot(destination / 'memory', report['memory_revision']).read()['lessons']
    assert read_json(destination / 'context.json')['candidate_sha256'] == report['candidate_sha256']
    assert report['qualification'] is None


def test_fixed_configuration_cannot_be_edited(seed):
    candidate, fixed, change = seed
    change('configuration', 'configuration.json', {'seed': 'different'})
    with pytest.raises(ValueError, match='fixed'):
        read_bundle(candidate, fixed=fixed)


def test_skill_path_cannot_escape_agent_input_directory(seed):
    candidate, fixed, change = seed
    change('skills', 'skills.json', {'files': {'../escape.py': 'pass'}})
    with pytest.raises(ValueError):
        read_bundle(candidate, fixed=fixed)


def test_memory_edit_creates_new_revision_preserving_old_snapshot(seed, tmp_path):
    candidate, fixed, change = seed
    first = materialize(candidate, fixed=fixed, destination=tmp_path / 'first')
    old = Snapshot(tmp_path / 'first/memory', first['memory_revision'])
    prior = deepcopy(old.read())
    change('memory_rules', 'memory.json', {'lessons': ['Reobserve after a suspected slip.']})
    second = materialize(candidate, fixed=fixed, destination=tmp_path / 'second')
    assert first['memory_revision'] != second['memory_revision']
    assert old.read() == prior


def test_existing_policy_mount_is_not_overwritten(seed, tmp_path):
    candidate, fixed, _ = seed
    destination = tmp_path / 'policy'
    materialize(candidate, fixed=fixed, destination=destination)
    with pytest.raises(FileExistsError):
        materialize(candidate, fixed=fixed, destination=destination)
