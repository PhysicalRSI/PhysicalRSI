"""Read records emitted by the pre-refactor implementation at fixture.source."""
from pathlib import Path

import pytest

from PhysicalRSI.Embodied_Harness.skills.composition import sequence
from PhysicalRSI_core.contracts import Context
from PhysicalRSI_core.infra.storage import atomic_json, digest, read_json
from PhysicalRSI_core.skill_graph.schema import SkillComposition
from physicalrsi import CallableSkill, SkillDescriptor, SkillRegistry, compose_sequence
from physicalrsi.reporting import ReportBundle
from physicalrsi.research.workflow import ResearchWorkflow


@pytest.fixture
def legacy():
    return read_json(Path(__file__).parent / 'fixtures/persistence_v1.json')


def increment():
    return CallableSkill(
        SkillDescriptor('increment', 'r1', {'name': 'number'}, {'name': 'number'}),
        lambda value, context: value + 1,
    )


@pytest.mark.parametrize('kind,expected', [('primitive', 2), ('composition', 3)])
def test_old_registry_executes_with_legacy_provider_without_rewriting(tmp_path, legacy, kind, expected):
    row = legacy[kind]
    path = tmp_path / 'skills' / (row['key'] + '.json')
    atomic_json(path, row['record'])
    before = path.read_bytes()
    provider = increment()
    if kind == 'composition':
        operation = provider.as_operation()
        provider = CallableSkill.from_operation(sequence('twice', operation, operation))
    assert provider.identity() == row['record']
    registry = SkillRegistry(tmp_path)
    loaded = registry.load(row['key'], implementation=provider)
    assert loaded.execute(1) == expected
    assert loaded.identity() == row['record']
    # v1 registry stored descriptors only; the graph was a separate record.
    assert loaded.composition_identity() is None
    new_key = registry.register(increment())
    assert {item['registry_revision'] for item in registry.list()} == {row['key'], new_key}
    assert path.read_bytes() == before


def test_new_composer_does_not_silently_rebind_old_descriptor(tmp_path, legacy):
    row = legacy['composition']
    path = tmp_path / 'skills' / (row['key'] + '.json')
    atomic_json(path, row['record'])
    before = path.read_bytes()
    skill = increment()
    rebuilt = compose_sequence('twice', skill, skill)
    with pytest.raises(ValueError, match='implementation identity differs'):
        SkillRegistry(tmp_path).load(row['key'], implementation=rebuilt)
    assert path.read_bytes() == before


def test_old_registry_keeps_historical_default_and_explicit_context(tmp_path, legacy):
    row = legacy['primitive']
    atomic_json(tmp_path / 'skills' / (row['key'] + '.json'), row['record'])
    descriptor = dict(row['record'])
    descriptor.pop('schema')
    seen = []
    provider = CallableSkill(SkillDescriptor(**descriptor),
                             lambda value, context: seen.append(context) or value + 1)
    loaded = SkillRegistry(tmp_path).load(row['key'], implementation=provider)
    assert loaded.execute(1) == 2
    assert isinstance(seen[0], Context) and seen[0].episode == 'skill-call'
    context = Context('caller-context')
    assert loaded.execute(1, context) == 2
    assert seen[1] is context


def test_old_graph_preserves_frozen_workflow_evidence(tmp_path, legacy):
    value = legacy['composition']['graph']
    path = tmp_path / 'composition.json'
    atomic_json(path, value)
    before = path.read_bytes()
    restored = SkillComposition.from_dict(read_json(path))
    assert restored.as_dict() == value
    workflow = ResearchWorkflow(restored)
    assert workflow.composition_revision == value['revision']
    workflow.evidence(path)
    assert path.read_bytes() == before


def test_v2_registry_envelope_can_reopen_an_existing_digest_key(tmp_path, legacy):
    provider = compose_sequence('twice', increment(), increment())
    record = provider.registry_record()
    record['composition'] = legacy['composition']['graph']
    key = digest(record)
    path = tmp_path / 'skills' / (key + '.json')
    atomic_json(path, record)
    before = path.read_bytes()
    loaded = SkillRegistry(tmp_path).load(key, implementation=provider)
    assert loaded.stored_record() == record
    assert loaded.registry_record() == provider.registry_record()
    assert loaded.execute(1) == 3
    assert path.read_bytes() == before


def test_old_report_bundle_reopens_and_finalizes_without_rewriting(tmp_path, legacy):
    old = legacy['bundle']
    for name, content in old['files'].items():
        (tmp_path / name).write_text(content)
    before = {name: (tmp_path / name).read_bytes() for name in old['files']}
    bundle = ReportBundle(tmp_path)
    manifest = read_json(tmp_path / 'manifest.json')
    assert bundle.verify() == manifest
    assert bundle.finalize(title=old['title'], summary=old['summary']) == manifest
    assert before == {name: (tmp_path / name).read_bytes() for name in before}


def test_legacy_operation_adapter_preserves_explicit_runtime_context():
    calls = []
    operation = increment().as_operation()
    context = Context('compatibility', execution=lambda op, value, ctx: calls.append((op, ctx)) or 42)
    assert CallableSkill.from_operation(operation).execute(1, context) == 42
    assert calls == [(operation, context)]


def test_registry_encodes_logical_keys_without_aliasing_or_hidden_records(tmp_path):
    registry = SkillRegistry(tmp_path)
    names = [('perception/pose', 'v1'), ('skill@variant', 'v1'), ('skill', 'variant@v1')]
    keys = []
    for name, revision in names:
        skill = CallableSkill(SkillDescriptor(name, revision, {}, {}), lambda value, context: value)
        key = registry.register(skill)
        assert registry.load(key).identity() == skill.identity()
        keys.append(key)
    assert len(set(keys)) == len(names)
    assert len(list((tmp_path / 'skills').glob('*.json'))) == len(names)
    assert {row['registry_revision'] for row in registry.list()} == set(keys)


def test_registry_retains_versions_when_the_same_descriptor_changes(tmp_path):
    descriptor = SkillDescriptor('increment', 'r1', {}, {})
    first = CallableSkill(descriptor, lambda value, context: value + 1)
    second = CallableSkill(descriptor, lambda value, context: value + 2)
    registry = SkillRegistry(tmp_path)
    first_key = registry.register(first)
    path = tmp_path / 'skills' / (first_key + '.json')
    before = path.read_bytes(), path.stat().st_mtime_ns
    second_key = registry.register(second)
    assert second_key != first_key
    assert registry.load(first_key).execute(1) == 2
    assert registry.load(second_key).execute(1) == 3
    restarted = SkillRegistry(tmp_path)
    assert restarted.register(first) == first_key
    assert restarted.register(second) == second_key
    assert len(restarted.list()) == 2
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
