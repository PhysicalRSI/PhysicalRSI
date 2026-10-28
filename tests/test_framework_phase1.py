from pathlib import Path
import pytest
from examples.robodyna.skills import estimate_motion, move_to_target, predict_intercept
from examples.robodyna.study import run
from physicalrsi import SkillRegistry, compose_sequence
from physicalrsi.reporting import ReportBundle
from PhysicalRSI_core.infra.storage import read_json, file_digest, digest, atomic_json
from physicalrsi import CallableSkill, SkillDescriptor
from PhysicalRSI_Autoresearch.records import ResearchRecord

OBSERVATION = dict(position=3., previous_position=1., sample_dt=1., horizon=2.)


def test_composed_skill_load_after_registry_restart(tmp_path):
    skill = compose_sequence('intercept', estimate_motion(), predict_intercept(), move_to_target())
    key = SkillRegistry(tmp_path).register(skill)
    assert key.startswith('record-')
    stored = read_json(tmp_path/'skills'/(key+'.json'))
    assert stored['schema'] == 'physicalrsi.skill-record/v2'
    assert stored['composition'] == skill.composition_identity()
    assert stored['implementation'] == skill.executable_identity()
    reopened = SkillRegistry(tmp_path)
    with pytest.raises(RuntimeError, match='trusted implementation'):
        reopened.load(key).execute(OBSERVATION)
    loaded = reopened.load(key, implementation=skill)
    assert loaded.execute(OBSERVATION) == {'target_position':7.}
    assert loaded.composition_identity() == skill.composition_identity()
    assert loaded.executable_identity() == skill.executable_identity()
    wrong = compose_sequence('intercept', estimate_motion(), predict_intercept(gain=.5), move_to_target())
    rebound = reopened.load(key, implementation=wrong)
    assert rebound.execute(OBSERVATION) == {'target_position': 5.}
    assert rebound.executable_identity() == wrong.executable_identity()
    assert rebound.stored_record() == stored
    path = tmp_path/'skills'/(key+'.json')
    path.write_text('{}')
    with pytest.raises(ValueError, match='Unknown skill schema'):
        reopened.load(key)


def test_registry_allows_same_descriptor_with_different_executable(tmp_path):
    descriptor = SkillDescriptor(
        'identity-check', 'revision-1', {'name': 'input'}, {'name': 'output'})
    first = CallableSkill(
        descriptor, lambda value, context: value,
        executable_identity={'schema': 'test-executable/v1', 'revision': 'first'})
    second = CallableSkill(
        descriptor, lambda value, context: value,
        executable_identity={'schema': 'test-executable/v1', 'revision': 'second'})
    key = SkillRegistry(tmp_path).register(first)
    assert SkillRegistry(tmp_path).load(key, implementation=second).execute(3) == 3


def test_registry_loads_legacy_descriptor_record(tmp_path):
    skill = estimate_motion()
    record = skill.identity()
    key = digest(record)
    atomic_json(tmp_path/'skills'/(key+'.json'), record)
    loaded = SkillRegistry(tmp_path).load(key, implementation=skill)
    assert loaded.execute(OBSERVATION) == {
        'position': 3., 'velocity': 2., 'horizon': 2.
    }


def test_skill_core_forwards_raw_values_without_contract_or_context_gate():
    descriptor = SkillDescriptor('raw-core', 'v1', {}, {})
    seen = []
    skill = CallableSkill(
        descriptor,
        lambda value, context: seen.append((value, context)) or value + 1,
    )
    assert skill.execute(2) == 3
    assert seen[0][0] == 2 and hasattr(seen[0][1], 'check')
    assert 'source_digest' not in skill.executable_identity()


def test_study_resume_without_new_trials(tmp_path, monkeypatch):
    root = tmp_path/'study'
    result = run(root)
    assert ResearchRecord(root/'research').status()['status'] == 'completed'
    before = {str(p): file_digest(p) for p in root.rglob('trajectory.json')}
    assert len(before) >= 6
    from physicalrsi.environments import RobodynaFixture
    def forbidden(*args, **kwargs):
        raise AssertionError('Restart must not reset the environment')
    monkeypatch.setattr(RobodynaFixture, 'reset', forbidden)
    assert run(root) == result
    assert before == {str(p):file_digest(p) for p in root.rglob('trajectory.json')}
    result_body = read_json(root/'campaign-result.json')
    assert result_body['current']['decision']['decision'] == 'inherit_child'
    config = read_json(root/'campaign/config.json')
    record = ResearchRecord(root/'research').read()
    declaration = read_json(root/'research/composition.json')
    assert config['context']['composition_revision'] == declaration['revision']
    assert str(root/'research/composition.json') in record['body']['evidence']
    bundle = ReportBundle(root/'report')
    assert 'report.md' in bundle.verify()['artifacts']
    assert not list(root.rglob('*.mp4'))
    with pytest.raises(ValueError):
        run(root, candidate_gain=.5)


def test_bundle_rejects_escape_and_post_finalization_write(tmp_path):
    bundle = ReportBundle(tmp_path)
    with pytest.raises(ValueError):
        bundle.write_text('../outside', 'no')
    bundle.write_text('run.log','actual log')
    bundle.finalize(title='Fixture', summary='No physical qualification.')
    with pytest.raises(ValueError, match='finalized'):
        bundle.write_text('new','no')
    (tmp_path/'run.log').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        bundle.verify()


def test_resume_after_campaign_before_result_collection(tmp_path, monkeypatch):
    root = tmp_path/'study'
    original = ResearchRecord.record_result
    def interrupted(*args, **kwargs):
        raise RuntimeError('simulated collection interruption')
    monkeypatch.setattr(ResearchRecord, 'record_result', interrupted)
    with pytest.raises(RuntimeError, match='collection interruption'):
        run(root)
    assert ResearchRecord(root/'research').status()['status'] == 'awaiting_result'
    before = {str(p):file_digest(p) for p in root.rglob('trajectory.json')}
    monkeypatch.setattr(ResearchRecord, 'record_result', original)
    from physicalrsi.environments import RobodynaFixture
    monkeypatch.setattr(RobodynaFixture, 'reset', lambda *args: pytest.fail('duplicate trial'))
    run(root)
    assert before.items() <= {str(p):file_digest(p) for p in root.rglob('trajectory.json')}.items()
    assert ResearchRecord(root/'research').status()['status'] == 'completed'


def test_only_predictor_component_changes(tmp_path):
    root = tmp_path/'study'
    run(root)
    parent = read_json(root/'campaign/rounds/round-0001/execution/parent.json')['harness']
    chosen = read_json(root/'campaign-result.json')['current']['harness']
    changed = {kind for kind in parent['components'] if parent['components'][kind] != chosen['components'][kind]}
    assert changed == {'skills'}
    assert set(chosen['components']['skills']) == {'PredictIntercept.json'}


def test_nested_compositions_reuse_core_contract_semantics():
    predictor = compose_sequence('predictor', estimate_motion(), predict_intercept())
    whole = compose_sequence('whole', predictor, move_to_target())
    assert whole.execute(OBSERVATION) == {'target_position': 7.}


def test_composition_core_accepts_untyped_values():
    first = CallableSkill(SkillDescriptor('first', 'v1', {}, {}), lambda value, context: value + 1)
    second = CallableSkill(SkillDescriptor('second', 'v1', {}, {}), lambda value, context: value * 2)
    assert compose_sequence('raw', first, second).execute(3) == 8
