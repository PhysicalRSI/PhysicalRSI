from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, read_json
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness import archive, campaign, research_campaign
from PhysicalRSI_core.self_harness.compatibility import same_execution
import PhysicalRSI_core.self_harness as loop_module
from PhysicalRSI_demos.rollout_evolution import setup as research_setup
from test_improvement_campaign import setup as improvement_setup


def setup(root, kind):
    return (improvement_setup(root, with_archive=True) if kind == 'improvement'
            else research_setup(root))


@contextmanager
def legacy_writer(monkeypatch):
    """Emit source-bound v1 configs without an explicit execution version."""
    with monkeypatch.context() as patch:
        for module in (loop_module, archive, campaign, research_campaign):
            source, original = Path(module.__file__), module.file_digest
            patch.setattr(module, 'file_digest',
                          lambda path, source=source, original=original:
                          '1' * 64 if Path(path) == source else original(path))
        initialize = SelfHarness.__init__

        def old_config(self, *args, **kwargs):
            initialize(self, *args, **kwargs)
            self.config.pop('engine_version', None)
            if 'campaign_policy' in self.config:
                self.config['campaign_policy'].pop('engine_version', None)
            self.policy = deepcopy(self.config.get('campaign_policy', self.config))
            atomic_json(self.root / 'config.json', self.config)

        patch.setattr(SelfHarness, '__init__', old_config)
        for module, method in ((campaign, 'atomic_json'), (research_campaign, 'freeze_json')):
            write = getattr(module, method)

            def old_campaign(path, value, write=write):
                config = value.get('config', value)
                if config.get('schema') in {'physicalrsi.campaign/v1', 'physicalrsi.research-campaign/v1'}:
                    config.pop('engine_version', None)
                return write(path, value)

            patch.setattr(module, method, old_campaign)
        yield


@pytest.mark.parametrize('kind', ['improvement', 'research'])
@pytest.mark.parametrize('boundary', ['completed', 'next_round', 'after_lineage'])
def test_legacy_campaign_upgrade_preserves_history_and_does_not_repeat_trials(tmp_path, monkeypatch, kind, boundary):
    with legacy_writer(monkeypatch):
        old, initial, old_suite, quota = setup(tmp_path, kind)
        if boundary == 'next_round':
            build = old.build_loop

            def interrupted(directory, *args):
                if 'round-0002' in Path(directory).parts:
                    raise RuntimeError('upgrade boundary')
                return build(directory, *args)

            old.build_loop = interrupted
        elif boundary == 'after_lineage':
            write = loop_module.atomic_json

            def interrupted_write(path, value):
                if Path(path).name == 'commit.json':
                    raise RuntimeError('upgrade boundary')
                return write(path, value)

            monkeypatch.setattr(loop_module, 'atomic_json', interrupted_write)
        if boundary == 'completed':
            previous = old.run(initial)
        else:
            with pytest.raises(RuntimeError, match='upgrade boundary'):
                old.run(initial)

    # The context above restores patch-owned methods; restore this interruption
    # separately because it belongs to the outer pytest monkeypatch fixture.
    if boundary == 'after_lineage':
        monkeypatch.setattr(loop_module, 'atomic_json', write)
    before = {path: path.read_bytes() for base in (old.root / 'rounds', tmp_path / 'archive', tmp_path / 'state/history')
              for path in base.rglob('*.json')}
    config_path = old.root / ('campaign.json' if kind == 'improvement' else 'config.json')
    frozen = read_json(config_path)
    frozen_config = frozen['config'] if kind == 'improvement' else frozen
    assert 'engine_version' not in frozen_config
    restarted, initial, new_suite, _ = setup(tmp_path, kind)
    result = restarted.run(initial)
    assert result['status'] == 'completed' and result['completed_rounds'] == 2
    assert len(old_suite.resets) + len(new_suite.resets) == (6 if kind == 'improvement' else 12)
    assert all(path.read_bytes() == content for path, content in before.items())
    current_config = read_json(config_path)
    assert (current_config['config'] if kind == 'improvement' else current_config) == frozen_config
    if boundary == 'completed':
        assert result == previous and new_suite.resets == []
    count = len(new_suite.resets)
    assert restarted.run(initial) == result
    assert len(new_suite.resets) == count


@pytest.mark.parametrize('change', ['engine', 'selector', 'budget'])
def test_execution_changes_still_require_a_new_campaign_while_history_remains_readable(tmp_path, monkeypatch, change):
    with legacy_writer(monkeypatch):
        old, initial, _, _ = setup(tmp_path, 'improvement')
        old.run(initial)
    entries = list((tmp_path / 'archive/entries').glob('*/record.json'))
    records = {path.parent.name: archive.CandidateArchive(tmp_path / 'archive').read(path.parent.name) for path in entries}
    fresh, initial, suite, _ = setup(tmp_path, 'improvement')
    if change == 'engine':
        monkeypatch.setattr(loop_module, 'ENGINE_VERSION', 2)
    elif change == 'selector':
        from test_improvement_campaign import Selector
        monkeypatch.setattr(Selector, 'identity', lambda self: {'selection': 'changed'})
    else:
        fresh.max_rounds += 1
    with pytest.raises(ValueError, match='[Ff]rozen.*changed'):
        fresh.run(initial)
    assert suite.resets == []
    for revision, record in records.items():
        assert archive.CandidateArchive(tmp_path / 'archive').read(revision) == record


def test_compatibility_keeps_selection_and_provider_semantics_bound():
    saved = dict(profile={'minimum_gain': 0.1}, identities={
        'core': 'legacy-source', 'scoring': 'scorer-v1', 'selector': {'mode': 'paired'},
        'proposer': {'revision': 'one'}, 'evaluator': {'revision': 'one'}},
        protocol={'identity': 'one'}, scope='software', max_candidates=1)
    current = deepcopy(saved)
    current.update(engine_version=1)
    current['identities']['core'] = 'new-source'
    assert same_execution(saved, current)
    for key in ('scoring', 'selector', 'proposer', 'evaluator'):
        changed = deepcopy(current)
        changed['identities'][key] = {'changed': True}
        assert not same_execution(saved, changed)
    changed = deepcopy(current)
    changed['profile']['minimum_gain'] = 0.2
    assert not same_execution(saved, changed)
    assert not same_execution(saved, dict(current, engine_version=2))
