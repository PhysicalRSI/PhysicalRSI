from copy import deepcopy

import pytest

from PhysicalRSI_baselines.behavior_baseline.revision import materialize
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness


def fixture(tmp_path):
    # Synthetic development feedback tests artifact handling, not admission.
    root = tmp_path / 'parent'
    atomic_json(root / 'fixed.json', {'scope': 'fixture'})
    (root / 'control.py').write_text('def initialize(task): return {}\ndef step(obs, state, primitive): return primitive(obs)\n')
    parent = dict(id='parent', root=str(root), components={
        kind: {'fixed.json': file_digest(root / 'fixed.json')} for kind in CLOSURE})
    parent['components']['control'] = {'control.py': file_digest(root / 'control.py')}
    feedback_root = tmp_path / 'evidence'
    name = 'development-feedback/episodes.json'
    summary = dict(schema='physicalrsi.behavior-development-feedback/v1',
                   parent_id='parent', parent_revision=verify_harness(parent),
                   evaluator_revision='fixture', episodes=[{'scope': 'synthetic'}])
    atomic_json(feedback_root / name, summary)
    feedback = dict(split='evolve', evaluator_revision='fixture', episodes=summary['episodes'],
                    evidence={name: file_digest(feedback_root / name)})
    code = tmp_path / 'proposed.py'
    code.write_text('def initialize(task): return {"steps": 0}\ndef step(obs, state, primitive):\n    state["steps"] += 1\n    return primitive(obs)\n')
    return dict(parent=parent, control=code, feedback=feedback, feedback_root=feedback_root,
                hypothesis='Fixture-only bookkeeping change; no improvement claim',
                environment={'kind': 'software-fixture'}, costs={'model_requests': 0},
                expected_effect='Preserve primitive actions while counting calls',
                required_checks=['independent paired native validation'])


def test_revision_preserves_foundation_and_parent_and_is_immutable(tmp_path):
    kwargs = fixture(tmp_path)
    parent = deepcopy(kwargs['parent'])
    parent_sha = verify_harness(parent)
    child = materialize(tmp_path / 'child', **kwargs)
    assert verify_harness(parent) == parent_sha
    assert verify_harness(child) != parent_sha
    for component in CLOSURE - {'control'}:
        assert child['components'][component] == parent['components'][component]
    proposal = read_json(tmp_path / 'child/proposal.json')
    assert proposal['parent_sha256'] == parent_sha
    assert proposal['candidate_sha256'] == verify_harness(child)
    assert child['parent_sha256'] == parent_sha
    assert child['costs'] == kwargs['costs']
    assert child['environment'] == kwargs['environment']
    assert child['evidence']['files'] == kwargs['feedback']['evidence']
    assert child['used_lessons'] == []
    with pytest.raises(FileExistsError):
        materialize(tmp_path / 'child', **kwargs)


def test_revision_rejects_wrong_split_parent_and_tampered_evidence(tmp_path):
    kwargs = fixture(tmp_path)
    with pytest.raises(ValueError, match='development feedback'):
        materialize(tmp_path / 'child', **dict(kwargs, feedback=dict(kwargs['feedback'], split='validation')))
    wrong = deepcopy(kwargs)
    wrong['parent']['id'] = 'different-parent'
    with pytest.raises(ValueError, match='another parent'):
        materialize(tmp_path / 'child', **wrong)
    with pytest.raises(ValueError, match='change the control'):
        materialize(tmp_path / 'child', **dict(kwargs, control=tmp_path / 'parent/control.py'))
    atomic_json(kwargs['feedback_root'] / 'development-feedback/episodes.json', {'tampered': True})
    with pytest.raises(ValueError, match='evidence changed'):
        materialize(tmp_path / 'child', **kwargs)
    assert not (tmp_path / 'child').exists()


def test_materialized_child_reaches_actual_core_admission_without_metadata_patch(tmp_path):
    kwargs = fixture(tmp_path)
    admitted = []

    class Proposal:
        def identity(self):
            return {'kind': 'fixture-proposal'}

        def develop(self, parent, output):
            return kwargs['feedback']

        def propose(self, parent, feedback, output):
            return [materialize(output / 'child', **kwargs)]

    class Admission:
        def identity(self):
            return {'kind': 'fixture-reject-child'}

        def admit(self, candidate, output):
            admitted.append(candidate['id'])
            # This fixture checks the core interface, not native admission.
            return dict(accepted=candidate['id'] == 'parent',
                        freeze_sha256=verify_harness(candidate),
                        evidence={'fixture': True}, reason='No native admission in software fixture')

    state = HarnessState(tmp_path / 'state')
    loop = SelfHarness(tmp_path / 'round', state=state, proposer=Proposal(),
        evaluator=Admission(), selector=Admission(), scope='software-interface-only',
        max_candidates=1, protocol={'evaluation_kind': 'policy_evaluation'},
        profile=dict(tasks={'fixture': dict(weight=1, episodes=1, score_range=[0, 1],
                                            maximum_regression=0)},
                     minimum_gain=0, tie_tolerance=0))
    initial = state.initialize(kwargs['parent'], policy=loop.config, scope=loop.config['scope'])
    assert loop.run()['revision'] == initial['revision']
    assert admitted == ['parent', 'child']
    assert read_json(loop.root / 'retained.json')['reason'] == 'all_candidates_rejected'


@pytest.mark.parametrize('change', [dict(costs={}), dict(environment={}),
    dict(expected_effect=''), dict(required_checks=[]), dict(used_lessons=['same', 'same'])])
def test_missing_research_provenance_rejected_before_writing(tmp_path, change):
    kwargs = dict(fixture(tmp_path), **change)
    with pytest.raises(ValueError):
        materialize(tmp_path / 'child', **kwargs)
    assert not (tmp_path / 'child').exists()
