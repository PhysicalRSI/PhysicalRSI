from copy import deepcopy
import json

import pytest

from PhysicalRSI_Autoresearch.heartbeat import ResearchHeartbeat
from PhysicalRSI_Autoresearch.records import ResearchRecord, main
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest
from PhysicalRSI_core.lineage import HarnessState, StateConflict
from PhysicalRSI_core.self_harness.artifacts import CLOSURE
from PhysicalRSI_core.self_harness.protocol import CaseProtocol
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory


def setup(tmp_path):
    harness = tmp_path / 'harness'
    for name in CLOSURE:
        atomic_json(harness / (name + '.json'), {'component': name})
    candidate = dict(id='parent', root=str(harness), components={
        name: {name + '.json': file_digest(harness / (name + '.json'))} for name in CLOSURE})
    state = HarnessState(tmp_path / 'lineage')
    state.initialize(candidate, policy={'fixture': True}, scope='software-contract-test')
    path = tmp_path / 'development.json'
    atomic_json(path, {'observation': 'fixture diagnostic'})
    evidence = {str(path): file_digest(path)}
    memory = ResearchMemory(tmp_path / 'memory')
    revision = memory.write({'lesson': dict(conditions={'task': 'counter'},
        hypothesis='Check an actual mechanism before running more cases', required_checks=['activation'],
        counterexample_checks=[], evidence=evidence)})
    protocol = CaseProtocol(tmp_path / 'protocol', splits={
        split: {'counter': [{'seed': i}]} for i, split in enumerate(('development', 'validation', 'test'))})
    record = ResearchRecord(tmp_path / 'research')
    created = record.create(question='Does the mechanism help?', hypothesis='It recovers the failure.',
        falsifier='No activation or regressing paired results.', alternatives=['The detector is wrong.'],
        budget={'max_trials': 6, 'trial_seconds': 30}, state=state, memory=memory,
        memory_revision=revision, context={'task': 'counter'}, evidence=evidence)
    args = dict(candidate=candidate, evaluator_sha256='1' * 64, protocol=protocol,
                workspace=tmp_path / 'experiment', used_lessons=['lesson'], evidence=evidence)
    return record, created, args, state, memory, revision, evidence


def dispatch(data):
    record, created, args, *_ = data
    ready = record.bind_experiment(created['revision'], **args)
    return record.claim_dispatch(ready['revision'], job_id='local-test-job')


def result_report(tmp_path, dispatched, evidence, **changes):
    path = tmp_path / 'result-report.json'
    report = dict(schema='physicalrsi.research-result/v1', research_revision=dispatched['revision'],
                  experiment_sha256=digest(dispatched['body']['experiment']),
                  job_id=dispatched['body']['job_id'], outcome='completed', evidence=evidence)
    report.update(changes)
    atomic_json(path, report)
    return path


def test_restart_results_decision_and_memory_preserve_one_history(tmp_path):
    data = setup(tmp_path)
    record, created, args, state, memory, revision, evidence = data
    assert record.status()['next_action'] == 'prepare_and_freeze_experiment'
    sent = dispatch(data)
    record = ResearchRecord(record.root)
    assert record.status()['job_id'] == 'local-test-job'
    result = record.record_result(sent['revision'], report=result_report(tmp_path, sent, evidence))
    assert record.status()['next_action'] == 'record_decision_and_memory'
    updated = memory.counterexample(revision, 'lesson', check='regression', evidence=evidence,
                                    explanation='A fixture counterexample was retained.')
    closed = record.close(result['revision'], disposition='rejected', explanation='Counterexample observed.',
                          evidence=evidence, memory_revision=updated, memory_explanation='Suspended the lesson.')
    assert closed['qualification'] is None
    assert record.status()['next_action'] == 'research_closed'
    assert record.read(created['revision']) == created
    assert state.resolve()['harness']['id'] == 'parent'  # No policy promotion.
    with pytest.raises(ValueError):
        record.close(closed['revision'], disposition='supported', explanation='Overwrite',
                     evidence=evidence, memory_revision=updated, memory_explanation='Overwrite')


def test_compare_and_swap_prevents_duplicate_dispatch(tmp_path):
    data = setup(tmp_path)
    record, created, args, *_ = data
    ready = record.bind_experiment(created['revision'], **args)
    record.claim_dispatch(ready['revision'], job_id='first')
    with pytest.raises(StateConflict, match='advanced'):
        record.claim_dispatch(ready['revision'], job_id='second')
    with pytest.raises(ValueError, match='not ready'):
        record.claim_dispatch(record.read()['revision'], job_id='third')


@pytest.mark.parametrize('field,value', [('research_revision', '0' * 64),
    ('experiment_sha256', '0' * 64), ('job_id', 'another-job'), ('schema', 'another-schema')])
def test_result_from_another_experiment_cannot_close_this_record(tmp_path, field, value):
    data = setup(tmp_path)
    record, *_, evidence = data
    sent = dispatch(data)
    with pytest.raises(ValueError, match='does not match'):
        record.record_result(sent['revision'], report=result_report(tmp_path, sent, evidence, **{field: value}))
    assert record.status()['status'] == 'awaiting_result'


def test_stale_selected_parent_stops_dispatch_but_allows_disposition(tmp_path):
    record, created, args, state, memory, revision, evidence = setup(tmp_path)
    ready = record.bind_experiment(created['revision'], **args)
    # Simulate another actor changing lineage under its writer protocol.
    current = state.resolve()
    changed = {k: v for k, v in current.items() if k != 'revision'}
    changed.update(action='rollback', previous=current['revision'])
    state._publish(changed)
    assert record.status()['next_action'] == 'supersede_stale_parent'
    with pytest.raises(StateConflict, match='parent is stale'):
        record.claim_dispatch(ready['revision'], job_id='wrong-parent')
    record.close(ready['revision'], disposition='superseded', explanation='Parent moved.',
                 evidence=evidence, memory_revision=revision, memory_explanation='No new lesson.')


def test_unrelated_memory_and_missing_result_never_support_a_hypothesis(tmp_path):
    record, created, args, state, memory, revision, evidence = setup(tmp_path)
    close = dict(disposition='supported', explanation='Unsupported assertion', evidence=evidence,
                 memory_revision=revision, memory_explanation='Retain')
    with pytest.raises(ValueError, match='completed experiment'):
        record.close(created['revision'], **close)
    unrelated = deepcopy(memory.read(revision)['lessons'])
    unrelated['lesson']['hypothesis'] = 'Unrelated hypothesis'
    independent = memory.write(unrelated)
    close.update(disposition='inconclusive', memory_revision=independent)
    with pytest.raises(ValueError, match='does not descend'):
        record.close(created['revision'], **close)


@pytest.mark.parametrize('fault', ['new-evidence', 'intermediate-revision'])
def test_closed_record_revalidates_the_selected_memory_chain(tmp_path, fault):
    record, created, args, state, memory, revision, evidence = setup(tmp_path)
    negative = tmp_path / 'new-counterexample.json'
    atomic_json(negative, {'development_observation': 'the mechanism failed'})
    updated = memory.counterexample(revision, 'lesson', check='new-counterexample',
        evidence={str(negative): file_digest(negative)}, explanation='Suspend the unsupported lesson.')
    refined = memory.refine(updated, 'lesson', conditions={'task': 'counter', 'condition': 'restricted'},
        hypothesis='Retest only with the new condition.', evidence=evidence)
    record.close(created['revision'], disposition='rejected', explanation='Negative development evidence.',
        evidence=evidence, memory_revision=refined, memory_explanation='Suspend and narrow the lesson.')
    assert record.status()['decision']['memory_revision'] == refined
    if fault == 'new-evidence':
        atomic_json(negative, {'changed': True})
    else:
        atomic_json(memory.root / (updated + '.json'), {})
    with pytest.raises(ValueError, match='changed'):
        ResearchRecord(record.root).status()


@pytest.mark.parametrize('fault', ['evidence', 'ancestor', 'candidate'])
def test_changed_inputs_fail_on_resume_or_dispatch(tmp_path, fault):
    record, created, args, state, memory, revision, evidence = setup(tmp_path)
    ready = record.bind_experiment(created['revision'], **args)
    if fault == 'evidence':
        atomic_json(tmp_path / 'development.json', {'changed': True})
    elif fault == 'ancestor':
        atomic_json(record.root / 'history' / (created['revision'] + '.json'), {})
    else:
        atomic_json(tmp_path / 'harness/control.json', {'changed': True})
    with pytest.raises(ValueError):
        record.claim_dispatch(ready['revision'], job_id='changed-inputs')


def test_unknown_used_memory_and_old_workspace_rejected(tmp_path):
    record, created, args, *_ = setup(tmp_path)
    with pytest.raises(ValueError, match='pinned retrieval'):
        record.bind_experiment(created['revision'], **dict(args, used_lessons=['unknown']))
    args['workspace'].mkdir()
    (args['workspace'] / 'old.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh experiment'):
        record.bind_experiment(created['revision'], **args)


def test_workspace_claim_cannot_reuse_results_written_after_binding(tmp_path):
    record, created, args, *_ = setup(tmp_path)
    ready = record.bind_experiment(created['revision'], **args)
    args['workspace'].mkdir()
    (args['workspace'] / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='workspace changed'):
        record.claim_dispatch(ready['revision'], job_id='duplicate')


def test_heartbeat_next_action_is_bound_to_record_and_resume_revision(tmp_path):
    record, created, args, state, memory, revision, evidence = setup(tmp_path)
    timer = ResearchHeartbeat(tmp_path / 'heartbeat')
    request = timer.tick(now=100)['pending']
    review = dict(evidence=evidence, reflection='Reviewed fixture evidence.',
                  alternatives='Test another detector.', decision='Prepare a bounded experiment.',
                  next_experiment='Freeze the candidate and case protocol.')
    completed = record.complete_review(timer, request, review, expected=created['revision'], now=101)
    action = json.loads(completed['review']['next_experiment'])
    assert action['research_revision'] == created['revision']
    assert action['next_action'] == 'prepare_and_freeze_experiment'
    assert str(record.root / 'history' / (created['revision'] + '.json')) in completed['review']['evidence']
    assert set(review['evidence']) == set(evidence)  # No caller mutation.


def test_index_cli_resolves_explicit_records_without_experiment_effects(tmp_path, monkeypatch, capsys):
    record, created, args, *_ = setup(tmp_path)
    index = tmp_path / 'index.json'
    atomic_json(index, dict(schema='physicalrsi.research-index/v1', records={'counter': str(record.root)}))
    monkeypatch.setattr('sys.argv', ['records', '--index', str(index)])
    main()
    status = json.loads(capsys.readouterr().out)['records']['counter']
    assert status['revision'] == created['revision']
    assert not args['workspace'].exists()


def test_real_self_harness_campaign_can_finish_record_after_agent_restart(tmp_path):
    """Exercise execution/selection, not just trusted result-envelope fixtures."""
    import shutil

    from test_improvement_campaign import setup as campaign_setup, manifest, Proposal, Selector
    from PhysicalRSI_core.embodiment import Revision
    from PhysicalRSI_core.experiments import Budget
    from PhysicalRSI_core.self_harness import SelfHarness
    from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
    from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
    from PhysicalRSI_core.self_harness.protocol import PreregisteredSuite

    _, parent, suite, quota = campaign_setup(tmp_path, max_trials=6, max_rounds=1)
    shutil.copytree(parent['root'], tmp_path / 'bound-child')
    atomic_json(tmp_path / 'bound-child/control.json', {'increment': 1})
    child = manifest(tmp_path / 'bound-child', 'child')
    protocol = CaseProtocol(tmp_path / 'protocol', splits={
        'development': {'counter': [{'seed': 0, 'initial': 0, 'target': 3}]},
        'validation': {'counter': [{'seed': 1, 'initial': 0, 'target': 5},
                                   {'seed': 2, 'initial': 0, 'target': 7}]},
        'test': {'counter': [{'seed': 3, 'initial': 0, 'target': 9}]}})
    evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, protocol),
        budgets={'counter': Budget(10, 30)}, scope='software-counter',
        system2=Revision('research-record-integration', '1'), quota=quota)
    state = HarnessState(tmp_path / 'state')

    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=Proposal(evaluator), evaluator=evaluator,
            selector=Selector(), scope='software-counter', max_candidates=1,
            profile=dict(tasks={'counter': dict(weight=1, episodes=2, score_range=[0, 1],
                maximum_regression=0)}, minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind='software-counter'))

    prepared = build_loop(tmp_path / 'preparation')
    state.initialize(parent, policy=prepared.config, scope='software-counter')
    context = tmp_path / 'task-context.json'
    atomic_json(context, {'scope': 'software-counter', 'targets': 'odd integers'})
    evidence = {str(context): file_digest(context)}
    memory = ResearchMemory(tmp_path / 'memory')
    revision = memory.write({'counter': dict(conditions={'task': 'counter'},
        hypothesis='An increment of one can reach odd targets.', required_checks=['paired-execution'],
        counterexample_checks=[], evidence=evidence)})
    record = ResearchRecord(tmp_path / 'research')
    created = record.create(question='Does the bound repair improve paired counter performance?',
        hypothesis='The repair reaches odd targets.', falsifier='Paired results fail to improve.',
        alternatives=['The existing increment is sufficient.'], budget={'max_trials': 6, 'trial_seconds': 30},
        state=state, memory=memory, memory_revision=revision, context={'task': 'counter'}, evidence=evidence)
    ready = record.bind_experiment(created['revision'], candidate=child,
        evaluator_sha256=evaluator.revision, protocol=protocol, workspace=tmp_path / 'experiment',
        used_lessons=['counter'], evidence=evidence)
    sent = record.claim_dispatch(ready['revision'], job_id='counter-job')
    result = ImprovementCampaign(tmp_path / 'experiment', state=state, build_loop=build_loop,
                                 max_rounds=1).run(parent)
    assert result['current']['decision']['decision'] == 'inherit_child'
    assert result['current']['freeze_sha256'] == sent['body']['experiment']['candidate_sha256']
    assert quota.status()['reserved_trials'] == len(suite.resets) == 5
    receipt = tmp_path / 'campaign-result.json'
    atomic_json(receipt, result)
    # Reopen after execution but before result collection: never launch again.
    record = ResearchRecord(record.root)
    collected = record.record_result(sent['revision'], report=result_report(
        tmp_path, sent, {str(receipt): file_digest(receipt)}))
    record.close(collected['revision'], disposition='supported', explanation='Software paired improvement.',
        evidence={str(receipt): file_digest(receipt)}, memory_revision=revision,
        memory_explanation='Keep the lesson; this integration test does not study memory improvement.')
    assert record.status()['next_action'] == 'research_closed'
    assert quota.status()['reserved_trials'] == len(suite.resets) == 5
