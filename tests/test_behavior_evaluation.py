"""Software integration fixtures; no simulator or benchmark qualification."""
from pathlib import Path
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from PhysicalRSI_baselines.behavior_baseline import native, service
from PhysicalRSI_baselines.behavior_baseline.evaluation import NativeEvaluator
from PhysicalRSI_baselines.behavior_baseline.revision import materialize
from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota, TrialLimitExceeded
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.protocol import CaseProtocol
from PhysicalRSI_core.self_harness.selection import select_survivor


def setup(tmp_path, monkeypatch, *, quota_limit=3):
    task = 'turning_on_radio'
    candidates = []
    for name in ('parent', 'child'):
        root = tmp_path / name
        atomic_json(root / 'backend.json', dict(kind='openpi-b1k/v1', task=task))
        (root / 'control.py').write_text('# fixture ' + name)
        candidate = dict(id=name, root=str(root), components={
            kind: {'backend.json': file_digest(root / 'backend.json')} for kind in CLOSURE})
        candidate['components']['control'] = {'control.py': file_digest(root / 'control.py')}
        candidates.append(candidate)
    def case(index, mode='train'):
        return dict(task=task, mode=mode, instance_index=index if mode == 'train' else index-301,
                    instance_id=index, rollout_id=0)
    protocol = CaseProtocol(tmp_path / 'cases', splits={
        'development': {task: [case(0)]}, 'validation': {task: [case(200)]},
        'test': {task: [case(301, 'public_test')]}}, identity_keys=('instance_id',))

    class Admission:
        def identity(self):
            return {'kind': 'software-fixture-only'}
        def check(self, candidate, directory):
            atomic_json(directory / 'check.json', dict(fixture=True, candidate=verify_harness(candidate)))
            return dict(accepted=True, evidence={'check.json': file_digest(directory / 'check.json')})

    quota = TrialQuota(tmp_path / 'quota', max_trials=quota_limit)
    evaluator = NativeEvaluator(tmp_path, protocol=protocol, admission=Admission(), quota=quota,
        python=tmp_path / 'python', system2=Revision('fixture', '1'), scope='software-fixture')
    output = tmp_path / 'round'
    events = []

    @contextmanager
    def served(runtime, path, folder, **kwargs):
        candidate = read_json(path)
        events.append(('service', candidate['id'], quota.status()['reserved_trials']))
        yield dict(host='127.0.0.1', port=kwargs['port'], policy_revision=verify_harness(candidate))

    def run(runtime, folder, *, expected_specification, **kwargs):
        spec = expected_specification
        case = spec['case']
        name = spec['system2']['candidate_id']
        events.append(('trial', name, case['instance_id']))
        result = dict(task=task, instance_id=case['instance_id'], rollout_id=0, success=False,
                      q_score={'final': .25 if name == 'parent' else .5})
        metadata = dict(schema='physicalrsi.behavior-policy-service/v1', task=task,
            candidate_sha256=spec['policy_revision'], policy_identity={'fixture': name},
            policy_identity_sha256=digest({'fixture': name}))
        for filename, value in [('specification.json', spec), ('result.json', result),
                                ('policy_service.json', metadata)]:
            atomic_json(folder / filename, value)
        video = f'native/videos/{task}_{case["instance_id"]}_0.mp4'
        (folder / video).parent.mkdir(parents=True, exist_ok=True)
        (folder / video).write_bytes(b'software fixture, not a video')
        receipt = dict(state='completed', specification_sha256=digest(spec), outcome='failure',
            native_result=result, native_result_file='result.json', evidence={
                n: file_digest(folder / n) for n in ['specification.json', 'result.json', 'policy_service.json', video]})
        atomic_json(folder / 'receipt.json', receipt)
        return receipt

    monkeypatch.setattr(service, 'policy_service', served)
    monkeypatch.setattr(native, 'run', run)
    comparison = dict(schema_version=1, round_id='fixture', parent_id='parent',
        frozen_at=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),
        candidates={c['id']: verify_harness(c) for c in candidates}, scope='software-fixture',
        profile=dict(tasks={task: dict(weight=1, episodes=1, score_range=[0, 1], maximum_regression=0)},
                     minimum_gain=0, tie_tolerance=0))
    return evaluator, candidates, comparison, output, events


def test_native_ports_development_pair_selection_and_resume(tmp_path, monkeypatch):
    evaluator, candidates, comparison, output, events = setup(tmp_path, monkeypatch)
    development = evaluator.development(candidates[0], output)
    assert development['split'] == 'evolve'
    assert len(development['episodes']) == 1
    evaluator.admit(candidates[1], output)
    cohort = evaluator.validation(comparison, output)
    assert evaluator.quota.status()['reserved_trials'] == 3
    results = [evaluator.evaluate(c, comparison, cohort, output) for c in candidates]
    assert select_survivor(comparison, cohort, results, evidence_root=output)['survivor_id'] == 'child'
    assert [e for e in events if e[0] == 'service'] == [
        ('service', 'parent', 1), ('service', 'child', 3), ('service', 'parent', 3)]
    before = deepcopy(events)
    assert evaluator.resume_evaluation(candidates[0], comparison, cohort, output) == results[0]
    assert events == before
    # Admission is checked again before any resumed execution.
    atomic_json(output / 'native-admission/child/check.json', {'tampered': True})
    with pytest.raises(ValueError, match='Admission evidence changed'):
        evaluator.evaluate(candidates[0], comparison, cohort, output)
    assert events == before


def test_whole_comparison_quota_rejection_has_no_partial_effects(tmp_path, monkeypatch):
    evaluator, candidates, comparison, output, events = setup(tmp_path, monkeypatch, quota_limit=1)
    for candidate in candidates:
        evaluator.admit(candidate, output)
    with pytest.raises(TrialLimitExceeded):
        evaluator.validation(comparison, output)
    assert evaluator.quota.status()['reserved_trials'] == 0
    assert events == []


def test_wrong_case_scope_and_unadmitted_candidate_rejected(tmp_path, monkeypatch):
    evaluator, candidates, comparison, output, events = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='scope'):
        evaluator.validation(dict(comparison, scope='official-full100'), output)
    evaluator.admit(candidates[0], output)
    with pytest.raises(FileNotFoundError):
        evaluator.validation(comparison, output)
    assert evaluator.quota.status()['reserved_trials'] == 0
    assert events == []


def test_core_iteration_commits_child_from_native_ports_software_fixture(tmp_path, monkeypatch):
    evaluator, candidates, comparison, output, events = setup(tmp_path, monkeypatch)
    control = tmp_path / 'proposed.py'
    control.write_text('def initialize(task): return {}\ndef step(obs, state, primitive): return primitive(obs)\n')

    class Proposal:
        def identity(self):
            return {'kind': 'fixture-proposer'}
        def develop(self, parent, output):
            return evaluator.development(parent, output)
        def propose(self, parent, feedback, output):
            return [materialize(output / 'new-child', parent=parent, control=control,
                feedback=feedback, feedback_root=output, hypothesis='Synthetic interface fixture',
                environment={'kind': 'fixture'}, costs={'model_requests': 0},
                expected_effect='Fixture returns a larger synthetic score',
                required_checks=['software contract only'])]

    class Selector:
        def identity(self):
            return {'kind': 'core-selector-fixture'}
        def select(self, *args, **kwargs):
            return select_survivor(*args, **kwargs)

    state = HarnessState(tmp_path / 'state')
    loop = SelfHarness(output, state=state, proposer=Proposal(), evaluator=evaluator,
        selector=Selector(), scope=evaluator.scope, profile=comparison['profile'],
        max_candidates=1, protocol={'evaluation_kind': 'policy_evaluation', 'identity': evaluator.revision})
    parent = state.initialize(candidates[0], policy=loop.config, scope=evaluator.scope)
    result = loop.run()
    assert result['previous'] == parent['revision']
    assert result['harness']['id'] == 'new-child'
    assert result['decision']['decision'] == 'inherit_child'
    before = deepcopy(events)
    assert loop.run() == result
    assert events == before


def test_task_bound_checkpoint_cannot_shrink_a_larger_comparison(tmp_path, monkeypatch):
    evaluator, candidates, comparison, output, events = setup(tmp_path, monkeypatch)
    manifest = read_json(evaluator.protocol.root / 'protocol.json')
    for tasks in manifest['splits'].values():
        tasks['other_task'] = [dict(c, task='other_task') for c in tasks['turning_on_radio']]
    protocol = CaseProtocol(tmp_path / 'larger-cases', splits=manifest['splits'],
                            identity_keys=('instance_id',))
    evaluator = NativeEvaluator(tmp_path, protocol=protocol, admission=evaluator.admission,
        quota=evaluator.quota, python=evaluator.python, system2=evaluator.system2, scope=evaluator.scope)
    comparison['profile']['tasks']['other_task'] = deepcopy(comparison['profile']['tasks']['turning_on_radio'])
    for candidate in candidates:
        evaluator.admit(candidate, output)
    with pytest.raises(ValueError, match='entire declared task set'):
        evaluator.validation(comparison, output)
    assert evaluator.quota.status()['reserved_trials'] == 0
    assert events == []


def test_multitask_plan_reserves_every_declared_case_without_shrinking(tmp_path, monkeypatch):
    evaluator, candidates, comparison, output, events = setup(tmp_path, monkeypatch, quota_limit=4)
    for candidate in candidates:
        path = Path(candidate['root']) / 'backend.json'
        atomic_json(path, dict(kind='openpi-b1k-multitask/v1',
            task_prompts={'turning_on_radio': 'turn on radio', 'opening_drawer': 'open drawer'},
            asset_id='full100', checkpoint_files={'assets/full100/norm_stats.json': 'fixture'}))
        for component in candidate['components'].values():
            if 'backend.json' in component: component['backend.json'] = file_digest(path)
        assert evaluator.admit(candidate, output)['accepted']
    cases={task:[dict(task=task,mode='train',instance_id=0,instance_index=0,rollout_id=0)]
           for task in ('turning_on_radio','opening_drawer')}
    plan=evaluator._plan(output,{c['id']:verify_harness(c) for c in candidates},cases,'development')
    assert len(plan.plan['trials'])==4
    assert {(r['inputs']['system2']['candidate_id'],r['inputs']['task']) for r in plan.plan['trials']} == {
        (name,task) for name in ('parent','child') for task in cases}
    assert evaluator.quota.status()['reserved_trials']==4 and events==[]


def test_group_option_routes_validation_and_preserves_single_case_development(tmp_path, monkeypatch):
    from PhysicalRSI_baselines.behavior_baseline.grouped import GroupedNativeTrials
    from PhysicalRSI_baselines.behavior_baseline.trials import NativeTrials

    original, candidates, comparison, output, _ = setup(tmp_path, monkeypatch)
    evaluator = NativeEvaluator(tmp_path, protocol=original.protocol, admission=original.admission,
        quota=original.quota, python=original.python, system2=original.system2,
        scope='software-fixture', native_group_size=2)
    candidate = candidates[0]
    evaluator.admit(candidate, output)
    revisions = {candidate['id']: verify_harness(candidate)}
    cases = {'turning_on_radio': [dict(instance_index=0)]}
    development = evaluator._plan(output, revisions, cases, 'development')
    validation = evaluator._plan(output, revisions, cases, 'validation', comparison, {'fixture': True})
    assert type(development) is NativeTrials
    assert type(validation) is GroupedNativeTrials
    assert evaluator.identity() != original.identity()
