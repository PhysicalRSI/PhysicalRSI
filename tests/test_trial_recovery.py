from pathlib import Path
import shutil
from dataclasses import replace

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialLimitExceeded, TrialQuota
from PhysicalRSI_core.infra.trial_recovery import TrialRecovery, inactive, reservations
from PhysicalRSI_core.trials import Evaluation, Trial
from PhysicalRSI_demos import rollout_evolution as demo


def interrupted(tmp_path, monkeypatch, *, task='axis', target=3, allowance=1, quota=15):
    campaign, initial, suite, budget = demo.setup(tmp_path, scheduled_workers=1,
        max_replacements=allowance, max_trials=quota)
    reset = demo.Environment.reset
    failed = False

    def crash(self, case, context):
        nonlocal failed
        if not failed and self.task == task and case['target'][0] == target:
            failed = True
            self.resets.append((self.task, case))
            raise RuntimeError('owned worker stopped during startup')
        return reset(self, case, context)

    with monkeypatch.context() as patch:
        patch.setattr(demo.Environment, 'reset', crash)
        with pytest.raises(RuntimeError, match='worker stopped'):
            campaign.run(initial)
    directory = campaign.root / 'rounds/round-0001'
    loop = campaign.build_loop(directory / 'execution', read_json(directory / 'context.json'))
    paths = [p for p in loop.root.glob('experiments/**/trials/*/receipt.json')
             if read_json(p)['state'] != 'completed']
    assert len(paths) == 1
    folder = paths[0].parent
    proof = tmp_path / 'stopping-proof.json'
    atomic_json(proof, dict(trial_id=folder.name, quiescent=True,
                           description='Synchronous software reset raised before returning; no worker remains'))
    options = dict(output=loop.root, split='development' if target == 1 else 'validation',
        run_id=folder.name, operator='test-operator', reason='Verified stopped startup',
        evidence={str(proof): file_digest(proof)}, quiescent=True)
    return campaign, initial, suite, budget, loop, folder, options


@pytest.mark.parametrize('task', ['axis', 'vector'])
@pytest.mark.parametrize('target', [1, 3])
def test_replacement_runs_original_cross_task_campaign_and_restarts_without_effects(tmp_path, monkeypatch, task, target):
    campaign, initial, suite, quota, loop, failed, options = interrupted(tmp_path, monkeypatch, task=task, target=target)
    before = {p.name: file_digest(p) for p in failed.iterdir() if p.is_file() and not p.name.endswith('.lock')}
    count, reserved = len(suite.resets), quota.status()['reserved_trials']
    with pytest.raises(ReconciliationRequired):
        campaign.run(initial)
    assert len(suite.resets) == count
    replacement = loop.evaluator.authorize_replacement(initial, **options)
    assert replacement['run_id'] != failed.name
    assert quota.status()['reserved_trials'] == reserved + 1
    assert loop.evaluator.authorize_replacement(initial, **options) == replacement
    assert quota.status()['reserved_trials'] == reserved + 1
    assert len(suite.resets) == count
    if target != 1:
        scheduler = loop.evaluator.scheduler
        jobs = scheduler.status()['jobs']
        key = next(k for k, job in jobs.items() if job['state'] == 'needs_reconciliation')
        with pytest.raises(ReconciliationRequired):
            campaign.run(initial)
        resumed = loop.evaluator.resume_replacements(key, operator=options['operator'],
            reason=options['reason'], evidence=options['evidence'])
        assert resumed['error'].endswith('owned worker stopped during startup')
        assert loop.evaluator.resume_replacements(key, operator=options['operator'],
            reason=options['reason'], evidence=options['evidence']) == resumed
    result = campaign.run(initial)
    report = campaign.report_test(initial)
    assert result['completed_rounds'] == 2
    assert all(row['success_rate'] == 1 for row in report['metrics'].values())
    assert len(suite.resets) == quota.status()['reserved_trials'] == 15
    assert {name: file_digest(failed / name) for name in before} == before
    value = Evaluation.read(ExperimentRuntime(failed.parent), replacement['run_id']).to_dict()
    assert value['recovery']['original_id'] == failed.name
    assert value['recovery']['index'] == 1
    assert value['trial']['case'] == read_json(failed / 'experiment.json')['case']
    restarted, seed, new_suite, new_quota = demo.setup(tmp_path, scheduled_workers=1, max_replacements=1, max_trials=15)
    assert restarted.run(seed) == result
    assert restarted.report_test(seed) == report
    assert new_suite.resets == [] and new_quota.status() == quota.status()
    # A raw completed Evaluation owns recovery evidence even after predecessor loss.
    shutil.move(str(failed), str(tmp_path / 'retained-failed-attempt'))
    assert Evaluation.read(ExperimentRuntime(failed.parent), replacement['run_id']).to_dict() == value


def test_default_refusal_and_quota_exhaustion_do_not_dispatch(tmp_path, monkeypatch):
    _, initial, suite, quota, loop, _, options = interrupted(tmp_path / 'disabled', monkeypatch, allowance=0)
    before = len(suite.resets)
    with pytest.raises(ValueError, match='allowance'):
        loop.evaluator.authorize_replacement(initial, **options)
    assert len(suite.resets) == before
    _, initial, suite, quota, loop, failed, options = interrupted(tmp_path / 'bounded', monkeypatch, quota=10)
    before = len(suite.resets)
    with pytest.raises(TrialLimitExceeded):
        loop.evaluator.authorize_replacement(initial, **options)
    assert len(suite.resets) == before and quota.status()['reserved_trials'] == 10
    assert not (failed.parent.parent / 'replacements.json').exists()


def test_recovery_publication_interruption_is_idempotent(tmp_path, monkeypatch):
    _, initial, suite, quota, loop, failed, options = interrupted(tmp_path, monkeypatch)
    from PhysicalRSI_core.infra import trial_recovery
    publish = trial_recovery.atomic_json
    before = len(suite.resets)

    def stop(path, value):
        if Path(path).name == 'replacements.json':
            raise RuntimeError('replacement publication interrupted')
        return publish(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(trial_recovery, 'atomic_json', stop)
        with pytest.raises(RuntimeError, match='publication interrupted'):
            loop.evaluator.authorize_replacement(initial, **options)
    count = quota.status()['reserved_trials']
    replacement = loop.evaluator.authorize_replacement(initial, **options)
    assert replacement['run_id'] != failed.name
    assert quota.status()['reserved_trials'] == count and len(suite.resets) == before


def test_changed_failed_evidence_blocks_replacement_execution(tmp_path, monkeypatch):
    campaign, initial, suite, quota, loop, failed, options = interrupted(tmp_path, monkeypatch, target=1)
    loop.evaluator.authorize_replacement(initial, **options)
    before = len(suite.resets)
    atomic_json(failed / 'extra.json', {'unexpected': True})
    with pytest.raises(ValueError, match='Failed Trial changed'):
        campaign.run(initial)
    assert len(suite.resets) == before


class StartupEnvironment(demo.Environment):
    def __init__(self, *, failures=1, leased=False, action_failure=False, invalid=False):
        super().__init__('axis', 2., [])
        self.failures, self.leased = failures, leased
        self.action_failure, self.invalid, self.starts = action_failure, invalid, 0

    def describe(self):
        return replace(super().describe(), resources=('fixture:simulator',) if self.leased else ())

    def reset(self, case, context):
        self.starts += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError('startup interrupted')
        if self.invalid:
            return dict(ready=False)
        return super().reset(case, context)

    def step(self, action, context):
        result = super().step(action, context)
        if self.action_failure:
            raise RuntimeError('reply lost after action')
        return result

    def quiesce(self, context):
        return dict(quiescent=True)


def standalone(tmp_path, *, failures=1, leased=False, action_failure=False, invalid=False,
               coefficient=.5, maximum=2, paid=True):
    directory = tmp_path / 'evaluation'
    registry = DeviceRegistry(tmp_path / 'devices') if leased else None
    runtime = ExperimentRuntime(directory / 'trials', device_registry=registry)
    trial = Trial('original', task='axis', case=dict(initial=[0], target=[2]), scope='software', budget=Budget(1, 30))
    spec = trial.to_dict()
    plan = dict(scope='software', trials=[dict(run_id='original', **{k: spec[k] for k in ('task', 'case', 'budget')})])
    quota = TrialQuota(tmp_path / 'quota', max_trials=1 + maximum)
    if paid:
        quota.reserve(reservations(directory, plan))
    recovery = TrialRecovery(directory, plan, quota=quota, device_registry=registry, maximum=maximum)
    environment = StartupEnvironment(failures=failures, leased=leased, action_failure=action_failure, invalid=invalid)
    adapters = dict(environment=environment, policy=demo.Policy('axis', '0'*64, coefficient), verifier=demo.Verifier())
    if failures or action_failure:
        with pytest.raises(RuntimeError):
            runtime.run_trial(trial, **adapters)
    else:
        runtime.run_trial(trial, **adapters)
    proof = tmp_path / 'stopped.json'
    atomic_json(proof, dict(quiescent=True, trial='original', environment='synchronous software fixture'))
    options = dict(operator='fixture-owner', reason='Synchronous call stopped',
                   evidence={str(proof): file_digest(proof)}, quiescent=True)
    return runtime, trial, recovery, quota, registry, adapters, options


def replacement_trial(trial, recovery):
    resolution = recovery.resolve(trial.to_dict()['id'])
    original = trial.arguments()
    original['run_id'] = resolution['run_id']
    original['replacement'] = resolution['replacement']
    return Trial(**original)


def test_prepared_replacement_requires_reference_before_reset(tmp_path):
    runtime, trial, recovery, quota, _, adapters, options = standalone(tmp_path)
    run_id = recovery.authorize('original', 'original', **options)
    arguments = dict(trial.arguments(), run_id=run_id)
    before = adapters['environment'].starts
    with pytest.raises(ReconciliationRequired, match='authorization reference'):
        runtime.evaluate(Trial(**arguments), **adapters)
    assert adapters['environment'].starts == before
    assert not (runtime.root / run_id / 'receipt.json').exists()
    assert quota.status()['reserved_trials'] == 2
    assert runtime.evaluate(replacement_trial(trial, recovery), **adapters).outcome == 'success'


@pytest.mark.parametrize('boundary', ['development', 'validation', 'queue', 'preflight'])
def test_completed_replacement_cannot_lose_planned_authorization(tmp_path, monkeypatch, boundary):
    target = 1 if boundary == 'development' else 3
    campaign, initial, suite, _, loop, failed, options = interrupted(tmp_path, monkeypatch, target=target)
    replacement = loop.evaluator.authorize_replacement(initial, **options)
    payload = None
    if target == 3:
        key, job = next((key, job) for key, job in loop.evaluator.scheduler.status()['jobs'].items()
                        if job['state'] == 'needs_reconciliation')
        payload = job['request']['payload']
        loop.evaluator.resume_replacements(key, operator=options['operator'],
            reason=options['reason'], evidence=options['evidence'])
    campaign.run(initial)
    folder = failed.parent / replacement['run_id']
    # A self-consistent ordinary receipt is insufficient when the plan requires
    # a replacement. Keep the authorization bundle and ledger intact.
    spec = read_json(folder / 'experiment.json')
    spec.pop('replacement')
    atomic_json(folder / 'experiment.json', spec)
    receipt = read_json(folder / 'receipt.json')
    receipt['experiment_sha256'] = digest(spec)
    receipt['evidence'] = {name: sha for name, sha in receipt['evidence'].items()
                           if not name.startswith('recovery/')}
    receipt['evidence']['experiment.json'] = file_digest(folder / 'experiment.json')
    atomic_json(folder / 'receipt.json', receipt)
    ordinary = Evaluation.read(ExperimentRuntime(failed.parent), replacement['run_id']).to_dict()
    assert 'replacement' not in ordinary['trial'] and 'recovery' not in ordinary
    before = len(suite.resets)
    message = 'Completed trial differs' if boundary == 'preflight' else 'replacement differs'
    with pytest.raises(ValueError, match=message):
        if boundary == 'development':
            loop.evaluator.read_development(initial, loop.root)
        elif boundary == 'validation':
            loop.evaluator.read_evaluation(*loop.evaluator._arguments(payload))
        elif boundary == 'queue':
            loop.evaluator.replacement_plan(payload)
        else:
            candidate, comparison, cohort, output = loop.evaluator._arguments(payload)
            _, _, directory, plan = loop.evaluator._evaluation_plan(candidate, comparison, cohort, output)
            loop.evaluator._run_plan(candidate, directory, plan, comparison=comparison)
    assert len(suite.resets) == before


@pytest.mark.parametrize('changes', [dict(failures=0), dict(failures=0, coefficient=1),
                                     dict(failures=0, invalid=True), dict(failures=0, action_failure=True)])
def test_completed_outcomes_and_dispatched_actions_cannot_be_replaced(tmp_path, changes):
    runtime, _, recovery, quota, _, adapters, options = standalone(tmp_path, **changes)
    before, receipt = adapters['environment'].starts, file_digest(runtime.root / 'original/receipt.json')
    with pytest.raises(ValueError, match='before reset readiness'):
        recovery.authorize('original', 'original', **options)
    assert adapters['environment'].starts == before and quota.status()['reserved_trials'] == 1
    assert file_digest(runtime.root / 'original/receipt.json') == receipt


def test_live_trial_owner_and_unpaid_original_are_rejected(tmp_path):
    runtime, _, recovery, quota, _, _, options = standalone(tmp_path / 'live')
    with inactive(runtime.root / 'original/.lock'):
        with pytest.raises(ReconciliationRequired, match='live execution'):
            recovery.authorize('original', 'original', **options)
    assert quota.status()['reserved_trials'] == 1
    _, _, recovery, quota, _, _, options = standalone(tmp_path / 'unpaid', paid=False)
    with pytest.raises(ValueError, match='reservations'):
        recovery.authorize('original', 'original', **options)
    assert quota.status()['reserved_trials'] == 0


def test_device_claim_must_be_reconciled_before_replacement(tmp_path):
    runtime, trial, recovery, quota, registry, adapters, options = standalone(tmp_path, leased=True)
    lease = read_json(runtime.root / 'original/lease.json')
    with pytest.raises(ReconciliationRequired, match='device lease'):
        recovery.authorize('original', 'original', **options)
    assert quota.status()['reserved_trials'] == 1 and registry.occupied()
    registry.reconcile(lease['token'], operator=options['operator'], reason=options['reason'], evidence=options['evidence'])
    recovery.authorize('original', 'original', **options)
    result = runtime.evaluate(replacement_trial(trial, recovery), **adapters)
    assert result.outcome == 'success'
    assert quota.status()['reserved_trials'] == 2 and registry.occupied() == {}
    assert read_json(runtime.root / 'original/receipt.json')['state'] == 'needs_reconciliation'


def test_second_startup_failure_needs_another_authorization_and_reservation(tmp_path):
    runtime, trial, recovery, quota, _, adapters, options = standalone(tmp_path, failures=2)
    first = recovery.authorize('original', 'original', **options)
    with pytest.raises(RuntimeError, match='startup interrupted'):
        runtime.run_trial(replacement_trial(trial, recovery), **adapters)
    with pytest.raises(ReconciliationRequired):
        runtime.run_trial(replacement_trial(trial, recovery), **adapters)
    assert adapters['environment'].starts == 2 and quota.status()['reserved_trials'] == 2
    proof = tmp_path / 'second-stopped.json'
    atomic_json(proof, dict(quiescent=True, trial=first))
    second = recovery.authorize('original', first, **dict(options, evidence={str(proof): file_digest(proof)}))
    assert second != first
    result = runtime.evaluate(replacement_trial(trial, recovery), **adapters)
    assert result.outcome == 'success' and result.to_dict()['recovery']['index'] == 2
    assert quota.status()['reserved_trials'] == adapters['environment'].starts == 3
    assert runtime.evaluate(replacement_trial(trial, recovery), **adapters) == result
    assert adapters['environment'].starts == 3
    with pytest.raises(ValueError, match='changed'):
        recovery.authorize('original', first, **options)


def test_replacement_cap_and_owned_proof_corruption_fail_closed(tmp_path):
    runtime, trial, recovery, quota, _, adapters, options = standalone(tmp_path / 'cap', failures=2, maximum=1)
    first = recovery.authorize('original', 'original', **options)
    with pytest.raises(RuntimeError):
        runtime.run_trial(replacement_trial(trial, recovery), **adapters)
    with pytest.raises(ReconciliationRequired, match='exhausted'):
        recovery.authorize('original', first, **options)
    assert quota.status()['reserved_trials'] == 2
    runtime, trial, recovery, _, _, adapters, options = standalone(tmp_path / 'corrupt')
    first = recovery.authorize('original', 'original', **options)
    result = runtime.evaluate(replacement_trial(trial, recovery), **adapters)
    record = result.to_dict()['recovery']
    owned = next(iter(record['proofs'].values()))['owned']
    (runtime.root / first / 'recovery' / owned).write_text('corrupt')
    with pytest.raises(ValueError, match='evidence changed'):
        Evaluation.read(runtime, first)


def test_missing_or_changed_phase_cannot_authorize_legacy_unknown_work(tmp_path):
    runtime, _, recovery, quota, _, _, options = standalone(tmp_path)
    path = runtime.root / 'original/receipt.json'
    receipt = read_json(path)
    assert receipt['execution_phase'] == 'reset'
    receipt.pop('execution_phase')
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match='before reset readiness'):
        recovery.authorize('original', 'original', **options)
    assert quota.status()['reserved_trials'] == 1


@pytest.mark.parametrize('completed', [False, True])
def test_missing_receipt_cannot_replay_existing_execution(tmp_path, completed):
    runtime, trial, _, _, _, adapters, _ = standalone(tmp_path, failures=0 if completed else 1)
    (runtime.root / 'original/receipt.json').unlink()
    before = adapters['environment'].starts
    with pytest.raises(ReconciliationRequired, match='Unpublished experiment'):
        runtime.run_trial(trial, **adapters)
    assert adapters['environment'].starts == before


@pytest.mark.parametrize('task', ['axis', 'vector'])
def test_screening_replacement_resumes_campaign_without_rerolling_completed_trials(tmp_path, monkeypatch, task):
    from PhysicalRSI_demos.staged_evolution import CalibrationSearch

    def setup():
        return demo.setup(tmp_path, evolver_factory=CalibrationSearch, max_candidates=4,
            scheduled_workers=1, screening_threshold=1.0, max_replacements=1, max_trials=23)

    campaign, initial, suite, quota = setup()
    reset = demo.Environment.reset

    def crash(self, case, context):
        if len(self.resets) >= 2 and self.task == task and case['target'][0] == 1:
            self.resets.append((self.task, case))
            raise RuntimeError('screening startup stopped')
        return reset(self, case, context)

    with monkeypatch.context() as patch:
        patch.setattr(demo.Environment, 'reset', crash)
        with pytest.raises(RuntimeError, match='screening startup stopped'):
            campaign.run(initial)
    round_root = campaign.root / 'rounds/round-0001'
    loop = campaign.build_loop(round_root / 'execution', read_json(round_root / 'context.json'))
    evaluator = loop.evaluator
    jobs = evaluator.scheduler.status()['jobs']
    key = next(k for k, job in jobs.items() if job['state'] == 'needs_reconciliation')
    candidate = jobs[key]['request']['payload']['candidate']
    failed = next(p.parent for p in loop.root.glob('experiments/screening/**/receipt.json')
                  if read_json(p)['state'] != 'completed')
    before = {p.name: file_digest(p) for p in failed.iterdir() if p.is_file() and not p.name.endswith('.lock')}
    proof = tmp_path / 'screening-stop.json'
    atomic_json(proof, dict(quiescent=True, description='Synchronous reset exited before ready'))
    audit = dict(operator='test-operator', reason='Stopped screening startup',
                 evidence={str(proof): file_digest(proof)})
    resets = len(suite.resets)
    with pytest.raises(ReconciliationRequired):
        evaluator.resume_screening_replacements(key, **audit)
    assert len(suite.resets) == resets
    reserved = quota.status()['reserved_trials']
    replacement = evaluator.authorize_screening_replacement(candidate, output=loop.root,
        run_id=failed.name, quiescent=True, **audit)
    assert quota.status()['reserved_trials'] == reserved + 1
    with pytest.raises(ValueError, match='handler identity'):
        evaluator.resume_replacements(key, **audit)
    resumed = evaluator.resume_screening_replacements(key, **audit)
    assert evaluator.resume_screening_replacements(key, **audit) == resumed
    assert len(suite.resets) == resets
    result = campaign.run(initial)
    report = campaign.report_test(initial)
    assert [row['outcome'] for row in result['rounds']] == ['inherit_child', 'no_changed_candidate']
    assert all(row['success_rate'] == 1 for row in report['metrics'].values())
    assert len(suite.resets) == quota.status()['reserved_trials'] == 23
    assert {name: file_digest(failed / name) for name in before} == before
    assert Evaluation.read(ExperimentRuntime(failed.parent), replacement['run_id']).to_dict()['recovery']['original_id'] == failed.name
    assert evaluator.scheduler.status()['used'] == {'cpu': 0}
    restarted, seed, new_suite, new_quota = setup()
    assert restarted.run(seed) == result and restarted.report_test(seed) == report
    assert new_suite.resets == [] and new_quota.status() == quota.status()
