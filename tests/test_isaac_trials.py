"""Contract doubles only; these tests provide no native simulator qualification."""
from pathlib import Path
import subprocess
import sys
from threading import Thread
import time

import pytest

from PhysicalRSI_core.contracts import Cancelled, Context, ReconciliationRequired
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.trials import Trial
from PhysicalRSI_demos.auto_engineering import research, trials
from PhysicalRSI_demos.auto_engineering.harness import create_seed
from PhysicalRSI_demos.auto_engineering.task import TASK, SCOPE, layout, judge
from PhysicalRSI_demos.auto_engineering.trial_bridge import receive_plan


class ContractWorker(trials.NativeWorker):
    resets = []
    success_height = .2
    phase = 9
    finger_position = .1
    disturb_on_failure = False

    def identity(self):
        return dict(kind="contract-test-double", source=file_digest(Path(__file__)))

    def reset(self, case, context):
        self.case = case
        self.output = self.root / (context.episode + "-" + context.harness_revision)
        self.output.mkdir(parents=True, exist_ok=False)
        self.resets.append(dict(case=case, candidate=context.harness_revision))
        public = {name: dict(status="estimated_surface", center_xy=[.4, .2], top_z=z)
                  for name, z in (("tube", .13), ("receiver", .02), ("obstacle", .18))}
        self.initial = dict(public_observation=public, joint_positions=[0.] * 9)
        return dict(ready=True, observation=self.initial)

    def execute(self, action, context):
        spec = layout(**self.case)
        success = action["memory"]["transit_height_m"] > self.success_height
        measured = dict(tube_position=[*spec["receiver"][:2], .081] if success else spec["tube"],
            tube_orientation_wxyz=[1, 0, 0, 0], tube_velocity=[0, 0, 0], obstacle_position=spec["obstacle"],
            distractor_position=spec["distractor"], gripper_opening=.08)
        if self.disturb_on_failure and not success:
            measured['obstacle_position'] = [spec['obstacle'][0]+.2, *spec['obstacle'][1:]]
        rollout = [dict(observation=self.initial), dict(step=0, action=dict(kind="joint_targets", joint_positions=[.1]*9),
            observation=dict(joint_positions=[.1]*7+[self.finger_position]*2), phase=self.phase)]
        for name, value in {"layout": spec, "private-evaluation": measured, "plan": action["plan"],
                            "public-rollout": rollout}.items():
            atomic_json(self.output / (name + ".json"), value)
        result = dict(state="completed", backend="isaacsim-5.1", **judge(spec, measured),
            runtime_version="5.1.0",
            seed=self.case["seed"], memory=action["memory"], public_observation=self.initial["public_observation"],
            layout_sha256=digest(spec), controller_completed=True, physics_steps=1,
            evidence={p.name: file_digest(p) for p in self.output.glob("*.json")})
        atomic_json(self.output / "result.json", result)
        return result, rollout


def test_native_adapter_campaign_attribution_restart_and_media_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "NativeWorker", ContractWorker)
    ContractWorker.resets = []
    result = research.run(tmp_path, sys.executable, episodes=2, rounds=2)
    assert result["state"] == "completed" and result["qualification"] is None
    assert result["scope"] == SCOPE
    assert [row["outcome"] for row in result["rounds"]] == ["inherit_child", "no_changed_candidate"]
    assert result["final_test"]["metrics"][TASK]["success_rate"] == 1
    assert result["quota"]["reserved_trials"] == len(ContractWorker.resets) == 8
    assert len({row["candidate"] for row in ContractWorker.resets}) == 2
    assert research.run(tmp_path, sys.executable, episodes=2, rounds=2) == result
    assert len(ContractWorker.resets) == 8
    receipt = next((tmp_path / "campaign/rounds").glob("**/trials/*/receipt.json"))
    payload = read_json(receipt)
    assert payload["binding"]["embodiment"]["mode"] == "simulation"
    assert "artifacts/native/public-rollout.json" in payload["evidence"]
    (receipt.parent / "artifacts/native/public-rollout.json").write_text("[]")
    with pytest.raises(ValueError):
        research.run(tmp_path, sys.executable, episodes=2, rounds=2)
    assert len(ContractWorker.resets) == 8


def test_worker_reset_waits_for_system1_action(tmp_path):
    observed = {}
    def worker():
        observed["action"] = receive_plan(tmp_path, {"joint_positions": [0]}, timeout=3)
    thread = Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 2
    while not (tmp_path / "ready.json").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert read_json(tmp_path / "ready.json")["ready"] is True
    assert thread.is_alive()
    request = dict(memory={"height": .3}, plan={"pick": [0, 0, 0]})
    atomic_json(tmp_path / "action.json", request)
    thread.join(timeout=3)
    assert not thread.is_alive() and observed["action"] == request
    with pytest.raises(TimeoutError):
        receive_plan(tmp_path / "timeout", {}, timeout=.01)


@pytest.mark.parametrize('screen_incumbent',[False,True])
def test_native_staged_contract_keeps_screening_separate_and_releases_queue(tmp_path,monkeypatch,screen_incumbent):
    monkeypatch.setattr(trials,"NativeWorker",ContractWorker)
    ContractWorker.resets=[]
    result=research.run(tmp_path,sys.executable,episodes=3,rounds=2,staged=True,screen_incumbent=screen_incumbent)
    assert result['scope']==SCOPE and result['qualification'] is None
    assert result['quota']['reserved_trials']==len(ContractWorker.resets)==14+int(screen_incumbent)
    assert [r['outcome'] for r in result['rounds']]==['inherit_child','no_changed_candidate']
    directory=tmp_path/'campaign/rounds/round-0001/execution'
    screening=read_json(directory/'screening.json')['output']
    assert screening['role']=='screening_only'
    assert screening['candidates']['initial']['measured'] is screen_incumbent
    assert screening['candidates']['initial']['per_task']==({TASK:0.} if screen_incumbent else None)
    assert {name for name,v in screening['candidates'].items() if not v['accepted']}=={'low-clearance-0-0','low-clearance-0-1'}
    comparison=read_json(directory/'freeze.json')['output']
    assert set(comparison['candidates'])=={'initial','clearance-0'}
    assert {x['case']['seed'] for x in ContractWorker.resets if x['case']['seed']<10000}=={5000,5001}
    assert result['scheduling']['used']=={'native_worker':0}
    assert len(result['scheduling']['jobs'])==5+int(screen_incumbent)
    assert all(j['state']=='completed' and not j['owner_active'] for j in result['scheduling']['jobs'].values())
    assert research.run(tmp_path,sys.executable,episodes=3,rounds=2,staged=True,screen_incumbent=screen_incumbent)==result
    assert len(ContractWorker.resets)==14+int(screen_incumbent)


def test_native_reset_timeout_kills_owned_worker_and_blocks_reexecution(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    processes = []
    def sleeper(command, **kwargs):
        process = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(trials.subprocess, "Popen", sleeper)
    suite = trials.Suite(sys.executable, tmp_path / "native")
    candidate = create_seed(tmp_path / "seed")
    runtime = ExperimentRuntime(tmp_path / "experiments", device_registry=DeviceRegistry(tmp_path / "devices"))
    trial = Trial("timeout", task=TASK, case=dict(seed=17), scope=SCOPE, budget=Budget(1, .5))
    def run():
        return runtime.run_trial(trial, environment=suite.environment(TASK),
                                 policy=suite.policy(candidate, TASK), verifier=suite.verifier(TASK))
    with pytest.raises(Cancelled):
        run()
    assert len(processes) == 1 and processes[0].poll() is not None
    with pytest.raises(ReconciliationRequired):
        run()
    assert len(processes) == 1


class RefinementContractWorker(ContractWorker):
    success_height = .4
    phase = 3
    finger_position = .018
    disturb_on_failure = True


def test_native_contract_recovers_declared_startup_replacement_through_research_entry(tmp_path, monkeypatch):
    class InterruptedWorker(ContractWorker):
        failed = False
        resets = []

        def reset(self, case, context):
            if case['seed'] == 20000 and not type(self).failed:
                type(self).failed = True
                self.resets.append(dict(case=case, candidate=context.harness_revision))
                raise RuntimeError('contract worker stopped before readiness')
            return super().reset(case, context)

    monkeypatch.setattr(trials, 'NativeWorker', InterruptedWorker)
    options = dict(episodes=1, rounds=1, staged=True, max_replacements=1, extra_trials=1)
    campaign, initial, quota = research.setup(tmp_path, sys.executable, **options)
    with pytest.raises(RuntimeError, match='before readiness'):
        campaign.run(initial)
    directory = campaign.root / 'rounds/round-0001'
    loop = campaign.build_loop(directory / 'execution', read_json(directory / 'context.json'))
    receipt = next(p for p in loop.root.glob('experiments/validation/**/trials/*/receipt.json')
                   if read_json(p)['state'] == 'needs_reconciliation')
    saved = file_digest(receipt)
    lease = read_json(receipt.parent / 'lease.json')
    registry = DeviceRegistry(tmp_path / 'devices')
    proof = tmp_path / 'stopped.json'
    atomic_json(proof, dict(quiescent=True, trial=receipt.parent.name, scope='contract worker; no simulator launched'))
    evidence = {str(proof): file_digest(proof)}
    registry.reconcile(lease['token'], operator='contract-test', reason='Synchronous worker stopped', evidence=evidence)
    loop.evaluator.authorize_replacement(initial, output=loop.root, split='validation', run_id=receipt.parent.name,
        operator='contract-test', reason='Synchronous worker stopped', evidence=evidence, quiescent=True)
    key = next(k for k, row in loop.evaluator.scheduler.status()['jobs'].items() if row['state'] == 'needs_reconciliation')
    loop.evaluator.resume_replacements(key, operator='contract-test', reason='Replacement authorized', evidence=evidence)
    result = research.run(tmp_path, sys.executable, **options)
    assert result['state'] == 'completed'
    assert result['final_test']['metrics'][TASK]['success_rate'] == 1
    assert result['quota']['max_trials'] == 10
    assert result['quota']['reserved_trials'] == len(InterruptedWorker.resets) == 8
    assert registry.occupied() == {} and result['scheduling']['used'] == {'native_worker': 0}
    assert file_digest(receipt) == saved
    assert research.run(tmp_path, sys.executable, **options) == result
    assert len(InterruptedWorker.resets) == 8


@pytest.mark.parametrize('options', [dict(max_replacements=True), dict(max_replacements=-1),
    dict(max_replacements=11), dict(extra_trials=True), dict(extra_trials=-1), dict(extra_trials=1.5)])
def test_native_recovery_budget_rejected_before_workspace_creation(tmp_path, options):
    root = tmp_path / 'unused'
    with pytest.raises(ValueError, match='Trial'):
        research.setup(root, sys.executable, **options)
    assert not root.exists()


def test_native_feedback_contract_reuses_rejected_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, 'NativeWorker', RefinementContractWorker)
    RefinementContractWorker.resets=[]
    result=research.run(tmp_path,sys.executable,episodes=3,rounds=2,feedback_search=True)
    assert [r['outcome'] for r in result['rounds']]==['all_candidates_rejected','inherit_child']
    fits=[read_json(p) for p in sorted((tmp_path/'campaign/rounds').glob('*/execution/search-fit.json'))]
    assert [f['proposed_value'] for f in fits]==[.34,.445]
    assert any(s['value']==.34 and s['source']['role']=='development_screening'
               for s in fits[1]['samples'] if isinstance(s['source'],dict))
    assert all(s['case']=={'seed':1000} for f in fits for s in f['samples'])
    assert len(RefinementContractWorker.resets)==result['quota']['reserved_trials']==13
    assert result['final_test']['metrics'][TASK]['success_rate']==1
    assert result['qualification'] is None
    assert result['scheduling']['used']=={'native_worker':0}
    assert result['screening']['proposal_roles']==['development-scalar-refinement']
    assert research.run(tmp_path,sys.executable,episodes=3,rounds=2,feedback_search=True)==result
    assert len(RefinementContractWorker.resets)==13


@pytest.mark.parametrize('aperture,disturbed,expected',[(.036,True,'insufficient'),(.08,True,'unresolved'),
    (.0,True,'unresolved'),(.036,False,'unresolved'),(float('nan'),True,'unresolved')])
def test_clearance_feedback_requires_grasp_and_disturbance(aperture,disturbed,expected):
    from PhysicalRSI_demos.auto_engineering.feedback import ClearanceFeedback
    value=dict(trial=dict(task=TASK,scope=SCOPE),outcome='failure',
        verdict=dict(measurements=dict(checks=dict(obstacle_undisturbed=not disturbed))),
        trajectory=[dict(native_rollout=[dict(phase=3,observation=dict(joint_positions=[0]*7+[aperture/2]*2))])])
    assert ClearanceFeedback().classify(value)==expected


class EmpiricalContractWorker(ContractWorker):
    """Give every changed vector a known success to test feedback plumbing.

    The runtime identity participates in sampler seeding. Contract assertions
    must not depend on an incidental random draw crossing a height threshold.
    Native studies use the unmodified physical verifier instead.
    """
    def execute(self, action, context):
        memory = action['memory']
        unchanged = (memory['transit_height_m'], memory['grasp_depth_fraction']) == (.13, .25)
        self.success_height = .55 if unchanged else .09
        return super().execute(action, context)


@pytest.mark.parametrize('sampler_seed', [0, 20261013])
@pytest.mark.parametrize('elite_weighting', ['uniform', 'fractional'])
@pytest.mark.parametrize('repetitions', [1, 2])
def test_native_empirical_contract_uses_complete_panels_and_two_mutable_parameters(tmp_path,monkeypatch,sampler_seed,elite_weighting,repetitions):
    from PhysicalRSI_demos.auto_engineering import empirical
    monkeypatch.setattr(trials,'NativeWorker',EmpiricalContractWorker)
    EmpiricalContractWorker.resets=[]
    panels=dict(development=[1000,50001],validation=[[62000,62001],[62100,62101]],test=[63000,63001,63002])
    options=dict(panels=panels,strategy='adaptive',seed=sampler_seed,shared_cache=True,
                 elite_weighting=elite_weighting,repetitions=repetitions)
    result=empirical.run(tmp_path,sys.executable,**options)
    fits=[read_json(p) for p in sorted((tmp_path/'campaign/rounds').glob('*/execution/adaptive-fit.json'))]
    assert all(f['identity']['settings']['elite_weighting']==elite_weighting for f in fits)
    assert len(fits)==2 and [f['model_ready'] for f in fits]==[False,True]
    assert len(fits[0]['panel'])==len(fits[1]['panel'])==2
    assert len(fits[1]['ranked'])==3
    assert [p['role'] for p in fits[0]['proposals']]==['global','global']
    assert [p['role'] for p in fits[1]['proposals']]==['global','elite']
    assert all(len(p['values'])==2 and .1<=p['values'][0]<=.55 and .15<=p['values'][1]<=.4
               for f in fits for p in f['proposals'])
    assert all(s['case']['seed'] in {1000,50001} for f in fits for s in f['samples'])
    assert result['quota']['max_trials']==27*repetitions and result['quota']['reserved_trials']==len(EmpiricalContractWorker.resets)
    assert len(fits[0]['samples'])==2*repetitions
    assert result['final_test']['metrics'][TASK]['planned']==3*repetitions
    screens=list((tmp_path/'campaign/rounds').glob('*/execution/screening.json'))
    assert screens
    for path in screens:
        for row in read_json(path)['output']['candidates'].values():
            if row['measured']:
                assert row['per_task'][TASK] == 1.
    assert result['qualification'] is None
    before=len(EmpiricalContractWorker.resets)
    assert empirical.run(tmp_path,sys.executable,**options)==result
    assert len(EmpiricalContractWorker.resets)==before
    scheduler=read_json(tmp_path/'campaign/rounds/round-0001/execution/config.json')['identities']['evaluator']['scheduling']
    assert scheduler['workers']==1
    candidate=result['current']['harness']
    assert read_json(Path(candidate['root'])/'memory_rules.json')['sample_height_m']==.12


def test_native_objective_gives_partial_feedback_without_claiming_success():
    from PhysicalRSI_demos.auto_engineering.empirical import Objective,CHECKS
    value=dict(trial=dict(task=TASK,scope=SCOPE),outcome='failure',
        verdict=dict(measurements=dict(checks={name:True for name in CHECKS})))
    assert Objective().score(value)==.9
    value['verdict']['measurements']['checks']['upright']=False
    assert Objective().score(value)==pytest.approx(.9*6/7)
    value['outcome']='success'
    with pytest.raises(ValueError,match='contradicts'):
        Objective().score(value)
    value['verdict']['measurements']['checks']['upright']=True
    assert Objective().score(value)==1
    value['verdict']['measurements']['checks']['upright']=1
    with pytest.raises(ValueError,match='boolean'):
        Objective().score(value)
    value['outcome']='uncertain'
    assert Objective().score(value) is None


@pytest.mark.parametrize('panels',[
    dict(development=[1],validation=[[2],[2]],test=[3]),
    dict(development=[1,1],validation=[[2]],test=[3]),
    dict(development=[1],validation=[[2]],test=[1]),
    dict(development=[True],validation=[[2]],test=[3]),
    dict(development=[1],validation=[[2],[4,5]],test=[3]),
])
def test_native_empirical_rejects_ambiguous_panels_before_workspace_creation(tmp_path,panels):
    from PhysicalRSI_demos.auto_engineering.empirical import setup
    root=tmp_path/'unused'
    with pytest.raises(ValueError):
        setup(root,sys.executable,panels=panels,strategy='uniform',seed=1)
    assert not root.exists()


def test_native_empirical_scenario_cases_reach_trials_and_lower_tail_fit(tmp_path, monkeypatch):
    from PhysicalRSI_demos.auto_engineering import empirical
    monkeypatch.setattr(trials, 'NativeWorker', ContractWorker)
    ContractWorker.resets=[]
    panels=dict(development=[1,dict(seed=2,scenario='tall-obstacle')],
                validation=[[3,dict(seed=4,scenario='tall-obstacle')]],
                test=[5,dict(seed=6,scenario='tall-obstacle')])
    result=empirical.run(tmp_path,sys.executable,panels=panels,strategy='adaptive',seed=1,
                         lower_tail_fraction=.5)
    fit=read_json(tmp_path/'campaign/rounds/round-0001/execution/adaptive-fit.json')
    assert fit['identity']['settings']['lower_tail_fraction']==.5
    assert {s['case']['seed'] for s in fit['samples']}=={1,2}
    assert any(r['case']==dict(seed=6,scenario='tall-obstacle') for r in ContractWorker.resets)
    assert result['quota']['max_trials']==14
    with pytest.raises(ValueError):
        empirical._panels(dict(development=[1],validation=[[dict(seed=1,scenario='tall-obstacle')]],test=[2]))


def test_empirical_screening_startup_recovery_preserves_search_and_budget(tmp_path, monkeypatch):
    from PhysicalRSI_demos.auto_engineering import empirical

    class InterruptedScreening(EmpiricalContractWorker):
        resets = []
        failed = False

        def reset(self, case, context):
            if case['seed'] == 1000 and self.resets and not type(self).failed:
                type(self).failed = True
                self.resets.append(dict(case=case, candidate=context.harness_revision))
                raise RuntimeError('screening worker stopped before readiness')
            return super().reset(case, context)

    monkeypatch.setattr(trials, 'NativeWorker', InterruptedScreening)
    options = dict(panels=dict(development=[1000], validation=[[71000]], test=[72000]),
                   strategy='adaptive', seed=20261013, max_replacements=1, extra_trials=1)
    campaign, initial, quota = empirical.setup(tmp_path, sys.executable, **options)
    with pytest.raises(RuntimeError, match='before readiness'):
        campaign.run(initial)
    directory = campaign.root / 'rounds/round-0001'
    loop = campaign.build_loop(directory / 'execution', read_json(directory / 'context.json'))
    evaluator = loop.evaluator
    jobs = evaluator.scheduler.status()['jobs']
    key = next(k for k, row in jobs.items() if row['state'] == 'needs_reconciliation')
    candidate = jobs[key]['request']['payload']['candidate']
    receipt = next(p for p in loop.root.glob('experiments/screening/**/receipt.json')
                   if read_json(p)['state'] == 'needs_reconciliation')
    saved = file_digest(receipt)
    lease = read_json(receipt.parent / 'lease.json')
    proof = tmp_path / 'stopped.json'
    atomic_json(proof, dict(quiescent=True, scope='Synchronous contract double; no simulator launched'))
    audit = dict(operator='contract-test', reason='Synchronous screening worker stopped',
                 evidence={str(proof): file_digest(proof)})
    registry = DeviceRegistry(tmp_path / 'devices')
    registry.reconcile(lease['token'], **audit)
    evaluator.authorize_screening_replacement(candidate, output=loop.root,
        run_id=receipt.parent.name, quiescent=True, **audit)
    evaluator.resume_screening_replacements(key, **audit)
    result = empirical.run(tmp_path, sys.executable, **options)
    assert result['status'] == 'completed'
    assert result['final_test']['metrics'][TASK]['success_rate'] == 1
    assert result['quota']['max_trials'] == 8
    assert result['quota']['reserved_trials'] == len(InterruptedScreening.resets)
    assert registry.occupied() == {} and evaluator.scheduler.status()['used'] == {'native_worker': 0}
    assert file_digest(receipt) == saved
    before = len(InterruptedScreening.resets)
    assert empirical.run(tmp_path, sys.executable, **options) == result
    assert len(InterruptedScreening.resets) == before


@pytest.mark.parametrize('options', [dict(max_replacements=True), dict(max_replacements=11),
                                    dict(extra_trials=-1), dict(extra_trials=1.5)])
def test_empirical_recovery_options_rejected_before_workspace_creation(tmp_path, options):
    from PhysicalRSI_demos.auto_engineering import empirical
    root = tmp_path / 'unused'
    with pytest.raises(ValueError, match='Trial'):
        empirical.setup(root, sys.executable, panels=dict(development=[1], validation=[[2]], test=[3]),
                        strategy='adaptive', seed=1, **options)
    assert not root.exists()


def test_native_shared_cache_reuses_only_derived_data_across_worker_roots(tmp_path, monkeypatch):
    launches = []

    def capture(command, **kwargs):
        launches.append(dict(command=command, **kwargs))
        raise RuntimeError('recorded launch boundary')

    monkeypatch.setattr(trials.subprocess, 'Popen', capture)
    cache = tmp_path / 'shared-cache'
    roots = [tmp_path / 'round-0', tmp_path / 'round-1-screening']
    for index, root in enumerate(roots):
        worker = trials.NativeWorker(sys.executable, root, cache_root=cache)
        assert worker.identity()['cache_root'] == str(cache)
        with pytest.raises(RuntimeError, match='launch boundary'):
            worker.reset(dict(seed=1), Context('episode-' + str(index), harness_revision='candidate'))
        assert (root / 'runtime/kit/cache').is_symlink()
        assert (root / 'runtime/kit/cache').resolve() == cache / 'kit'
        if index == 0:
            (cache / 'kit/compiled-probe').write_text('derived cache only')
        assert (root / 'runtime/kit/cache/compiled-probe').read_text() == 'derived cache only'
    for key in ('XDG_CACHE_HOME', 'CUDA_CACHE_PATH', '__GL_SHADER_DISK_CACHE_PATH'):
        assert launches[0]['env'][key] == launches[1]['env'][key]
    for key in ('TMPDIR', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME'):
        assert launches[0]['env'][key] != launches[1]['env'][key]
    assert launches[0]['command'] != launches[1]['command']
    assert not (cache / 'ipc').exists() and not (cache / 'episode').exists()


@pytest.mark.parametrize('conflict', ['existing-directory', 'different-link', 'nested-target', 'ancestor-target'])
def test_native_shared_cache_refuses_to_replace_existing_cache_or_create_cycle(tmp_path, conflict):
    root, shared = tmp_path / 'native', tmp_path / 'shared'
    local = root / 'runtime/kit/cache'
    local.parent.mkdir(parents=True)
    if conflict == 'existing-directory':
        local.mkdir()
        (local / 'retained').write_text('keep')
    elif conflict == 'different-link':
        old = tmp_path / 'previous-cache'
        old.mkdir()
        local.symlink_to(old, target_is_directory=True)
    elif conflict == 'nested-target':
        shared = local / 'nested'
    else:
        shared = root / 'runtime'
    worker = trials.NativeWorker(sys.executable, root, cache_root=shared)
    with pytest.raises(ValueError, match='cache'):
        worker._prepare_cache()
    if conflict != 'ancestor-target':
        assert not shared.exists()
    else:
        assert not local.exists()
    if conflict == 'existing-directory':
        assert (local / 'retained').read_text() == 'keep' and not local.is_symlink()
    elif conflict == 'different-link':
        assert local.resolve() == old


def test_native_cache_is_isolated_by_default(tmp_path):
    worker = trials.NativeWorker(sys.executable, tmp_path / 'native')
    assert worker._prepare_cache() == tmp_path / 'native/runtime'
    assert worker.identity()['cache_root'] is None
    assert not (tmp_path / 'native').exists()


def test_native_python_failure_survives_clean_worker_exit(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    processes = []
    def failed_worker(command, **kwargs):
        bridge = command[command.index('--bridge') + 1]
        script = ('from PhysicalRSI_demos.auto_engineering.trial_bridge import report_failure; '
                  'import sys; report_failure(sys.argv[1], TimeoutError("RGB-D sensors unavailable"))')
        process = real_popen([sys.executable, '-c', script, bridge], **kwargs)
        # Model Kit shutdown masking the original exception with exit status zero.
        assert process.wait(timeout=10) == 0
        processes.append(process)
        return process
    monkeypatch.setattr(trials.subprocess, 'Popen', failed_worker)
    suite = trials.Suite(sys.executable, tmp_path / 'native')
    candidate = create_seed(tmp_path / 'seed')
    registry = DeviceRegistry(tmp_path / 'devices')
    runtime = ExperimentRuntime(tmp_path / 'experiments', device_registry=registry)
    trial = Trial('sensor-error', task=TASK, case=dict(seed=17), scope=SCOPE, budget=Budget(1, 30))
    def run():
        return runtime.run_trial(trial, environment=suite.environment(TASK),
            policy=suite.policy(candidate, TASK), verifier=suite.verifier(TASK))
    with pytest.raises(RuntimeError, match='TimeoutError: RGB-D sensors unavailable'):
        run()
    receipt = read_json(tmp_path / 'experiments/sensor-error/receipt.json')
    assert receipt['state'] == 'needs_reconciliation'
    assert registry.occupied()
    with pytest.raises(ReconciliationRequired):
        run()
    assert len(processes) == 1


def test_native_reported_error_takes_precedence_over_reset_readiness(tmp_path):
    from types import SimpleNamespace
    from PhysicalRSI_demos.auto_engineering.trial_bridge import report_failure
    worker = trials.NativeWorker(sys.executable, tmp_path)
    worker.directory = tmp_path
    worker.ipc = tmp_path / 'ipc'
    atomic_json(worker.ipc / 'ready.json', dict(ready=True, observation={}))
    report_failure(worker.ipc, ValueError('invalid camera output'))
    worker.process = SimpleNamespace(poll=lambda: 0)
    with pytest.raises(RuntimeError, match='ValueError: invalid camera output'):
        worker._wait(SimpleNamespace(check=lambda: None), path=worker.ipc / 'ready.json')


def test_native_signal_exit_retains_exit_code_without_python_error(tmp_path):
    from types import SimpleNamespace
    worker = trials.NativeWorker(sys.executable, tmp_path)
    worker.directory = tmp_path
    worker.ipc = tmp_path / 'ipc'
    worker.process = SimpleNamespace(poll=lambda: -11)
    with pytest.raises(RuntimeError, match='exited with code -11'):
        worker._wait(SimpleNamespace(check=lambda: None), path=worker.ipc / 'ready.json')


@pytest.mark.parametrize('threshold,validated', [(1., False), (.5, True)])
def test_empirical_partial_screening_keeps_full_validation_gate(tmp_path, monkeypatch, threshold, validated):
    from PhysicalRSI_demos.auto_engineering import empirical

    class PartialWorker(EmpiricalContractWorker):
        def execute(self, action, context):
            if self.case['seed'] % 2:
                self.success_height = 1.
                return ContractWorker.execute(self, action, context)
            return super().execute(action, context)

    monkeypatch.setattr(trials, 'NativeWorker', PartialWorker)
    PartialWorker.resets = []
    options = dict(panels=dict(development=[100, 101], validation=[[200, 201]], test=[300, 301]),
                   strategy='adaptive', seed=1, repetitions=2,
                   screening_minimum_success_rate=threshold)
    result = empirical.run(tmp_path, sys.executable, **options)
    root = tmp_path / 'campaign/rounds/round-0001/execution'
    screening = read_json(root / 'screening.json')['output']
    children = [v for v in screening['candidates'].values() if v['measured']]
    assert len(children) == 2
    assert all(v['per_task'][TASK] == .5 and v['accepted'] == validated for v in children)
    assert screening['policy']['minimum_success_rate'] == threshold
    assert any(200 <= row['case']['seed'] < 300 for row in PartialWorker.resets) == validated
    # Partial screening gives access to fresh validation, never promotion or test credit.
    assert result['current']['harness']['id'] == 'initial'
    assert result['final_test']['metrics'][TASK]['success_rate'] == 0
    assert result['final_test']['metrics'][TASK]['planned'] == 4
    before = len(PartialWorker.resets)
    assert empirical.run(tmp_path, sys.executable, **options) == result
    assert len(PartialWorker.resets) == before
    with pytest.raises(ValueError):
        empirical.run(tmp_path, sys.executable,
                      **dict(options, screening_minimum_success_rate=.5 if threshold == 1. else 1.))
    assert len(PartialWorker.resets) == before


@pytest.mark.parametrize('threshold', [True, -.1, 1.1, float('nan'), float('inf'), '0.5', None])
def test_empirical_screening_threshold_rejected_before_workspace_creation(tmp_path, threshold):
    from PhysicalRSI_demos.auto_engineering import empirical
    root = tmp_path / 'unused'
    with pytest.raises(ValueError):
        empirical.setup(root, sys.executable, panels=dict(development=[1], validation=[[2]], test=[3]),
                        strategy='adaptive', seed=1, screening_minimum_success_rate=threshold)
    assert not root.exists()


def test_empirical_cli_preparation_resume_and_frozen_invocation(tmp_path, monkeypatch, capsys):
    from PhysicalRSI_demos.auto_engineering import empirical
    monkeypatch.setattr(trials, 'NativeWorker', EmpiricalContractWorker)
    EmpiricalContractWorker.resets = []
    plan = tmp_path / 'plan.json'
    options = dict(panels=dict(development=[100, 101], validation=[[200, 201]], test=[300, 301]),
                   strategy='adaptive', seed=1, repetitions=2, screening_minimum_success_rate=.5)
    atomic_json(plan, options)
    root = tmp_path / 'workspace'
    argv = ['--workspace', str(root), '--python', sys.executable, '--plan', str(plan)]
    prepared = empirical.main([*argv, '--prepare-only'])
    assert prepared['execution_started'] is False
    assert prepared['quota'] == dict(max_trials=28, reserved_trials=0, remaining_trials=28)
    assert EmpiricalContractWorker.resets == []
    assert not (root / 'native').exists()
    assert read_json(root / 'empirical-plan.json')['options'] == options
    assert empirical.main([*argv, '--prepare-only']) == prepared
    result = empirical.main(argv)
    assert result['final_test']['metrics'][TASK]['planned'] == 4
    assert result['final_test']['metrics'][TASK]['success_rate'] == 1
    count = len(EmpiricalContractWorker.resets)
    assert count == result['quota']['reserved_trials']
    assert empirical.main(argv) == result
    assert len(EmpiricalContractWorker.resets) == count
    atomic_json(plan, dict(options, seed=2))
    with pytest.raises(ValueError):
        empirical.main([*argv, '--prepare-only'])
    assert len(EmpiricalContractWorker.resets) == count
    assert read_json(root / 'empirical-plan.json')['options'] == options
    assert '"execution_started": false' in capsys.readouterr().out


@pytest.mark.parametrize('plan', [[], {}, {'strategy': 'adaptive', 'seed': 1, 'panels': {}, 'unknown': 1},
                                 {'workspace': 'elsewhere'}, {'python': 'elsewhere'}])
def test_empirical_cli_malformed_plan_has_no_workspace(tmp_path, plan):
    from PhysicalRSI_demos.auto_engineering import empirical
    source = tmp_path / 'plan.json'
    atomic_json(source, plan)
    root = tmp_path / 'unused'
    with pytest.raises(SystemExit) as caught:
        empirical.main(['--workspace', str(root), '--python', sys.executable, '--plan', str(source), '--prepare-only'])
    assert caught.value.code == 2
    assert not root.exists()
