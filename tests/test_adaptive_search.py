from copy import deepcopy
from pathlib import Path

import pytest

from PhysicalRSI_core.embodiment import Revision,System2Trial
from PhysicalRSI_core.experiments import Budget,ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json,digest,file_digest,read_json
from PhysicalRSI_core.self_harness.adaptive_search import AdaptiveParameterEvolver,measurement_context
from PhysicalRSI_core.self_harness.artifacts import CLOSURE,verify_harness
from PhysicalRSI_core.self_harness.search_space import ParameterSpace,SearchField
from PhysicalRSI_core.trials import Trial
from PhysicalRSI_demos import adaptive_control as demo
from PhysicalRSI_demos.recursive_research import manifest


class Archive:
    def __init__(self,rows=()):self.rows=list(rows)
    def identity(self):return dict(kind='unit-observation-container')
    def development(self,view):return deepcopy(self.rows)


def candidate(root,values=(.3,.1),fixed='same'):
    for kind in CLOSURE:
        atomic_json(root/(kind+'.json'),dict(proportional=values[0],damping=values[1]) if kind=='control' else dict(kind=kind,fixed=fixed))
    return manifest(root,root.name)


def space():
    return ParameterSpace([SearchField('control','control.json',(k,),.1,8) for k in ('proportional','damping')])


def values(root,harness,*,tasks=demo.TASKS,steps=100,environment_factory=demo.Environment):
    suite=demo.Suite();result=[]
    for task in tasks:
        trial=Trial('trial-'+task,task=task,case=dict(initial=0.,target=1.),scope='software',budget=Budget(steps,30),
            system2=System2Trial(Revision('unit-development','1'),harness['id'],verify_harness(harness),'development'))
        result.append(ExperimentRuntime(root).evaluate(trial,environment=environment_factory(task,[]),
            policy=suite.policy(harness,task),verifier=suite.verifier(task)).to_dict())
    return result


def rows(harness,evaluations):
    return [dict(candidate=harness,evaluation=v,archive_revision='history',role='development_screening',archive_evidence={}) for v in evaluations]


def build(root,*,history=(),**kwargs):
    parent=candidate(root/'parent')
    current=values(root/'current',parent)
    evolver=AdaptiveParameterEvolver(archive=Archive(history),space=space(),objective=demo.Objective(),
        task_weights={t:1 for t in demo.TASKS},population=4,**kwargs)
    return evolver,(parent,current,dict(index=1),dict(view=dict(scope='software')),root/'proposal')


def test_cpu_two_gain_search_uses_rejected_trajectories_and_restarts(tmp_path):
    campaign,initial,suite,quota=demo.setup(tmp_path,rounds=3,population=4)
    result=campaign.run(initial)
    test=campaign.report_test(initial)
    fits=[read_json(p) for p in sorted((campaign.root/'rounds').glob('*/execution/adaptive-fit.json'))]
    assert len(fits)==3 and len(fits[1]['ranked'])>len(fits[0]['ranked'])
    assert any(isinstance(s['source'],dict) and s['source']['role']=='development_screening' and s['outcome']=='failure'
               for s in fits[1]['samples'])
    assert all(s['case'] in [dict(initial=0.,target=1.,load_scale=.9),dict(initial=0.,target=-1.,load_scale=1.1)]
               for fit in fits for s in fit['samples'])
    assert all(fit['qualification'] is None for fit in fits)
    assert all(len(p['values'])==2 for fit in fits for p in fit['proposals'])
    assert any(r['outcome']=='inherit_child' for r in result['rounds'])
    assert all(v['success_rate']==1 for v in test['metrics'].values())
    executions=len(suite.resets)
    assert executions==quota.status()['reserved_trials']
    assert campaign.run(initial)==result and campaign.report_test(initial)==test
    assert len(suite.resets)==executions


def test_history_changes_elite_sampling_but_not_global_control_points(tmp_path):
    historical=candidate(tmp_path/'observed',(5.,3.))
    history=rows(historical,values(tmp_path/'history',historical))
    current,args=build(tmp_path/'current-only')
    informed,more=build(tmp_path/'history-aware',history=history)
    current.propose_from_archive(*args);informed.propose_from_archive(*more)
    a,b=(read_json(v[-1]/'adaptive-fit.json') for v in (args,more))
    assert a['random_seed']==b['random_seed']
    assert a['center']!=b['center']
    assert a['proposals'][:2]==b['proposals'][:2]
    assert a['proposals'][2:]!=b['proposals'][2:]
    assert b['elite'][0]['values']==[5.,3.]


@pytest.mark.parametrize('difference',['budget','environment','verifier','fixed'])
def test_incomparable_observations_do_not_enter_fit(tmp_path,difference):
    historical=candidate(tmp_path/'observed',(5.,3.),fixed='different' if difference=='fixed' else 'same')
    factory=demo.Environment
    if difference=='environment':
        class Different(demo.Environment):
            def __init__(self,*a):super().__init__(*a);self.base_mass*=2
        factory=Different
    history=values(tmp_path/'history',historical,steps=99 if difference=='budget' else 100,environment_factory=factory)
    if difference=='verifier':
        # Rebind a well-formed unit observation to another verifier identity.
        for v in history:
            path=next(Path(p) for p in v['evidence'] if Path(p).name=='experiment.json')
            spec=read_json(path);spec['adapters']['verifier']={'kind':'another-verifier'}
            atomic_json(path,spec);v['evidence'][str(path)]=file_digest(path);v['receipt']['experiment_sha256']=digest(spec)
    evolver,args=build(tmp_path/'study',history=rows(historical,history))
    evolver.propose_from_archive(*args)
    fit=read_json(args[-1]/'adaptive-fit.json')
    assert len(fit['ranked'])==1
    assert len(fit['excluded'])==2
    reason='other_controller_inputs_changed' if difference=='fixed' else 'measurement_context_changed'
    assert all(s['reason']==reason for s in fit['excluded'])


def test_partial_task_coverage_cannot_become_an_elite(tmp_path):
    historical=candidate(tmp_path/'observed',(5.,3.))
    history=rows(historical,values(tmp_path/'history',historical,tasks=('light-load',)))
    evolver,args=build(tmp_path/'study',history=history)
    evolver.propose_from_archive(*args)
    fit=read_json(args[-1]/'adaptive-fit.json')
    assert len(fit['ranked'])==1
    assert fit['incomplete']==[dict(values=[5.,3.],reason='incomplete_development_panel')]


@pytest.mark.parametrize('damage',['validation','candidate','trial','specification'])
def test_unbound_development_is_rejected(tmp_path,damage):
    evolver,args=build(tmp_path)
    v=args[1][0]
    if damage=='validation':v['trial']['system2']['split']='validation';v['trial_sha256']=digest(v['trial'])
    elif damage=='candidate':v['trial']['system2']['candidate_id']='foreign';v['trial_sha256']=digest(v['trial'])
    elif damage=='trial':v['trial_sha256']='0'*64
    else:
        path=next(Path(p) for p in v['evidence'] if Path(p).name=='experiment.json');path.write_text('{}')
    with pytest.raises(ValueError):evolver.propose_from_archive(*args)
    assert not (args[-1]/'adaptive-fit.json').exists()


@pytest.mark.parametrize('score',[float('nan'),float('inf'),True,1.1,-.1])
def test_malformed_objective_scores_are_rejected(tmp_path,score):
    evolver,args=build(tmp_path)
    evolver.objective.score=lambda value:score
    with pytest.raises(ValueError):evolver.propose_from_archive(*args)


def test_unknown_feedback_abstains_without_treating_it_as_failure(tmp_path):
    evolver,args=build(tmp_path)
    args[1][0]['outcome']='uncertain'
    assert evolver.propose_from_archive(*args)==[]
    fit=read_json(args[-1]/'adaptive-fit.json')
    assert fit['reason']=='incomplete_current_feedback'
    assert fit['samples'][0]['score'] is None


@pytest.mark.parametrize('boundary',['fit','rename'])
def test_partial_candidate_pool_publication_resumes(tmp_path,monkeypatch,boundary):
    from PhysicalRSI_core.self_harness import adaptive_search,search_space
    evolver,args=build(tmp_path)
    obj=adaptive_search if boundary=='fit' else search_space.os
    attribute='freeze_json' if boundary=='fit' else 'rename'
    original=getattr(obj,attribute)
    def interrupt(*a,**k):
        original(*a,**k)
        raise RuntimeError('interrupted durable candidate pool')
    with monkeypatch.context() as patch:
        patch.setattr(obj,attribute,interrupt)
        with pytest.raises(RuntimeError,match='durable candidate'):evolver.propose_from_archive(*args)
    children=evolver.resume_proposal_from_archive(*args)
    assert len(children)==4 and children==evolver.resume_proposal_from_archive(*args)
    assert len({verify_harness(c) for c in children})==4
    assert all(c['parent_sha256']==verify_harness(args[0]) for c in children)
    assert read_json(Path(args[0]['root'])/'control.json')==dict(proportional=.3,damping=.1)


def test_parameter_space_across_files_preserves_every_fixed_input(tmp_path):
    parent=candidate(tmp_path/'parent')
    atomic_json(Path(parent['root'])/'memory_rules.json',dict(nested=dict(steps=2),fixed='retained'))
    parent=manifest(Path(parent['root']),'parent')
    declared=ParameterSpace([SearchField('control','control.json',('proportional',),.1,8),
        SearchField('memory_rules','memory_rules.json',('nested','steps'),1,10,precision=0)])
    inspected=declared.inspect(parent)
    child,changes=declared.candidate(parent,inspected,[4.,6],output=tmp_path/'child',name='child')
    assert set(changes)=={'control','memory_rules'}
    assert read_json(Path(child['root'])/'memory_rules.json')==dict(nested=dict(steps=6),fixed='retained')
    assert read_json(Path(child['root'])/'control.json')['damping']==.1
    assert declared.inspect(child)['condition_sha256']==inspected['condition_sha256']
    for kind in CLOSURE-{'control','memory_rules'}:assert child['components'][kind]==parent['components'][kind]


@pytest.mark.parametrize('keys',[(('x',),('x',)),(('x',),('x','y')),(('x','y'),('x',))])
def test_overlapping_search_fields_are_rejected(keys):
    with pytest.raises(ValueError,match='overlap'):
        ParameterSpace([SearchField('control','control.json',k,0,1) for k in keys])


@pytest.mark.parametrize('options',[dict(component='foundation'),dict(component='weights'),dict(lower=float('nan')),
    dict(file='../bad.json'),dict(precision=True),dict(precision=-1),dict(key=()),dict(lower=8,upper=1)])
def test_invalid_parameter_domains_fail(options):
    with pytest.raises((ValueError,TypeError)):
        SearchField(**(dict(component='control',file='control.json',key=('x',),lower=0,upper=1)|options))


def test_exhausted_coarse_grid_terminates(tmp_path):
    parent=candidate(tmp_path/'parent',(1.,.1))
    historical=candidate(tmp_path/'other',(2.,.1))
    history=rows(historical,values(tmp_path/'history',historical))
    declared=ParameterSpace([SearchField('control','control.json',('proportional',),1,2,precision=0)])
    evolver=AdaptiveParameterEvolver(archive=Archive(history),space=declared,objective=demo.Objective(),
        task_weights={t:1 for t in demo.TASKS},population=4)
    args=(parent,values(tmp_path/'current',parent),dict(index=1),dict(view=dict(scope='software')),tmp_path/'output')
    assert evolver.propose_from_archive(*args)==[]
    assert read_json(args[-1]/'adaptive-fit.json')['reason']=='no_unseen_sample_within_attempt_budget'


def test_foreign_or_mutated_parameter_inspection_cannot_change_fixed_inputs(tmp_path):
    parent=candidate(tmp_path/'parent')
    foreign=candidate(tmp_path/'foreign',(4.,2.),fixed='different')
    declared=space()
    inspected=declared.inspect(foreign)
    with pytest.raises(ValueError,match='current parent'):
        declared.candidate(parent,inspected,[5.,3.],output=tmp_path/'child',name='child')
    assert not (tmp_path/'child').exists()
    inspected=declared.inspect(parent)
    inspected['documents']['control.json']['undeclared']='changed'
    with pytest.raises(ValueError,match='current parent'):
        declared.candidate(parent,inspected,[5.,3.],output=tmp_path/'child',name='child')


def test_search_context_ignores_unrelated_export_named_experiment(tmp_path):
    parent=candidate(tmp_path/'parent')
    value=values(tmp_path/'current',parent,tasks=('light-load',))[0]
    expected=measurement_context(value)
    extra=tmp_path/'export/experiment.json';atomic_json(extra,dict(provider='auxiliary artifact'))
    value['evidence'][str(extra)]=file_digest(extra)
    assert measurement_context(value)==expected


def test_same_owned_receipt_is_not_weighted_twice(tmp_path):
    historical=candidate(tmp_path/'observed',(5.,3.))
    evaluations=values(tmp_path/'history',historical)
    history=rows(historical,evaluations)*2
    evolver,args=build(tmp_path/'study',history=history)
    evolver.propose_from_archive(*args)
    fit=read_json(args[-1]/'adaptive-fit.json')
    assert len(fit['samples'])==4
    assert len(fit['excluded'])==2
    assert all(r['reason']=='duplicate_observation' for r in fit['excluded'])


def test_current_task_panel_and_round_index_are_required(tmp_path):
    evolver,args=build(tmp_path)
    with pytest.raises(ValueError,match='round index'):
        evolver.propose_from_archive(args[0],args[1],None,*args[3:])
    with pytest.raises(ValueError,match='task panel'):
        evolver.propose_from_archive(args[0],args[1][:1],*args[2:])


def test_changed_evolver_identity_and_existing_candidate_are_rejected(tmp_path):
    evolver,args=build(tmp_path)
    result=evolver.propose_from_archive(*args)
    (Path(result[0]['root'])/'control.json').write_text('{}')
    with pytest.raises(ValueError):evolver.resume_proposal_from_archive(*args)
    evolver.settings['seed']+=1
    with pytest.raises(ValueError,match='implementation changed'):evolver.resume_proposal_from_archive(*args)
