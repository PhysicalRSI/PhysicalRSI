from copy import deepcopy
from pathlib import Path
import math

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


@pytest.mark.parametrize('upper', [1., .125])
@pytest.mark.parametrize('unit', [1., .9999999999999999])
def test_wide_search_range_preserves_upper_boundary_and_nearby_values(tmp_path, upper, unit):
    from decimal import Decimal, localcontext
    field = SearchField('control', 'control.json', ('proportional',), -1e16, upper, precision=9)
    with localcontext() as context:
        context.prec = 80
        u = Decimal.from_float(unit)
        exact = Decimal.from_float(field.lower) * (1 - u) + Decimal.from_float(upper) * u
        expected = float(round(exact, 9))
    value = field.quantize(unit)
    assert value == expected
    parent = candidate(tmp_path / 'parent', (field.lower, .1))
    declared = ParameterSpace([field])
    child, _ = declared.candidate(parent, declared.inspect(parent), [value],
                                  output=tmp_path / 'child', name='boundary')
    assert declared.inspect(child)['values'] == [expected]
    assert declared.quantize(declared.unit([upper])) == [upper]


def values(root,harness,*,tasks=demo.TASKS,steps=100,environment_factory=demo.Environment,case=None):
    suite=demo.Suite();result=[]
    for task in tasks:
        trial=Trial('trial-'+task,task=task,case=case if case is not None else dict(initial=0.,target=1.),scope='software',budget=Budget(steps,30),
            system2=System2Trial(Revision('unit-development','1'),harness['id'],verify_harness(harness),'development'))
        result.append(ExperimentRuntime(root).evaluate(trial,environment=environment_factory(task,[]),
            policy=suite.policy(harness,task),verifier=suite.verifier(task)).to_dict())
    return result


def rows(harness,evaluations):
    return [dict(candidate=harness,evaluation=v,archive_revision='history',role='development_screening',archive_evidence={}) for v in evaluations]


def build(root,*,history=(),objective=None,**kwargs):
    parent=candidate(root/'parent')
    current=values(root/'current',parent)
    evolver=AdaptiveParameterEvolver(archive=Archive(history),space=space(),objective=objective or demo.Objective(),
        task_weights={t:1 for t in demo.TASKS},population=4,**kwargs)
    return evolver,(parent,current,dict(index=1),dict(view=dict(scope='software')),root/'proposal')


class PanelObjective:
    def __init__(self, scores=None, default=0.):
        self.scores, self.default = scores or {}, default

    def identity(self):
        return dict(kind='declared-panel-objective', scores=self.scores, default=self.default)

    def score(self, value):
        return self.scores.get(value['trial']['system2']['candidate_id'], self.default)


@pytest.mark.parametrize('fraction,expected', [(1., 'brittle'), (.75, 'robust'), (.5, 'robust')])
def test_lower_tail_ranks_complete_multicase_development_panels(tmp_path, fraction, expected):
    class CaseObjective(PanelObjective):
        def score(self, value):
            name = value['trial']['system2']['candidate_id']
            if name == 'brittle':
                return 1. if value['trial']['case']['target'] == 1. else .2
            return .5 if name == 'robust' else .1

    candidates = {name: candidate(tmp_path/name, point) for name, point in
                  [('parent', (.3, .1)), ('brittle', (5., 3.)), ('robust', (3., 5.))]}
    panels = {}
    for name, harness in candidates.items():
        panels[name] = [value for index, target in enumerate((1., 2.))
                       for value in values(tmp_path/(name+'-case-'+str(index)), harness,
                                           case=dict(initial=0., target=target))]
    # Repeating one case must not give that scenario extra mass in the tail.
    repeated = values(tmp_path/'repeated', candidates['brittle'],
                      case=dict(initial=0., target=1.))
    history = rows(candidates['brittle'], panels['brittle'] + repeated)
    history += rows(candidates['robust'], panels['robust'])
    evolver = AdaptiveParameterEvolver(archive=Archive(history), space=space(),
        objective=CaseObjective(), task_weights={t:1 for t in demo.TASKS},
        population=4, lower_tail_fraction=fraction)
    output = tmp_path/'proposal'
    evolver.propose_from_archive(candidates['parent'], panels['parent'], dict(index=1),
                                 dict(view=dict(scope='software')), output)
    fit = read_json(output/'adaptive-fit.json')
    assert fit['ranked'][0]['values'] == space().inspect(candidates[expected])['values']
    assert not fit['incomplete'] and fit['model_ready']
    brittle = next(row for row in fit['ranked'] if row['values'] == [5., 3.])
    assert brittle['score'] == pytest.approx({1.: .6, .75: .7/1.5, .5: .2}[fraction])
    assert fit['identity']['settings']['lower_tail_fraction'] == fraction


@pytest.mark.parametrize('fraction', [0, -1, 1.01, True, float('nan'), float('inf')])
def test_invalid_lower_tail_fraction_is_rejected(tmp_path, fraction):
    with pytest.raises(ValueError):
        build(tmp_path, lower_tail_fraction=fraction)


@pytest.mark.parametrize('fraction', [1., .75, .1, 5e-324])
def test_lower_tail_preserves_flat_scores_and_fractional_boundary(fraction):
    from PhysicalRSI_core.self_harness.adaptive_search import _lower_tail_mean
    assert _lower_tail_mean([.3] * 7, fraction) == .3
    assert _lower_tail_mean([1., .2], .75) == pytest.approx(.7/1.5)
    assert _lower_tail_mean([1., .2], 5e-324) == .2


@pytest.mark.parametrize('kind', ['adaptive', 'scalar'])
@pytest.mark.parametrize('before,after', [(1, True), (True, 1), (1, 1.0), (False, 0)])
def test_search_rejects_typed_provider_drift_before_reading_history(tmp_path, monkeypatch, kind, before, after):
    if kind == 'adaptive':
        original, args = build(tmp_path)
        provider = original.objective
        def create():
            return AdaptiveParameterEvolver(archive=original.archive, space=original.space,
                objective=provider, task_weights=original.weights, **original.settings)
    else:
        from test_parameter_search import search
        from PhysicalRSI_core.self_harness.parameter_search import ScalarRefinementEvolver
        original, args = search(tmp_path)
        provider = original.classifier
        def create():
            return ScalarRefinementEvolver(archive=original.archive, parameter=original.parameter,
                                           classifier=provider)
    identity = dict(provider.identity(), settings={'version': before})
    monkeypatch.setattr(provider, 'identity', lambda: identity)
    evolver = create()
    identity['settings']['version'] = after
    assert evolver.identity() == evolver._identity
    with monkeypatch.context() as patch:
        patch.setattr(original.archive, 'development', lambda view: pytest.fail('Changed search must not read history'))
        with pytest.raises(ValueError, match='search implementation changed'):
            evolver.propose_from_archive(*args)
    assert not args[-1].exists()
    identity['settings']['version'] = before
    children = evolver.propose_from_archive(*args)
    assert children
    assert evolver.resume_proposal_from_archive(*args) == children


@pytest.mark.parametrize('score', [0., .5])
@pytest.mark.parametrize('elite_weighting', ['uniform', 'fractional'])
def test_flat_feedback_keeps_global_exploration_without_arbitrary_elites(tmp_path, score, elite_weighting):
    history = []
    for index, point in enumerate(((1., 7.), (7., 1.))):
        previous = candidate(tmp_path / ('observed-' + str(index)), point)
        history.extend(rows(previous, values(tmp_path / ('history-' + str(index)), previous)))
    objective = PanelObjective(default=score)
    adaptive, args = build(tmp_path / 'adaptive', history=history, objective=objective, minimum_observations=3, elite_weighting=elite_weighting)
    uniform, control = build(tmp_path / 'uniform', history=history, objective=objective,
                             minimum_observations=3, strategy='uniform')
    adaptive.propose_from_archive(*args)
    uniform.propose_from_archive(*control)
    a, b = [read_json(v[-1] / 'adaptive-fit.json') for v in (args, control)]
    assert len(a['ranked']) == 3 and not a['model_ready']
    assert a['reason'] == 'flat_development_scores'
    assert a['proposals'] == b['proposals']
    assert all(row['role'] == 'global' for row in a['proposals'])


@pytest.mark.parametrize('elite_weighting', ['uniform', 'fractional'])
def test_tied_elite_cutoff_includes_both_parameter_directions(tmp_path, elite_weighting):
    history = []
    for name, point in [('tied-a', (1., 7.)), ('tied-b', (7., 1.)), ('inferior', (2., 2.))]:
        previous = candidate(tmp_path / name, point)
        history.extend(rows(previous, values(tmp_path / (name + '-trials'), previous)))
    objective = PanelObjective({'tied-a': .8, 'tied-b': .8, 'inferior': .2})
    evolver, args = build(tmp_path / 'study', history=history, objective=objective, elite_fraction=.25, elite_weighting=elite_weighting)
    evolver.propose_from_archive(*args)
    fit = read_json(args[-1] / 'adaptive-fit.json')
    assert fit['model_ready']
    assert {tuple(row['values']) for row in fit['elite']} == {(1., 7.), (7., 1.)}
    assert fit['center'] == pytest.approx(space().unit([4., 4.]))


@pytest.mark.parametrize('covariance', ['diagonal', 'full'])
@pytest.mark.parametrize('elite_weighting', ['uniform', 'fractional'])
def test_single_elite_respects_minimum_exploration_scale(tmp_path, covariance, elite_weighting):
    previous = candidate(tmp_path / 'best', (5., 3.))
    history = rows(previous, values(tmp_path / 'history', previous))
    evolver, args = build(tmp_path / 'study', history=history,
        objective=PanelObjective({'best': 1.}), minimum_scale=.6, covariance=covariance, elite_weighting=elite_weighting)
    evolver.propose_from_archive(*args)
    fit = read_json(args[-1] / 'adaptive-fit.json')
    assert fit['model_ready'] and len(fit['elite']) == 1
    assert fit['scale'] == [.6, .6]
    assert all(fit['covariance'][i][i] >= .6**2 for i in range(2))


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
    current,args=build(tmp_path/'current-only',minimum_observations=1)
    informed,more=build(tmp_path/'history-aware',history=history,minimum_observations=1)
    current.propose_from_archive(*args);informed.propose_from_archive(*more)
    a,b=(read_json(v[-1]/'adaptive-fit.json') for v in (args,more))
    assert a['random_seed']==b['random_seed']
    assert a['center']!=b['center']
    assert a['proposals'][:2]==b['proposals'][:2]
    assert a['proposals'][2:]!=b['proposals'][2:]
    assert b['elite'][0]['values']==[5.,3.]


def test_sparse_feedback_uses_the_same_global_population_as_uniform_search(tmp_path):
    adaptive,args=build(tmp_path/'adaptive',minimum_observations=None)
    uniform,control=build(tmp_path/'uniform',strategy='uniform')
    adaptive.propose_from_archive(*args);uniform.propose_from_archive(*control)
    a,b=(read_json(v[-1]/'adaptive-fit.json') for v in (args,control))
    assert adaptive.settings['minimum_observations']==5
    assert uniform.settings['minimum_observations']==1
    assert a['reason']=='global_warmup' and not a['model_ready']
    assert a['proposals']==b['proposals']
    assert all(p['role']=='global' for p in a['proposals'])


def test_full_covariance_preserves_elite_geometry_and_global_control_points(tmp_path):
    history=[]
    for index,point in enumerate(((1.,1.),(2.,2.),(3.,3.))):
        historical=candidate(tmp_path/('observed-'+str(index)),point)
        history.extend(rows(historical,values(tmp_path/('history-'+str(index)),historical)))
    diagonal,args=build(tmp_path/'diagonal',history=history,elite_fraction=1)
    full,other=build(tmp_path/'full',history=history,elite_fraction=1,covariance='full')
    diagonal.propose_from_archive(*args);full.propose_from_archive(*other)
    a,b=[read_json(v[-1]/'adaptive-fit.json') for v in (args,other)]
    assert a['center']==b['center'] and a['scale']==b['scale']
    assert a['proposals'][:2]==b['proposals'][:2]
    assert a['proposals'][2:]!=b['proposals'][2:]
    assert a['covariance_factor'] is None and a['covariance'][0][1]==0
    matrix,factor=b['covariance'],b['covariance_factor']
    assert matrix[0][1]>0 and matrix[0][1]==matrix[1][0]
    for i in range(2):
        assert matrix[i][i]==pytest.approx(a['scale'][i]**2)
        for j in range(2):
            assert sum(factor[i][k]*factor[j][k] for k in range(2))==pytest.approx(matrix[i][j])
    assert all(.1<=v<=8 and round(v,3)==v for row in b['proposals'] for v in row['values'])


def test_full_covariance_keeps_sparse_warmup_identical_to_uniform(tmp_path):
    full,args=build(tmp_path/'full',minimum_observations=None,covariance='full')
    uniform,other=build(tmp_path/'uniform',strategy='uniform')
    full.propose_from_archive(*args);uniform.propose_from_archive(*other)
    a,b=[read_json(v[-1]/'adaptive-fit.json') for v in (args,other)]
    assert not a['model_ready'] and a['reason']=='global_warmup'
    assert a['proposals']==b['proposals']


def test_regularized_collinear_covariance_remains_usable_at_dimension_limit():
    from PhysicalRSI_core.self_harness.adaptive_search import _cholesky
    # Perfectly collinear elite data, shrunk 10% toward its diagonal.
    matrix=[[.01 if i==j else .009 for j in range(32)] for i in range(32)]
    factor=_cholesky(matrix)
    assert all(math.isfinite(v) for row in factor for v in row)
    assert all(factor[i][i]>0 for i in range(32))
    for i in range(32):
        for j in range(32):
            assert math.fsum(factor[i][k]*factor[j][k] for k in range(32))==pytest.approx(matrix[i][j])


@pytest.mark.parametrize('mode',['dense',None,True,{},0])
def test_invalid_covariance_mode_is_rejected_before_proposal(mode):
    with pytest.raises(ValueError,match='covariance'):
        AdaptiveParameterEvolver(archive=Archive(),space=space(),objective=demo.Objective(),
            task_weights={t:1 for t in demo.TASKS},covariance=mode)


def test_full_covariance_requires_representable_minimum_variance():
    with pytest.raises(ValueError,match='minimum variance'):
        AdaptiveParameterEvolver(archive=Archive(),space=space(),objective=demo.Objective(),
            task_weights={t:1 for t in demo.TASKS},covariance='full',minimum_scale=1e-200)


def test_warmup_counts_distinct_complete_vectors_not_receipts(tmp_path):
    history=[]
    for i,point in enumerate(((5.,3.),(4.,3.),(6.,4.),(5.,4.))):
        historical=candidate(tmp_path/('observed-'+str(i)),point)
        history.extend(rows(historical,values(tmp_path/('history-'+str(i)),historical)))
    incomplete,args=build(tmp_path/'incomplete',history=history[:-1]*3,minimum_observations=None)
    informed,full=build(tmp_path/'complete',history=history,minimum_observations=None)
    incomplete.propose_from_archive(*args);informed.propose_from_archive(*full)
    a,b=(read_json(v[-1]/'adaptive-fit.json') for v in (args,full))
    assert len(a['ranked'])==4 and len(a['incomplete'])==1
    assert not a['model_ready'] and a['reason']=='global_warmup'
    assert len(b['ranked'])==5 and b['model_ready']
    assert b['reason']=='sample_development_distribution'
    assert a['proposals'][:2]==b['proposals'][:2]
    assert [p['role'] for p in b['proposals']]==['global','global','elite','elite']


@pytest.mark.parametrize('minimum',[0,-1,True,False,1.5,float('nan'),'5'])
def test_invalid_warmup_threshold_fails_before_execution(tmp_path,minimum):
    with pytest.raises(ValueError,match='minimum_observations'):
        AdaptiveParameterEvolver(archive=Archive(),space=space(),objective=demo.Objective(),
            task_weights={t:1 for t in demo.TASKS},minimum_observations=minimum)


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
@pytest.mark.parametrize('covariance',['diagonal','full'])
def test_partial_candidate_pool_publication_resumes(tmp_path,monkeypatch,boundary,covariance):
    from PhysicalRSI_core.self_harness import adaptive_search,search_space
    evolver,args=build(tmp_path,covariance=covariance)
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


@pytest.mark.parametrize('score', [.1, .7])
def test_identical_repeated_scores_do_not_create_a_search_signal(tmp_path, score):
    observed = candidate(tmp_path / 'observed', (5., 3.))
    history = []
    for repeat in range(3):
        history.extend(rows(observed, values(tmp_path / ('repeat-' + str(repeat)), observed)))
    evolver, args = build(tmp_path / 'study', history=history,
        objective=PanelObjective(default=score), minimum_observations=2)
    evolver.propose_from_archive(*args)
    fit = read_json(args[-1] / 'adaptive-fit.json')
    assert len(fit['samples']) == 8  # Independent repeats are retained, not deduplicated.
    assert len(fit['ranked']) == 2
    assert all(row['score'] == score for row in fit['ranked'])
    assert not fit['score_contrast'] and not fit['model_ready']
    assert fit['reason'] == 'flat_development_scores'
    assert all(proposal['role'] == 'global' for proposal in fit['proposals'])


def test_repeated_feedback_mean_is_independent_of_archive_order(tmp_path):
    history = []
    scores = {'large': 1., 'tiny-a': 1e-16, 'tiny-b': 1e-16}
    for name in scores:
        previous = candidate(tmp_path / name, (5., 3.))
        history.extend(rows(previous, values(tmp_path / (name + '-trials'), previous)))
    fits = []
    for index, observed in enumerate((history, list(reversed(history)))):
        evolver, args = build(tmp_path / str(index), history=observed,
                             objective=PanelObjective(scores))
        evolver.propose_from_archive(*args)
        fits.append(read_json(args[-1] / 'adaptive-fit.json'))
    assert fits[0]['ranked'] == fits[1]['ranked']
    assert fits[0]['elite'] == fits[1]['elite']
    assert fits[0]['proposals'] == fits[1]['proposals']


def test_independent_identical_rollouts_survive_deduplication_but_copies_do_not(tmp_path, monkeypatch):
    from PhysicalRSI_core import contracts, experiments
    monkeypatch.setattr(experiments, 'monotonic', lambda: 1000.)
    monkeypatch.setattr(contracts, 'monotonic', lambda: 1000.)
    observed = candidate(tmp_path / 'observed', (5., 3.))
    first = values(tmp_path / 'first-execution', observed)
    second = values(tmp_path / 'second-execution', observed)
    assert [v['trial_sha256'] for v in first] == [v['trial_sha256'] for v in second]
    history = rows(observed, first) + rows(observed, second) + rows(observed, deepcopy(first))
    evolver, args = build(tmp_path / 'study', history=history, objective=PanelObjective(default=.1))
    evolver.propose_from_archive(*args)
    fit = read_json(args[-1] / 'adaptive-fit.json')
    assert len(fit['samples']) == 6  # Current panel plus two genuinely executed panels.
    assert len(fit['excluded']) == 2  # Only the copied first panel is redundant.
    assert all(row['reason'] == 'duplicate_observation' for row in fit['excluded'])


@pytest.mark.parametrize('weight', [1., 5e-324, 5e307, 1e308])
def test_relative_weight_scale_preserves_feedback_contrast(tmp_path, weight):
    observed = candidate(tmp_path / 'observed', (5., 3.))
    history = rows(observed, values(tmp_path / 'history', observed))
    parent = candidate(tmp_path / 'parent')
    current = values(tmp_path / 'current', parent)
    evolver = AdaptiveParameterEvolver(archive=Archive(history), space=space(),
        objective=PanelObjective({'parent': .2, 'observed': .24}),
        task_weights={t: weight for t in demo.TASKS}, population=4, minimum_observations=2)
    output = tmp_path / 'proposal'
    evolver.propose_from_archive(parent, current, dict(index=1), dict(view=dict(scope='software')), output)
    fit = read_json(output / 'adaptive-fit.json')
    assert [row['score'] for row in fit['ranked']] == [.24, .2]
    assert fit['score_contrast'] and fit['model_ready']
    assert any(row['role'] != 'global' for row in fit['proposals'])


def test_changed_development_cases_are_explicitly_excluded_before_scoring(tmp_path):
    previous = candidate(tmp_path / 'previous', (5., 3.))
    history = rows(previous, values(tmp_path / 'history', previous, case=dict(initial=0., target=2.)))

    class CurrentPanelOnly(PanelObjective):
        def score(self, value):
            assert value['trial']['case']['target'] == 1., 'Out-of-panel feedback reached the objective'
            return super().score(value)

    evolver, args = build(tmp_path / 'study', history=history, objective=CurrentPanelOnly(default=.2),
                         minimum_observations=2)
    evolver.propose_from_archive(*args)
    fit = read_json(args[-1] / 'adaptive-fit.json')
    assert len(fit['samples']) == 2
    assert len(fit['ranked']) == 1 and not fit['model_ready']
    assert fit['reason'] == 'global_warmup'
    assert len(fit['excluded']) == 2
    assert all(row['reason'] == 'outside_current_development_panel' for row in fit['excluded'])
    assert {row['trial_sha256'] for row in fit['excluded']} == {row['evaluation']['trial_sha256'] for row in history}
    assert all(row['case']['target'] == 2. and row['source']['archive_revision'] == 'history'
               for row in fit['excluded'])


@pytest.mark.parametrize('covariance', ['diagonal', 'full'])
@pytest.mark.parametrize('reverse', [False, True])
def test_fractional_elites_preserve_rank_mass_at_tied_cutoff(tmp_path, covariance, reverse):
    history = []
    for name, point in [('best', (7., 7.)), ('boundary', (1., 1.))]:
        previous = candidate(tmp_path / name, point)
        history.extend(rows(previous, values(tmp_path / (name + '-trials'), previous)))
    if reverse:
        history.reverse()
    objective = PanelObjective({'best': 1.}, default=.2)
    evolver, args = build(tmp_path / 'study', history=history, objective=objective,
        elite_fraction=.5, elite_weighting='fractional', covariance=covariance)
    evolver.propose_from_archive(*args)
    fit = read_json(args[-1] / 'adaptive-fit.json')
    assert len(fit['elite']) == 3
    assert fit['elite_weights'] == [.5, .25, .25]
    points = [space().unit(v) for v in [(7., 7.), (.3, .1), (1., 1.)]]
    expected = [sum(w*p[i] for w,p in zip([.5, .25, .25], points)) for i in range(2)]
    assert fit['center'] == pytest.approx(expected)
    variance = [sum(w*(p[i]-expected[i])**2 for w,p in zip([.5, .25, .25], points)) for i in range(2)]
    assert fit['scale'] == pytest.approx([math.sqrt(v) for v in variance])
    if covariance == 'full':
        cross = .9*sum(w*(p[0]-expected[0])*(p[1]-expected[1]) for w,p in zip([.5, .25, .25], points))
        assert fit['covariance'][0][1] == pytest.approx(cross)
    # Explicit uniform weighting retains the original equal-weight fit.
    control, other = build(tmp_path / 'control', history=history, objective=objective,
        elite_fraction=.5, elite_weighting='uniform', covariance=covariance)
    control.propose_from_archive(*other)
    old = read_json(other[-1] / 'adaptive-fit.json')
    assert old['elite_weights'] == [1/3]*3
    assert fit['proposals'][:2] == old['proposals'][:2]


def test_sampling_stream_ignores_runtime_location_but_keeps_feedback_context(tmp_path):
    fits = []
    for label in ('first-cache', 'second-cache'):
        class Located(demo.Environment):
            def identity(self):
                return dict(super().identity(), runtime_cache=str(tmp_path / label))
        evolver, args = build(tmp_path / label)
        parent, _, context, search, output = args
        evaluations = values(tmp_path / (label + '-trials'), parent, environment_factory=Located)
        evolver.propose_from_archive(parent, evaluations, context, search, output)
        fits.append(read_json(output / 'adaptive-fit.json'))
    assert fits[0]['panel'] != fits[1]['panel']
    assert fits[0]['random_seed'] == fits[1]['random_seed']
    assert fits[0]['proposals'] == fits[1]['proposals']
