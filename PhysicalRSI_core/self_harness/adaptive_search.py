"""Elite-guided numeric search using complete, comparable development panels.

Global exploration remains in every mixed population. No monotonicity or causal
repair classifier is required. Scores guide proposals, never inheritance.
"""
from copy import deepcopy
from hashlib import sha256
from itertools import chain
import json
import math
from pathlib import Path
from random import Random

from ..infra.storage import digest, file_digest
from .artifacts import verify_harness
from .evaluation import freeze_json
from .search_space import ParameterSpace, finite


def measurement_context(value):
    """Pin measured dynamics, verifier, budget and contracts across fidelities.

    Policy parameters vary within the declared harness space. Their closure is
    checked separately; suite identity/admission attests policy construction.
    """
    paths=[p for p in value['evidence'] if Path(p).name=='experiment.json'
           and Path(p).parent.name==value['trial']['id']]
    if len(paths)!=1:
        raise ValueError('Development evaluation requires one bound experiment specification')
    raw=Path(paths[0]).read_bytes()
    if sha256(raw).hexdigest()!=value['evidence'][paths[0]]:
        raise ValueError('Development experiment specification changed')
    spec=json.loads(raw)
    if digest(spec)!=value['receipt']['experiment_sha256']:
        raise ValueError('Experiment specification differs from its Trial receipt')
    if any(spec[k]!=value['trial'][k] for k in ('task','case','scope','budget','system2')):
        raise ValueError('Experiment specification differs from the attributed Trial')
    adapters={k:v for k,v in spec['adapters'].items() if k!='policy'}
    binding=deepcopy(spec.get('binding'))
    if binding is not None:
        binding['system1'].pop('revision',None)
    return digest(dict(adapters=adapters,binding=binding,budget=spec['budget'],
        artifact_contract=spec.get('artifact_contract'),policy_lifecycle=spec.get('policy_lifecycle')))


class AdaptiveParameterEvolver:
    def __init__(self,*,archive,space,objective,task_weights,population=6,seed=0,
                 elite_fraction=.25,exploration_fraction=.5,minimum_scale=.025,
                 strategy='adaptive',stop_on_success=True,minimum_observations=1):
        if not isinstance(space,ParameterSpace):
            raise ValueError('Declare a ParameterSpace')
        if minimum_observations is None:
            minimum_observations=2*len(space.fields)+1
        if type(minimum_observations) is not int or minimum_observations<1:
            raise ValueError('Declare a positive integer minimum_observations')
        if (type(population) is not int or not 1<=population<=64 or type(seed) is not int
                or not 0<finite(elite_fraction)<=1 or not 0<finite(exploration_fraction)<=1
                or not 0<finite(minimum_scale)<=1 or strategy not in ('adaptive','uniform')
                or type(stop_on_success) is not bool or not isinstance(task_weights,dict) or not task_weights
                or any(not isinstance(t,str) or not t or finite(w)<=0 for t,w in task_weights.items())):
            raise ValueError('Declare a bounded population, sampling policy and positive task weights')
        self.archive,self.space,self.objective=archive,space,objective
        self.weights=deepcopy(task_weights)
        finite(sum(self.weights.values()))
        self.settings=dict(population=population,seed=seed,elite_fraction=elite_fraction,
            exploration_fraction=exploration_fraction,minimum_scale=minimum_scale,strategy=strategy,
            stop_on_success=stop_on_success,minimum_observations=minimum_observations)
        self._identity=deepcopy(self.identity())

    def identity(self):
        return dict(kind='development-adaptive-parameter-search/v1',archive=self.archive.identity(),
            space=self.space.identity(),objective=self.objective.identity(),task_weights=self.weights,
            settings=self.settings,implementation=file_digest(Path(__file__)),
            dependencies={n:file_digest(Path(__file__).with_name(n+'.py')) for n in ('search_space','artifacts')})

    def _stable(self):
        if self.identity()!=self._identity:
            raise ValueError('Frozen adaptive search implementation changed')

    def propose_from_archive(self,parent,evaluations,context,search,output):
        self._stable()
        if not isinstance(context,dict) or type(context.get('index')) is not int or context['index']<0:
            raise ValueError('Adaptive search requires a nonnegative round index')
        inspected=self.space.inspect(parent)
        samples,excluded,evidence=[],[],{}
        panel={}
        observed=set()

        def observe(candidate,value,source,extra=None,*,current=False):
            origin=value['trial']['system2']
            if (origin is None or origin['split']!='development' or origin['candidate_id']!=candidate['id']
                    or origin['candidate_sha256']!=verify_harness(candidate)
                    or value['trial_sha256']!=digest(value['trial']) or value['trial']['scope']!=search['view']['scope']):
                raise ValueError('Adaptive search requires attributed development Trials')
            case=digest({k:value['trial'][k] for k in ('task','case','scope')})
            if not current and case not in panel:return
            point=self.space.inspect(candidate)
            condition=measurement_context(value)
            row=dict(source=source,trial_sha256=value['trial_sha256'],receipt_sha256=digest(value['receipt']),
                task=value['trial']['task'],case=value['trial']['case'],case_sha256=case,
                values=point['values'],measurement_context=condition,outcome=value['outcome'])
            if current:
                if case in panel:raise ValueError('Current development has duplicate cases')
                panel[case]=dict(task=value['trial']['task'],measurement_context=condition)
            if point['condition_sha256']!=inspected['condition_sha256']:
                excluded.append(dict(row,reason='other_controller_inputs_changed'));return
            if condition!=panel[case]['measurement_context']:
                excluded.append(dict(row,reason='measurement_context_changed'));return
            key=(value['trial_sha256'],row['receipt_sha256'])
            if key in observed:
                excluded.append(dict(row,reason='duplicate_observation'));return
            observed.add(key)
            score=self.objective.score(deepcopy(value)) if value['outcome'] in ('success','failure') else None
            if score is not None and not 0<=finite(score)<=1:
                raise ValueError('Development objective must return a score in [0, 1] or None')
            samples.append(dict(row,score=score))
            for path,sha in chain(value['evidence'].items(),(extra or {}).items()):
                if path in evidence and evidence[path]!=sha:raise ValueError('Conflicting development evidence digests')
                evidence[path]=sha

        for value in evaluations:observe(parent,value,'current_development',current=True)
        if not panel or {p['task'] for p in panel.values()}!=set(self.weights):
            raise ValueError('Current development must cover the declared task panel')
        if len(samples)!=len(evaluations):raise ValueError('Current development differs from its parameter conditions')
        current_complete=all(s['score'] is not None for s in samples)
        solved=all(v['outcome']=='success' for v in evaluations)
        for row in self.archive.development(search['view']):
            observe(row['candidate'],row['evaluation'],dict(archive_revision=row['archive_revision'],role=row['role']),row['archive_evidence'])
        grouped={}
        for row in samples:
            key=tuple(row['values'])
            grouped.setdefault(key,{}).setdefault(row['case_sha256'],[]).append(row['score'])
        ranked,incomplete=[],[]
        for values,cases in grouped.items():
            if set(cases)!=set(panel) or any(any(s is None for s in scores) for scores in cases.values()):
                incomplete.append(dict(values=list(values),reason='incomplete_development_panel'));continue
            means={case:sum(scores)/len(scores) for case,scores in cases.items()}
            per_task={t:sum(means[c] for c,p in panel.items() if p['task']==t)/sum(p['task']==t for p in panel.values()) for t in self.weights}
            score=sum(per_task[t]*w for t,w in self.weights.items())/sum(self.weights.values())
            ranked.append(dict(values=list(values),score=score,per_task=per_task,case_scores=means))
        ranked.sort(key=lambda r:(-r['score'],r['values']))
        model_ready=len(ranked)>=self.settings['minimum_observations']
        elite=ranked[:max(1,math.ceil(len(ranked)*self.settings['elite_fraction']))]
        units=[self.space.unit(r['values']) for r in elite]
        center=[sum(v[i] for v in units)/len(units) for i in range(len(self.space.fields))] if units else self.space.unit(inspected['values'])
        scale=[max(self.settings['minimum_scale'],math.sqrt(sum((v[i]-center[i])**2 for v in units)/len(units)))
               if len(units)>1 else .25 for i in range(len(center))]
        rng_seed=digest(dict(seed=self.settings['seed'],round=context['index'],panel=panel,condition=inspected['condition_sha256']))
        rng=Random(int(rng_seed,16))
        proposals=[]
        reason='sample_development_distribution'
        if not current_complete:reason='incomplete_current_feedback'
        elif solved and self.settings['stop_on_success']:reason='current_panel_satisfied'
        elif not ranked:reason='no_complete_development_panel'
        else:
            seen={tuple(inspected['values']),*grouped.keys()}
            population=self.settings['population']
            global_only=self.settings['strategy']=='uniform' or not model_ready
            global_count=population if global_only else max(1,math.ceil(population*self.settings['exploration_fraction']))
            if self.settings['strategy']=='adaptive' and not model_ready:
                reason='global_warmup'
            # A fixed attempt bound makes even an exhausted coarse grid terminate.
            for slot in range(population):
                role='global' if slot<global_count else 'elite'
                for attempt in range(256):
                    vector=([rng.random() for _ in center] if role=='global' else [rng.gauss(m,s) for m,s in zip(center,scale)])
                    values=self.space.quantize(vector)
                    if tuple(values) not in seen:
                        proposals.append(dict(values=values,role=role));seen.add(tuple(values));break
            if not proposals:reason='no_unseen_sample_within_attempt_budget'
        report=dict(schema='physicalrsi.adaptive-parameter-fit/v1',identity=self.identity(),
            parent_sha256=verify_harness(parent),condition_sha256=inspected['condition_sha256'],
            archive_view_sha256=digest(search['view']),panel=panel,samples=samples,excluded=excluded,
            ranked=ranked,incomplete=incomplete,elite=elite,center=center,scale=scale,model_ready=model_ready,
            random_seed=rng_seed,proposals=proposals,reason=reason,qualification=None)
        output=Path(output)
        freeze_json(output/'adaptive-fit.json',report)
        self._stable()
        evidence[str(output/'adaptive-fit.json')]=file_digest(output/'adaptive-fit.json')
        children=[]
        for index,proposal in enumerate(proposals):
            name='adaptive-'+digest(report)[:12]+'-'+str(index)
            candidate,changes=self.space.candidate(parent,inspected,proposal['values'],output=output/name,name=name)
            children.append(dict(candidate,parent_sha256=verify_harness(parent),method='development-adaptive-parameter-search',
                changes=changes,environment=dict(scope=search['view']['scope']),evidence=deepcopy(evidence),
                costs=dict(proposals=1),used_lessons=[],
                hypothesis='Explore globally and near parameter vectors with stronger comparable development scores',
                expected_effect='Improve independently evaluated task outcomes without assuming parameter monotonicity',
                required_checks=['fresh-paired-evaluation','complete-task-success']))
        self._stable()
        return children

    resume_proposal_from_archive=propose_from_archive
