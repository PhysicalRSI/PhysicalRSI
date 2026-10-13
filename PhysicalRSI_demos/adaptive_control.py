"""Measured CPU study of two-parameter feedback control under two load dynamics.

Each Trial integrates damped motion; the objective uses measured trajectory
error and overshoot. Adaptive and uniform proposals use the same space, quota,
case protocol and independent survivor selection. No simulator is implied.
"""
from pathlib import Path
import math

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, Revision, System1
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.adaptive_search import AdaptiveParameterEvolver
from PhysicalRSI_core.self_harness.search_space import ParameterSpace, SearchField
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.fidelity import MultiFidelityEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.rollout import RolloutProposer
from .recursive_research import Selector, manifest

TASKS=('light-load','heavy-load')
OBSERVATION=Contract('position-velocity-target',embodiment='cpu-damped-motion')
ACTION=Contract('force',embodiment='cpu-damped-motion')


def identity(kind,**values):
    return dict(kind=kind,implementation=file_digest(Path(__file__)),**values)


class Environment:
    def __init__(self,task,resets):
        self.task,self.resets=task,resets
        self.base_mass=1. if task=='light-load' else 1.7

    def identity(self):
        return identity('damped-motion',task=self.task,mass=self.base_mass,friction=.12,force_limit=4.,dt=.05,steps=100)

    def describe(self):
        return Embodiment(self.task,'1','software',OBSERVATION,ACTION)

    def observation(self):
        return dict(position=self.position,velocity=self.velocity,target=self.target)

    def reset(self,case,context):
        self.resets.append((self.task,case))
        self.position,self.velocity,self.target=case['initial'],0.,case['target']
        self.mass=self.base_mass*case.get('load_scale',1.)
        self.steps=0
        return dict(ready=True,observation=self.observation())

    def step(self,action,context):
        applied=max(-4.,min(4.,action))
        self.velocity+=.05*(applied-.12*self.velocity)/self.mass
        self.position+=.05*self.velocity
        self.steps+=1
        return dict(observation=self.observation(),terminated=self.steps==100)


class Policy:
    def __init__(self,candidate):
        self.revision=verify_harness(candidate)
        self.gains=read_json(Path(candidate['root'])/'control.json')

    def identity(self):
        return identity('pd-control',revision=self.revision,gains=self.gains)

    def describe(self):
        return System1('pd-control',self.revision,OBSERVATION,ACTION)

    def begin_episode(self,*args):pass

    def act(self,observation,context):
        return self.gains['proportional']*(observation['target']-observation['position'])-self.gains['damping']*observation['velocity']


def measurements(case,trace):
    distance=abs(case['target']-case['initial'])
    direction=1 if case['target']>case['initial'] else -1
    errors=[abs(row['observation']['position']-case['target'])/distance for row in trace[1:]]
    overshoot=max(0.,max(direction*(row['observation']['position']-case['target'])/distance for row in trace[1:]))
    tail=sum(errors[-10:])/10
    quality=1/(1+sum(errors)/len(errors)+4*overshoot+8*tail)
    return dict(quality=quality,mean_error=sum(errors)/len(errors),tail_error=tail,overshoot=overshoot)


class Verifier:
    def identity(self):return identity('trajectory-error-verifier')

    def verify(self,case,trace):
        values=measurements(case,trace)
        passed=values['tail_error']<.025 and values['overshoot']<.15
        return dict(outcome='success' if passed else 'failure',reason='Measured settling error and overshoot',measurements=values)


class Objective:
    def identity(self):return identity('development-trajectory-quality')

    def score(self,value):return value['verdict']['measurements']['quality']


class Suite:
    def __init__(self):self.resets=[]

    def identity(self):return identity('two-load-cpu-control')

    def admit(self,candidate):
        gains=read_json(Path(candidate['root'])/'control.json')
        passed=set(gains)=={'proportional','damping'} and all(type(v) in (int,float) and math.isfinite(v) and .1<=v<=8 for v in gains.values())
        return dict(accepted=passed,reason='Two bounded feedback gains',evidence=dict(closure=verify_harness(candidate)))

    def environment(self,task):return Environment(task,self.resets)

    def policy(self,candidate,task):return Policy(candidate)

    def verifier(self,task):return Verifier()


class Reflector:
    def identity(self):return identity('fixed-development-objective')

    def reflect(self,context,development,output):
        return dict(schema='physicalrsi.memory-delta/v1',operations=[])


def setup(root,*,strategy='adaptive',seed=0,rounds=4,population=6):
    if type(rounds) is not int or not 1<=rounds<=6:
        raise ValueError('Use one to six rounds in this fixed CPU protocol')
    root=Path(root).resolve()
    folder=root/'seed'
    if not folder.exists():
        for kind in CLOSURE:
            atomic_json(folder/(kind+'.json'),dict(proportional=.3,damping=.1) if kind=='control' else dict(kind=kind))
    initial=manifest(folder,'initial')
    state,suite,archive=HarnessState(root/'state'),Suite(),CandidateArchive(root/'archive')
    space=ParameterSpace([SearchField('control','control.json',(key,),.1,8.,precision=3) for key in ('proportional','damping')])
    evolver=AdaptiveParameterEvolver(archive=archive,space=space,objective=Objective(),task_weights={t:1 for t in TASKS},
        population=population,seed=seed,strategy=strategy,stop_on_success=False)
    memory=ResearchMemory(root/'memory')
    revision=memory.write({'trajectory-quality':dict(conditions=dict(study='adaptive-control'),
        hypothesis='Use observed error and overshoot to explore both feedback gains',required_checks=['paired-quality'],
        counterexample_checks=[],evidence={str(folder/'control.json'):file_digest(folder/'control.json')})})
    # Reserve the same worst-case budget for either proposal strategy.
    quota=TrialQuota(root/'quota',max_trials=rounds*(4+population*4+(population+1)*4)+4)
    scheduler=JobScheduler(root/'scheduler',capacities={'cpu':2})

    def build(directory,context):
        def cases(settings):
            return {t:[dict(initial=0.,target=target,load_scale=load) for target,load in settings] for t in TASKS}
        index=context['index']
        protocol=CaseProtocol(directory/'cases',splits=dict(development=cases([(1.,.9),(-1.,1.1)]),
            validation=cases([(.8+.03*index,.95),(-1.2-.03*index,1.05)]),
            test=cases([(1.4,1.15),(-.7,.85)])),identity_keys=('initial','target','load_scale'))
        declared=PreregisteredSuite(suite,protocol)
        common=dict(suite=declared,budgets={t:Budget(100,30) for t in TASKS},scope='software',quota=quota,score_measurement='quality')
        screening=ExperimentEvaluator(**common,system2=Revision('pd-development-screening','1'))
        evaluator=MultiFidelityEvaluator(**common,system2=Revision('pd-validation','1'),screening=screening,
            scheduler=scheduler,resources={'cpu':1},screening_resources={'cpu':1},workers=2,minimum_success_rate=1.,screen_incumbent=False)
        return SelfHarness(directory,state=state,proposer=RolloutProposer(evaluator,evolver,archive=archive),evaluator=evaluator,
            selector=Selector(),evolution=EvolutionSpec(('control',)),archive=archive,max_candidates=population,scope='software',round_context=context,
            profile=dict(tasks={t:dict(weight=1,episodes=2,score_range=[0,1],maximum_regression=0,
                minimum_success_rate=1.,maximum_success_regression=0) for t in TASKS},minimum_gain=0,tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision,evaluation_kind='local_contract_evaluation'))

    campaign=ResearchCampaign(root/'campaign',state=state,build_loop=build,memory=memory,memory_revision=revision,
        context=dict(study='adaptive-control'),reflector=Reflector(),max_rounds=rounds)
    return campaign,initial,suite,quota


def run(root,**options):
    campaign,initial,_,quota=setup(root,**options)
    result=campaign.run(initial)
    return dict(result,final_test=campaign.report_test(initial),quota=quota.status())
