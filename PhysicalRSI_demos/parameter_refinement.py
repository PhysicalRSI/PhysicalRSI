"""CPU acceptance: refine a Newton iteration budget using rejected rollouts.

Two square-root tasks execute numerical iterations and expose their residuals.
The search sees development evaluations, never the verifier's validation cases.
This demonstrates shared search infrastructure, not model training or robotics.
"""
from pathlib import Path
import math

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, Revision, System1
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.fidelity import MultiFidelityEvaluator
from PhysicalRSI_core.self_harness.parameter_search import ScalarParameter, ScalarRefinementEvolver
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.rollout import RolloutProposer
from .recursive_research import Selector, manifest

TASKS = ('small-root', 'large-root')
OBSERVATION = Contract('positive-square-root-residual', embodiment='software-numerical')
ACTION = Contract('newton-iterations', embodiment='software-numerical')


def identity(kind):
    return dict(kind=kind, implementation=file_digest(Path(__file__)))


class Environment:
    def __init__(self, task, resets):
        self.task, self.resets = task, resets

    def identity(self):
        return dict(identity('newton-environment'), task=self.task)

    def describe(self):
        return Embodiment('newton', '1', 'software', OBSERVATION, ACTION)

    def reset(self, case, context):
        self.resets.append((self.task, case))
        self.number, self.estimate = case['number'], case['initial']
        return dict(ready=True, observation=dict(number=self.number, estimate=self.estimate))

    def step(self, action, context):
        for _ in range(action['iterations']):
            self.estimate = (self.estimate + self.number / self.estimate) / 2
        return dict(observation=dict(number=self.number, estimate=self.estimate), terminated=True)


class Policy:
    def __init__(self, candidate):
        self.revision = verify_harness(candidate)
        self.iterations = read_json(Path(candidate['root']) / 'control.json')['iterations']

    def identity(self):
        return dict(identity('newton-policy'), revision=self.revision, iterations=self.iterations)

    def describe(self):
        return System1('newton', self.revision, OBSERVATION, ACTION)

    def begin_episode(self, *args):
        pass

    def act(self, observation, context):
        return dict(iterations=int(self.iterations))


class Verifier:
    def identity(self):
        return identity('root-residual-verifier')

    def verify(self, case, trace):
        residual = abs(trace[-1]['observation']['estimate'] ** 2 - case['number'])
        return dict(outcome='success' if residual < 1e-9 else 'failure',
                    reason='Independently measured square-root residual', measurements=dict(residual=residual))


class Feedback:
    def identity(self):
        return identity('root-residual-feedback')

    def classify(self, value):
        residual = value['verdict']['measurements']['residual']
        if not math.isfinite(residual) or residual < 0:
            return 'unresolved'
        return 'satisfied' if value['outcome'] == 'success' else 'insufficient'


class Suite:
    def __init__(self):
        self.resets = []

    def identity(self):
        return identity('two-root-suite')

    def admit(self, candidate):
        value = read_json(Path(candidate['root']) / 'control.json')['iterations']
        valid = type(value) in (int, float) and math.isfinite(value) and 1 <= value <= 9 and int(value) == value
        return dict(accepted=valid, reason='An integer iteration budget between 1 and 9',
                    evidence=dict(closure=verify_harness(candidate)))

    def environment(self, task):
        return Environment(task, self.resets)

    def policy(self, candidate, task):
        return Policy(candidate)

    def verifier(self, task):
        return Verifier()


class Reflector:
    def identity(self):
        return identity('fixed-numerical-hypothesis')

    def reflect(self, context, development, output):
        return dict(schema='physicalrsi.memory-delta/v1', operations=[])


def setup(root):
    root = Path(root).resolve()
    seed = root / 'seed'
    if not seed.exists():
        for kind in CLOSURE:
            atomic_json(seed / (kind+'.json'), dict(iterations=1) if kind == 'control' else dict(kind=kind))
    initial = manifest(seed, 'one-iteration')
    state, suite, archive = HarnessState(root/'state'), Suite(), CandidateArchive(root/'archive')
    evolver = ScalarRefinementEvolver(archive=archive, classifier=Feedback(),
        parameter=ScalarParameter('control', 'control.json', ('iterations',), 1, 9, precision=0))
    memory = ResearchMemory(root/'memory')
    memory_revision = memory.write({'numerical-residual':dict(conditions=dict(study='newton-scalar-refinement'),
        hypothesis='Increase iterations only after a finite residual fails its tolerance',
        required_checks=['complete-task-success'], counterexample_checks=[],
        evidence={str(seed/'control.json'):file_digest(seed/'control.json')})})
    quota = TrialQuota(root/'quota', max_trials=30)
    scheduler = JobScheduler(root/'scheduler', capacities={'cpu': 1})

    def build(directory, context):
        def cases(offsets):
            return {task: [dict(initial=10., number=base+n) for n in offsets]
                    for task, base in zip(TASKS, (2., 3.))}
        protocol = CaseProtocol(directory/'cases', splits=dict(development=cases([0]),
            validation=cases([.1+context['index']*.2, .15+context['index']*.2]), test=cases([.8])),
            identity_keys=('initial', 'number'))
        declared = PreregisteredSuite(suite, protocol)
        common = dict(suite=declared, budgets={t:Budget(1,30) for t in TASKS}, scope='software', quota=quota)
        screening = ExperimentEvaluator(**common, system2=Revision('newton-screening','1'))
        evaluator = MultiFidelityEvaluator(**common, system2=Revision('newton-validation','1'),
            screening=screening, scheduler=scheduler, resources={'cpu':1}, screening_resources={'cpu':1},
            workers=1, minimum_success_rate=1., screen_incumbent=False)
        return SelfHarness(directory, state=state, proposer=RolloutProposer(evaluator,evolver,archive=archive),
            evaluator=evaluator, selector=Selector(), evolution=EvolutionSpec(('control',)),
            archive=archive, max_candidates=1, scope='software', round_context=context,
            profile=dict(tasks={t:dict(weight=1,episodes=2,score_range=[0,1],maximum_regression=0) for t in TASKS},
                         minimum_gain=0,tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision,evaluation_kind='local_contract_evaluation'))

    campaign = ResearchCampaign(root/'campaign',state=state,build_loop=build,memory=memory,
        memory_revision=memory_revision,context=dict(study='newton-scalar-refinement'),reflector=Reflector(),max_rounds=2)
    return campaign, initial, suite, quota


def run(root):
    campaign, initial, _, quota = setup(root)
    result = campaign.run(initial)
    return dict(result,final_test=campaign.report_test(initial),quota=quota.status())
