"""Native two-parameter search through the ordinary task-independent core ports.

The operator declares all case panels. Adaptive and uniform modes share the
same numeric space and independent success gates; neither implies a model update.
"""
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.self_harness.adaptive_search import AdaptiveParameterEvolver
from PhysicalRSI_core.self_harness.search_space import ParameterSpace, SearchField
from . import research as r

CHECKS = ('correct_tube_in_receiver', 'upright', 'supported', 'released',
          'settled', 'obstacle_undisturbed', 'distractor_undisturbed')


class Objective:
    def identity(self):
        return dict(kind='native-development-intended-checks/v1', implementation=r.implementation(),
                    checks=list(CHECKS), failure_weight=.9, success_score=1.)

    def score(self, value):
        if value['trial']['task'] != r.TASK or value['trial']['scope'] != r.SCOPE:
            raise ValueError('Native search requires the declared task and scope')
        if value['outcome'] not in ('success', 'failure'):
            return None
        checks = value['verdict']['measurements']['checks']
        if not isinstance(checks, dict) or any(type(checks.get(name)) is not bool for name in CHECKS):
            raise ValueError('Native objective requires all seven independent boolean checks')
        if value['outcome'] == 'success':
            if not all(checks[name] for name in CHECKS):
                raise ValueError('Native success contradicts the intended-task checks')
            return 1.
        return .9 * sum(checks[name] for name in CHECKS) / len(CHECKS)


class Reflector:
    def identity(self):
        return dict(kind='fixed-native-intended-task/v1', implementation=r.implementation())

    def reflect(self, context, development, output):
        # Parameter observations live in the CandidateArchive. Do not invent a
        # clearance diagnosis for failures that may instead depend on the grip.
        return dict(schema='physicalrsi.memory-delta/v1', operations=[])


def _panels(value):
    if not isinstance(value, dict) or set(value) != {'development', 'validation', 'test'}:
        raise ValueError('Declare development, per-round validation and final test seed panels')
    value = deepcopy(value)
    rounds = value['validation']
    if not isinstance(rounds, list) or not 1 <= len(rounds) <= 20:
        raise ValueError('Declare one to twenty validation panels')
    groups = [value['development'], *rounds, value['test']]
    if any(not isinstance(seeds, list) or not 1 <= len(seeds) <= 50
           or any(type(seed) is not int for seed in seeds) for seeds in groups):
        raise ValueError('Each native panel requires one to fifty integer seeds')
    if len({len(seeds) for seeds in rounds}) != 1:
        raise ValueError('Every round requires the same paired validation count')
    seeds = [seed for group in groups for seed in group]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Native seed panels must be disjoint and duplicate-free')
    return value


def setup(workspace, python, *, panels, strategy, seed, population=2,
          device_registry=None, resource='isaac-gpu-0', gui=False,
          max_replacements=0, extra_trials=0):
    panels = _panels(panels)
    if strategy not in ('adaptive', 'uniform') or type(seed) is not int:
        raise ValueError('Declare adaptive or uniform search and an integer sampler seed')
    if type(population) is not int or not 2 <= population <= 8:
        raise ValueError('Declare two to eight native proposals per round')
    if (type(max_replacements) is not int or not 0 <= max_replacements <= 10
            or type(extra_trials) is not int or extra_trials < 0):
        raise ValueError('Declare a bounded Trial replacement allowance and extra Trial capacity')
    root = Path(workspace).resolve()
    initial = r.manifest(root/'seed', 'initial') if (root/'seed').exists() else r.create_seed(root/'seed')
    state, memory = r.HarnessState(root/'state'), r.ResearchMemory(root/'memory')
    archive = r.CandidateArchive(root/'archive')
    parameter_space = ParameterSpace([
        SearchField('memory_rules', 'memory_rules.json', ('transit_height_m',), .10, .55),
        SearchField('memory_rules', 'memory_rules.json', ('grasp_depth_fraction',), .15, .40)])
    evolver = AdaptiveParameterEvolver(archive=archive, space=parameter_space, objective=Objective(),
        task_weights={r.TASK:1}, population=population, seed=seed, strategy=strategy,
        minimum_observations=3, elite_fraction=.5, exploration_fraction=.5,
        minimum_scale=.025, covariance='diagonal', stop_on_success=False)
    revision = memory.write({'native-intended-task':dict(conditions=dict(task=r.TASK),
        hypothesis='Explore carrying height and grasp depth using complete development check panels',
        required_checks=['complete-native-transfer-and-disturbance'], counterexample_checks=[],
        evidence={str(root/'seed/source/task.py'):r.file_digest(root/'seed/source/task.py')})})
    rounds, episodes = len(panels['validation']), len(panels['validation'][0])
    quota = r.TrialQuota(root/'quota', max_trials=rounds*(len(panels['development'])*(1+population)
        + episodes*(1+population))+len(panels['test'])+extra_trials)
    registry = r.DeviceRegistry(device_registry or root/'devices')
    scheduler = r.JobScheduler(root/'scheduler', capacities={'native_worker':1})

    def build(directory, context):
        def cases(seeds):return {r.TASK:[dict(seed=seed) for seed in seeds]}
        protocol = r.CaseProtocol(directory/'cases', splits=dict(development=cases(panels['development']),
            validation=cases(panels['validation'][context['index']]), test=cases(panels['test'])))
        suite = r.Suite(python, root/'native'/('round-'+str(context['index'])), resource=resource, gui=gui)
        declared = r.PreregisteredSuite(suite, protocol)
        screening = r.ExperimentEvaluator(suite=r.PreregisteredSuite(r.Suite(python, suite.root/'screening',
            resource=resource, gui=gui), protocol), budgets={r.TASK:r.Budget(1,1200)}, scope=r.SCOPE,
            system2=r.Revision('isaac-empirical-screening','1'), quota=quota, device_registry=registry,
            max_replacements=max_replacements)
        evaluator = r.MultiFidelityEvaluator(suite=declared, budgets={r.TASK:r.Budget(1,1200)}, scope=r.SCOPE,
            system2=r.Revision('isaac-empirical-validation','1'), quota=quota, device_registry=registry,
            max_replacements=max_replacements, screening=screening, screening_resources={'native_worker':1}, minimum_success_rate=1.,
            screen_incumbent=False, scheduler=scheduler, resources={'native_worker':1}, workers=1, wait_seconds=7200)
        return r.SelfHarness(directory, state=state, proposer=r.RolloutProposer(evaluator,evolver,archive=archive),
            evaluator=evaluator, selector=r.Selector(), evolution=r.EvolutionSpec(('memory_rules',)),
            archive=archive, max_candidates=population, scope=r.SCOPE, round_context=context,
            profile=dict(tasks={r.TASK:dict(weight=1,episodes=episodes,score_range=[0,1],maximum_regression=0,
                minimum_success_rate=1.,maximum_success_regression=0)}, minimum_gain=0,tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision,evaluation_kind='local_contract_evaluation',
                          backend='isaacsim-5.1',qualification=None))

    campaign = r.ResearchCampaign(root/'campaign',state=state,build_loop=build,memory=memory,memory_revision=revision,
        context=dict(task=r.TASK,study='native-empirical-parameters'),reflector=Reflector(),max_rounds=rounds)
    return campaign, initial, quota


def run(workspace, python, **options):
    campaign, initial, quota = setup(workspace, python, **options)
    result = campaign.run(initial)
    return dict(result, final_test=campaign.report_test(initial), quota=quota.status(), qualification=None)
