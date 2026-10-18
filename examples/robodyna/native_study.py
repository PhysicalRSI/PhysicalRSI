"""Ramp-only evolution and frozen, zero-tune two-task transfer evaluation.

The supplied suite executes real RoboDyna; CPU test suites only test orchestration.
All reset seeds are committed before development. No final observations reach
proposal generation. A2 is the selected survivor and may be identical to A1.
"""
from copy import deepcopy
import difflib
import math
from pathlib import Path
import shutil
from statistics import mean, median

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget, trial_adapters
from PhysicalRSI_Autoresearch.records import ResearchRecord
from physicalrsi.research import ResearchWorkflow
from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import digest, file_digest, locked, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator, freeze_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.scheduled import ScheduledEvaluator
from .study import Reflector, Selector

RAMP = 'catch_ramp_ball'
VALLEY = 'catch_valley_ball'


def manifest(root, name, **metadata):
    root = Path(root)
    return dict(id=name, root=str(root), components={kind: {
        ('dynamic_interception.json' if kind == 'skills' else kind + '.json'):
        file_digest(root / ('dynamic_interception.json' if kind == 'skills' else kind + '.json'))
        } for kind in CLOSURE}, **metadata)


def seed_candidate(root, name, parameters, implementation):
    root = Path(root)
    freeze_json(root / 'dynamic_interception.json', parameters)
    for kind in CLOSURE - {'skills'}:
        freeze_json(root / (kind + '.json'), dict(component=kind, implementation=implementation))
    return manifest(root, name)


def estimate_rolling_energy(evaluations):
    """Infer energy coefficient from observed acceleration, never native oracle fields.

    The guide supplies a local tangent and gravity supplies its downhill
    component. Repeated sliding observation windows count each time triple once
    per episode. Corners, uphill/flat guides and out-of-domain estimates abstain.
    """
    estimates = []
    for evaluation in evaluations:
        seen = set()
        for row in evaluation['trajectory']:
            observation = row.get('observation', {}).get('policy_observation', {})
            if observation.get('phase') != 'rolling':
                continue
            samples = observation.get('samples', [])
            geometry = observation.get('geometry', {})
            path = geometry.get('path_m', [])
            gravity = geometry.get('gravity_m_s2', 9.81)
            if not isinstance(gravity, (int, float)) or not math.isfinite(gravity) or gravity <= 0:
                continue
            segments = []
            for start, end in zip(path, path[1:]):
                delta = [b-a for a, b in zip(start, end)]
                length = math.sqrt(sum(x*x for x in delta))
                if len(delta) != 3 or not math.isfinite(length) or length <= 1e-9:
                    continue
                segments.append((start, [x/length for x in delta], length))
            if not segments:
                continue
            def closest(point):
                distances = []
                for start, tangent, length in segments:
                    distance = min(length, max(0., sum((p-a)*t for p, a, t in zip(point, start, tangent))))
                    distances.append(sum((p-a-distance*t)**2 for p, a, t in zip(point, start, tangent)))
                return min(range(len(distances)), key=distances.__getitem__)
            for triple in zip(samples, samples[1:], samples[2:]):
                if any(sample.get('phase', 'rolling') != 'rolling' for sample in triple):
                    continue
                times = [sample['timestamp_s'] for sample in triple]
                positions = [sample['position_m'] for sample in triple]
                if any(not math.isfinite(t) for t in times) or not times[0] < times[1] < times[2]:
                    continue
                if any(len(p) != 3 or any(not math.isfinite(x) for x in p) for p in positions):
                    continue
                key = tuple(times)
                if key in seen:
                    continue
                seen.add(key)
                indices = [closest(p) for p in positions]
                if len(set(indices)) != 1:
                    continue
                tangent = segments[indices[0]][1]
                downhill = -tangent[2]
                if downhill <= 1e-4:
                    continue
                dt0, dt1 = times[1]-times[0], times[2]-times[1]
                acceleration = [2*((positions[2][i]-positions[1][i])/dt1
                                  -(positions[1][i]-positions[0][i])/dt0)/(dt0+dt1) for i in range(3)]
                factor = sum(a*t for a, t in zip(acceleration, tangent))/(gravity*downhill)
                if math.isfinite(factor) and 0 <= factor <= 1:
                    estimates.append(factor)
    return dict(method='median-observed-tangential-acceleration/v1', samples=len(estimates),
                rolling_energy_factor=median(estimates) if len(estimates) >= 3 else None,
                observed_range=[min(estimates), max(estimates)] if estimates else None)


class BoundedProposer:
    """Infer one shared parameter after failure, or use a preregistered override.

    Inference reads verified development observations only. Its bounded model,
    source and optional override are frozen before development; the resulting
    candidate freezes before validation. Insufficient evidence retains parent.
    """
    def __init__(self, evaluator, parameters, hypothesis, *, tasks=(RAMP,), trigger="failure"):
        self.tasks, self.trigger = tuple(tasks), trigger
        self.evaluator, self.parameters, self.hypothesis = evaluator, deepcopy(parameters), hypothesis

    def identity(self):
        return dict(kind='bounded-native-interception/v1', parameters=self.parameters,
                    hypothesis=self.hypothesis, tasks=list(self.tasks), trigger=self.trigger, source=file_digest(Path(__file__)))

    def develop(self, parent, output):
        return self.evaluator.development(parent, output)

    resume_development = develop

    def propose_with_context(self, parent, feedback, context, output):
        episodes = feedback['episodes']
        if set(row['task'] for row in episodes) != set(self.tasks):
            raise ValueError('Development feedback differs from preregistered evolution tasks')
        if self.trigger == 'failure' and not any(row['outcome'] == 'failure' for row in episodes):
            return []
        from PhysicalRSI_core.self_harness.rollout import development_evaluations
        evaluations = development_evaluations(parent, feedback, output)
        inference = estimate_rolling_energy(evaluations)
        parameters = deepcopy(self.parameters)
        if parameters is None:
            parameters = read_json(Path(parent['root']) / 'dynamic_interception.json')
            coefficient = inference['rolling_energy_factor']
            if coefficient is None:
                freeze_json(output / 'development-analysis.json', dict(episodes=episodes,
                    hypothesis=self.hypothesis, inference=inference, disposition='insufficient-observed-acceleration',
                    evidence=feedback['evidence']))
                return []
            parameters['rolling_energy_factor'] = coefficient
        if parameters == read_json(Path(parent['root']) / 'dynamic_interception.json'):
            return []
        root = output / 'candidate'
        shutil.copytree(parent['root'], root)
        from PhysicalRSI_core.infra.storage import atomic_json
        atomic_json(root / 'dynamic_interception.json', parameters)
        before = (Path(parent['root']) / 'dynamic_interception.json').read_text()
        after = (root / 'dynamic_interception.json').read_text()
        (output / 'skill.diff').write_text(''.join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile='parent', tofile='candidate')))
        freeze_json(output / 'development-analysis.json', dict(episodes=episodes,
                    hypothesis=self.hypothesis, inference=inference, explicit_override=self.parameters is not None,
                    evidence=feedback['evidence']))
        return [manifest(root, 'candidate', parent_sha256=verify_harness(parent),
            method='bounded-shared-prediction-revision', changes=dict(skills=['dynamic_interception']),
            environment='native-RoboDyna', evidence=feedback['evidence'], costs=dict(proposals=1),
            used_lessons=[], hypothesis=self.hypothesis, expected_effect='Improved paired native success',
            required_checks=['paired-success', 'zero-success-regression'])]


def validate_seeds(development, validation, heldout):
    groups = [list(development), list(validation), list(heldout)]
    if any(not group for group in groups):
        raise ValueError('Every split requires seeds')
    flat = [seed for group in groups for seed in group]
    if any(type(seed) is not int or seed < 0 for seed in flat) or len(set(flat)) != len(flat):
        raise ValueError('Seeds must be distinct nonnegative integers across all splits')
    return groups


def setup(root, *, suite, parent_parameters, baseline_parameters, candidate_parameters=None,
          development=(101,), validation=(201, 202), heldout=(10000, 10001, 10002), scenario='default',
          mode='frozen', max_steps=800, seconds=600, evolution_tasks=(RAMP,), proposal_trigger='failure',
          predecessor=None,
          hypothesis='A bounded shared prediction revision reduces interception failure'):
    root = Path(root).resolve()
    development, validation, heldout = validate_seeds(development, validation, heldout)
    if mode not in {'frozen', 'async'}:
        raise ValueError('Declare frozen or async evaluation')
    evolution_tasks = tuple(evolution_tasks)
    if evolution_tasks not in ((RAMP,), (RAMP, VALLEY)):
        raise ValueError('Declare Ramp-only or two-task joint evolution')
    if proposal_trigger not in ('failure', 'prediction-calibration'):
        raise ValueError('Unknown proposal trigger')
    implementation = dict(suite=suite.identity(), study=file_digest(Path(__file__)))
    native = implementation['suite'].get('kind') == 'native-robodyna'
    scope = dict(control_mode=implementation['suite'].get('control_mode', 'contract-fixture'),
        observation_mode=implementation['suite'].get('observation_mode', 'CPU-contract-only'),
        privileged_inputs=['simulator ball position', 'static guide geometry', 'robot/catcher pose'] if native else [],
        scoring='upstream task check_success/get_score' if native else 'CPU-contract-only',
        latency_scope=dict(latency_seconds='shared skill inference only',
                           episode_wall_seconds='native rollout wall time excluding reset'),
        comparison='state-observation code-policy pilot; not published VLA protocol' if native else 'CPU contract test')
    config = dict(implementation=implementation, scope=scope, parent=parent_parameters, candidate=candidate_parameters,
                  baseline=baseline_parameters, development=development, validation=validation,
                  heldout=heldout, scenario=scenario, mode=mode, max_steps=max_steps,
                  seconds=seconds, hypothesis=hypothesis, evolution_tasks=list(evolution_tasks),
                  proposal_trigger=proposal_trigger, predecessor=predecessor)
    freeze_json(root / 'config.json', config)
    parent = seed_candidate(root / 'A1', 'parent', parent_parameters, implementation)
    baseline = seed_candidate(root / 'A0', 'baseline', baseline_parameters, implementation)
    memory = ResearchMemory(root / 'memory')
    memory_revision = memory.write({'native-interception': dict(
        conditions={'study': 'native-dynamic-interception'}, hypothesis=hypothesis,
        required_checks=['paired-success'], counterexample_checks=[],
        evidence={str(root / 'config.json'): file_digest(root / 'config.json')})})
    state = HarnessState(root / 'lineage')
    # Reserve both paired arms and all three final groups; identical A1/A2
    # reuse one report, so unused quota remains available but is not consumed.
    quota = TrialQuota(root / 'quota', max_trials=len(evolution_tasks)*(len(development) + 2*len(validation)) + 6*len(heldout))
    scheduler = JobScheduler(root / 'scheduler', capacities={'simulator': 1})
    cases = lambda seeds: [dict(seed=s, scenario=scenario, mode=mode) for s in seeds]
    splits = {split: {task: cases(seeds) for task in evolution_tasks} for split, seeds in
              [('development', development), ('validation', validation), ('test', heldout)]}
    def build(directory, context):
        protocol = CaseProtocol(directory / 'cases', splits=splits, identity_keys=('seed', 'scenario', 'mode'))
        evaluator = ScheduledEvaluator(suite=PreregisteredSuite(suite, protocol),
            budgets={task: Budget(max_steps, seconds) for task in evolution_tasks}, scope='native-RoboDyna',
            system2=Revision('bounded-shared-interception', '1'), quota=quota,
            scheduler=scheduler, resources={'simulator': 1}, workers=1,
            wait_seconds=max(3600, len(evolution_tasks)*2*len(validation)*seconds+300))
        return SelfHarness(directory, state=state,
            proposer=BoundedProposer(evaluator, candidate_parameters, hypothesis,
                tasks=evolution_tasks, trigger=proposal_trigger), evaluator=evaluator,
            selector=Selector(), scope='native-RoboDyna', evolution=EvolutionSpec(mutable=('skills',)),
            max_candidates=1, round_context=context,
            profile=dict(tasks={task: dict(weight=1, episodes=len(validation), score_range=[0, 1],
                         maximum_regression=0, maximum_success_regression=0) for task in evolution_tasks},
                         minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind='native-RoboDyna'))
    workflow = ResearchWorkflow(trial_adapters(suite, parent, RAMP).policy.skill)
    campaign = workflow.campaign(root / 'campaign', state=state, build_loop=build, memory=memory,
        memory_revision=memory_revision, context={'study': 'native-dynamic-interception'},
        reflector=Reflector(), max_rounds=1)
    return campaign, parent, baseline, quota, config


def wilson(successes, count):
    """95% Wilson binomial interval; does not imply seed independence."""
    if count <= 0:
        raise ValueError('Nonempty denominator required')
    p, z = successes/count, 1.959963984540054
    center = (p + z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0., center-half), min(1., center+half)]


def summarize(report):
    result = deepcopy(report['metrics'])
    for task, row in result.items():
        row['success_interval_95_wilson'] = wilson(row['successes'], row['planned'])
        measurements = [r['measurements'] for r in report['episodes'] if r['task'] == task]
        for key in ('native_progress', 'latency_seconds', 'inference_latency_seconds', 'episode_wall_seconds'):
            values = [m[key] for m in measurements if isinstance(m.get(key), (int, float))]
            row[key] = dict(observed=len(values), mean=mean(values) if values else None)
    return result


def run(root, **kwargs):
    """Resume verified stages; interrupted native effects require reconciliation."""
    root = Path(root).resolve()
    with locked(root / '.native-study.lock'):
        campaign, parent, baseline, quota, config = setup(root, **kwargs)
        suite = kwargs['suite']
        workflow = ResearchWorkflow(trial_adapters(suite, parent, RAMP).policy.skill)
        context = campaign._context(0, '0'*64, campaign.initial_memory, None)
        prepared = campaign.build_loop(root / 'preparation', context)
        campaign.state.initialize(parent, policy=prepared.policy, scope=prepared.config['scope'])
        record = ResearchRecord(root / 'research')
        evidence = {str(root / 'config.json'): file_digest(root / 'config.json')}
        if not (record.root / 'current.json').exists():
            workflow.create_record(record,
                question=('Can one shared interception revision improve the joint task average without task regression?'
                          if len(config['evolution_tasks']) == 2 else
                          'Can a shared interception revision improve Ramp and transfer without Valley tuning?'),
                hypothesis=config['hypothesis'],
                falsifier='Preregistered paired selection or heldout comparison fails to support improvement.',
                alternatives=['Execution bottlenecks dominate prediction.',
                              'A Ramp correction does not transfer to Valley.'],
                budget=dict(max_trials=quota.max_trials, trial_seconds=math.ceil(config['seconds'])),
                state=campaign.state, memory=campaign.memory, memory_revision=campaign.initial_memory,
                context=campaign.context, evidence=evidence)
        snapshot = record.read()
        if snapshot['body']['status'] == 'preparing':
            snapshot = workflow.bind_experiment(record, snapshot['revision'], candidate=parent,
                evaluator_sha256=prepared.evaluator.revision, protocol=prepared.evaluator.suite.protocol,
                workspace=root / 'campaign', used_lessons=[], evidence=evidence)
        if snapshot['body']['status'] == 'ready':
            snapshot = record.claim_dispatch(snapshot['revision'], job_id='local-native-interception-campaign')
        # Core reconciles existing stages. A dispatch claim alone never proves
        # a worker is live or authorizes replay of an uncertain native effect.
        result = campaign.run(parent)
        freeze_json(root / 'campaign-result.json', result)
        if result['status'] != 'completed':
            return result
        selected = result['current']['harness']
        # Commit ALL report groups before any held-out access.
        groups = {'A0': baseline, 'A1': parent, 'A2': selected}
        freeze_json(root / 'final-groups.json', dict(selection_revision=result['revision'],
            groups={name: verify_harness(value) for name, value in groups.items()},
            transfer=('Joint-selected shared revision; this is not zero-tune transfer'
                      if len(config['evolution_tasks']) == 2 else
                      'Ramp-selected revision applied to Valley without tuning')))
        suite = kwargs['suite']
        def cases(seeds):
            return [dict(seed=s, scenario=config['scenario'], mode=config['mode']) for s in seeds]
        protocol = CaseProtocol(root / 'final-cases', splits={split: {
            task: cases(config[key]) for task in (RAMP, VALLEY)} for split, key in
            [('development', 'development'), ('validation', 'validation'), ('test', 'heldout')]},
            identity_keys=('seed', 'scenario', 'mode'))
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, protocol),
            budgets={task: Budget(config['max_steps'], config['seconds']) for task in (RAMP, VALLEY)},
            scope='native-RoboDyna-report-only', system2=Revision('zero-tune-transfer', '1'), quota=quota)
        reports, by_revision = {}, {}
        for name, candidate in groups.items():
            revision = verify_harness(candidate)
            if revision not in by_revision:
                by_revision[revision] = evaluator.report_test(candidate,
                    selection_sha256=result['revision'], output=root / 'final' / name)
            reports[name] = by_revision[revision]
        summary = dict(schema='physicalrsi.native-transfer/v1', selection_revision=result['revision'],
            groups={name: dict(candidate_sha256=verify_harness(groups[name]),
                              metrics=summarize(report)) for name, report in reports.items()},
            same_A1_A2=verify_harness(parent) == verify_harness(selected),
            qualification=None, evaluation_mode=config['mode'], scope=config['scope'])
        freeze_json(root / 'summary.json', summary)
        lines = ['# RoboDyna shared skill experiment', '',
                 f"Control mode: {config['scope']['control_mode']}.",
                 f"Observations: {config['scope']['observation_mode']}.",
                 f"Scoring: {config['scope']['scoring']}.",
                 f"Scope: {config['scope']['comparison']}.", '',
                 ('Both-task development and paired selection; independent heldout reporting.'
                  if len(config['evolution_tasks']) == 2 else
                  'Ramp development and paired selection; Valley transfer without tuning.'), '',
                 '| Group | Task | Success | Uncertain / invalid | Rate | 95% Wilson interval | Progress | Inference (s) | Episode wall (s) |',
                 '|---|---|---:|---:|---:|---|---:|---:|---:|']
        for name, group in summary['groups'].items():
            for task, values in group['metrics'].items():
                low, high = values['success_interval_95_wilson']
                lines.append(f"| {name} | {task} | {values['successes']}/{values['planned']} | "
                    f"{values['uncertain']} / {values['invalid']} | {values['success_rate']:.3f} | [{low:.3f}, {high:.3f}] | "
                    f"{values['native_progress']['mean']} | {values['latency_seconds']['mean']} | "
                    f"{values['episode_wall_seconds']['mean']} |")
        lines.extend(['', 'A2 is the selected survivor; retention of A1 is a valid result.',
                      'Native outcome evidence is retained in each final report workspace.',
                      f"Evaluation mode: {config['mode']}.", ''])
        report_text = '\n'.join(lines)
        report_path = root / 'report.md'
        if report_path.exists() and report_path.read_text() != report_text:
            raise ValueError('Frozen Markdown report changed')
        if not report_path.exists():
            report_path.write_text(report_text)
        if snapshot['body']['status'] == 'awaiting_result':
            report_path = root / 'research-result.json'
            result_evidence = {str(path): file_digest(path) for path in
                [root / 'campaign-result.json', root / 'summary.json', root / 'report.md',
                 root / 'final-groups.json', root / 'final-cases/protocol.json']}
            result_evidence.update({str(path): file_digest(path) for path in (root / 'final').rglob('report.json')})
            freeze_json(report_path, dict(schema='physicalrsi.research-result/v1',
                research_revision=snapshot['revision'], experiment_sha256=digest(snapshot['body']['experiment']),
                job_id=snapshot['body']['job_id'], outcome='completed', evidence=result_evidence))
            snapshot = record.record_result(snapshot['revision'], report=report_path)
        if snapshot['body']['status'] == 'awaiting_decision':
            paired_improved = (result['current'].get('decision') or {}).get('decision') == 'inherit_child'
            a1, a2 = summary['groups']['A1']['metrics'], summary['groups']['A2']['metrics']
            heldout_supported = (all(a2[t]['success_rate'] >= a1[t]['success_rate'] for t in (RAMP, VALLEY))
                                 and any(a2[t]['success_rate'] > a1[t]['success_rate'] for t in config['evolution_tasks']))
            snapshot = record.close(snapshot['revision'],
                disposition='supported' if paired_improved and heldout_supported else 'inconclusive',
                explanation='Preregistered paired selection and descriptive heldout evidence support the bounded hypothesis.'
                if paired_improved and heldout_supported else
                'This pilot does not establish shared improvement; retained native outcomes remain reportable.',
                evidence={str(root / 'summary.json'): file_digest(root / 'summary.json'),
                          str(root / 'campaign-result.json'): file_digest(root / 'campaign-result.json')},
                memory_revision=result['memory_revision'],
                memory_explanation='Retain development memory; heldout outcomes do not update the skill.')
        record.status()
        return summary


def run_joint(root, *, predecessor_root, suite, **kwargs):
    """Start a separately preregistered joint calibration from a completed pilot.

    The predecessor is read and hash-verified, never rerun under changed source.
    Shared parameter inheritance is explicit; new execution source pins form a
    new experiment closure and never relabel the historical zero-tune result.
    """
    predecessor_root = Path(predecessor_root).resolve()
    if Path(root).resolve() == predecessor_root:
        raise ValueError('Joint study requires a fresh workspace')
    record = ResearchRecord(predecessor_root / 'research')
    snapshot = record.read()
    if snapshot['body']['status'] != 'completed':
        raise ValueError('Complete predecessor reporting before joint evolution')
    for name, sha in snapshot['body']['evidence'].items():
        if file_digest(Path(name)) != sha:
            raise ValueError('Predecessor evidence changed: ' + name)
    result = read_json(predecessor_root / 'campaign-result.json')
    if result['status'] != 'completed':
        raise ValueError('Predecessor campaign is incomplete')
    selected = result['current']['harness']
    previous_config = read_json(predecessor_root / 'config.json')
    parameters = read_json(Path(selected['root']) / 'dynamic_interception.json')
    options = dict(development=(301,302), validation=(401,402,403), heldout=(10003,10004,10005),
        parent_parameters=parameters, baseline_parameters=previous_config['baseline'], evolution_tasks=(RAMP,VALLEY),
        proposal_trigger='prediction-calibration',
        hypothesis='Calibrating shared rolling energy from observed motion may improve prediction without reducing either task success.',
        predecessor=dict(root=str(predecessor_root), research_revision=snapshot['revision'],
                         campaign_revision=result['revision'], inherited_harness_sha256=verify_harness(selected),
                         inherited_parameters_sha256=digest(parameters)))
    options.update(kwargs)
    old_seeds = set(previous_config['development'] + previous_config['validation'] + previous_config['heldout'])
    if old_seeds.intersection(seed for key in ('development','validation','heldout') for seed in options[key]):
        raise ValueError('Joint study must use fresh development, validation and heldout seeds')
    if options['evolution_tasks'] != (RAMP, VALLEY) or options['proposal_trigger'] != 'prediction-calibration':
        raise ValueError('Joint study requires both-task calibration protocol')
    if options['parent_parameters'] != parameters:
        raise ValueError('Joint parent must inherit the selected shared parameters')
    return run(root, suite=suite, **options)
