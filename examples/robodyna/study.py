"""A bounded skill experiment using unmodified Core and Autoresearch engines."""
import difflib
import json
import shutil
from pathlib import Path
from physicalrsi import SkillRegistry, compose_sequence
from physicalrsi.environments import RobodynaFixture
from physicalrsi.research import ResearchWorkflow
from physicalrsi.reporting import ReportBundle
from PhysicalRSI_Autoresearch.records import ResearchRecord
from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import Revision, System1
from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json, locked
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator, freeze_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from .skills import estimate_motion, move_to_target, predict_intercept


def composition(gain):
    return compose_sequence('intercept', estimate_motion(), predict_intercept(gain=gain), move_to_target(),
        ontology_revision='constant-velocity/v1', embodiment_profile='software-point/v1')


def identity(kind):
    from physicalrsi.skills import compose, protocol
    from physicalrsi.environments import robodyna
    return dict(kind=kind, sources={str(p): file_digest(p) for p in
        [Path(__file__), Path(__file__).with_name('skills.py'), Path(compose.__file__),
         Path(protocol.__file__), Path(robodyna.__file__)]})


def manifest(root, name, **metadata):
    return dict(id=name, root=str(root), components={kind: {
        p.name: file_digest(p) for p in ([root/'PredictIntercept.json'] if kind == 'skills'
        else [root/(kind+'.json')])} for kind in CLOSURE}, **metadata)


class Policy:
    def __init__(self, candidate):
        self.revision = verify_harness(candidate)
        self.skill = composition(read_json(Path(candidate['root'])/'PredictIntercept.json')['gain'])
    def identity(self):
        return dict(identity('composed-policy'), revision=self.revision, skill=self.skill.revision)
    def begin_episode(self, task, case, observation, context):
        context.check()
    def describe(self):
        return System1('intercept', self.revision, Contract('observation'), Contract('action'))
    def act(self, observation, context):
        return self.skill.execute(observation, context)


class Verifier:
    def identity(self):
        return identity('position-verifier')
    def verify(self, case, trace):
        final = trace[-1]['observation']
        error = abs(final['robot_position']-final['target_position'])
        return dict(outcome='success' if error <= .01 else 'failure',
                    reason='Observed intercept error', measurements=dict(error=error))


class Suite:
    def identity(self):
        return identity('interception-suite')
    def admit(self, candidate):
        verify_harness(candidate)
        gain = read_json(Path(candidate['root'])/'PredictIntercept.json')['gain']
        composition(gain)
        return dict(accepted=True, evidence=dict(composition=composition(gain).composition.revision))
    def environment(self, task):
        return RobodynaFixture()
    def policy(self, candidate, task):
        return Policy(candidate)
    def verifier(self, task):
        return Verifier()


class Proposer:
    def __init__(self, evaluator, gain):
        self.evaluator, self.gain = evaluator, gain
    def identity(self):
        return dict(identity('bounded-skill-parameter-proposer'), gain=self.gain)
    def develop(self, parent, output):
        return self.evaluator.development(parent, output)
    resume_development = develop
    def propose_with_context(self, parent, feedback, context, output):
        root = output/'candidate'
        shutil.copytree(parent['root'], root)
        atomic_json(root/'PredictIntercept.json', dict(gain=self.gain))
        return [manifest(root, 'candidate', parent_sha256=verify_harness(parent),
            method='bounded-PredictIntercept-parameter', changes=dict(skills=['PredictIntercept']),
            environment='software-contract-fixture', evidence=feedback['evidence'], costs=dict(proposals=1),
            used_lessons=[], hypothesis='Velocity extrapolation reduces observed intercept error',
            expected_effect='Higher paired success', required_checks=['paired-success'])]


class Selector:
    def identity(self):
        return dict(source=file_digest(Path(selection.__file__)))
    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


class Reflector:
    def identity(self):
        return identity('retain-memory-reflector')
    def reflect(self, context, development, output):
        return dict(schema='physicalrsi.memory-delta/v1', operations=[])


def setup(root, incumbent_gain=0.0, candidate_gain=1.0):
    root = Path(root).resolve()
    freeze_json(root/'config.json', dict(incumbent_gain=incumbent_gain, candidate_gain=candidate_gain,
                                       implementation=identity('fixture-study')))
    seed = root/'seed'
    for kind in CLOSURE:
        if kind == 'skills':
            freeze_json(seed/'PredictIntercept.json', dict(gain=incumbent_gain))
        else:
            freeze_json(seed/(kind+'.json'), dict(component=kind, implementation=identity('fixed-skills'),
                order=['EstimateMotion','PredictIntercept','MoveToTarget']))
    parent = manifest(seed, 'parent')
    memory = ResearchMemory(root/'memory')
    memory_revision = memory.write({'baseline': dict(conditions={'study':'interception-fixture'},
        hypothesis='Measure intercept error before changing PredictIntercept.',
        required_checks=['paired-success'], counterexample_checks=[],
        evidence={str(root/'config.json'): file_digest(root/'config.json')})})
    state = HarnessState(root/'lineage')
    quota = TrialQuota(root/'quota', max_trials=7)
    workflow = ResearchWorkflow(composition(incumbent_gain))

    def build(directory, context):
        def case(p, v):
            return dict(position=p, velocity=v, horizon=1.0)
        plan = CaseProtocol(directory/'cases', splits=dict(
            development={'fixture': [case(0., 1.)]},
            validation={'fixture': [case(2., 2.), case(-2., -1.)]},
            test={'fixture': [case(7., -3.)]}), identity_keys=('position','velocity','horizon'))
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(Suite(), plan),
            budgets={'fixture': Budget(1, 30)}, scope='software-contract-fixture',
            system2=Revision('skill-parameter-experiment','1'), quota=quota)
        return SelfHarness(directory, state=state, proposer=Proposer(evaluator, candidate_gain),
            evaluator=evaluator, selector=Selector(), scope='software-contract-fixture',
            evolution=EvolutionSpec(mutable=('skills',)), max_candidates=1, round_context=context,
            profile=dict(tasks={'fixture': dict(weight=1, episodes=2, score_range=[0,1], maximum_regression=0)},
                         minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind='software-contract-fixture'))

    campaign = workflow.campaign(root/'campaign', context={'study':'interception-fixture'},
        state=state, build_loop=build, memory=memory, memory_revision=memory_revision,
        reflector=Reflector(), max_rounds=1)
    return campaign, parent, workflow, quota


def run(root, *, incumbent_gain=0.0, candidate_gain=1.0):
    root = Path(root).resolve()
    with locked(root/'.study.lock'):
        return _run(root, incumbent_gain, candidate_gain)


def _run(root, incumbent_gain, candidate_gain):
    campaign, parent, workflow, quota = setup(root, incumbent_gain, candidate_gain)
    registry = SkillRegistry(root/'registry')
    for gain in (incumbent_gain, candidate_gain):
        for skill in [estimate_motion(), predict_intercept(gain=gain), move_to_target(), composition(gain)]:
            registry.register(skill)
    context = campaign._context(0, '0'*64, campaign.initial_memory, None)
    prepared = campaign.build_loop(root/'preparation', context)
    campaign.state.initialize(parent, policy=prepared.policy, scope=prepared.config['scope'])
    record = ResearchRecord(root/'research')
    evidence = {str(root/'config.json'): file_digest(root/'config.json')}
    if not (record.root/'current.json').exists():
        workflow.create_record(record, question='Does changing PredictIntercept improve interception?',
            hypothesis='Velocity extrapolation improves paired success.',
            falsifier='No paired success improvement.', alternatives=['The current predictor is sufficient.'],
            budget={'max_trials':7,'trial_seconds':30}, state=campaign.state, memory=campaign.memory,
            memory_revision=campaign.initial_memory, context=campaign.context, evidence=evidence)
    snapshot = record.read()
    if snapshot['body']['status'] == 'preparing':
        protocol = prepared.evaluator.suite.protocol
        snapshot = workflow.bind_experiment(record, snapshot['revision'], candidate=parent,
            evaluator_sha256=prepared.evaluator.revision, protocol=protocol,
            workspace=root/'campaign', used_lessons=[], evidence=evidence)
    if snapshot['body']['status'] == 'ready':
        snapshot = record.claim_dispatch(snapshot['revision'], job_id='local-skill-campaign')
    if snapshot['body']['status'] == 'awaiting_result':
        # Core resumes verified stages; uncertain external calls remain blocked.
        result = campaign.run(parent)
        test = campaign.report_test(parent, require_completed=True)
        freeze_json(root/'campaign-result.json', result)
        freeze_json(root/'test-result.json', test)
        report_path = root/'research-result.json'
        freeze_json(report_path, dict(schema='physicalrsi.research-result/v1',
            research_revision=snapshot['revision'], experiment_sha256=digest(snapshot['body']['experiment']),
            job_id=snapshot['body']['job_id'], outcome='completed', evidence={
                str(root/'campaign-result.json'):file_digest(root/'campaign-result.json'),
                str(root/'test-result.json'):file_digest(root/'test-result.json')}))
        snapshot = record.record_result(snapshot['revision'], report=report_path)
    if snapshot['body']['status'] == 'awaiting_decision':
        result = read_json(root/'campaign-result.json')
        improved = result['current']['decision']['decision'] == 'inherit_child'
        snapshot = record.close(snapshot['revision'], disposition='supported' if improved else 'rejected',
            explanation='Core paired evaluation selected the candidate.' if improved else 'No paired improvement.',
            evidence={str(root/'campaign-result.json'):file_digest(root/'campaign-result.json')},
            memory_revision=result['memory_revision'], memory_explanation='Retain memory; parameter experiment only.')
    record.status()
    result = campaign.run(parent)  # Audit completed campaign on restart, without new trials.
    bundle = ReportBundle(root/'report')
    for path in sorted((root/'campaign').rglob('trajectory.json')):
        bundle.add_file(path, str(path.relative_to(root)))
    for name in ['campaign-result.json','test-result.json','research-result.json','config.json']:
        bundle.add_file(root/name, name)
    bundle.add_file(record.root/'history'/(snapshot['revision']+'.json'), 'research-record.json')
    bundle.write_text('changes.diff', ''.join(difflib.unified_diff(
        (json.dumps(dict(gain=incumbent_gain), sort_keys=True)+'\n').splitlines(True),
        (json.dumps(dict(gain=candidate_gain), sort_keys=True)+'\n').splitlines(True),
        fromfile='parent/PredictIntercept.json', tofile='candidate/PredictIntercept.json')))
    bundle.write_text('run.log', json.dumps(dict(completed_rounds=result['completed_rounds'],
        decision=result['current']['decision'], quota=quota.status()), indent=2)+'\n')
    bundle.write_json('composition.json', composition(candidate_gain).composition.as_dict())
    bundle.finalize(title='Interception skill contract experiment',
        summary='Software fixture only. Only PredictIntercept gain can evolve. Videos unsupported. '
                'No RoboDyna benchmark or held-out-task generalization claim.')
    return dict(root=str(root), research_revision=snapshot['revision'],
                composition_revision=workflow.composition_revision, report=str(root/'report/report.md'),
                quota=quota.status(), qualification=None)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
