"""CPU orchestration contracts; these are not RoboDyna benchmark results."""
from pathlib import Path
import pytest
from examples.robodyna import native_study as study
from examples.robodyna.study import Suite, Policy, composition
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness


class ContractPolicy(Policy):
    def __init__(self, candidate):
        self.revision = verify_harness(candidate)
        self.skill = composition(read_json(Path(candidate['root'])/'dynamic_interception.json')['gain'])


class ContractSuite(Suite):
    def identity(self):
        return {'kind': 'CPU-native-study-orchestration-fixture'}
    def environment(self, task):
        # Existing fixture uses position/velocity/horizon in reset case; map
        # preregistered seed deterministically without native benchmark claims.
        from physicalrsi.environments import RobodynaFixture
        class Environment(RobodynaFixture):
            def reset(self, case, context):
                return super().reset(dict(position=float(case['seed']), velocity=1., horizon=1.), context)
        return Environment()
    def policy(self, candidate, task):
        return ContractPolicy(candidate)
    def admit(self, candidate):
        verify_harness(candidate)
        return {'accepted': True, 'evidence': {'qualification': 'CPU contract only'}}


def arguments():
    return dict(suite=ContractSuite(), parent_parameters={'gain': 0.},
                candidate_parameters={'gain': 1.}, baseline_parameters={'gain': 0.},
                development=(11,), validation=(21,22), heldout=(31,32), max_steps=1, seconds=10)


def test_seed_splits_reject_leakage():
    with pytest.raises(ValueError, match='distinct'):
        study.validate_seeds([1], [2], [1])


def test_frozen_configuration_rejects_mutation(tmp_path):
    study.setup(tmp_path, **arguments())
    args = arguments()
    args['candidate_parameters'] = {'gain': .5}
    with pytest.raises(ValueError, match='Frozen'):
        study.setup(tmp_path, **args)


def test_real_core_campaign_resume_and_transfer_contract(tmp_path):
    result = study.run(tmp_path, **arguments())
    assert result['same_A1_A2'] is False
    from PhysicalRSI_Autoresearch.records import ResearchRecord
    record = ResearchRecord(tmp_path/'research').read()
    assert record['body']['status'] == 'completed'
    assert (tmp_path/'research/composition.json').exists()
    assert result['scope']['scoring'] == 'CPU-contract-only'
    assert result['groups']['A2']['metrics'][study.VALLEY]['successes'] == 2
    assert result['groups']['A1']['metrics'][study.RAMP]['successes'] == 0
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('receipt.json')}
    assert study.run(tmp_path, **arguments()) == result
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('receipt.json')}
    plan = read_json(tmp_path/'campaign/rounds/round-0001/execution/cases/protocol.json')
    assert set(plan['splits']['development']) == {study.RAMP}


def test_wilson_and_missing_native_measurements():
    summary = study.summarize({'metrics': {'task': {'successes': 0, 'planned': 2}},
                              'episodes': [{'task':'task', 'measurements':{}}]})
    assert summary['task']['native_progress'] == {'observed': 0, 'mean': None}
    low, high = summary['task']['success_interval_95_wilson']
    assert low == 0
    assert 0 < high < 1


def test_no_failure_retains_parent_and_reuses_final_trials(tmp_path):
    args = arguments()
    args['parent_parameters'] = {'gain': 1.}
    args['candidate_parameters'] = {'gain': .5}
    result = study.run(tmp_path, **args)
    assert result['same_A1_A2'] is True
    assert result['groups']['A2'] == result['groups']['A1']
    assert not (tmp_path/'final/A2').exists()
    assert (tmp_path/'report.md').exists()


def test_resume_after_selection_before_final_dispatch(tmp_path, monkeypatch):
    original = study.ExperimentEvaluator.report_test
    def interrupted(*args, **kwargs):
        raise RuntimeError('controller interrupted before final dispatch')
    monkeypatch.setattr(study.ExperimentEvaluator, 'report_test', interrupted)
    with pytest.raises(RuntimeError, match='controller interrupted'):
        study.run(tmp_path, **arguments())
    from PhysicalRSI_Autoresearch.records import ResearchRecord
    assert ResearchRecord(tmp_path/'research').read()['body']['status'] == 'awaiting_result'
    before = {str(p): p.read_bytes() for p in (tmp_path/'campaign').rglob('receipt.json')}
    assert before
    monkeypatch.setattr(study.ExperimentEvaluator, 'report_test', original)
    result = study.run(tmp_path, **arguments())
    assert result['same_A1_A2'] is False
    assert before == {str(p): p.read_bytes() for p in (tmp_path/'campaign').rglob('receipt.json')}


def observed_acceleration(factor=.04, timestamps=(0., .1, .25, .4, .6, .8)):
    tangent = [0., .8, -.6]
    acceleration = factor*9.81*.6
    samples = []
    for t in timestamps:
        distance = .2*t + .5*acceleration*t*t
        samples.append(dict(timestamp_s=t, position_m=[distance*x for x in tangent], phase='rolling'))
    observation = dict(phase='rolling', samples=samples,
        geometry=dict(path_m=[[0.,0.,0.], [0.,8.,-6.]], gravity_m_s2=9.81))
    return [{'trajectory': [{'observation': {'policy_observation': observation}}]}]


def test_acceleration_inference_nonuniform_observation_times_and_deduplication():
    evaluations = observed_acceleration(.037)
    evaluations[0]['trajectory'] *= 2
    inferred = study.estimate_rolling_energy(evaluations)
    assert inferred['samples'] == 4
    assert inferred['rolling_energy_factor'] == pytest.approx(.037)


def test_acceleration_inference_abstains_for_no_evidence_and_wrong_phase():
    assert study.estimate_rolling_energy([])['rolling_energy_factor'] is None
    evaluations = observed_acceleration()
    evaluations[0]['trajectory'][0]['observation']['policy_observation']['phase'] = 'flight'
    assert study.estimate_rolling_energy(evaluations)['samples'] == 0
    assert study.estimate_rolling_energy(observed_acceleration(timestamps=(0., .1, .2)))['rolling_energy_factor'] is None


def test_automatic_proposal_without_rolling_evidence_retains_parent(tmp_path):
    args = arguments()
    args['candidate_parameters'] = None
    result = study.run(tmp_path, **args)
    assert result['same_A1_A2'] is True
    analysis = read_json(tmp_path/'campaign/rounds/round-0001/execution/development-analysis.json')
    assert analysis['disposition'] == 'insufficient-observed-acceleration'
