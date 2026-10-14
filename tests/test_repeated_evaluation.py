"""Independent executions of one case must not become independent layouts."""
from copy import deepcopy
from pathlib import Path
import shutil
import math

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.experiments import ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, read_json
from PhysicalRSI_core.infra.trial_quota import TrialLimitExceeded
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.experiments import experiment_result
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.rollout import development_evaluations
from PhysicalRSI_core.self_harness.selection import select_survivor
from PhysicalRSI_demos.experiment import CounterVerifier
from test_improvement_campaign import setup, manifest


def prepare(root, *, repetitions=3, max_trials=30, mixed=False, uncertain=False):
    campaign, parent, suite, quota = setup(root, max_trials=max_trials)
    original = campaign.build_loop(root/'unused').evaluator
    if mixed or uncertain:
        class MixedVerifier(CounterVerifier):
            def identity(self):
                return dict(kind='declared-repeat-fixture', mixed=mixed, uncertain=uncertain)

            def verify(self, case, trace):
                result = super().verify(case, trace)
                if len(suite.resets) % 3 == 2:
                    result['outcome'] = 'uncertain' if uncertain else 'failure'
                return result
        suite.verifier = lambda task: MixedVerifier()
    protocol = CaseProtocol(root/'cases', identity_keys=('target',), splits={
        split: {'counter': [dict(initial=0, target=target) for target in targets]}
        for split, targets in [('development', [3]), ('validation', [5, 7]), ('test', [9])]})
    evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, protocol),
        budgets=original.budgets, scope=original.scope, system2=original.system2,
        quota=quota, repetitions=repetitions)
    child_root = root/'child'
    shutil.copytree(parent['root'], child_root)
    atomic_json(child_root/'control.json', dict(increment=1))
    child = manifest(child_root, 'child')
    comparison = dict(schema_version=1, round_id='repeated', parent_id=parent['id'], frozen_at=now(),
        candidates={v['id']: verify_harness(v) for v in [parent, child]}, scope='software',
        evaluation_kind='local_contract_evaluation', profile=dict(
            tasks={'counter': dict(weight=1, episodes=2, score_range=[0, 1], maximum_regression=0,
                                   minimum_success_rate=1, cost_limits={'position': 10})},
            minimum_gain=0, tie_tolerance=1e-10))
    return evaluator, parent, child, comparison, suite, quota


def test_repeated_validation_retains_failures_and_unique_case_denominator(tmp_path):
    evaluator, parent, child, comparison, suite, quota = prepare(tmp_path, mixed=True)
    comparison['profile']['statistics'] = dict(method='paired-hoeffding', alpha=.1, max_comparisons=1)
    output = tmp_path/'evaluation'
    feedback = evaluator.development(parent, output)
    assert len(development_evaluations(parent, feedback, output)) == 3
    cohort = evaluator.validation(comparison, output)
    assert len(cohort['layouts']['counter']) == 2 and cohort['repetitions'] == 3
    assert len(suite.resets) == 3 and quota.status()['reserved_trials'] == 15
    results = [evaluator.evaluate(v, comparison, cohort, output) for v in [parent, child]]
    child_result = results[1]
    assert len(child_result['episodes']) == 2
    for row in child_result['episodes']:
        assert row['score'] == pytest.approx(2/3)
        assert row['success'] is False and row['success_fraction'] == pytest.approx(2/3)
        assert len(row['replicates']) == row['repetitions'] == 3
        assert len({x['trial_sha256'] for x in row['replicates']}) == 3
    decision = select_survivor(comparison, cohort, results, evidence_root=output)
    assert decision['survivor_id'] == parent['id']
    assert decision['metrics']['child']['success_rate']['counter'] == 0
    assert decision['metrics']['child']['mean_cost']['counter']['position'] == 6
    improvement = decision['metrics']['child']['improvement_evidence']
    assert improvement['paired_wins'] == 2
    assert improvement['lower_bound'] == pytest.approx(2/3 - math.sqrt(math.log(10)))
    assert evaluator.evaluate(child, comparison, cohort, output) == child_result
    assert len(suite.resets) == quota.status()['reserved_trials'] == 15
    evidence = next(k for k in child_result['episodes'][0]['evidence_sha256'] if k.endswith('trajectory.json'))
    (output/evidence).write_text('[]')
    with pytest.raises(ValueError, match='evidence'):
        select_survivor(comparison, cohort, results, evidence_root=output)


def test_missing_repeat_cannot_be_hidden_by_complete_layout_coverage(tmp_path):
    evaluator, parent, child, comparison, _, _ = prepare(tmp_path)
    output = tmp_path/'evaluation'
    evaluator.development(parent, output)
    cohort = evaluator.validation(comparison, output)
    evaluator.evaluate(child, comparison, cohort, output)
    folder = output/'experiments/validation/child'
    runs = [row['run_id'] for row in read_json(folder/'plan.json')['trials']]
    with pytest.raises(ValueError, match='Incomplete experiment cohort'):
        experiment_result(ExperimentRuntime(folder/'trials'), runs[1:], candidate_id='child',
            comparison=comparison, cohort=cohort, evaluator_revision=evaluator.revision, evidence_root=output)
    changed = deepcopy(cohort)
    changed['repetitions'] = 2
    with pytest.raises(ValueError, match='cohort changed'):
        evaluator.read_evaluation(child, comparison, changed, output)


def test_test_report_counts_every_repeat_including_uncertain(tmp_path):
    evaluator, _, child, _, suite, quota = prepare(tmp_path, uncertain=True)
    output = tmp_path/'report'
    report = evaluator.report_test(child, selection_sha256='a'*64, output=output)
    metric = report['metrics']['counter']
    assert metric == dict(planned=3, successes=2, failures=0, uncertain=1, invalid=0,
                          success_rate=2/3, cases=1, repetitions=3, case_success_rate=0)
    assert len(report['episodes']) == len(suite.resets) == quota.status()['reserved_trials'] == 3
    assert evaluator.report_test(child, selection_sha256='a'*64, output=output) == report
    assert len(suite.resets) == 3


def test_uncertain_validation_repeat_blocks_before_remaining_effects(tmp_path):
    evaluator, parent, child, comparison, suite, quota = prepare(tmp_path, uncertain=True)
    output = tmp_path/'evaluation'
    evaluator.development(parent, output)
    cohort = evaluator.validation(comparison, output)
    with pytest.raises(ReconciliationRequired, match='no definite task outcome'):
        evaluator.evaluate(child, comparison, cohort, output)
    assert len(suite.resets) == 5 and quota.status()['reserved_trials'] == 15
    with pytest.raises(ReconciliationRequired):
        evaluator.resume_evaluation(child, comparison, cohort, output)
    assert len(suite.resets) == 5


def test_all_repeats_reserved_before_first_effect_and_restart_preserves_completed(tmp_path, monkeypatch):
    evaluator, parent, _, _, suite, quota = prepare(tmp_path, max_trials=2)
    with pytest.raises(TrialLimitExceeded):
        evaluator.development(parent, tmp_path/'too-small')
    assert not suite.resets and quota.status()['reserved_trials'] == 0
    evaluator, parent, _, _, suite, quota = prepare(tmp_path/'resume')
    original = ExperimentRuntime.run_trial
    def interrupted(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError('after durable repeat')
    with monkeypatch.context() as patch:
        patch.setattr(ExperimentRuntime, 'run_trial', interrupted)
        with pytest.raises(RuntimeError, match='durable repeat'):
            evaluator.development(parent, tmp_path/'resume/output')
    assert len(suite.resets) == 1 and quota.status()['reserved_trials'] == 3
    feedback = evaluator.development(parent, tmp_path/'resume/output')
    assert len(feedback['episodes']) == len(suite.resets) == 3
    assert evaluator.development(parent, tmp_path/'resume/output') == feedback
    assert len(suite.resets) == 3


@pytest.mark.parametrize('value', [0, -1, True, 1.0, 101, None])
def test_repeat_count_is_an_explicit_bounded_integer(tmp_path, value):
    with pytest.raises(ValueError, match='repetitions'):
        prepare(tmp_path, repetitions=value)
