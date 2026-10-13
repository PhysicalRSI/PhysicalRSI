"""A selection must survive canonical JSON storage and candidate-result reordering."""
from copy import deepcopy
from itertools import permutations
import json

import pytest

from PhysicalRSI_core.infra.storage import digest,file_digest
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.selection import select_survivor


@pytest.mark.parametrize('policy',['task_regression','success_constraint','cost_constraint','weighted_score'])
def test_selection_is_independent_of_task_and_measurement_dictionary_order(tmp_path,policy):
    evidence=tmp_path/'observation.json';evidence.write_text('{}')
    names=['z-task','b-task','a-task']
    tasks={t:dict(weight=(1. if t=='z-task' else 1e-16) if policy=='weighted_score' else 1.,
        episodes=1,score_range=[0,1],maximum_regression=0,
        **(dict(maximum_success_regression=0) if policy=='success_constraint' else {}),
        **(dict(cost_limits={'z-cost':1.,'a-cost':1.}) if policy=='cost_constraint' else {})) for t in names}
    comparison=dict(schema_version=1,round_id='comparison',frozen_at=now(),parent_id='parent',
        candidates={'parent':'1'*64,'child':'2'*64},profile=dict(tasks=tasks,minimum_gain=0,tie_tolerance=1e-10))
    cohort=dict(comparison_sha256=digest(comparison),generated_at=now(),layouts={t:[digest(t)] for t in tasks})
    results=[]
    for name in ('parent','child'):
        episodes=[]
        for t in tasks:
            score=(0. if name=='parent' else {'z-task':.9,'b-task':.2,'a-task':.1}[t]) if policy=='weighted_score' else (
                .9 if name=='parent' else .1 if policy=='task_regression' else .95)
            episodes.append(dict(task=t,layout_sha256=digest(t),state='completed',score=score,
                success=not (policy=='success_constraint' and name=='child'),
                measurements={'z-cost':2 if name=='child' else 0,'a-cost':2 if name=='child' else 0},
                evidence_sha256={'observation.json':file_digest(evidence)}))
        results.append(dict(candidate_id=name,kind='policy_evaluation',native_exit_code=0,
            comparison_sha256=digest(comparison),cohort_sha256=digest(cohort),freeze_sha256=comparison['candidates'][name],
            evaluator_revision='test',episodes=episodes))
    expected=select_survivor(comparison,cohort,results,evidence_root=tmp_path)
    stored=json.loads(json.dumps(comparison,sort_keys=True))
    assert select_survivor(stored,cohort,results[::-1],evidence_root=tmp_path)==expected
    for order in permutations(names):
        variant=deepcopy(comparison)
        variant['profile']['tasks']={t:deepcopy(tasks[t]) for t in order}
        for spec in variant['profile']['tasks'].values():
            if 'cost_limits' in spec:spec['cost_limits']=dict(reversed(list(spec['cost_limits'].items())))
        assert select_survivor(variant,cohort,results,evidence_root=tmp_path)==expected


def paired_comparison(tmp_path, parent_scores, child_scores, *, costs=None, cost_limit=None):
    evidence = tmp_path / 'observation.json'
    evidence.write_text('{}')
    cases = [digest(str(i)) for i in range(len(parent_scores))]
    task = dict(weight=1., episodes=len(cases), score_range=[0, 1], maximum_regression=0)
    if cost_limit is not None:
        task['cost_limits'] = {'energy': cost_limit}
    comparison = dict(schema_version=1, round_id='comparison', frozen_at=now(), parent_id='parent',
        candidates={'parent':'1'*64, 'child':'2'*64},
        profile=dict(tasks={'task':task}, minimum_gain=0, tie_tolerance=0))
    cohort = dict(comparison_sha256=digest(comparison), generated_at=now(), layouts={'task':cases})
    results = []
    for name, scores in [('parent', parent_scores), ('child', child_scores)]:
        episodes = [dict(task='task', layout_sha256=case, state='completed', score=score, success=True,
            measurements={'energy':costs[i]} if costs is not None else {},
            evidence_sha256={'observation.json':file_digest(evidence)})
            for i, (case, score) in enumerate(zip(cases, scores))]
        results.append(dict(candidate_id=name, kind='policy_evaluation', native_exit_code=0,
            comparison_sha256=digest(comparison), cohort_sha256=digest(cohort),
            freeze_sha256=comparison['candidates'][name], evaluator_revision='test', episodes=episodes))
    return comparison, cohort, results


def test_equal_paired_scores_cannot_promote_by_reordering_episodes(tmp_path):
    comparison, cohort, results = paired_comparison(tmp_path, [1., 1e-16, 1e-16], [1., 1e-16, 1e-16])
    expected = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    for order in permutations(range(3)):
        reordered = deepcopy(results)
        reordered[1]['episodes'] = [results[1]['episodes'][i] for i in order]
        selected = select_survivor(comparison, cohort, reordered, evidence_root=tmp_path)
        assert selected['decision'] == 'retain_parent'
        assert selected['metrics'] == expected['metrics']
        assert selected['rejected'] == expected['rejected']


@pytest.mark.parametrize('cost', [.1, 1e308])
def test_constant_cost_at_limit_is_not_rejected_by_aggregation(tmp_path, cost):
    comparison, cohort, results = paired_comparison(tmp_path, [0.]*3, [1.]*3,
                                                   costs=[cost]*3, cost_limit=cost)
    selected = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    assert selected['metrics']['child']['mean_cost']['task']['energy'] == cost
    assert selected['decision'] == 'inherit_child'
