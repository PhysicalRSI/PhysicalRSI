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
