"""Ablation attribution and stale-evidence checks; no simulator qualification."""
from copy import deepcopy
import json

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.factorial_proposals import FactorialEdits
from PhysicalRSI_baselines.embodied_goodharts_law.candidate_program import edit_contract, MEMORY
from PhysicalRSI_core.infra.storage import file_digest
from PhysicalRSI_core.self_harness.proposals import ProposalLimits
from test_egl_candidate_program import fixture_candidate


def setup_hypothesis(tmp_path):
    candidate = fixture_candidate(tmp_path / 'parent')
    inputs = edit_contract().inputs(candidate, ProposalLimits())
    replacements = {name: deepcopy(row['value']) for name, row in inputs.items()}
    for name, value in replacements.items():
        if name == MEMORY:
            value['memory']['lessons'].append('new observation')
        else:
            value['source'] += '\n# reviewed hypothesis\n'
    evidence = tmp_path / 'observation.json'
    evidence.write_text('{"outcome":"failure"}')
    strategy = FactorialEdits(inputs=inputs, replacements=replacements,
                             evidence={str(evidence): file_digest(evidence)},
                             rationale='Test independent contributions')
    return strategy, dict(editable=inputs, parent_sha256='parent'), evidence


def test_complete_factorial_edits_pass_existing_contract(tmp_path):
    strategy, request, _ = setup_hypothesis(tmp_path)
    reply = json.loads(strategy.generate(request, ProposalLimits(max_candidates=7))['text'])
    changed = [frozenset(edit_contract().replacements(p, request['editable']))
               for p in reply['candidates']]
    assert len(set(changed)) == 7
    assert sorted(map(len, changed)) == [1, 1, 1, 2, 2, 2, 3]
    assert reply['parent_sha256'] == 'parent'


def test_no_silent_subset_when_budget_too_small(tmp_path):
    strategy, request, _ = setup_hypothesis(tmp_path)
    with pytest.raises(ValueError, match='every component ablation'):
        strategy.generate(request, ProposalLimits(max_candidates=3))


def test_reject_stale_parent_and_changed_evidence(tmp_path):
    strategy, request, evidence = setup_hypothesis(tmp_path)
    stale = deepcopy(request)
    stale['editable'][MEMORY]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='different parent'):
        strategy.generate(stale, ProposalLimits(max_candidates=7))
    evidence.write_text('{"outcome":"success"}')
    with pytest.raises(ValueError, match='evidence changed'):
        strategy.generate(request, ProposalLimits(max_candidates=7))
