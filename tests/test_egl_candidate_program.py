"""Candidate binding checks, not simulator or benchmark qualification."""
from copy import deepcopy
from pathlib import Path

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.candidate_program import (
    COMBINATIONS, MEMORY, PRIMITIVES, bind_candidate, edit_contract,
)
from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.proposals import ProposalLimits


def fixture_candidate(root):
    values = {
        PRIMITIVES: {"schema": "physicalrsi.egl-primitives/v1", "source": "raise RuntimeError('must not execute on host')\n"},
        COMBINATIONS: {"schema": "physicalrsi.egl-combinations/v1", "source": "def policy(robot, memory):\n    return memory\n"},
        MEMORY: {"schema": "physicalrsi.egl-memory/v1", "memory": {"lessons": ["one"]}},
    }
    components = {}
    for kind in CLOSURE:
        name = kind + ".json"
        atomic_json(root / name, {"kind": kind, "scope": "binding-test-fixture"})
        components[kind] = {name: file_digest(root / name)}
    for name, value in values.items():
        atomic_json(root / name, value)
        kind = edit_contract().public()[name]["component"]
        components[kind][name] = file_digest(root / name)
    return {"id": "test-parent", "root": str(root), "components": components}


def test_binding_is_data_only_and_memory_is_copied(tmp_path):
    candidate = fixture_candidate(tmp_path)
    bound = bind_candidate(candidate)  # The primitive source raises if executed.
    assert bound.harness_sha256 == verify_harness(candidate)
    memory = bound.memory
    memory["lessons"].append("transient episode update")
    assert bound.memory == {"lessons": ["one"]}
    assert set(bound.identity()["artifacts"]) == {PRIMITIVES, COMBINATIONS, MEMORY}


def test_changed_artifact_and_wrong_component_are_rejected(tmp_path):
    candidate = fixture_candidate(tmp_path)
    wrong = deepcopy(candidate)
    wrong["components"]["assets"][MEMORY] = wrong["components"]["memory_rules"].pop(MEMORY)
    with pytest.raises(ValueError, match="exclusively"):
        bind_candidate(wrong)
    (tmp_path / MEMORY).write_text('{}')
    with pytest.raises(ValueError, match="Changed or missing"):
        bind_candidate(candidate)


def test_all_three_artifacts_have_separate_parent_bound_edits(tmp_path):
    candidate = fixture_candidate(tmp_path)
    contract = edit_contract()
    inputs = contract.inputs(candidate, ProposalLimits())
    edits = []
    for name, entry in inputs.items():
        value = deepcopy(entry['value'])
        if name == MEMORY:
            value['memory']['lessons'].append('revised')
        else:
            value['source'] += '\n# revision\n'
        edits.append(dict(path=name, before_sha256=entry['sha256'], value=value))
    proposal = dict(rationale='Exercise all declared edit types', edits=edits)
    assert set(contract.replacements(proposal, inputs)) == {PRIMITIVES, COMBINATIONS, MEMORY}
    proposal['edits'][0]['before_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='different file revision'):
        contract.replacements(proposal, inputs)
    proposal['edits'][0]['path'] = 'foundation.json'
    with pytest.raises(ValueError, match='undeclared'):
        contract.replacements(proposal, inputs)


@pytest.mark.parametrize('source', ['def incomplete(', 'def other(robot, memory):\n    return 1\n'])
def test_invalid_combinations_fail_before_execution(tmp_path, source):
    candidate = fixture_candidate(tmp_path)
    atomic_json(tmp_path / COMBINATIONS, dict(schema='physicalrsi.egl-combinations/v1', source=source))
    candidate['components']['skill_selection'][COMBINATIONS] = file_digest(tmp_path / COMBINATIONS)
    with pytest.raises((ValueError, SyntaxError)):
        bind_candidate(candidate)
