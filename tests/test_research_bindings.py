from copy import deepcopy

import pytest

from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_memory import MemoryBoundSuite
from test_research_memory import fixture, gate_for
from test_research_protocol import splits


def test_fresh_validation_preserves_policy_but_changed_final_test_does_not(tmp_path):
    cases = splits()
    initial = CaseProtocol(tmp_path / "first", splits=cases, identity_keys=("initial", "target"))
    changed = deepcopy(cases)
    changed["validation"]["counter"] = [dict(initial=0, target=13), dict(initial=0, target=15)]
    fresh = CaseProtocol(tmp_path / "fresh", splits=changed, identity_keys=("initial", "target"))
    assert fresh.policy_identity() == initial.policy_identity()
    changed["test"]["counter"] = [dict(initial=0, target=17)]
    other_test = CaseProtocol(tmp_path / "other", splits=changed, identity_keys=("initial", "target"))
    assert other_test.policy_identity() != initial.policy_identity()


@pytest.mark.parametrize("field", ["memory_revision", "scope", "parent_revision"])
def test_memory_gate_must_match_the_proposer_round_context(tmp_path, field):
    data = fixture(tmp_path)
    parent, _, suite, memory, revision, *_ = data
    plan = CaseProtocol(tmp_path / "cases", splits=splits(), identity_keys=("initial", "target"))
    wrapped = MemoryBoundSuite(PreregisteredSuite(suite, plan), gate_for(tmp_path, data))
    state = HarnessState(tmp_path / "binding-state")
    anchor = state.initialize(parent, policy={}, scope="software")
    context = dict(parent_revision=anchor["revision"], memory_revision=revision, scope={"task": "counter"})
    wrapped.register_round(state, "round", context=context)
    changed = deepcopy(context)
    if field == "scope":
        changed[field] = {"task": "other"}
    elif field == "memory_revision":
        lessons = memory.read(revision)["lessons"]
        lessons["recovery"]["hypothesis"] = "A revised research hypothesis"
        changed[field] = memory.write(lessons, parent=revision)
    else:
        # A valid lineage revision for a different executable parent.
        other_state = HarnessState(tmp_path / "other-state")
        changed[field] = other_state.initialize(data[1], policy={}, scope="software")["revision"]
    with pytest.raises((ValueError, FileNotFoundError)):
        wrapped.register_round(state, "other-round", context=changed)
