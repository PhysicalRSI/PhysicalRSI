from copy import deepcopy

import pytest

from PhysicalRSI_core.self_harness import SelfHarness, now
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from test_improvement_campaign import setup


def profile():
    return dict(tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1],
                                      maximum_regression=0)}, minimum_gain=0, tie_tolerance=1e-10)


@pytest.mark.parametrize("field,value", [
    ("weight", 0), ("weight", True), ("episodes", False),
    ("score_range", [1, 1]), ("score_range", [2, 1]),
    ("maximum_regression", 2), ("minimum_gain", -1), ("tie_tolerance", 1),
])
def test_invalid_profile_fails_before_port_calls_or_workspace_creation(tmp_path, field, value):
    invalid = profile()
    target = invalid if field in {"minimum_gain", "tie_tolerance"} else invalid["tasks"]["counter"]
    target[field] = value

    class Uncalled:
        def identity(self):
            pytest.fail("Invalid profile must be rejected before calling a port")

    root = tmp_path / "round"
    with pytest.raises(ValueError):
        SelfHarness(root, state=None, proposer=Uncalled(), evaluator=Uncalled(), selector=Uncalled(),
                    profile=invalid, protocol={}, scope="software")
    assert not root.exists()


@pytest.mark.parametrize("change", ["range", "pool", "task", "scope"])
@pytest.mark.parametrize("entry", ["validation", "resume_validation", "evaluate", "resume_evaluation"])
def test_direct_evaluator_rejects_bad_comparison_before_sampling_or_reserving(tmp_path, change, entry):
    campaign, parent, suite, quota = setup(tmp_path)
    loop = campaign.build_loop(tmp_path / "loop")
    comparison = dict(schema_version=1, round_id="round-one", parent_id=parent["id"], frozen_at=now(),
                      candidates={parent["id"]: verify_harness(parent), "child": "0" * 64},
                      profile=profile(), scope="software")
    if change == "range":
        comparison["profile"]["tasks"]["counter"]["score_range"] = [1, 1]
    elif change == "pool":
        del comparison["candidates"]["child"]
    elif change == "task":
        comparison["profile"]["tasks"]["other"] = comparison["profile"]["tasks"].pop("counter")
    else:
        comparison["scope"] = "different"
    original = deepcopy(comparison)
    suite.validation_cases = lambda comparison: pytest.fail("Invalid comparison must not sample cases")
    output = tmp_path / "evaluation"
    method = getattr(loop.evaluator, entry)
    with pytest.raises(ValueError):
        if entry in {"evaluate", "resume_evaluation"}:
            method(parent, comparison, {}, output)
        else:
            method(comparison, output)
    assert comparison == original
    assert not output.exists() and suite.resets == []
    assert quota.status()["reserved_trials"] == 0
