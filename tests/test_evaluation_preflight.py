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


@pytest.mark.parametrize('before,after', [(True, 1), (1, True), (1, 1.0), (False, 0)])
def test_frozen_artifact_rejects_changed_json_type_without_overwriting(tmp_path, before, after):
    from PhysicalRSI_core.self_harness.evaluation import freeze_json
    path = tmp_path / 'frozen.json'
    freeze_json(path, {'settings': {'value': before}})
    original = path.read_bytes()
    with pytest.raises(ValueError, match='Frozen evaluation artifact changed'):
        freeze_json(path, {'settings': {'value': after}})
    assert path.read_bytes() == original


def test_frozen_artifact_accepts_equivalent_json_sequences_and_key_order(tmp_path):
    from PhysicalRSI_core.self_harness.evaluation import freeze_json
    path = tmp_path / 'frozen.json'
    freeze_json(path, {'values': (1, True), 'name': 'same'})
    original = path.read_bytes()
    freeze_json(path, {'name': 'same', 'values': [1, True]})
    freeze_json(path, {'values': (1, True), 'name': 'same'})
    assert path.read_bytes() == original


@pytest.mark.parametrize('before,after', [(1, True), (True, 1), (1, 1.0), (False, 0)])
def test_evaluator_rejects_suite_identity_type_drift_before_sampling(tmp_path, monkeypatch, before, after):
    from PhysicalRSI_core.embodiment import Revision
    from PhysicalRSI_core.experiments import Budget
    from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
    _, parent, suite, quota = setup(tmp_path)
    settings = {'mode': before}
    identity = suite.identity()
    monkeypatch.setattr(suite, 'identity', lambda: dict(identity, settings=settings))
    evaluator = ExperimentEvaluator(suite=suite, budgets={'counter': Budget(8, 30)},
        scope='software', system2=Revision('typed-suite', '1'), quota=quota)
    settings['mode'] = after
    assert evaluator.identity() == evaluator._identity
    output = tmp_path / 'typed-evaluation'
    with monkeypatch.context() as patch:
        patch.setattr(suite, 'development_cases', lambda parent: pytest.fail('Changed suite must not sample'))
        with pytest.raises(ValueError, match='evaluator identity changed'):
            evaluator.development(parent, output)
    assert not output.exists()
    assert suite.resets == [] and quota.status()['reserved_trials'] == 0
    settings['mode'] = before
    result = evaluator.development(parent, output)
    assert len(result['episodes']) == 1
    assert len(suite.resets) == quota.status()['reserved_trials'] == 1
    assert evaluator.development(parent, output) == result
    assert len(suite.resets) == 1


@pytest.mark.parametrize('entry', ['reopen', 'run'])
@pytest.mark.parametrize('weight', [True, 1.0])
def test_round_rejects_typed_config_drift_without_overwriting(tmp_path, entry, weight):
    from PhysicalRSI_core.infra.storage import atomic_json, read_json
    campaign, parent, suite, quota = setup(tmp_path)
    directory = tmp_path / 'loop'
    loop = campaign.build_loop(directory)
    loop.state.initialize(parent, policy=loop.policy, scope='software')
    path = directory / 'config.json'
    original = read_json(path)
    changed = deepcopy(original)
    changed['profile']['tasks']['counter']['weight'] = weight
    assert changed == original
    atomic_json(path, changed)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Frozen experiment configuration changed'):
        campaign.build_loop(directory) if entry == 'reopen' else loop.run()
    assert path.read_bytes() == before
    assert suite.resets == [] and quota.status()['reserved_trials'] == 0
    atomic_json(path, original)
    restored = campaign.build_loop(directory)
    result = restored.run()
    count = len(suite.resets)
    assert count > 0 and restored.run() == result
    assert len(suite.resets) == count
