import pytest

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from test_core_versions import seed
from test_improvement_campaign import Proposal, Selector, Suite, manifest


@pytest.mark.parametrize("overlap", ["same_split", "other_task_development"])
def test_task_local_cases_share_payloads_without_cross_task_collisions(tmp_path, overlap):
    from PhysicalRSI_core.infra.storage import atomic_json
    initial = seed(tmp_path / "seed")
    atomic_json(tmp_path / "seed/control.json", dict(increment=2))
    initial = manifest(tmp_path / "seed", initial["id"])
    case = lambda target: dict(initial=0, target=target)
    splits = dict(development={"left": [case(3)], "right": [case(3 if overlap == "same_split" else 5)]},
                  validation={"left": [case(5), case(7)],
                              "right": [case(5), case(7)] if overlap == "same_split" else [case(3), case(9)]},
                  test={"left": [case(23)], "right": [case(23)]})
    protocol = CaseProtocol(tmp_path / "cases", splits=splits, identity_keys=("initial", "target"))
    suite, state = Suite(), HarnessState(tmp_path / "state")
    quota = TrialQuota(tmp_path / "quota", max_trials=10)
    evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, protocol),
        budgets={task: Budget(10, 30) for task in ("left", "right")}, scope="software",
        system2=Revision("case-identity-test", "1"), quota=quota)
    loop = SelfHarness(tmp_path / "round", state=state, proposer=Proposal(evaluator), evaluator=evaluator,
        selector=Selector(), scope="software", max_candidates=1,
        profile=dict(tasks={task: dict(weight=1, episodes=2, score_range=[0, 1], maximum_regression=0)
                            for task in ("left", "right")}, minimum_gain=0, tie_tolerance=1e-10),
        protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"))
    state.initialize(initial, policy=loop.policy, scope="software")
    assert loop.run()["harness"]["id"] == "child"
    cohort = read_json(loop.root / "validation.json")["output"]
    assert set(cohort["layouts"]["left"]).isdisjoint(cohort["layouts"]["right"])
    assert len(suite.resets) == quota.status()["reserved_trials"] == 10
    assert loop.run() == state.resolve() and len(suite.resets) == 10
    with pytest.raises(ValueError, match="Repeated case"):
        evaluator._cases({"left": [case(5), case(5)], "right": [case(5)]})


def test_task_namespace_does_not_allow_within_task_split_aliases(tmp_path):
    splits = dict(development={"left": [dict(seed=1)]},
                  validation={"left": [dict(seed=1, annotation="different JSON")]},
                  test={"left": [dict(seed=3)]})
    with pytest.raises(ValueError, match="Reset identity reused"):
        CaseProtocol(tmp_path, splits=splits)
