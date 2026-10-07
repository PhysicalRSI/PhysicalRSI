import pytest

from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.experiments import ExperimentRuntime
from PhysicalRSI_core.infra.storage import digest
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.experiments import experiment_result
from PhysicalRSI_core.self_harness.selection import select_survivor
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


@pytest.fixture
def comparison_run(tmp_path):
    policies = {"parent": CounterPolicy(2), "candidate": CounterPolicy(1)}
    comparison = dict(schema_version=1, round_id="round", parent_id="parent", frozen_at=now(),
                      candidates={name: policy.describe().revision for name, policy in policies.items()},
                      scope="software", protocol_sha256=digest("binary-counter-v1"),
                      evaluation_kind="local_contract_evaluation", profile=dict(
                          tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1], maximum_regression=0)},
                          minimum_gain=0, tie_tolerance=1e-10))
    cases = [{"initial": 0, "target": target} for target in (3, 5)]
    cohort = dict(split="validation", comparison_sha256=digest(comparison), generated_at=now(),
                  layouts={"counter": [digest(case) for case in cases]})
    runtime = ExperimentRuntime(tmp_path / "trials")

    def evaluate(name, verifier=None, comparison_override=None, **changes):
        origin = System2Trial(Revision("scripted-search", "1"), name, policies[name].describe().revision,
                              "validation", digest(comparison), digest(cohort))
        run_ids = []
        for index, case in enumerate(cases):
            run_id = f"{name}-{index}"
            runtime.run(run_id, task="counter", case=case, scope="software",
                        environment=CounterEnvironment(), policy=policies[name],
                        verifier=verifier or CounterVerifier(), system2=origin)
            run_ids.append(run_id)
        options = dict(candidate_id=name, comparison=comparison_override or comparison,
                       cohort=cohort, evaluator_revision="binary-counter-v1", evidence_root=tmp_path)
        options.update(changes)
        return experiment_result(runtime, run_ids, **options)

    return runtime, comparison, cohort, evaluate


def test_paired_experiments_feed_existing_self_harness_selection(comparison_run, tmp_path):
    runtime, comparison, cohort, evaluate = comparison_run
    results = [evaluate(name) for name in ("parent", "candidate")]
    decision = select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    assert decision["survivor_id"] == "candidate"
    assert decision["metrics"]["candidate"]["normalized_score"] == 1
    assert decision["metrics"]["parent"]["normalized_score"] == 0
    assert any(name.endswith("receipt.json") for name in results[1]["episodes"][0]["evidence_sha256"])
    (runtime.root / "candidate-0/trajectory.json").write_text("[]")
    with pytest.raises(ValueError, match="evidence"):
        select_survivor(comparison, cohort, results, evidence_root=tmp_path)


@pytest.mark.parametrize("outcome", ["uncertain", "invalid"])
def test_unscored_outcomes_never_become_failed_episodes(comparison_run, outcome):
    runtime, _, _, evaluate = comparison_run

    class Uncertain(CounterVerifier):
        def verify(self, case, trace):
            return dict(outcome="uncertain", reason="Sensor unavailable", measurements={})

    if outcome == "invalid":
        # Invalid reset has a distinct verdict produced by the runtime.
        original = runtime.run

        class Unready(CounterEnvironment):
            def reset(self, case, context):
                return {"ready": False}

        def unready(*args, **kwargs):
            kwargs["environment"] = Unready()
            return original(*args, **kwargs)

        runtime.run = unready
    with pytest.raises(ValueError, match="no definite task outcome"):
        evaluate("candidate", verifier=Uncertain())


def test_candidate_identity_and_case_coverage_are_required(comparison_run, tmp_path):
    runtime, comparison, cohort, evaluate = comparison_run
    evaluate("parent")
    options = dict(comparison=comparison, cohort=cohort,
                   evaluator_revision="binary-counter-v1", evidence_root=tmp_path)
    with pytest.raises(ValueError, match="candidate/comparison/cohort"):
        experiment_result(runtime, ["parent-0", "parent-1"], candidate_id="candidate", **options)
    with pytest.raises(ValueError, match="Incomplete experiment cohort"):
        experiment_result(runtime, ["parent-0"], candidate_id="parent", **options)
    with pytest.raises(ValueError, match="unique"):
        experiment_result(runtime, ["parent-0", "parent-0"], candidate_id="parent", **options)


def test_score_mapping_is_explicit_and_bounded(comparison_run):
    _, _, _, evaluate = comparison_run
    with pytest.raises(ValueError, match="finite scalar"):
        evaluate("candidate", score_measurement="missing_score")
    with pytest.raises(ValueError, match="declared native range"):
        evaluate("candidate", score_measurement="position")


def test_candidate_gain_cannot_hide_a_changed_verifier(comparison_run, tmp_path):
    _, comparison, cohort, evaluate = comparison_run

    class DifferentVerifier(CounterVerifier):
        def identity(self):
            return {"name": "different-grading-protocol", "revision": "1"}

    results = [evaluate("parent"), evaluate("candidate", verifier=DifferentVerifier())]
    with pytest.raises(ValueError, match="protocols differ"):
        select_survivor(comparison, cohort, results, evidence_root=tmp_path)
    del results[1]["experiment_protocol_sha256"]
    with pytest.raises(ValueError, match="SHA256"):
        select_survivor(comparison, cohort, results, evidence_root=tmp_path)


def test_malformed_cohort_cannot_hide_missing_trials(comparison_run, tmp_path):
    runtime, comparison, cohort, evaluate = comparison_run
    evaluate("parent")
    options = dict(candidate_id="parent", comparison=comparison,
                   evaluator_revision="binary-counter-v1", evidence_root=tmp_path)
    cohort["layouts"]["counter"] = [cohort["layouts"]["counter"][0]]
    with pytest.raises(ValueError, match="cohort coverage"):
        experiment_result(runtime, ["parent-0"], cohort=cohort, **options)
    cohort["layouts"]["counter"] *= 2
    with pytest.raises(ValueError, match="repeated cases"):
        experiment_result(runtime, ["parent-0"], cohort=cohort, **options)
