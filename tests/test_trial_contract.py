from copy import deepcopy

import pytest

from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.trials import Evaluation, Trial
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


def test_trial_is_frozen_and_evaluation_exposes_actual_rollout(tmp_path):
    case = dict(initial=0, target=3)
    trial = Trial("attempt", task="counter", case=case, scope="software", budget=Budget(5, 30))
    frozen = trial.to_dict()
    case["target"] = 99
    value = trial.to_dict()
    value["case"]["target"] = 97
    assert trial.to_dict() == frozen
    runtime = ExperimentRuntime(tmp_path)
    result = runtime.evaluate(trial, environment=CounterEnvironment(), policy=CounterPolicy(2), verifier=CounterVerifier())
    assert isinstance(result, Evaluation) and result.outcome == "failure"
    payload = result.to_dict()
    assert payload["trial"] == frozen
    assert payload["trial_sha256"] == trial.revision
    assert payload["trajectory"][1]["action"] == 2
    assert payload["trajectory"][-1]["observation"]["position"] == 4
    assert payload["verdict"]["measurements"]["position"] == 4
    payload["trajectory"].clear()
    assert result == Evaluation.read(runtime, "attempt")


@pytest.mark.parametrize("outcome", ["success", "failure", "uncertain", "invalid"])
def test_unified_evaluation_preserves_native_outcomes(tmp_path, outcome):
    class Environment(CounterEnvironment):
        def reset(self, case, context):
            return dict(ready=False) if outcome == "invalid" else super().reset(case, context)

    class Verifier(CounterVerifier):
        def verify(self, case, trace):
            return dict(outcome=outcome, reason="measured outcome", measurements={})

    result = ExperimentRuntime(tmp_path).evaluate(Trial("attempt", task="counter", case=dict(initial=0, target=1),
        scope="software"), environment=Environment(), policy=CounterPolicy(), verifier=Verifier())
    assert result.outcome == outcome
    assert result.to_dict()["qualification"] is None
    assert (result.to_dict()["trajectory"] == []) == (outcome == "invalid")


def test_evaluation_rejects_changed_raw_trace(tmp_path):
    runtime = ExperimentRuntime(tmp_path)
    runtime.evaluate(Trial("attempt", task="counter", case=dict(initial=0, target=1), scope="software"),
                     environment=CounterEnvironment(), policy=CounterPolicy(), verifier=CounterVerifier())
    (tmp_path / "attempt" / "trajectory.json").write_text("[]")
    with pytest.raises(ValueError, match="evidence changed"):
        Evaluation.read(runtime, "attempt")
