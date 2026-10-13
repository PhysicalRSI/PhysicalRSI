"""Reject native adapter evidence substitution using software-only doubles."""
import sys

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.trials import Trial
from PhysicalRSI_demos.auto_engineering import trials
from PhysicalRSI_demos.auto_engineering.harness import create_seed
from PhysicalRSI_demos.auto_engineering.task import TASK, SCOPE
from test_isaac_trials import ContractWorker


@pytest.mark.parametrize("corruption", ["success", "plan", "initial_observation", "steps", "runtime_version", "scenario"])
def test_native_evidence_must_match_independent_measurements_and_dispatched_plan(tmp_path, monkeypatch, corruption):
    class AlteredWorker(ContractWorker):
        def execute(self, action, context):
            if corruption == 'scenario':self.case=dict(seed=self.case['seed'])
            result, rollout = super().execute(action, context)
            if corruption == "success":
                result["success"] = not result["success"]
            elif corruption == "plan":
                value = read_json(self.output / "plan.json")
                value["transit_height_m"] = .5
                atomic_json(self.output / "plan.json", value)
                result["evidence"]["plan.json"] = file_digest(self.output / "plan.json")
            elif corruption == "initial_observation":
                rollout[0] = dict(observation=dict(joint_positions=[10]*9))
            elif corruption == "runtime_version":
                result["runtime_version"] = "6.0.0"
            elif corruption == 'steps':
                result["physics_steps"] += 1
            atomic_json(self.output / "result.json", result)
            return result, rollout

    monkeypatch.setattr(trials, "NativeWorker", AlteredWorker)
    suite = trials.Suite(sys.executable, tmp_path / "native")
    candidate = create_seed(tmp_path / "seed")
    runtime = ExperimentRuntime(tmp_path / "experiments", device_registry=DeviceRegistry(tmp_path / "devices"))
    case=dict(seed=17)
    if corruption=='scenario':case['scenario']='tall-obstacle'
    trial = Trial("corrupt", task=TASK, case=case, scope=SCOPE, budget=Budget(1, 30))
    def run():
        return runtime.run_trial(trial, environment=suite.environment(TASK),
                                 policy=suite.policy(candidate, TASK), verifier=suite.verifier(TASK))
    with pytest.raises(ValueError, match="disagrees|differs|incomplete"):
        run()
    with pytest.raises(ReconciliationRequired):
        run()


def test_native_capture_uses_the_verified_manifest_after_worker_exit(tmp_path):
    from PhysicalRSI_core.infra.trial_artifacts import capture
    worker = trials.NativeWorker(sys.executable, tmp_path / "worker")
    worker.output = tmp_path / "episode"
    atomic_json(worker.output / "sensor.json", {"reading": 1})
    expected = {"evidence": {"sensor.json": file_digest(worker.output / "sensor.json")}}
    atomic_json(worker.output / "result.json", expected)
    atomic_json(worker.output / "sensor.json", {"reading": 2})
    # A new file digest must not silently replace the previously verified one.
    with pytest.raises(ValueError, match="changed during capture"):
        capture(tmp_path / "trial", worker.exports(expected))
    atomic_json(worker.output / "result.json", dict(expected, changed=True))
    with pytest.raises(ValueError, match="changed before evidence capture"):
        worker.exports(expected)
