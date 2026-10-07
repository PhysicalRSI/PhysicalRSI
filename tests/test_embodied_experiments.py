from dataclasses import replace

import pytest

from PhysicalRSI_core.contracts import Cancelled, ReconciliationRequired
from PhysicalRSI_core.embodiment import Dependency, Embodiment, Revision, System2Trial
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.devices import DeviceBusy, DeviceRegistry
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_demos.experiment import ACTION, OBSERVATION, CounterEnvironment, CounterPolicy, CounterVerifier


class LeasedCounter(CounterEnvironment):
    resets = 0
    quiet_calls = 0

    def describe(self):
        # A declared software resource tests ownership without claiming hardware.
        return replace(super().describe(), resources=("counter:one",))

    def reset(self, case, context):
        self.resets += 1
        return super().reset(case, context)

    def quiesce(self, context):
        self.quiet_calls += 1
        return {"quiescent": True, "reason": "Synchronous CPU controller"}


def run(runtime, run_id="trial", **changes):
    options = dict(task="counter", case={"initial": 0, "target": 3}, scope="software",
                   environment=LeasedCounter(), policy=CounterPolicy(),
                   verifier=CounterVerifier(), budget=Budget(max_steps=5, seconds=10))
    options.update(changes)
    return runtime.run(run_id, **options)


@pytest.fixture
def runtime(tmp_path):
    return ExperimentRuntime(tmp_path / "trials", device_registry=DeviceRegistry(tmp_path / "devices"))


@pytest.mark.parametrize("change", [dict(unit="meter"), dict(frame="different"),
                                    dict(embodiment="other-robot"), dict(name="other-action")])
def test_contract_mismatch_rejected_before_reset(runtime, change):
    class WrongPolicy(CounterPolicy):
        def describe(self):
            return replace(super().describe(), action=replace(ACTION, **change))
    env = LeasedCounter()
    with pytest.raises(ValueError, match="action contract"):
        run(runtime, environment=env, policy=WrongPolicy())
    assert env.resets == 0
    assert runtime.device_registry.occupied() == {}
    assert not (runtime.root / "trial/receipt.json").exists()


def test_physical_declaration_requires_explicit_devices_and_registry(tmp_path):
    with pytest.raises(ValueError, match="named device"):
        Embodiment("arm", "calibration-1", "physical", OBSERVATION, ACTION)
    env = LeasedCounter()
    with pytest.raises(ValueError, match="shared device registry"):
        run(ExperimentRuntime(tmp_path), environment=env)
    assert env.resets == 0


def test_receipt_binds_system1_system2_and_device_evidence(runtime):
    policy = CounterPolicy()
    origin = System2Trial(Revision("scripted-search", "1"), "candidate", policy.describe().revision,
                          "validation", digest("comparison"), digest("cohort"))
    env = LeasedCounter()
    receipt = run(runtime, policy=policy, environment=env, system2=origin)
    assert receipt["binding"]["system1"]["revision"] == origin.candidate_sha256
    assert receipt["system2"]["comparison_sha256"] == origin.comparison_sha256
    assert receipt["qualification"] is None
    assert {"lease.json", "quiescence.json"}.issubset(receipt["evidence"])
    claim = read_json(runtime.root / "trial/lease.json")
    assert runtime.device_registry.inspect(claim["token"])["state"] == "released"
    assert env.quiet_calls == 1
    # Reading completed work needs no new device lease, even during another trial.
    with runtime.device_registry.lease(["counter:one"], owner="another-trial"):
        assert run(runtime, policy=policy, environment=env, system2=origin) == receipt
    assert env.resets == 1 and env.quiet_calls == 1
    with pytest.raises(ValueError, match="different experiment"):
        run(runtime, system2=replace(origin, system2=Revision("other-search", "1")))


@pytest.mark.parametrize("failure", ["raise", "unconfirmed"])
def test_policy_cleanup_failure_keeps_experiment_and_device_unresolved(runtime, failure):
    class OwnedPolicy(CounterPolicy):
        stops = 0

        def end_episode(self, context):
            self.stops += 1
            if failure == "raise":
                raise RuntimeError("Owned worker did not stop")
            return dict(stopped=False)

    policy, env = OwnedPolicy(), LeasedCounter()
    with pytest.raises((RuntimeError, ReconciliationRequired)):
        run(runtime, policy=policy, environment=env)
    assert policy.stops == 1 and env.quiet_calls == 0
    assert runtime.device_registry.occupied()
    assert read_json(runtime.root / "trial/receipt.json")["state"] == "needs_reconciliation"
    assert read_json(runtime.root / "trial/policy.json")["stopped"] is False


def test_unready_reset_does_not_start_or_close_an_unallocated_policy_worker(runtime):
    class NotReady(LeasedCounter):
        def reset(self, case, context):
            return dict(ready=False)

    class NotStarted(CounterPolicy):
        def begin_episode(self, *args):
            pytest.fail("Unready reset must not start the policy")

        def end_episode(self, context):
            pytest.fail("No policy was allocated")

    receipt = run(runtime, policy=NotStarted(), environment=NotReady())
    assert receipt["outcome"] == "invalid"
    assert read_json(runtime.root / "trial/policy.json") == dict(stopped=True, started=False)
    assert not runtime.device_registry.occupied()


def test_system2_cannot_misattribute_the_executing_candidate(runtime):
    wrong = System2Trial(Revision("search", "1"), "other-candidate", digest("other"), "development")
    env = LeasedCounter()
    with pytest.raises(ValueError, match="candidate differs"):
        run(runtime, environment=env, system2=wrong)
    assert env.resets == 0
    with pytest.raises(ValueError, match="frozen comparison and cohort"):
        System2Trial(Revision("search", "1"), "candidate", digest("candidate"), "validation")


def test_stateful_system1_starts_each_episode_with_frozen_inputs(runtime):
    class StatefulPolicy(CounterPolicy):
        starts = 0

        def begin_episode(self, task, case, observation, context):
            self.starts += 1
            assert context.harness_revision == self.describe().revision
            self.remaining = case["target"] - observation["position"]
            case["target"] = -1  # Cannot edit the verifier's case or the specification.

        def act(self, observation, context):
            self.remaining -= 1
            return 1 if self.remaining >= 0 else 0

    policy = StatefulPolicy()
    for name in ("first", "second"):
        assert run(runtime, name, policy=policy)["outcome"] == "success"
    assert policy.starts == 2
    assert read_json(runtime.root / "second/experiment.json")["case"]["target"] == 3


def test_memory_revision_is_immutable_during_action_dispatch(runtime):
    class MutablePolicy(CounterPolicy):
        memory = digest("before")

        def describe(self):
            return replace(super().describe(), dependencies=(Dependency("history", self.memory, "memory"),))

        def act(self, observation, context):
            self.memory = digest("after")
            return 1

    env = LeasedCounter()
    with pytest.raises(ValueError, match="adapter changed"):
        run(runtime, environment=env, policy=MutablePolicy())
    assert env.position == 0  # The changed action never reached the environment.
    assert runtime.device_registry.occupied()
    assert read_json(runtime.root / "trial/receipt.json")["state"] == "needs_reconciliation"
    with pytest.raises(ValueError, match="SHA256"):
        Dependency("history", "mutable-latest", "memory")


def test_interrupted_trial_blocks_other_workspaces_and_new_ids(runtime, tmp_path):
    class Interrupted(LeasedCounter):
        def step(self, action, context):
            super().step(action, context)
            raise RuntimeError("Connection lost after action")

    env = Interrupted()
    with pytest.raises(RuntimeError, match="Connection lost"):
        run(runtime, environment=env)
    other = ExperimentRuntime(tmp_path / "other", device_registry=DeviceRegistry(runtime.device_registry.root))
    with pytest.raises(ReconciliationRequired):
        run(other, "new-id")
    assert not (other.root / "new-id/receipt.json").exists()
    claim = read_json(runtime.root / "trial/lease.json")
    runtime.device_registry.reconcile(claim["token"], operator="test-controller",
                                      reason="CPU fixture returned; no external effects",
                                      evidence={"mock_position": env.position})
    with pytest.raises(ReconciliationRequired):
        run(runtime, environment=env)  # Recovery does not retry the interrupted trial.
    assert run(other, "new-id")["outcome"] == "success"


def test_busy_device_does_not_start_a_trial(runtime):
    env = LeasedCounter()
    with runtime.device_registry.lease(["counter:one"], owner="other-client"):
        with pytest.raises(DeviceBusy):
            run(runtime, environment=env)
    assert env.resets == 0
    assert not (runtime.root / "trial/receipt.json").exists()


def test_unconfirmed_quiescence_cannot_release_or_complete(runtime):
    class StillRunning(LeasedCounter):
        def quiesce(self, context):
            return {"quiescent": False, "reason": "Controller did not confirm stop"}
    with pytest.raises(ReconciliationRequired, match="quiescence"):
        run(runtime, environment=StillRunning())
    assert runtime.device_registry.occupied()
    with pytest.raises(ReconciliationRequired):
        runtime.read("trial")


def test_returned_observation_survives_cancellation(runtime):
    class CancelAfterEffect(LeasedCounter):
        def step(self, action, context):
            result = super().step(action, context)
            context.cancelled.set()
            return result
    with pytest.raises(Cancelled):
        run(runtime, environment=CancelAfterEffect())
    trace = read_json(runtime.root / "trial/trajectory.json")
    assert trace[-1]["observation"]["position"] == 1
    assert runtime.device_registry.occupied()


@pytest.mark.parametrize("tamper", ["manifest", "scope", "qualification", "binding", "termination"])
def test_completed_receipt_cannot_drop_evidence_or_change_claims(runtime, tamper):
    run(runtime)
    path = runtime.root / "trial/receipt.json"
    receipt = read_json(path)
    if tamper == "manifest":
        del receipt["evidence"]["trajectory.json"]
        (runtime.root / "trial/trajectory.json").write_text("[]")
    elif tamper == "scope":
        receipt["scope"] = "physical"
    elif tamper == "qualification":
        receipt["qualification"] = "robot-qualified"
    elif tamper == "termination":
        receipt["termination"] = "step_budget"
    else:
        receipt["binding"]["system1"]["revision"] = digest("unexecuted-candidate")
    atomic_json(path, receipt)
    with pytest.raises(ValueError):
        runtime.read("trial")


def test_initial_termination_and_step_budget_are_separate(runtime):
    receipt = run(runtime, "already-done", case={"initial": 3, "target": 3})
    assert receipt["steps"] == 0 and receipt["termination"] == "environment"
    assert receipt["outcome"] == "success"
    receipt = run(runtime, "limited", budget=Budget(max_steps=1, seconds=10))
    assert receipt["steps"] == 1 and receipt["termination"] == "step_budget"
    assert receipt["outcome"] == "failure"


def test_legacy_ports_still_execute_without_claiming_an_embodiment(runtime):
    class LegacyEnvironment(CounterEnvironment):
        describe = None
    class LegacyPolicy(CounterPolicy):
        describe = None
    receipt = run(runtime, environment=LegacyEnvironment(), policy=LegacyPolicy())
    assert receipt["binding"] is None and receipt["qualification"] is None
    assert receipt["outcome"] == "success"


def test_historical_v1_receipt_is_read_without_reexecution(tmp_path):
    folder = tmp_path / "historical"
    specification = dict(schema="physicalrsi.experiment/v1", task="counter", scope="software",
                         case={"initial": 0, "target": 1}, budget={"max_steps": 1, "seconds": 5},
                         adapters={"environment": {"revision": "1"}, "policy": {"increment": 1},
                                   "verifier": {"revision": "1"}, "runtime": "historical-runtime"})
    verdict = dict(outcome="success", reason="Measured position", measurements={"position": 1})
    files = {"experiment.json": specification, "reset.json": {"ready": True, "observation": 0},
             "trajectory.json": [{"observation": 0}, {"step": 0, "action": 1,
                                                      "observation": 1, "terminated": True}],
             "verdict.json": verdict}
    for name, value in files.items():
        atomic_json(folder / name, value)
    receipt = dict(schema="physicalrsi.experiment-receipt/v1", id="historical", state="completed",
                   scope="software", qualification=None, experiment_sha256=digest(specification),
                   outcome="success", verdict=verdict, steps=1, elapsed_seconds=0.1,
                   evidence={name: file_digest(folder / name) for name in files})
    atomic_json(folder / "receipt.json", receipt)
    assert ExperimentRuntime(tmp_path).read("historical") == receipt
