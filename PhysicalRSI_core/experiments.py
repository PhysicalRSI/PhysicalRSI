"""Versioned reset → execute → verify experiments, independent of any policy.

Adapters own physical resets and outcome measurement. Deadlines are cooperative:
an adapter must honor Context.check() and bound its own device/network calls.
Device ownership and trial receipts survive interruption without automatic retry.
"""
import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol

from .contracts import Context, ReconciliationRequired
from .embodiment import Embodiment, ExecutionBinding, System1, System2Trial
from .infra import devices, storage
from .infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json


def _json(value):
    """Freeze JSON values, including nested adapter-owned mutable objects."""
    return json.loads(storage.canonical(value))


@dataclass(frozen=True)
class Budget:
    max_steps: int = 100
    seconds: float = 60

    def __post_init__(self):
        import math
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if isinstance(self.seconds, bool) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be finite and positive")


class Environment(Protocol):
    def identity(self) -> dict: ...
    def reset(self, case: dict, context: Context) -> dict:
        """Return {ready: bool, observation: JSON value}; verify reset readiness."""
        ...
    def step(self, action, context: Context) -> dict:
        """Return {observation: JSON value, terminated: bool}."""
        ...


class Policy(Protocol):
    def identity(self) -> dict: ...
    def act(self, observation, context: Context): ...


class EmbodiedEnvironment(Environment, Protocol):
    def describe(self) -> Embodiment:
        """Side-effect-free declaration; include calibration in the revision."""
        ...
    def quiesce(self, context: Context) -> dict:
        """For leased devices, confirm {quiescent: True} before normal release.

        Bound all device calls. On an interrupted call the adapter must use its
        own stop/watchdog mechanism; core retains the device claim for recovery.
        """
        ...


class System1Policy(Policy, Protocol):
    def describe(self) -> System1: ...
    def begin_episode(self, task: str, case: dict, observation, context: Context):
        """Initialize history/action chunks from the frozen public task inputs."""
        ...

    def end_episode(self, context: Context) -> dict:
        """Optional owned-worker cleanup: return {stopped: True} after teardown.

        Cleanup also runs on failure and must have its own bounded allowance
        when the episode context is already expired. It never stops the device.
        """
        ...


class Verifier(Protocol):
    def identity(self) -> dict: ...
    def verify(self, case: dict, trace: list[dict]) -> dict:
        """Return outcome (success/failure/uncertain), reason and measurements."""
        ...


def _binding(environment, policy):
    declarations = [getattr(adapter, "describe", None) for adapter in (environment, policy)]
    if all(declaration is None for declaration in declarations):
        return None  # Legacy software adapters have no declared embodiment.
    if not all(callable(declaration) for declaration in declarations):
        raise ValueError("Both environment and System 1 must declare their interfaces")
    binding = ExecutionBinding(*(declaration() for declaration in declarations))
    if not callable(getattr(policy, "begin_episode", None)):
        raise ValueError("Declared System 1 requires begin_episode")
    if binding.embodiment.resources and not callable(getattr(environment, "quiesce", None)):
        raise ValueError("Leased environments require quiesce")
    return binding


class ExperimentRuntime:
    """Freeze a trial, own its devices, execute System 1, preserve the outcome.

    System 2 may supply candidate/comparison attribution but cannot change the
    executing revision. Completed runs read verified evidence without device
    calls. Uncertain runs require reconciliation, even after device recovery.
    """
    def __init__(self, root, *, device_registry=None):
        self.root = Path(root).resolve()
        self.device_registry = device_registry

    def read(self, run_id):
        folder = self.root / identifier(run_id)
        receipt = read_json(folder / "receipt.json")
        if receipt["state"] != "completed":
            raise ReconciliationRequired(f"Inspect interrupted experiment: {folder}")
        specification = read_json(folder / "experiment.json")
        if specification["schema"] not in {"physicalrsi.experiment/v1", "physicalrsi.experiment/v2"}:
            raise ValueError("Unsupported experiment schema")
        version = specification["schema"].rsplit("/", 1)[1]
        if receipt.get("schema") != "physicalrsi.experiment-receipt/" + version:
            raise ValueError("Receipt schema differs from the experiment")
        if digest(specification) != receipt["experiment_sha256"]:
            raise ValueError("Experiment specification changed")
        required = {"experiment.json", "reset.json", "trajectory.json", "verdict.json"}
        binding = specification.get("binding")
        if binding and binding["embodiment"]["resources"]:
            required.update({"lease.json", "quiescence.json"})
        if specification.get("policy_lifecycle") == "episode":
            required.add("policy.json")
        elif specification.get("policy_lifecycle") is not None:
            raise ValueError("Unknown policy lifecycle")
        if set(receipt["evidence"]) != required:
            raise ValueError("Experiment evidence manifest is incomplete or unexpected")
        for name, expected in receipt["evidence"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or file_digest(path) != expected:
                raise ValueError("Experiment evidence changed: " + name)
        if (receipt["id"] != run_id or receipt["scope"] != specification["scope"]
                or receipt["qualification"] is not None):
            raise ValueError("Receipt identity or qualification differs from the experiment")
        if version == "v2" and any(
            receipt.get(key) != specification.get(key) for key in ("binding", "system2")
        ):
            raise ValueError("Receipt execution binding changed")
        verdict = read_json(folder / "verdict.json")
        if receipt["verdict"] != verdict or receipt["outcome"] != verdict["outcome"]:
            raise ValueError("Receipt disagrees with the recorded verdict")
        trace = read_json(folder / "trajectory.json")
        if receipt["steps"] != max(0, len(trace) - 1):
            raise ValueError("Receipt step count differs from the trajectory")
        if version == "v2":
            initial = read_json(folder / "reset.json")
            if initial["ready"] is False:
                termination = "reset_not_ready"
            elif initial.get("terminated", False) or (len(trace) > 1 and trace[-1]["terminated"]):
                termination = "environment"
            else:
                termination = "step_budget"
                if receipt["steps"] != specification["budget"]["max_steps"]:
                    raise ValueError("Step budget termination lacks complete execution evidence")
            if receipt.get("termination") != termination:
                raise ValueError("Receipt termination differs from the recorded observations")
        if "lease.json" in required:
            lease = read_json(folder / "lease.json")
            if (lease["resources"] != binding["embodiment"]["resources"]
                    or lease["registry"] != specification["device_registry"]
                    or read_json(folder / "quiescence.json").get("quiescent") is not True):
                raise ValueError("Device evidence differs from the execution binding")
        if "policy.json" in required and read_json(folder / "policy.json").get("stopped") is not True:
            raise ValueError("Policy termination lacks confirmed cleanup evidence")
        return receipt

    def run(self, run_id, *, task, case, scope, environment, policy, verifier,
            budget=None, system2: System2Trial | None = None):
        budget = budget or Budget()
        folder = self.root / identifier(run_id)
        if not scope or not task:
            raise ValueError("Declare the task and evidence scope")
        binding = _binding(environment, policy)
        finish_policy = getattr(policy, "end_episode", None)
        if finish_policy is not None and not callable(finish_policy):
            raise ValueError("Optional policy end_episode must be callable")
        if system2 is not None:
            if not isinstance(system2, System2Trial) or binding is None:
                raise ValueError("System 2 attribution requires a declared System 1")
            system2.bind(binding.system1)
        resources = binding.embodiment.resources if binding else ()
        if resources and self.device_registry is None:
            raise ValueError("Configure one shared device registry for all device clients")

        def identities():
            return _json(dict(environment=environment.identity(), policy=policy.identity(),
                              verifier=verifier.identity(), runtime=file_digest(Path(__file__)),
                              contracts=file_digest(Path(__file__).with_name("contracts.py")),
                              embodiment=file_digest(Path(__file__).with_name("embodiment.py")),
                              timing=file_digest(Path(__file__).with_name("timing.py")),
                              devices=file_digest(Path(devices.__file__)),
                              storage=file_digest(Path(storage.__file__))))

        specification = _json(dict(
            schema="physicalrsi.experiment/v2", task=task, case=case,
            scope=scope, budget=asdict(budget), adapters=identities(),
            binding=asdict(binding) if binding else None,
            policy_lifecycle="episode" if finish_policy is not None else None,
            system2=asdict(system2) if system2 else None,
            device_registry=self.device_registry.identity() if resources else None))

        def stable():
            current = _binding(environment, policy)
            if (identities() != specification["adapters"]
                    or callable(getattr(policy, "end_episode", None)) != bool(specification["policy_lifecycle"])
                    or _json(asdict(current) if current else None) != specification["binding"]):
                raise ValueError("An adapter changed during the experiment")

        frozen = digest(specification)
        with locked(folder / ".lock"):
            if (folder / "receipt.json").exists():
                if read_json(folder / "receipt.json")["experiment_sha256"] != frozen:
                    raise ValueError("Run ID belongs to a different experiment")
                return self.read(run_id)
            lease = (self.device_registry.lease(resources, owner=str(folder))
                     if resources else nullcontext(None))
            # Admission/occupancy failures occur before any trial or reset starts.
            with lease as claim:
                return self._execute(run_id, specification, frozen, binding, claim,
                                     environment, policy, verifier, budget, stable)

    def _execute(self, run_id, specification, frozen, binding, claim,
                 environment, policy, verifier, budget, stable):
        folder = self.root / run_id
        atomic_json(folder / "experiment.json", specification)
        evidence_names = {"experiment.json", "reset.json", "trajectory.json", "verdict.json"}
        if claim:
            atomic_json(folder / "lease.json", claim)
            evidence_names.update({"lease.json", "quiescence.json"})
        receipt = dict(schema="physicalrsi.experiment-receipt/v2", id=run_id,
                       state="started", experiment_sha256=frozen, scope=specification["scope"],
                       qualification=None, binding=specification["binding"],
                       system2=specification["system2"])
        atomic_json(folder / "receipt.json", receipt)
        start = monotonic()
        context = Context(run_id, deadline=start + budget.seconds,
                          harness_revision=binding.system1.revision if binding else "")
        case = specification["case"]
        trace = []
        policy_started, policy_stop_attempted = False, False
        if specification["policy_lifecycle"]:
            evidence_names.add("policy.json")

        def stop_policy():
            nonlocal policy_stop_attempted
            if not specification["policy_lifecycle"] or policy_stop_attempted:
                return
            policy_stop_attempted = True
            try:
                report = (_json(policy.end_episode(context)) if policy_started
                          else dict(stopped=True, started=False))
                if report.get("stopped") is not True:
                    raise ReconciliationRequired("Policy worker termination is unconfirmed")
                atomic_json(folder / "policy.json", report)
            except BaseException as error:
                atomic_json(folder / "policy.json", dict(stopped=False, error_type=type(error).__name__))
                raise

        try:
            stable()
            context.check()
            initial = _json(environment.reset(deepcopy(case), context))
            atomic_json(folder / "reset.json", initial)
            context.check()
            if type(initial.get("ready")) is not bool:
                raise ValueError("Reset must declare readiness")
            if not initial["ready"]:
                termination = "reset_not_ready"
                verdict = dict(outcome="invalid", reason=termination, measurements={})
            else:
                trace.append(dict(observation=initial["observation"]))
                atomic_json(folder / "trajectory.json", trace)
                if type(initial.get("terminated", False)) is not bool:
                    raise ValueError("Reset termination must be a boolean")
                if binding:
                    stable()
                    policy_started = True
                    policy.begin_episode(specification["task"], deepcopy(case),
                                         deepcopy(initial["observation"]), context)
                    context.check()
                termination = "environment" if initial.get("terminated", False) else "step_budget"
                for step in range(0 if termination == "environment" else budget.max_steps):
                    stable()
                    context.check()
                    policy_started = True
                    action = _json(policy.act(deepcopy(trace[-1]["observation"]), context))
                    context.check()
                    stable()
                    # Persist the intended effect before entering the environment.
                    pending = dict(step=step, action=action, state="dispatched")
                    atomic_json(folder / "pending-action.json", pending)
                    result = _json(environment.step(deepcopy(action), context))
                    if (type(result.get("terminated")) is not bool or "observation" not in result
                            or {"step", "action"}.intersection(result)):
                        raise ValueError("Environment must return observation and termination without action metadata")
                    trace.append(dict(result, step=step, action=action))
                    # Retain returned observations even if the call exceeded its deadline.
                    atomic_json(folder / "trajectory.json", trace)
                    atomic_json(folder / "pending-action.json", dict(pending, state="observed"))
                    context.check()
                    if result["terminated"]:
                        termination = "environment"
                        break
                stop_policy()
                stable()
                verdict = _json(verifier.verify(deepcopy(case), deepcopy(trace)))
                context.check()
                if verdict.get("outcome") not in {"success", "failure", "uncertain"}:
                    raise ValueError("Verifier must return success, failure or uncertain")
                if not verdict.get("reason") or not isinstance(verdict.get("measurements"), dict):
                    raise ValueError("Verifier must supply a reason and measured evidence")
            stop_policy()
            if claim:
                context.check()
                quiet = _json(environment.quiesce(context))
                atomic_json(folder / "quiescence.json", quiet)
                if quiet.get("quiescent") is not True:
                    raise ReconciliationRequired("Environment did not confirm device quiescence")
                context.check()
            stable()
            atomic_json(folder / "trajectory.json", trace)
            atomic_json(folder / "verdict.json", verdict)
            receipt.update(state="completed", outcome=verdict["outcome"], verdict=verdict,
                           steps=max(0, len(trace) - 1), elapsed_seconds=monotonic() - start,
                           termination=termination,
                           evidence={name: file_digest(folder / name) for name in sorted(evidence_names)})
            atomic_json(folder / "receipt.json", receipt)
            return self.read(run_id)
        except BaseException as error:
            try:
                stop_policy()
            except BaseException as cleanup_error:
                receipt["policy_cleanup_error"] = type(cleanup_error).__name__
            receipt.update(state="needs_reconciliation", error=type(error).__name__ + ": " + str(error))
            if getattr(error, "request_id", None):
                receipt["controller_request_id"] = error.request_id
            if isinstance(getattr(error, "receipt", None), dict):
                receipt["controller_rejection"] = _json(error.receipt)
            atomic_json(folder / "receipt.json", receipt)
            raise
