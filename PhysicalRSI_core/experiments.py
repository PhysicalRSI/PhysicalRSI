"""Versioned reset → execute → verify experiments, independent of any policy.

Adapters own physical resets and outcome measurement. Deadlines are cooperative:
an adapter must honor Context.check() and bound its own device/network calls.
Device ownership and trial receipts survive interruption without automatic retry.
"""
import json
import secrets
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol

from .contracts import Context, ReconciliationRequired
from .embodiment import Embodiment, ExecutionBinding, System1, System2Trial
from .infra import devices, evidence, storage, trial_artifacts
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


@dataclass(frozen=True)
class TrialAdapters:
    """One trial's environment, policy and verifier, constructed without effects.

    A suite may implement trial(candidate, task) to bind stateful adapters as one
    bundle. Factories must create independent bundles for concurrent trials.
    Construction does not reset, actuate, start workers or reserve devices.
    """
    environment: Environment
    policy: Policy
    verifier: Verifier

    def __post_init__(self):
        for port, methods in ((self.environment, ("identity", "reset", "step")),
                              (self.policy, ("identity", "act")), (self.verifier, ("identity", "verify"))):
            if any(not callable(getattr(port, name, None)) for name in methods):
                raise ValueError("Trial adapters require complete environment, policy and verifier ports")


def trial_factory_kind(suite):
    factory = getattr(suite, "trial", None)
    if factory is not None and not callable(factory):
        raise ValueError("Optional trial factory must be callable")
    return "bundled" if factory is not None else "legacy"


def trial_adapters(suite, candidate, task):
    """Prefer an indivisible factory; retain existing independent legacy ports."""
    if trial_factory_kind(suite) == "bundled":
        adapters = suite.trial(deepcopy(candidate), task)
        if not isinstance(adapters, TrialAdapters):
            raise ValueError("Trial factory must return TrialAdapters")
        return adapters
    environment = suite.environment(task)
    return TrialAdapters(environment, suite.policy(deepcopy(candidate), task), suite.verifier(task))


def runtime_implementation():
    """Owned execution dependencies, shared by Trial and evaluator identities.

    Read actual source bytes on every identity check. Historical evidence can
    still be audited independently; executing under changed code needs a new
    frozen evaluator and Trial. External libraries remain provider-owned.
    """
    root = Path(__file__).parent
    return {name: file_digest(root / path) for name, path in {
        "runtime": "experiments.py", "contracts": "contracts.py", "trials": "trials.py",
        "trial_artifacts": "infra/trial_artifacts.py", "evidence": "infra/evidence.py",
        "embodiment": "embodiment.py", "timing": "timing.py",
        "devices": "infra/devices.py", "storage": "infra/storage.py",
        "trial_recovery": "infra/trial_recovery.py", "trial_quota": "infra/trial_quota.py",
    }.items()}


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

    def require_unstarted(self, run_id, *, replacement=None):
        """A missing receipt is not proof that an earlier execution never began."""
        folder = self.root / identifier(run_id)
        if (folder / 'recovery').exists() and replacement is None:
            raise ReconciliationRequired('Prepared replacement requires its authorization reference: ' + str(folder))
        if folder.exists() and any(p.name not in {'.lock', 'recovery'} for p in folder.iterdir()):
            raise ReconciliationRequired('Unpublished experiment evidence requires reconciliation: ' + str(folder))

    def run_trial(self, trial, *, environment, policy, verifier):
        """Execute the shared Trial contract; legacy run() uses the same path."""
        from .trials import Trial
        if not isinstance(trial, Trial):
            raise ValueError("Expected a Trial")
        return self.run(**trial.arguments(), environment=environment, policy=policy, verifier=verifier)

    def evaluate(self, trial, *, environment, policy, verifier):
        """Return a verified Evaluation with the actual observations/actions."""
        from .trials import Evaluation
        receipt = self.run_trial(trial, environment=environment, policy=policy, verifier=verifier)
        return Evaluation.read(self, receipt["id"])

    def read(self, run_id):
        return self.read_evidence(run_id)["receipt"]

    def read_evidence(self, run_id):
        """Verify once and return receipt, parsed documents and raw-file hashes.

        No execution or adapter callback occurs during this audit. The reader
        is local to this call; every later read performs a fresh byte audit.
        """
        folder = self.root / identifier(run_id)
        reader = evidence.EvidenceReader(folder)
        receipt = reader.document("receipt.json")
        if "execution_id" in receipt:
            execution_id = receipt["execution_id"]
            if (not isinstance(execution_id, str) or len(execution_id) != 32
                    or any(c not in "0123456789abcdef" for c in execution_id)):
                raise ValueError("Invalid Trial execution identity")
        if receipt["state"] != "completed":
            raise ReconciliationRequired(f"Inspect interrupted experiment: {folder}")
        specification = reader.document("experiment.json", receipt["evidence"].get("experiment.json"))
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
        if specification.get("artifact_contract") == "files/v1":
            artifacts = trial_artifacts.artifact_entries(folder,
                reader.document("artifacts.json", receipt["evidence"].get("artifacts.json")))
            required.update({"artifacts.json", *artifacts})
        elif specification.get("artifact_contract") is not None:
            raise ValueError("Unknown adapter artifact contract")
        if specification.get("policy_evidence_contract") == "json/v1":
            required.add("policy-evidence.json")
        elif specification.get("policy_evidence_contract") is not None:
            raise ValueError("Unknown policy evidence contract")
        recovery = None
        if 'replacement' in specification:
            from .infra.trial_recovery import audit_bundle
            recovery, recovery_files = audit_bundle(reader, run_id, specification)
            required.update(recovery_files)
        if set(receipt["evidence"]) != required:
            raise ValueError("Experiment evidence manifest is incomplete or unexpected")
        if specification.get("artifact_contract") == "files/v1" and any(
                receipt["evidence"][name] != sha for name, sha in artifacts.items()):
            raise ValueError("Trial artifact evidence changed: manifest disagrees with receipt")
        # These are the exact JSON bytes used for semantic checks and exposed
        # to Evaluation. Binary artifacts only need a streaming digest.
        for name in required.intersection({"reset.json", "trajectory.json", "verdict.json",
                                           "lease.json", "quiescence.json", "policy.json"}):
            reader.document(name, receipt["evidence"][name])
        reader.verify(receipt["evidence"])
        if (receipt["id"] != run_id or receipt["scope"] != specification["scope"]
                or receipt["qualification"] is not None):
            raise ValueError("Receipt identity or qualification differs from the experiment")
        if version == "v2" and any(
            receipt.get(key) != specification.get(key) for key in ("binding", "system2")
        ):
            raise ValueError("Receipt execution binding changed")
        verdict = reader.document("verdict.json")
        if receipt["verdict"] != verdict or receipt["outcome"] != verdict["outcome"]:
            raise ValueError("Receipt disagrees with the recorded verdict")
        trace = reader.document("trajectory.json")
        if receipt["steps"] != max(0, len(trace) - 1):
            raise ValueError("Receipt step count differs from the trajectory")
        if version == "v2":
            initial = reader.document("reset.json")
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
            lease = reader.document("lease.json")
            if (lease["resources"] != binding["embodiment"]["resources"]
                    or lease["registry"] != specification["device_registry"]
                    or reader.document("quiescence.json").get("quiescent") is not True):
                raise ValueError("Device evidence differs from the execution binding")
        if "policy.json" in required and reader.document("policy.json").get("stopped") is not True:
            raise ValueError("Policy termination lacks confirmed cleanup evidence")
        reader.unchanged_document("receipt.json")
        result = dict(receipt=receipt, specification=specification,
                    documents={name:reader.document(name) for name in ("reset.json", "trajectory.json", "verdict.json")},
                    evidence={str(reader.path(name)):sha for name,sha in reader.hashes.items()})
        if recovery is not None:
            result['recovery'] = recovery
        return result

    def run(self, run_id, *, task, case, scope, environment, policy, verifier,
            budget=None, system2: System2Trial | None = None, replacement=None):
        budget = budget or Budget()
        folder = self.root / identifier(run_id)
        if not scope or not task:
            raise ValueError("Declare the task and evidence scope")
        binding = _binding(environment, policy)
        finish_policy = getattr(policy, "end_episode", None)
        if finish_policy is not None and not callable(finish_policy):
            raise ValueError("Optional policy end_episode must be callable")
        export_evidence = getattr(environment, "export_evidence", None)
        if export_evidence is not None and not callable(export_evidence):
            raise ValueError("Optional environment export_evidence must be callable")
        policy_evidence = getattr(policy, "execution_evidence", None)
        if policy_evidence is not None and not callable(policy_evidence):
            raise ValueError("Optional policy execution_evidence must be callable")
        if system2 is not None:
            if not isinstance(system2, System2Trial) or binding is None:
                raise ValueError("System 2 attribution requires a declared System 1")
            system2.bind(binding.system1)
        resources = binding.embodiment.resources if binding else ()
        if resources and self.device_registry is None:
            raise ValueError("Configure one shared device registry for all device clients")

        def identities():
            return _json(dict(environment=environment.identity(), policy=policy.identity(),
                              verifier=verifier.identity(), **runtime_implementation()))

        specification = _json(dict(
            schema="physicalrsi.experiment/v2", task=task, case=case,
            scope=scope, budget=asdict(budget), adapters=identities(),
            binding=asdict(binding) if binding else None,
            policy_lifecycle="episode" if finish_policy is not None else None,
            artifact_contract="files/v1" if export_evidence is not None else None,
            system2=asdict(system2) if system2 else None,
            device_registry=self.device_registry.identity() if resources else None))
        if policy_evidence is not None:
            specification["policy_evidence_contract"] = "json/v1"
        if replacement is not None:
            specification['replacement'] = _json(replacement)

        def stable():
            current = _binding(environment, policy)
            if (identities() != specification["adapters"]
                    or callable(getattr(policy, "end_episode", None)) != bool(specification["policy_lifecycle"])
                    or callable(getattr(environment, "export_evidence", None)) != bool(specification["artifact_contract"])
                    or callable(getattr(policy, "execution_evidence", None)) != bool(specification.get("policy_evidence_contract"))
                    or _json(asdict(current) if current else None) != specification["binding"]):
                raise ValueError("An adapter changed during the experiment")

        frozen = digest(specification)
        with locked(folder / ".lock"):
            if (folder / "receipt.json").exists():
                if read_json(folder / "receipt.json")["experiment_sha256"] != frozen:
                    raise ValueError("Run ID belongs to a different experiment")
                return self.read(run_id)
            self.require_unstarted(run_id, replacement=replacement)
            if replacement is not None:
                from .infra.trial_recovery import before_execution
                before_execution(folder, run_id, specification, self.device_registry)
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
        if 'replacement' in specification:
            from .infra.trial_recovery import audit_bundle
            _, recovery_files = audit_bundle(evidence.EvidenceReader(folder), run_id, specification)
            evidence_names.update(recovery_files)
        if claim:
            atomic_json(folder / "lease.json", claim)
            evidence_names.update({"lease.json", "quiescence.json"})
        # A local run ID identifies a plan within one runtime root. Independent
        # executions may otherwise have byte-identical receipts (including time).
        # Mint once before effects; read/resume/archive copies retain this token.
        receipt = dict(schema="physicalrsi.experiment-receipt/v2", id=run_id,
                       execution_id=secrets.token_hex(16),
                       state="started", experiment_sha256=frozen, scope=specification["scope"],
                       qualification=None, binding=specification["binding"],
                       system2=specification["system2"], execution_phase='admitted')
        atomic_json(folder / "receipt.json", receipt)
        start = monotonic()
        phase, phase_start = "preparation", start
        phase_seconds = dict.fromkeys(("preparation", "reset", "rollout", "verification", "finalization", "cleanup"), 0.)

        def enter_phase(name):
            nonlocal phase, phase_start
            current = monotonic()
            phase_seconds[phase] += current - phase_start
            phase, phase_start = name, current

        def record_timing():
            enter_phase(phase)
            receipt["timing"] = dict(schema="physicalrsi.trial-timing/v1",
                                     phase_seconds=dict(phase_seconds))
            receipt["elapsed_seconds"] = phase_start - start

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

        def record_policy_evidence():
            if specification.get("policy_evidence_contract"):
                value = _json(policy.execution_evidence())
                if not isinstance(value, dict) or not value:
                    raise ValueError("Policy execution evidence must be a nonempty JSON object")
                atomic_json(folder / "policy-evidence.json", value)
                evidence_names.add("policy-evidence.json")

        try:
            stable()
            context.check()
            enter_phase('reset')
            receipt['execution_phase'] = 'reset'
            atomic_json(folder / 'receipt.json', receipt)
            initial = _json(environment.reset(deepcopy(case), context))
            atomic_json(folder / "reset.json", initial)
            enter_phase('rollout')
            receipt['execution_phase'] = 'episode'
            atomic_json(folder / 'receipt.json', receipt)
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
                    # Retain initial policy evidence before the first
                    # environment action, including when that action is lost.
                    record_policy_evidence()
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
                enter_phase("verification")
                verdict = _json(verifier.verify(deepcopy(case), deepcopy(trace)))
                context.check()
                if verdict.get("outcome") not in {"success", "failure", "uncertain"}:
                    raise ValueError("Verifier must return success, failure or uncertain")
                if not verdict.get("reason") or not isinstance(verdict.get("measurements"), dict):
                    raise ValueError("Verifier must supply a reason and measured evidence")
            enter_phase("finalization")
            stop_policy()
            if claim:
                context.check()
                quiet = _json(environment.quiesce(context))
                atomic_json(folder / "quiescence.json", quiet)
                if quiet.get("quiescent") is not True:
                    raise ReconciliationRequired("Environment did not confirm device quiescence")
                context.check()
            stable()
            if specification["artifact_contract"]:
                evidence_names.update(trial_artifacts.capture(folder, environment.export_evidence()))
                context.check()
                stable()
            if specification.get("policy_evidence_contract"):
                record_policy_evidence()
                context.check()
                stable()
            atomic_json(folder / "trajectory.json", trace)
            atomic_json(folder / "verdict.json", verdict)
            receipt.update(state="completed", outcome=verdict["outcome"], verdict=verdict,
                           execution_phase='completed',
                           steps=max(0, len(trace) - 1),
                           termination=termination,
                           evidence={name: file_digest(folder / name) for name in sorted(evidence_names)})
            record_timing()
            atomic_json(folder / "receipt.json", receipt)
            return self.read(run_id)
        except BaseException as error:
            enter_phase("cleanup")
            try:
                stop_policy()
            except BaseException as cleanup_error:
                receipt["policy_cleanup_error"] = type(cleanup_error).__name__
            record_timing()
            receipt.update(state="needs_reconciliation", error=type(error).__name__ + ": " + str(error))
            if getattr(error, "request_id", None):
                receipt["controller_request_id"] = error.request_id
            if isinstance(getattr(error, "receipt", None), dict):
                receipt["controller_rejection"] = _json(error.receipt)
            atomic_json(folder / "receipt.json", receipt)
            raise
