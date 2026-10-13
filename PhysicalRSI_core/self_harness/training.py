"""Optional training port: completed jobs produce candidates, never fitness.

Providers own execution, hard resource limits and cancellation. Core freezes
requests, reconciles uncertain submission, verifies owned data and inference
bytes, and accounts for reported costs. It does not import a training framework.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from ..checkpoints import CheckpointStore, validate_budget, validate_model
from ..contracts import ReconciliationRequired
from ..infra.storage import atomic_json, digest, file_digest, locked, read_json, relative_path
from .artifacts import CLOSURE, EvolutionSpec, verify_harness
from .bindings import policy_identity
from .evaluation import freeze_json
from .rollout import development_evaluations


class TrainingPending(ReconciliationRequired):
    """Known pending job; observe this same job before resuming the proposal."""


class TrainingRejected(ValueError):
    """Provider reports a terminal, quiescent training failure."""


@dataclass(frozen=True, init=False)
class CheckpointEvolutionSpec(EvolutionSpec):
    def __init__(self, store, model, *, binding_path="weights.json"):
        validate_model(model)
        relative_path(binding_path)
        object.__setattr__(self, "mutable", ("weights",))
        object.__setattr__(self, "store", deepcopy(store.identity()))
        object.__setattr__(self, "model", deepcopy(model))
        object.__setattr__(self, "binding_path", binding_path)

    def identity(self):
        return dict(schema="physicalrsi.checkpoint-evolution/v1", mutable=["weights"],
                    fixed=sorted(CLOSURE - {"weights"}), store=deepcopy(self.store),
                    model=deepcopy(self.model), binding_path=self.binding_path,
                    implementation=file_digest(Path(__file__)))

    @classmethod
    def from_identity(cls, value):
        result = cls(CheckpointStore(value["store"]["root"]), value["model"], binding_path=value["binding_path"])
        if result.identity() != value:
            raise ValueError("Frozen checkpoint evolution specification changed")
        return result

    def compare(self, parent, child):
        changed = super().compare(parent, child)
        store = CheckpointStore(self.store["root"])
        if store.identity() != self.store:
            raise ValueError("Checkpoint store implementation changed")
        before = store.binding(parent, binding_path=self.binding_path, model=self.model)
        after = store.binding(child, binding_path=self.binding_path, model=self.model)
        if before["revision"] != after["revision"] and after["record"]["metadata"]["training"] is None:
            # A seed is eligible only when restoring an actual ancestor. A
            # separately imported model is not a trained child of this family.
            ancestor = before["record"]
            while ancestor["metadata"]["training"] is not None:
                revision = ancestor["metadata"]["training"]["request"]["parent_checkpoint"]
                if revision == after["revision"]:
                    break
                ancestor = store.read(revision)
            else:
                raise ValueError("Checkpoint update requires verified training provenance")
        return changed


class TrainingCoordinator:
    def __init__(self, *, store, provider, backend):
        self.store, self.provider, self.backend = store, provider, backend
        self.frozen = deepcopy(self.identity())

    def identity(self):
        return dict(schema="physicalrsi.training-coordinator/v1", store=self.store.identity(),
                    provider=self.provider.identity(), backend=self.backend.identity(),
                    implementation=file_digest(Path(__file__)))

    def _stable(self):
        if self.identity() != self.frozen:
            raise ValueError("Frozen training provider or backend changed")

    def advance(self, request, output):
        """Submit once, or perform read-only recovery/poll/collection on resume."""
        request = deepcopy(request)
        self._stable()
        if (set(request) != {"schema", "store", "model", "provider", "kind", "budget", "parameters",
                            "parent_harness_sha256", "parent_checkpoint", "experience_revision", "binding_path"}
                or request["schema"] != "physicalrsi.training-request/v1"
                or request["store"] != self.store.identity() or request["provider"] != self.provider.identity()
                or request["model"]["backend"] != self.backend.identity()
                or request["kind"] not in {"rl", "supervised"} or not isinstance(request["parameters"], dict)):
            raise ValueError("Invalid frozen training request")
        validate_model(request["model"])
        validate_budget(request["budget"])
        parent = self.store.read(request["parent_checkpoint"])
        if parent["metadata"]["model"] != request["model"]:
            raise ValueError("Training changes the fixed model specification")
        self.store.training_source(request)
        output = Path(output).resolve()
        if output.is_relative_to(self.store.root) or self.store.root.is_relative_to(output):
            raise ValueError("Keep training workspace outside the checkpoint store")
        sha = digest(request)
        with locked(output / ".training.lock"):
            freeze_json(output / "request.json", request)
            completed = output / "checkpoint.json"
            if completed.exists():
                result = read_json(completed)
                record = self.store.read(result["checkpoint_revision"])
                if result["request_sha256"] != sha or record["metadata"]["training"]["request"] != request:
                    raise ValueError("Published checkpoint differs from the training request")
                return result["checkpoint_revision"]
            receipt = output / "submission.json"
            handle = None
            if receipt.exists():
                saved = read_json(receipt)
                if saved["request_sha256"] != sha:
                    raise ValueError("Training submission request changed")
                if saved["state"] == "completed":
                    handle = saved["handle"]
                    if digest(handle) != saved["handle_sha256"]:
                        raise ValueError("Training submission handle changed")
                elif saved["state"] == "started":
                    recover = getattr(self.provider, "recover_submission", None)
                    if not callable(recover):
                        raise ReconciliationRequired("Reconcile the original training submission before retry")
                    handle = recover(deepcopy(request), output)
                    if handle is None:
                        raise ReconciliationRequired("Original training submission remains unknown")
                else:
                    raise ValueError("Unknown training submission receipt")
            else:
                atomic_json(receipt, dict(state="started", request_sha256=sha))
                handle = self.provider.submit(deepcopy(request), output)
            # Providers may retain and update their own reply objects later.
            # Published provenance must own the values observed at each boundary.
            handle = deepcopy(handle)
            self._stable()
            if (not isinstance(handle, dict) or handle.get("request_sha256") != sha
                    or not isinstance(handle.get("job_id"), str) or not handle["job_id"]):
                raise ValueError("Training handle does not bind its request")
            atomic_json(receipt, dict(state="completed", request_sha256=sha,
                                     handle=handle, handle_sha256=digest(handle)))
            status_path = output / "terminal.json"
            status = deepcopy(read_json(status_path) if status_path.exists() else self.provider.poll(deepcopy(handle)))
            self._stable()
            if (status.get("request_sha256") != sha or status.get("job_id") != handle["job_id"]
                    or status.get("state") not in {"queued", "running", "completed", "failed", "unknown"}):
                raise ValueError("Training status differs from its job")
            if status["state"] in {"queued", "running"}:
                raise TrainingPending("Training job remains " + status["state"] + ": " + handle["job_id"])
            if status["state"] == "unknown" or status.get("quiescent") is not True:
                raise ReconciliationRequired("Reconcile uncertain training effects: " + handle["job_id"])
            freeze_json(status_path, status)
            if status["state"] == "failed":
                raise TrainingRejected("Training job failed and is quiescent: " + handle["job_id"])
            # Collection is read-only, so an interrupted collection can safely
            # be repeated against this same completed, quiescent job.
            result = deepcopy(self.provider.collect(deepcopy(handle)))
            self._stable()
            if (set(result) != {"root", "entrypoint", "files", "costs", "metrics", "role", "request_sha256", "job_id"}
                    or result["request_sha256"] != sha or result["job_id"] != handle["job_id"]
                    or result["role"] != "training_only" or not isinstance(result["metrics"], dict)):
                raise ValueError("Training output does not bind the completed job")
            validate_budget(request["budget"], result["costs"])
            digest(result)
            source = result["root"]
            recorded = {k:deepcopy(v) for k,v in result.items() if k != "root"}
            freeze_json(output / "collection.json", recorded)
            revision = self.store.publish(model=request["model"], backend=self.backend, root=source,
                entrypoint=result["entrypoint"], files=result["files"],
                training=dict(request=request, handle=handle, status=status, result=recorded))
            atomic_json(completed, dict(request_sha256=sha, checkpoint_revision=revision))
            return revision


class CheckpointProposer:
    """Development-only planner plus a durable optional RL/supervised trainer."""
    def __init__(self, evaluator, coordinator, planner, *, model, budget, kind="rl", binding_path="weights.json"):
        validate_model(model)
        validate_budget(budget)
        if kind not in {"rl", "supervised"}:
            raise ValueError("Declare the training method")
        self.evaluator, self.coordinator, self.planner = evaluator, coordinator, planner
        self.model, self.budget, self.kind, self.binding_path = deepcopy(model), deepcopy(budget), kind, binding_path

    def identity(self):
        return dict(schema="physicalrsi.checkpoint-proposer/v1", evaluator=self.evaluator.identity(),
                    coordinator=self.coordinator.identity(), planner=self.planner.identity(),
                    model=self.model, budget=self.budget, kind=self.kind, binding_path=self.binding_path,
                    implementation=file_digest(Path(__file__)))

    def policy_identity(self):
        return dict(self.identity(), evaluator=policy_identity(self.evaluator))

    def develop(self, parent, output):
        return self.evaluator.development(parent, output)

    resume_development = develop

    def _propose(self, parent, feedback, context, output):
        values = development_evaluations(parent, feedback, output)
        store = self.coordinator.store
        binding = store.binding(parent, binding_path=self.binding_path, model=self.model)
        directory = Path(output) / "training-proposal"
        inputs = dict(parent=verify_harness(parent), feedback=digest(feedback), context=context, planner=self.planner.identity())
        with locked(directory / ".proposal.lock"):
            path = directory / "plan.json"
            plan = None
            if path.exists():
                receipt = read_json(path)
                if receipt["input_sha256"] != digest(inputs):
                    raise ValueError("Training planner inputs changed")
                if receipt["state"] == "completed":
                    plan = receipt["output"]
                    if digest(plan) != receipt["output_sha256"]:
                        raise ValueError("Training plan changed")
                elif receipt["state"] == "started":
                    method = getattr(self.planner, "resume_plan", None)
                    if not callable(method):
                        raise ReconciliationRequired("Reconcile interrupted training planner")
                    plan = method(deepcopy(parent), deepcopy(values), deepcopy(context), directory)
                else:
                    raise ValueError("Unknown training planner receipt")
            else:
                atomic_json(path, dict(state="started", input_sha256=digest(inputs)))
                plan = self.planner.plan(deepcopy(parent), deepcopy(values), deepcopy(context), directory)
            if self.planner.identity() != inputs["planner"]:
                raise ValueError("Training planner changed")
            atomic_json(path, dict(state="completed", input_sha256=digest(inputs), output=plan, output_sha256=digest(plan)))
            if plan is None:
                return []
            if (set(plan) != {"parameters", "hypothesis", "expected_effect", "required_checks", "used_lessons"}
                    or not isinstance(plan["parameters"], dict)):
                raise ValueError("Training planner must declare parameters and testable provenance")
            experience = store.experiences.capture(parent, feedback, output)
            request = dict(schema="physicalrsi.training-request/v1", store=store.identity(), model=self.model,
                provider=self.coordinator.provider.identity(), kind=self.kind, budget=self.budget,
                parameters=plan["parameters"], parent_harness_sha256=verify_harness(parent),
                parent_checkpoint=binding["revision"], experience_revision=experience, binding_path=self.binding_path)
            try:
                revision = self.coordinator.advance(request, directory / "job")
            except TrainingRejected:
                return []
            development_evaluations(parent, feedback, output)
            record = store.read(revision)
            child = store.install(parent, revision, directory / "publication", candidate_id="trained-" + revision[:16],
                binding_path=self.binding_path, parent_sha256=verify_harness(parent), method=self.kind + "-checkpoint",
                changes=dict(weights=dict(before=binding["revision"], after=revision)), environment="declared-training-provider",
                evidence=deepcopy(feedback["evidence"]), costs=record["metadata"]["training"]["result"]["costs"],
                **{key:plan[key] for key in ("hypothesis", "expected_effect", "required_checks", "used_lessons")})
            return [child]

    def propose(self, parent, feedback, output):
        return self._propose(parent, feedback, None, output)

    def propose_with_context(self, parent, feedback, context, output):
        return self._propose(parent, feedback, context, output)

    resume_proposal = propose
    resume_proposal_with_context = propose_with_context
