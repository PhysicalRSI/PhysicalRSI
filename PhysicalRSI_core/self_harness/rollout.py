"""Development-rollout input for task-specific or model-backed evolvers.

The core verifies trajectory attribution before calling the evolver. It does
not interpret task observations, invoke a model, or expose validation results.
Providers are trusted executable adapters, not sandboxed by this interface.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Protocol

from ..experiments import ExperimentRuntime
from ..infra.storage import file_digest, read_json, relative_path
from ..trials import Evaluation
from .artifacts import verify_harness
from .bindings import policy_identity


def development_evaluations(parent, feedback, output):
    """Rehydrate exact development trials; reject substituted or stale traces."""
    if feedback.get("split") != "evolve" or not feedback.get("evaluations"):
        raise ValueError("Evolver requires completed development rollouts")
    values = read_development_evaluations(parent, feedback, output)
    if values != feedback["evaluations"]:
        raise ValueError("Development evaluation differs from verified rollout")
    return values


def read_development_evaluations(parent, feedback, output):
    """Rehydrate a complete plan from disk; no caller-supplied trace is trusted."""
    root = Path(output).resolve()
    if feedback.get("split") != "evolve" or not feedback.get("episodes"):
        raise ValueError("Evolver requires completed development rollouts")
    expected_sha = verify_harness(parent)
    allowed = {}
    for name, sha in feedback["evidence"].items():
        relative = relative_path(name)
        if str(relative) != name:
            raise ValueError("Development evidence paths must be canonical")
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "experiments" / "development"):
            raise ValueError("Evolver evidence must be verified development data")
        if str(path) in allowed:
            raise ValueError("Duplicate resolved development evidence path")
        allowed[str(path)] = sha
    runtime = ExperimentRuntime(root / "experiments" / "development" / parent["id"] / "trials")
    plan_path = runtime.root.parent / "plan.json"
    if str(plan_path) not in allowed:
        raise ValueError("Development trial plan is missing from evidence")
    payload = plan_path.read_bytes()
    if sha256(payload).hexdigest() != allowed[str(plan_path)]:
        raise ValueError("Evolver evidence must be verified development data")
    plan = json.loads(payload)
    if (plan["candidate_id"] != parent["id"] or plan["candidate_sha256"] != expected_sha
            or plan["split"] != "development" or plan["evaluator_revision"] != feedback["evaluator_revision"]):
        raise ValueError("Development plan differs from the candidate or evaluator")
    values, seen, audited = [], set(), {str(plan_path)}
    for row in feedback["episodes"]:
        run_id = row["run_id"]
        current = Evaluation.read(runtime, run_id).to_dict()
        origin = current["trial"]["system2"]
        if (origin is None or origin["split"] != "development"
                or origin["candidate_id"] != parent["id"] or origin["candidate_sha256"] != expected_sha):
            raise ValueError("Development rollout belongs to another candidate or split")
        if run_id in seen or any(allowed.get(path) != sha for path, sha in current["evidence"].items()):
            raise ValueError("Duplicate or unbound development rollout")
        audited.update(current["evidence"])
        seen.add(run_id)
        values.append(current)
    rows = [dict(task=value["trial"]["task"], case=value["trial"]["case"],
                 run_id=value["trial"]["id"], outcome=value["outcome"],
                 measurements=value["verdict"]["measurements"]) for value in values]
    if rows != feedback["episodes"]:
        raise ValueError("Development episodes differ from the actual rollouts")
    trials = [{key: value["trial"][key] for key in ("task", "case", "budget")}
              | {"run_id": value.get('recovery', {}).get('original_id', value["trial"]["id"])} for value in values]
    if trials != plan["trials"] or any(value["trial"]["scope"] != plan["scope"] for value in values):
        raise ValueError("Development rollouts do not cover the frozen plan")
    # Every Trial file above was read and verified by Evaluation. Auxiliary
    # declared files still require their own audit, even when no Trial uses them.
    for path in set(allowed) - audited:
        if file_digest(Path(path)) != allowed[path]:
            raise ValueError("Evolver evidence must be verified development data")
    return values


class RolloutEvolver(Protocol):
    def identity(self) -> dict: ...
    def propose(self, parent: dict, evaluations: list[dict], context: dict | None,
                output: Path) -> list[dict]:
        """Materialize candidates with normal Self-Harness provenance fields.

        A research context additionally requires scoped lesson IDs, hypothesis,
        expected effect and required checks. Unknown provider calls need an
        explicit resume_proposal method with this signature.
        """
        ...
    # Archive-aware providers may instead implement propose_from_archive(
    # parent, evaluations, research_context, search_context, output), with a
    # corresponding resume_proposal_from_archive recovery method. This keeps
    # the strict round-context/v1 contract unchanged for existing providers.


class RolloutProposer:
    def __init__(self, evaluator, evolver: RolloutEvolver, *, archive=None):
        self.evaluator, self.evolver, self.archive = evaluator, evolver, archive

    def identity(self):
        result = dict(kind="rollout-proposer/v1", evaluator=self.evaluator.identity(),
                      evolver=self.evolver.identity(), implementation=file_digest(Path(__file__)))
        if self.archive is not None:
            result["archive"] = self.archive.identity()
        return result

    def policy_identity(self):
        return dict(self.identity(), evaluator=policy_identity(self.evaluator),
                    evolver=policy_identity(self.evolver))

    def develop(self, parent, output):
        return self.evaluator.development(parent, output)

    resume_development = develop

    def _propose(self, parent, feedback, context, output, *, resume=False):
        values = development_evaluations(parent, feedback, output)
        use_archive = self.archive is not None and callable(getattr(self.evolver, "propose_from_archive", None))
        method_name = ("resume_proposal_from_archive" if resume else "propose_from_archive") if use_archive else (
            "resume_proposal" if resume else "propose")
        method = getattr(self.evolver, method_name, None)
        if not callable(method):
            from ..contracts import ReconciliationRequired
            raise ReconciliationRequired("Evolver must reconcile its interrupted proposal")
        view = None
        arguments = [deepcopy(parent), values, deepcopy(context)]
        if self.archive is not None:
            view = read_json(output / "archive-view.json")
            search = dict(schema="physicalrsi.archive-search/v1", view=deepcopy(view),
                          candidates=self.archive.summaries(view))
            if use_archive:
                arguments.append(search)
        children = method(*arguments, output)
        development_evaluations(parent, feedback, output)
        if view is not None:
            if read_json(output / "archive-view.json") != view:
                raise ValueError("Frozen archive view changed during proposal")
            self.archive.summaries(view)
        return children

    def propose(self, parent, feedback, output):
        return self._propose(parent, feedback, None, output)

    def propose_with_context(self, parent, feedback, context, output):
        return self._propose(parent, feedback, context, output)

    def resume_proposal(self, parent, feedback, output):
        return self._propose(parent, feedback, None, output, resume=True)

    def resume_proposal_with_context(self, parent, feedback, context, output):
        return self._propose(parent, feedback, context, output, resume=True)
