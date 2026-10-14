"""Recursive research campaigns with independently versioned execution and memory.

Only a complete round receipt advances the research cursor. Harness selection
may finish earlier; a restart reconciles that exact pending round before any
new proposal. Reflection sees development evidence, never validation outcomes.
Providers remain trusted code; this boundary does not provide OS isolation.
"""
from copy import deepcopy
from pathlib import Path
from typing import Protocol

from ..contracts import ReconciliationRequired
from ..infra.storage import atomic_json, digest, file_digest, locked, read_json, relative_path
from ..infra.trial_quota import TrialLimitExceeded
from ..lineage import StateConflict
from .artifacts import verify_harness
from .campaign import _manifest
from .evaluation import freeze_json


class Reflector(Protocol):
    def identity(self) -> dict: ...
    def reflect(self, research_context: dict, development: dict, output: Path) -> dict: ...
    # Optional resume_reflection has the same signature and must reconcile its
    # own provider receipts. Unknown model calls are not automatically repeated.


class ResearchCampaign:
    def __init__(self, root, *, state, build_loop, memory, memory_revision, context,
                 reflector: Reflector, max_rounds, update_memory=True):
        if type(max_rounds) is not int or not 1 <= max_rounds <= 10000:
            raise ValueError("Declare a bounded positive round count")
        if (not isinstance(context, dict) or not context
                or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
                       for k, v in context.items())):
            raise ValueError("Declare nonempty research scope conditions")
        if type(update_memory) is not bool:
            raise ValueError("Declare whether research memory updates are enabled")
        self.root = Path(root).resolve()
        if any(store.root.resolve().is_relative_to(self.root / "rounds") for store in (state, memory)):
            raise ValueError("Keep mutable state and memory stores outside round evidence")
        self.state, self.build_loop, self.memory = state, build_loop, memory
        self.initial_memory, self.context = memory_revision, deepcopy(context)
        self.reflector, self.max_rounds, self.update_memory = reflector, max_rounds, update_memory
        self.memory.read(memory_revision)
        self.reflector_identity = deepcopy(reflector.identity())

    def _context(self, index, parent, memory, previous):
        return dict(schema="physicalrsi.round-context/v1", index=index, parent_revision=parent,
                    memory_revision=memory, previous_round=previous, scope=deepcopy(self.context),
                    lessons=self.memory.retrieve(memory, self.context),
                    budget=dict(max_rounds=self.max_rounds, remaining_rounds=self.max_rounds-index))

    def _stable(self):
        if digest(self.reflector.identity()) != digest(self.reflector_identity):
            raise ValueError("Frozen reflector identity changed")

    def _publish(self, body):
        revision = digest(body)
        freeze_json(self.root / "history" / (revision + ".json"), body)
        atomic_json(self.root / "current.json", dict(revision=revision))
        return revision

    def _read(self, revision):
        from .protocol import _sha
        path = self.root / "history" / (_sha(revision) + ".json")
        value = read_json(path)
        if digest(value) != revision:
            raise ValueError("Research campaign history changed")
        return value

    def _development(self, loop):
        record = read_json(loop.root / "develop.json")
        if record["state"] != "completed" or digest(record["output"]) != record["output_sha256"]:
            raise ValueError("Development receipt changed")
        feedback = deepcopy(record["output"])
        if "evaluations" in feedback:
            from .rollout import development_evaluations
            development_evaluations(read_json(loop.root / "parent.json")["harness"], feedback, loop.root)
        allowed = {}
        for name, sha in feedback["evidence"].items():
            path = (loop.root / relative_path(name)).resolve()
            if (not path.is_relative_to(loop.root / "experiments" / "development")
                    or file_digest(path) != sha):
                raise ValueError("Reflection requires verified development-only evidence")
            allowed[str(path)] = sha
        if feedback.get("split") != "evolve" or not allowed:
            raise ValueError("Development evidence is required for reflection")
        feedback["evidence"] = allowed
        return feedback

    def _reflect(self, context, development, directory):
        path = directory / "reflection.json"
        inputs = digest(dict(context=context, development=development, reflector=self.reflector_identity))
        method = self.reflector.reflect
        if path.exists():
            record = read_json(path)
            if record["input_sha256"] != inputs:
                raise ValueError("Frozen reflection inputs changed")
            if record["state"] == "completed":
                if digest(record["output"]) != record["output_sha256"]:
                    raise ValueError("Reflection output changed")
                return deepcopy(record["output"])
            method = getattr(self.reflector, "resume_reflection", None)
            if record["state"] != "started" or not callable(method):
                raise ReconciliationRequired("Reconcile uncertain reflection before retry")
        self._stable()
        atomic_json(path, dict(state="started", input_sha256=inputs))
        delta = method(deepcopy(context), deepcopy(development), directory / "reflection")
        self._stable()
        atomic_json(path, dict(state="completed", input_sha256=inputs,
                              output=delta, output_sha256=digest(delta)))
        return delta

    def run(self, initial):
        with locked(self.root / ".campaign.lock"):
            return self._run(initial)

    def _run(self, initial):
        self._stable()
        # Factories only declare adapters and materialize frozen plans. They
        # must not execute experiments or invoke proposal/reflection providers.
        template_context = self._context(0, "0" * 64, self.initial_memory, None)
        template = self.build_loop(self.root / "template", template_context)
        if template.archive is not None and template.archive.root.is_relative_to(self.root / "rounds"):
            raise ValueError("Keep the candidate archive outside campaign round evidence")
        if template.round_context != template_context:
            raise ValueError("Research loop must use the supplied round context")
        statistics = template.policy["profile"].get("statistics")
        if statistics and self.max_rounds * template.policy["max_candidates"] > statistics["max_comparisons"]:
            raise ValueError("Campaign exceeds preregistered statistical comparison budget")
        config = dict(schema="physicalrsi.research-campaign/v1", policy=template.policy,
                      initial_sha256=verify_harness(initial), initial_memory=self.initial_memory,
                      state_root=str(self.state.root), memory_root=str(self.memory.root.resolve()),
                      context=self.context, reflector=self.reflector_identity,
                      max_rounds=self.max_rounds, update_memory=self.update_memory,
                      implementation=file_digest(Path(__file__)),
                      memory_implementation=file_digest(Path(__file__).with_name("research_memory.py")))
        freeze_json(self.root / "config.json", config)
        current = self.state.initialize(initial, policy=template.policy, scope=template.config["scope"])
        pointer = self.root / "current.json"
        if not pointer.exists():
            if current["freeze_sha256"] != config["initial_sha256"]:
                raise StateConflict("Research campaign requires its declared initial harness")
            self._publish(dict(schema="physicalrsi.research-cursor/v1", config_sha256=digest(config),
                               previous=None, round=None, index=0, harness_revision=current["revision"],
                               memory_revision=self.initial_memory))
        revision = read_json(pointer)["revision"]
        cursor = self._read(revision)
        history, seen, archived_rounds = [], set(), []
        check, key = cursor, revision
        while True:
            if key in seen or check["config_sha256"] != digest(config):
                raise ValueError("Research cursor cycle or policy mismatch")
            seen.add(key)
            self.state.read(check["harness_revision"])
            self.memory.read(check["memory_revision"])
            if check["previous"] is None:
                if check["index"] != 0 or check["round"] is not None or check["memory_revision"] != self.initial_memory:
                    raise ValueError("Invalid research bootstrap")
                break
            record = check["round"]
            directory = self.root / "rounds" / f"round-{check['index']:04d}"
            if record["files"] != _manifest(directory):
                raise ValueError("Completed research round changed")
            archived_rounds.append(directory / "execution")
            prior = self._read(check["previous"])
            if (check["index"] != prior["index"] + 1
                    or record["parent_revision"] != prior["harness_revision"]
                    or record["parent_memory"] != prior["memory_revision"]):
                raise ValueError("Research round does not descend from its parent")
            history.append(record)
            key, check = check["previous"], prior
        from .archive import verify_archive_receipts
        verify_archive_receipts(archived_rounds)
        if cursor["index"] > self.max_rounds:
            raise ValueError("Campaign exceeds its round budget")
        history.reverse()
        for index in range(cursor["index"], self.max_rounds):
            self._stable()
            directory = self.root / "rounds" / f"round-{index + 1:04d}"
            context = self._context(index, cursor["harness_revision"], cursor["memory_revision"], revision)
            freeze_json(directory / "context.json", context)
            loop = self.build_loop(directory / "execution", deepcopy(context))
            if (loop.policy != config["policy"] or loop.round_context != context
                    or loop.state.root != self.state.root or loop.root != directory / "execution"):
                raise ValueError("Research loop changed policy, context or destination")
            current = self.state.resolve()
            if current["revision"] != cursor["harness_revision"]:
                selection = loop.root / "selection.json"
                if (current["previous"] != cursor["harness_revision"] or not selection.exists()
                        or current["evidence"].get(str(selection)) != file_digest(selection)
                        or current.get("decision") != read_json(selection).get("output")):
                    raise StateConflict("Harness changed outside the pending research round")
            try:
                selected = loop.run()
            except TrialLimitExceeded:
                # Reservation exhaustion has no unknown external effect. Keep
                # the incomplete round and immutable plans for inspection.
                return self._result(cursor, revision, history, "budget_exhausted")
            development = self._development(loop)
            delta = self._reflect(context, development, directory)
            if self._development(loop) != development:
                raise ValueError("Development evidence changed during reflection")
            # Validate even the frozen-memory control arm; discard its update.
            proposed_memory = self.memory.apply_delta(cursor["memory_revision"], delta,
                                allowed_evidence=development["evidence"], context=self.context)
            memory_revision = proposed_memory if self.update_memory else cursor["memory_revision"]
            if self.state.resolve()["revision"] != selected["revision"]:
                raise StateConflict("Harness changed before research round commit")
            selection_path = loop.root / "selection.json"
            outcome = (read_json(selection_path)["output"]["decision"] if selection_path.exists()
                       else read_json(loop.root / "retained.json")["reason"])
            receipt = dict(schema="physicalrsi.research-round/v1", index=index,
                           parent_revision=cursor["harness_revision"], result_revision=selected["revision"],
                           parent_memory=cursor["memory_revision"], result_memory=memory_revision,
                           proposed_memory=proposed_memory, outcome=outcome,
                           context_sha256=digest(context), reflection_sha256=digest(delta),
                           qualification=None)
            freeze_json(directory / "receipt.json", receipt)
            record = dict(receipt, files=_manifest(directory))
            cursor = dict(schema="physicalrsi.research-cursor/v1", config_sha256=digest(config),
                          previous=revision, round=record, index=index + 1,
                          harness_revision=selected["revision"], memory_revision=memory_revision)
            revision = self._publish(cursor)
            history.append(record)
        return self._result(cursor, revision, history, "completed")

    def report_test(self, initial, *, require_completed=False):
        """Complete/audit the campaign, then evaluate its frozen final revision.

        The claim and report live outside immutable round evidence. Test feedback
        is never exposed to reflection or to a subsequent round in this campaign.
        With require_completed, reject unfinished research before any provider
        or Trial is dispatched; completed evidence still receives the full audit.
        """
        if type(require_completed) is not bool:
            raise ValueError("require_completed must be boolean")
        with locked(self.root / ".campaign.lock"):
            if require_completed:
                pointer = self.root / "current.json"
                if (not pointer.exists()
                        or self._read(read_json(pointer)["revision"])["index"] != self.max_rounds):
                    raise ValueError("Final tests require an already completed research campaign")
            result = self._run(initial)
            if result["status"] != "completed":
                raise ValueError("Final tests require a completed research campaign")
            directory = self.root / "rounds" / f"round-{self.max_rounds:04d}"
            context = read_json(directory / "context.json")
            loop = self.build_loop(directory / "execution", context)
            if loop.policy != read_json(self.root / "config.json")["policy"]:
                raise ValueError("Final test evaluator changed its policy")
            report = loop.evaluator.report_test(result["current"]["harness"],
                selection_sha256=result["revision"], output=self.root / "final-test")
            if self.state.resolve()["revision"] != result["current"]["revision"]:
                raise StateConflict("Selected harness changed during final test")
            return report

    def _result(self, cursor, revision, history, status):
        current = self.state.resolve()
        if current["revision"] != cursor["harness_revision"]:
            raise StateConflict("Research cursor and harness state differ")
        return dict(status=status, revision=revision, completed_rounds=cursor["index"],
                    current=current, memory_revision=cursor["memory_revision"],
                    rounds=deepcopy(history), qualification=None)
