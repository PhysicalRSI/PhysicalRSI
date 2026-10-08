"""Resumable research coordination; execution and selection remain external.

Records bind research intent to existing lineage and memory. Artifact hashes
verify identity, not experimental validity. This module never launches jobs,
selects a policy, or certifies a benchmark result.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json
from PhysicalRSI_core.lineage import HarnessState, StateConflict
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.protocol import _sha
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory, _evidence


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty research explanation required")
    return value


def _reference(path):
    path = Path(path).resolve()
    return {str(path): file_digest(path)}


def _memory_lineage(memory, revision, ancestor):
    cursor = revision
    while cursor != ancestor:
        if cursor is None:
            raise ValueError("Memory revision does not descend from pinned memory")
        cursor = memory.read(cursor)["parent"]
    memory.read(ancestor)


class ResearchRecord:
    """One hypothesis per workspace, with immutable snapshots and CAS updates.

    The caller supplies reviewed development artifacts. A completed record is
    a research disposition, not a successful experiment or a policy promotion.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()

    def _publish(self, body, previous):
        value = dict(schema="physicalrsi.research-record/v1", previous=previous,
                     body=deepcopy(body), qualification=None)
        revision = digest(value)
        path = self.root / "history" / (revision + ".json")
        if path.exists() and read_json(path) != value:
            raise ValueError("Research history changed")
        atomic_json(path, value)
        atomic_json(self.root / "current.json", {"revision": revision})
        return self._view(revision, value)

    def _load(self, revision):
        _sha(revision)
        value = read_json(self.root / "history" / (revision + ".json"))
        if digest(value) != revision or value.get("schema") != "physicalrsi.research-record/v1":
            raise ValueError("Research record changed")
        _evidence(value["body"]["evidence"])
        return value

    @staticmethod
    def _view(revision, value):
        return dict(revision=revision, **deepcopy(value))

    def read(self, revision=None):
        revision = revision or read_json(self.root / "current.json")["revision"]
        value = self._load(revision)
        cursor = value["previous"]
        seen = {revision}
        while cursor is not None:
            if cursor in seen:
                raise ValueError("Research history cycle")
            seen.add(cursor)
            cursor = self._load(cursor)["previous"]
        return self._view(revision, value)

    def create(self, *, question, hypothesis, falsifier, alternatives, budget,
               state, memory, memory_revision, context, evidence):
        for value in (question, hypothesis, falsifier):
            _text(value)
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("Declare competing explanations")
        for alternative in alternatives:
            _text(alternative)
        if (set(budget) != {"max_trials", "trial_seconds"}
                or any(type(v) is not int or v < 1 for v in budget.values())):
            raise ValueError("Declare positive integer max_trials and trial_seconds")
        _evidence(evidence)
        parent = state.resolve()
        lessons = memory.retrieve(memory_revision, context)
        body = dict(question=question, hypothesis=hypothesis, falsifier=falsifier,
                    alternatives=deepcopy(alternatives), budget=deepcopy(budget),
                    parent=dict(state_root=str(state.root), revision=parent["revision"],
                                freeze_sha256=parent["freeze_sha256"]),
                    memory=dict(root=str(memory.root.resolve()), revision=memory_revision,
                                context=deepcopy(context), retrieved_lessons=sorted(lessons)),
                    evidence=deepcopy(evidence), experiment=None, result=None, decision=None,
                    status="preparing")
        with locked(self.root / ".lock"):
            if (self.root / "current.json").exists():
                raise ValueError("Research record exists; resume or use a new workspace")
            return self._publish(body, None)

    def _current(self, expected):
        current = self.read()
        if current["revision"] != expected:
            raise StateConflict("Research record advanced; reload before updating")
        return current["body"]

    @staticmethod
    def _parent_current(body):
        binding = body["parent"]
        current = HarnessState(binding["state_root"]).resolve()
        if (current["revision"] != binding["revision"]
                or current["freeze_sha256"] != binding["freeze_sha256"]):
            raise StateConflict("Research parent is stale; register a new hypothesis record")

    @staticmethod
    def _memory(body):
        binding = body["memory"]
        memory = ResearchMemory(binding["root"])
        lessons = memory.retrieve(binding["revision"], binding["context"])
        if sorted(lessons) != binding["retrieved_lessons"]:
            raise ValueError("Research memory retrieval changed")
        if body["decision"] is not None:
            _memory_lineage(memory, body["decision"]["memory_revision"], binding["revision"])
        return memory

    @staticmethod
    def _add_evidence(body, evidence):
        _evidence(evidence)
        for path, sha in evidence.items():
            if path in body["evidence"] and body["evidence"][path] != sha:
                raise ValueError("Evidence identity changed")
            body["evidence"][path] = sha

    def bind_experiment(self, expected, *, candidate, evaluator_sha256, protocol,
                        workspace, used_lessons, evidence):
        """Freeze a prepared experiment before dispatch; this is not admission.

        Protocol is an existing CaseProtocol. Actual evaluators still enforce
        admission, case separation, trial quotas, and survivor selection.
        """
        from PhysicalRSI_core.self_harness.protocol import CaseProtocol
        if not isinstance(protocol, CaseProtocol):
            raise ValueError("Bind an existing CaseProtocol")
        with locked(self.root / ".lock"):
            body = self._current(expected)
            if body["status"] != "preparing":
                raise ValueError("Experiment already bound or research closed")
            self._parent_current(body)
            self._memory(body)
            if (not isinstance(used_lessons, list) or len(set(used_lessons)) != len(used_lessons)
                    or not set(used_lessons) <= set(body["memory"]["retrieved_lessons"])):
                raise ValueError("Used lessons must come from pinned retrieval")
            workspace = Path(workspace).resolve()
            if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
                raise ValueError("Bind a fresh experiment workspace before dispatch")
            self._add_evidence(body, evidence)
            self._add_evidence(body, _reference(protocol.root / "protocol.json"))
            body["experiment"] = dict(candidate=deepcopy(candidate),
                candidate_sha256=verify_harness(candidate), evaluator_sha256=_sha(evaluator_sha256),
                protocol=protocol.identity(), workspace=str(workspace),
                used_lessons=deepcopy(used_lessons))
            body["status"] = "ready"
            return self._publish(body, expected)

    def claim_dispatch(self, expected, *, job_id):
        """Durably claim one dispatch before external effects; never auto-retry.

        A crash after this call requires checking the job/workspace. This token
        does not prove a remote process exists or enforce remote ownership.
        """
        _text(job_id)
        with locked(self.root / ".lock"):
            body = self._current(expected)
            if body["status"] != "ready":
                raise ValueError("Research is not ready for dispatch")
            self._parent_current(body)
            self._memory(body)
            experiment = body["experiment"]
            workspace = Path(experiment["workspace"])
            if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
                raise ValueError("Experiment workspace changed before dispatch")
            if verify_harness(experiment["candidate"]) != experiment["candidate_sha256"]:
                raise ValueError("Research candidate changed")
            body["job_id"] = job_id
            body["status"] = "awaiting_result"
            return self._publish(body, expected)

    def record_result(self, expected, *, report):
        """Record a reviewed completed/failed job, preserving all evidence.

        A result is not a selection. Infrastructure failures can be recorded
        without claiming that the research hypothesis has been disproved.
        """
        with locked(self.root / ".lock"):
            body = self._current(expected)
            if body["status"] != "awaiting_result":
                raise ValueError("No dispatched experiment awaits a result")
            reference = _reference(report)
            result = read_json(report)
            binding = dict(schema="physicalrsi.research-result/v1", research_revision=expected,
                           experiment_sha256=digest(body["experiment"]), job_id=body["job_id"])
            if any(result.get(k) != v for k, v in binding.items()):
                raise ValueError("Result does not match the dispatched research experiment")
            if result.get("outcome") not in {"completed", "failed", "inconclusive"}:
                raise ValueError("Unknown experiment outcome")
            self._add_evidence(body, result["evidence"])
            self._add_evidence(body, reference)
            body["result"] = dict(outcome=result["outcome"], report=reference)
            body["status"] = "awaiting_decision"
            return self._publish(body, expected)

    def close(self, expected, *, disposition, explanation, evidence,
              memory_revision, memory_explanation):
        """Close with a research disposition and explicit memory handling.

        Does not commit harness lineage. Retaining memory is a valid explicit
        decision. Changed memory must descend from the pinned revision.
        """
        if disposition not in {"supported", "rejected", "inconclusive", "superseded"}:
            raise ValueError("Unknown research disposition")
        _text(explanation)
        _text(memory_explanation)
        with locked(self.root / ".lock"):
            body = self._current(expected)
            if body["status"] in {"completed", "awaiting_result"}:
                raise ValueError("Reconcile the dispatched job before closing research")
            if disposition == "supported" and (body["result"] or {}).get("outcome") != "completed":
                raise ValueError("Support requires completed experiment evidence")
            memory = self._memory(body)
            _memory_lineage(memory, memory_revision, body["memory"]["revision"])
            self._add_evidence(body, evidence)
            self._add_evidence(body, _reference(memory.root / (memory_revision + ".json")))
            body["decision"] = dict(disposition=disposition, explanation=explanation,
                memory_revision=memory_revision, memory_explanation=memory_explanation)
            body["status"] = "completed"
            return self._publish(body, expected)

    def status(self):
        record = self.read()
        body = record["body"]
        self._memory(body)
        stale = False
        try:
            self._parent_current(body)
        except StateConflict:
            stale = True
        actions = dict(preparing="prepare_and_freeze_experiment", ready="claim_then_dispatch",
                       awaiting_result="inspect_job_and_collect_evidence",
                       awaiting_decision="record_decision_and_memory", completed="research_closed")
        action = actions[body["status"]]
        if stale and body["status"] in {"preparing", "ready"}:
            action = "supersede_stale_parent"
        return dict(workspace=str(self.root), revision=record["revision"], status=body["status"],
                    question=body["question"], hypothesis=body["hypothesis"],
                    next_action=action, parent_stale=stale, parent=body["parent"],
                    memory=body["memory"], budget=body["budget"], job_id=body.get("job_id"),
                    decision=body["decision"],
                    experiment_workspace=(body["experiment"] or {}).get("workspace"),
                    qualification=None)

    def complete_review(self, heartbeat, request_id, review, *, expected, now=None):
        """Bind the existing heartbeat's next action to an immutable record.

        A timer is not altered or restarted. The caller must still inspect live
        processes and write an evidence-backed scientific review.
        """
        with locked(self.root / ".lock"):
            self._current(expected)
            status = self.status()
            review = deepcopy(review)
            review["evidence"].update(_reference(self.root / "history" / (expected + ".json")))
            review["next_experiment"] = json.dumps(dict(
                research_workspace=str(self.root), research_revision=expected,
                next_action=status["next_action"], explanation=_text(review["next_experiment"])),
                sort_keys=True)
            return heartbeat.complete(request_id, review, now=now)


def index_status(path):
    """Inspect explicitly registered records; do not guess from directory age."""
    index = read_json(path)
    if index.get("schema") != "physicalrsi.research-index/v1" or not index.get("records"):
        raise ValueError("Expected a nonempty research index")
    result = {}
    for name, root in index["records"].items():
        identifier(name)
        if not Path(root).is_absolute():
            raise ValueError("Research index requires absolute workspace paths")
        result[name] = ResearchRecord(root).status()
    return dict(records=result, qualification=None)


def main():
    parser = argparse.ArgumentParser(description="Inspect a persisted research record without launching work.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--workspace")
    group.add_argument("--index", help="JSON index of explicitly registered research workspaces")
    args = parser.parse_args()
    result = index_status(args.index) if args.index else ResearchRecord(args.workspace).status()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
