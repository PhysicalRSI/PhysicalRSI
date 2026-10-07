"""Bounded sequential campaigns using the existing Self-Harness selection loop.

The campaign records retained rounds as well as inherited revisions. Completed
rounds are audited without reopening their stages, whose evidence may already
belong to immutable lineage. No scheduler or autonomous proposer is supplied.
"""

from copy import deepcopy
from pathlib import Path

from ..infra.storage import atomic_json, file_digest, locked, read_json
from ..infra.trial_quota import TrialLimitExceeded
from ..lineage import StateConflict
from .artifacts import verify_harness


def _manifest(directory):
    entries = {}
    for path in directory.rglob("*"):
        name = path.relative_to(directory)
        if path.is_symlink():
            raise ValueError("Campaign evidence must be physical files")
        if path.is_file():
            entries[str(name)] = file_digest(path)
    return entries


class ImprovementCampaign:
    """The loop factory constructs ports; it must not execute experiments."""

    def __init__(self, root, *, state, build_loop, max_rounds):
        if type(max_rounds) is not int or not 1 <= max_rounds <= 10000:
            raise ValueError("Declare a bounded positive round count")
        self.root = Path(root).resolve()
        self.state, self.build_loop, self.max_rounds = state, build_loop, max_rounds

    def run(self, initial):
        with locked(self.root / ".campaign.lock"):
            return self._run(initial)

    def _directory(self, index):
        return self.root / "rounds" / f"round-{index + 1:04d}"

    def _run(self, initial):
        first = self.build_loop(self._directory(0))
        config = dict(schema="physicalrsi.campaign/v1", max_rounds=self.max_rounds,
                      initial_sha256=verify_harness(initial), state_root=str(self.state.root),
                      policy=first.config, implementation=file_digest(Path(__file__)))
        current = self.state.initialize(initial, policy=first.config, scope=first.config["scope"])
        path = self.root / "campaign.json"
        if path.exists():
            ledger = read_json(path)
            if ledger["config"] != config:
                raise ValueError("Frozen campaign configuration changed")
        else:
            ledger = dict(config=config, anchor_revision=current["revision"], rounds=[])
            atomic_json(path, ledger)
        expected = ledger["anchor_revision"]
        self.state.read(expected)
        if len(ledger["rounds"]) > self.max_rounds:
            raise ValueError("Campaign exceeds its round budget")
        for index, record in enumerate(ledger["rounds"]):
            directory = self._directory(index)
            if (record["index"] != index or record["parent_revision"] != expected
                    or record["files"] != _manifest(directory)):
                raise ValueError("Completed campaign round changed")
            saved = self.state.read(record["result_revision"])
            if (saved["revision"] != expected and saved["previous"] != expected):
                raise StateConflict("Campaign result does not descend from its parent")
            expected = saved["revision"]
        if "stop" in ledger:
            if (current["revision"] != expected or
                    ledger["stop"]["files"] != _manifest(self._directory(len(ledger["rounds"])))):
                raise StateConflict("Budget-stopped campaign state changed")
            return self._result(ledger, current)
        for index in range(len(ledger["rounds"]), self.max_rounds):
            directory = self._directory(index)
            parent = directory / "parent.json"
            if parent.exists() and read_json(parent)["revision"] != expected:
                raise StateConflict("Pending round belongs to a different committed parent")
            current = self.state.resolve()
            if current["revision"] != expected:
                # A crash may occur after lineage CAS but before commit.json or
                # this ledger. Only the exact pending round can explain it.
                selection = directory / "selection.json"
                if (current["previous"] != expected or not selection.exists() or not parent.exists()
                        or current["evidence"].get(str(selection)) != file_digest(selection)
                        or current.get("decision") != read_json(selection).get("output")):
                    raise StateConflict("Committed state changed outside this campaign")
            loop = first if index == 0 else self.build_loop(directory)
            if loop.config != config["policy"] or loop.state.root != self.state.root or loop.root != directory:
                raise ValueError("Campaign loop changed its policy, state or directory")
            try:
                result = loop.run()
            except TrialLimitExceeded as error:
                if self.state.resolve()["revision"] != expected:
                    raise StateConflict("Committed parent changed before quota stop") from error
                ledger["stop"] = dict(status="budget_exhausted", reason=str(error),
                                      round_index=index, files=_manifest(directory))
                atomic_json(path, ledger)
                return self._result(ledger, self.state.resolve())
            if result["revision"] != expected and result["previous"] != expected:
                raise StateConflict("Unexpected campaign lineage transition")
            ledger["rounds"].append(dict(
                index=index, parent_revision=expected, result_revision=result["revision"],
                outcome="retained" if result["revision"] == expected else "inherited",
                files=_manifest(directory)))
            atomic_json(path, ledger)
            expected = result["revision"]
        current = self.state.resolve()
        if current["revision"] != expected:
            raise StateConflict("Committed state changed after this campaign")
        return self._result(ledger, current)

    def _result(self, ledger, current):
        return dict(status=ledger.get("stop", {}).get("status", "completed"),
                    completed_rounds=len(ledger["rounds"]), max_rounds=self.max_rounds,
                    rounds=deepcopy(ledger["rounds"]), current=current,
                    stop=deepcopy(ledger.get("stop")), qualification=None)
