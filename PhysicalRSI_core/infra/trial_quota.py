"""Durable admission budget for experiments on one cooperating POSIX host.

A reservation charges a planned trial, not a claimed successful execution.
Interrupted or never-started reservations are not automatically refunded.
"""

from pathlib import Path

from .storage import atomic_json, digest, file_digest, identifier, locked, read_json


class TrialLimitExceeded(RuntimeError):
    """The complete requested plan cannot fit in the frozen trial allowance."""


class TrialQuota:
    def __init__(self, root, *, max_trials):
        if type(max_trials) is not int or max_trials < 0:
            raise ValueError("A nonnegative integer trial allowance is required")
        self.root = Path(root).resolve()
        self.max_trials = max_trials
        with locked(self.root / ".lock"):
            path = self.root / "reservations.json"
            if path.exists():
                self._read()
            else:
                atomic_json(path, dict(max_trials=max_trials, trials={}))

    def identity(self):
        return dict(root=str(self.root), max_trials=self.max_trials,
                    implementation=file_digest(Path(__file__)))

    def _read(self):
        ledger = read_json(self.root / "reservations.json")
        if ledger["max_trials"] != self.max_trials:
            raise ValueError("Frozen trial allowance changed")
        if not isinstance(ledger["trials"], dict) or len(ledger["trials"]) > self.max_trials:
            raise ValueError("Invalid trial reservation ledger")
        for key, sha in ledger["trials"].items():
            identifier(key)
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("Invalid trial reservation digest")
        return ledger

    def reserve(self, trials):
        """Admit an entire plan atomically; repeated identical IDs cost nothing.

        IDs must identify the durable experiment destination, and values must
        bind its frozen inputs. The returned receipt excludes mutable totals so
        later plans cannot change earlier evidence.
        """
        if not isinstance(trials, dict) or not trials:
            raise ValueError("Declare a nonempty trial plan")
        requested = {identifier(key): digest(value) for key, value in trials.items()}
        with locked(self.root / ".lock"):
            ledger = self._read()
            for key, sha in requested.items():
                if key in ledger["trials"] and ledger["trials"][key] != sha:
                    raise ValueError("Trial reservation inputs changed: " + key)
            new = set(requested) - set(ledger["trials"])
            if len(ledger["trials"]) + len(new) > self.max_trials:
                raise TrialLimitExceeded(
                    f"Trial plan needs {len(new)} new reservations; "
                    f"{self.max_trials - len(ledger['trials'])} remain")
            if new:
                ledger["trials"].update(requested)
                atomic_json(self.root / "reservations.json", ledger)
        return dict(schema="physicalrsi.trial-reservations/v1", quota=self.identity(), trials=requested)

    def status(self):
        with locked(self.root / ".lock"):
            count = len(self._read()["trials"])
        return dict(max_trials=self.max_trials, reserved_trials=count,
                    remaining_trials=self.max_trials - count)
