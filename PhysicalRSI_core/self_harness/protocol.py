"""Opt-in, frozen case splits for experiment-backed Self-Harness suites.

This is a local reproducibility boundary, not a secret test set or a proof
that two simulator resets produce different physical layouts.
"""

from copy import deepcopy
from pathlib import Path

from ..infra.storage import atomic_json, digest, file_digest, locked, read_json


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(
        c not in "0123456789abcdef" for c in value
    ):
        raise ValueError("Expected a SHA256 identity")
    return value


class CaseProtocol:
    """Freeze all splits before development, including case identity keys.

    ``identity_keys`` names the task-local reset identity (usually seed,
    or layout_id). Extra case metadata cannot disguise reuse of that identity.
    The provider remains responsible for declaring all reset conditions.
    """

    def __init__(self, root, *, splits, identity_keys=("seed",)):
        keys = tuple(identity_keys)
        if not keys or len(set(keys)) != len(keys) or any(
            not isinstance(k, str) or not k for k in keys
        ):
            raise ValueError("Declare unique reset identity keys")
        if set(splits) != {"development", "validation", "test"}:
            raise ValueError("Declare development, validation and test splits")
        tasks = set(splits["development"])
        if not tasks or any(not isinstance(t, str) or not t for t in tasks):
            raise ValueError("Declare nonempty tasks")
        seen = set()
        for split in splits.values():
            if set(split) != tasks:
                raise ValueError("All splits must cover the same tasks")
            for task, cases in split.items():
                if not isinstance(cases, list) or not cases:
                    raise ValueError("Every task needs nonempty case lists")
                for case in cases:
                    if not isinstance(case, dict) or any(k not in case for k in keys):
                        raise ValueError("Case lacks its reset identity")
                    key = digest([task, {k: case[k] for k in keys}])
                    if key in seen:
                        raise ValueError("Reset identity reused within or across splits")
                    seen.add(key)
        manifest = dict(schema="physicalrsi.case-protocol/v1",
                        identity_keys=list(keys), splits=deepcopy(splits),
                        qualification=None, test_role="report_only")
        digest(manifest)  # Reject non-JSON and nonfinite values before writing.
        self.root = Path(root)
        self._manifest = manifest
        self.revision = digest(manifest)
        with locked(self.root / ".lock"):
            path = self.root / "protocol.json"
            if path.exists():
                if read_json(path) != manifest:
                    raise ValueError("Frozen case protocol changed; use a new workspace")
            else:
                atomic_json(path, manifest)

    def identity(self):
        if read_json(self.root / "protocol.json") != self._manifest:
            raise ValueError("Frozen case protocol changed")
        return dict(kind="case-protocol/v1", revision=self.revision,
                    implementation=file_digest(Path(__file__)))

    def cases(self, split):
        self.identity()
        if split not in {"development", "validation"}:
            raise ValueError("Test cases require a frozen final-candidate claim")
        return deepcopy(self._manifest["splits"][split])

    def claim_test(self, *, candidate_sha256, selection_sha256):
        """Bind report-only test access to one final selection, resumably.

        The caller verifies the actual selection and executes test trials.
        This records a commitment; it does not itself certify a selection,
        prevent a privileged reader accessing JSON, or execute held-out tests.
        """
        self.identity()
        claim = dict(protocol_sha256=self.revision,
                     candidate_sha256=_sha(candidate_sha256),
                     selection_sha256=_sha(selection_sha256))
        with locked(self.root / ".lock"):
            self.identity()
            path = self.root / "test-claim.json"
            if path.exists():
                if read_json(path) != claim:
                    raise ValueError("Test set already bound to another final selection")
            else:
                atomic_json(path, claim)
        return deepcopy(self._manifest["splits"]["test"])

    def claim_validation(self, comparison):
        """One paired comparison per plan; a new round needs fresh cases."""
        from .selection import comparison_identity
        identity = comparison_identity(comparison)
        claim = dict(protocol_sha256=self.revision, comparison_sha256=identity)
        with locked(self.root / ".lock"):
            self.identity()
            path = self.root / "validation-claim.json"
            if path.exists():
                if read_json(path) != claim:
                    raise ValueError("Validation cases already bound; preregister a fresh round")
            else:
                atomic_json(path, claim)
        return self.cases("validation")


class PreregisteredSuite:
    """ExperimentSuite adapter; existing evaluators need no modification."""

    def __init__(self, suite, protocol, *, shadow_gate=None):
        if not isinstance(protocol, CaseProtocol):
            raise ValueError("A frozen CaseProtocol is required")
        self.suite, self.protocol = suite, protocol
        self.shadow_gate = shadow_gate

    def identity(self):
        return dict(kind="preregistered-suite/v1", suite=self.suite.identity(),
                    protocol=self.protocol.identity(),
                    shadow_gate=self.shadow_gate.identity() if self.shadow_gate else None)

    def development_cases(self, parent):
        return self.protocol.cases("development")

    def validation_cases(self, comparison):
        return self.protocol.claim_validation(comparison)

    def admit(self, candidate):
        self.protocol.identity()
        shadow = self.shadow_gate.check(candidate) if self.shadow_gate else None
        if shadow is not None and not shadow["accepted"]:
            return shadow
        result = deepcopy(self.suite.admit(candidate))
        if shadow is not None:
            result.setdefault("evidence", {})["shadow"] = shadow["evidence"]
        return result

    def environment(self, task):
        return self.suite.environment(task)

    def policy(self, candidate, task):
        return self.suite.policy(candidate, task)

    def verifier(self, task):
        return self.suite.verifier(task)
