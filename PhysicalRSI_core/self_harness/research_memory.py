"""Immutable System 2 lessons with evidence-bound candidate admission.

This opt-in adapter admits experiments, not promotion. Check reports are trusted
producer attestations: hashes verify identity, not scientific truth or isolation.
"""
from copy import deepcopy
from pathlib import Path

from ..infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json
from .artifacts import verify_harness
from .protocol import _sha


def _evidence(files, *, verified=None):
    if not isinstance(files, dict) or not files:
        raise ValueError("Nonempty evidence required")
    for name, expected in files.items():
        _sha(expected)
        if verified is not None and name in verified:
            if verified[name] != expected:
                raise ValueError("Memory evidence changed: " + name)
            continue
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or file_digest(path) != expected:
            raise ValueError("Memory evidence changed: " + name)
        if verified is not None:
            verified[name] = expected


def _lesson(value, *, verified=None):
    if value.get("status", "active") not in {"active", "suspended"}:
        raise ValueError("Invalid lesson status")
    if not isinstance(value.get("conditions"), dict) or not value["conditions"]:
        raise ValueError("Explicit applicability conditions required")
    if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
           for k, v in value["conditions"].items()):
        raise ValueError("Conditions must be nonempty string pairs")
    if not isinstance(value.get("hypothesis"), str) or not value["hypothesis"].strip():
        raise ValueError("A testable hypothesis is required")
    checks = value.get("required_checks")
    counterexamples = value.get("counterexample_checks")
    if not isinstance(checks, list) or not checks or not isinstance(counterexamples, list):
        raise ValueError("Declare checks and counterexample checks")
    for group in (checks, counterexamples):
        if len(group) != len(set(group)):
            raise ValueError("Duplicate memory checks")
        for check in group:
            identifier(check)
    _evidence(value["evidence"], verified=verified)
    digest(value)


class ResearchMemory:
    """Append content-addressed snapshots; callers explicitly pin a revision."""
    def __init__(self, root):
        self.root = Path(root)

    def write(self, lessons, *, parent=None):
        if parent is not None:
            self.read(parent)
        if not isinstance(lessons, dict) or not lessons:
            raise ValueError("Nonempty lesson mapping required")
        verified = {}
        for name, lesson in lessons.items():
            identifier(name)
            _lesson(lesson, verified=verified)
        value = dict(schema="physicalrsi.research-memory/v1", parent=parent,
                     lessons=deepcopy(lessons), qualification=None)
        revision = digest(value)
        path = self.root / (revision + ".json")
        with locked(self.root / ".lock"):
            if path.exists():
                if read_json(path) != value:
                    raise ValueError("Immutable memory snapshot changed")
            else:
                atomic_json(path, value)
        return revision

    def read(self, revision):
        _sha(revision)
        value = read_json(self.root / (revision + ".json"))
        if digest(value) != revision or value.get("schema") != "physicalrsi.research-memory/v1":
            raise ValueError("Memory snapshot changed")
        # Deduplicate shared evidence only within this read. Later reads must
        # observe changed or missing files, including through bound skills.
        verified = {}
        for lesson in value["lessons"].values():
            _lesson(lesson, verified=verified)
        return deepcopy(value)

    def retrieve(self, revision, context):
        """Exact declared-condition matching; no semantic similarity expansion."""
        lessons = self.read(revision)["lessons"]
        return {name: lesson for name, lesson in lessons.items()
                if lesson.get("status", "active") == "active"
                and all(context.get(k) == v for k, v in lesson["conditions"].items())}

    def apply_delta(self, revision, delta, *, allowed_evidence, context):
        """Apply development-only add/suspend/refine operations in one snapshot.

        Existing lessons and counterexamples cannot be deleted or overwritten.
        The campaign supplies the verified evidence allowlist, never the reflector.
        Empty operations explicitly retain the prior memory revision.
        """
        if (not isinstance(delta, dict) or delta.get("schema") != "physicalrsi.memory-delta/v1"
                or set(delta) != {"schema", "operations"} or not isinstance(delta["operations"], list)):
            raise ValueError("Expected a structured MemoryDelta")
        lessons = self.read(revision)["lessons"]
        used = set()
        for operation in delta["operations"]:
            name = identifier(operation["lesson_id"])
            if name in used:
                raise ValueError("One memory operation per lesson per round")
            used.add(name)
            kind = operation["kind"]
            evidence = operation["lesson"]["evidence"] if kind == "add" else operation["evidence"]
            _evidence(evidence)
            if any(allowed_evidence.get(path) != sha for path, sha in evidence.items()):
                raise ValueError("Memory updates require permitted development evidence")
            if kind == "add":
                if name in lessons:
                    raise ValueError("A new lesson cannot overwrite an existing lesson")
                lesson = deepcopy(operation["lesson"])
                if lesson.get("status", "active") != "active" or "counterexamples" in lesson:
                    raise ValueError("Add a new active hypothesis without fabricated counterexamples")
                _lesson(lesson)
                if any(lesson["conditions"].get(k) != v for k, v in context.items()):
                    raise ValueError("A new lesson cannot broaden the research scope")
                lessons[name] = lesson
                continue
            if kind not in {"counterexample", "refine"} or name not in lessons:
                raise ValueError("Unknown memory operation or lesson")
            lesson = lessons[name]
            if any(context.get(k) != v for k, v in lesson["conditions"].items()):
                raise ValueError("Memory operation targets a lesson outside research scope")
            if kind == "counterexample":
                check = identifier(operation["check"])
                explanation = operation["explanation"]
                if not isinstance(explanation, str) or not explanation.strip():
                    raise ValueError("Explain the counterexample")
                lesson["status"] = "suspended"
                lesson.setdefault("counterexamples", []).append(dict(
                    check=check, explanation=explanation, evidence=deepcopy(evidence)))
                if check not in lesson["counterexample_checks"]:
                    lesson["counterexample_checks"].append(check)
            else:
                conditions = operation["conditions"]
                if any(conditions.get(k) != v for k, v in lesson["conditions"].items()):
                    raise ValueError("Refinement cannot broaden or change existing scope")
                lesson.update(conditions=deepcopy(conditions), hypothesis=operation["hypothesis"], status="active")
            for path, sha in evidence.items():
                if path in lesson["evidence"] and lesson["evidence"][path] != sha:
                    raise ValueError("Evidence path reused with different content")
                lesson["evidence"][path] = sha
        return self.write(lessons, parent=revision) if used else revision

    def counterexample(self, revision, lesson_id, *, check, evidence, explanation):
        """Suspend a contradicted recommendation in a new revision.

        System 2 supplies reviewed development evidence. This transition cannot
        promote a policy or silently reactivate an older recommendation.
        """
        identifier(check)
        _evidence(evidence)
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Explain the counterexample")
        lessons = self.read(revision)["lessons"]
        lesson = lessons[lesson_id]
        lesson["status"] = "suspended"
        lesson.setdefault("counterexamples", []).append(dict(
            check=check, evidence=deepcopy(evidence), explanation=explanation))
        if check not in lesson["counterexample_checks"]:
            lesson["counterexample_checks"].append(check)
        # Keep all supporting and contradicting evidence verifiable on read.
        for path, sha in evidence.items():
            if path in lesson["evidence"] and lesson["evidence"][path] != sha:
                raise ValueError("Evidence path reused with different content")
            lesson["evidence"][path] = sha
        return self.write(lessons, parent=revision)

    def refine(self, revision, lesson_id, *, conditions, hypothesis, evidence):
        """Propose a narrower lesson while retaining its counterexample checks.

        Reactivation here means eligibility for experiment admission, never
        scientific confirmation. Scope expansion requires a separate lesson.
        """
        _evidence(evidence)
        lessons = self.read(revision)["lessons"]
        lesson = lessons[lesson_id]
        if any(conditions.get(k) != v for k, v in lesson["conditions"].items()):
            raise ValueError("Refinement cannot broaden or change existing scope")
        lesson.update(conditions=deepcopy(conditions), hypothesis=hypothesis, status="active")
        for path, sha in evidence.items():
            if path in lesson["evidence"] and lesson["evidence"][path] != sha:
                raise ValueError("Evidence path reused with different content")
            lesson["evidence"][path] = sha
        return self.write(lessons, parent=revision)


class ResearchMemoryGate:
    """Require all applicable checks, including recorded counterexamples.

    Reports bind candidate, selected parent, memory revision and trusted context.
    A report may be added after evaluator construction; its bytes must match the
    predeclared digest. Parent admission still verifies the pinned memory.
    """
    def __init__(self, *, memory, revision, context, parent_sha256, producer, reports):
        self.memory = memory
        self.revision = _sha(revision)
        self.context = deepcopy(context)
        digest(self.context)
        self.parent = _sha(parent_sha256)
        self.producer = identifier(producer)
        self.reports = {str(Path(p).resolve()): _sha(h) for p, h in reports.items()}
        self._identity = self.identity()

    def identity(self):
        self.memory.read(self.revision)
        return dict(kind="research-memory-gate/v1", memory_revision=self.revision,
                    context=deepcopy(self.context), parent_sha256=self.parent,
                    producer=self.producer, reports=deepcopy(self.reports),
                    implementation=file_digest(Path(__file__)))

    def check(self, candidate):
        if self.identity() != self._identity:
            raise ValueError("Frozen memory gate changed")
        freeze = verify_harness(candidate)
        lessons = self.memory.retrieve(self.revision, self.context)
        evidence = dict(memory_revision=self.revision, applicable_lessons=sorted(lessons),
                        candidate_sha256=freeze, parent_sha256=self.parent,
                        qualification=None, scope="experiment admission only")
        if freeze == self.parent:
            return dict(accepted=True, evidence=evidence)
        matches = []
        for name, expected in self.reports.items():
            path = Path(name)
            if not path.exists():
                continue
            if file_digest(path) != expected:
                raise ValueError("Memory check report changed")
            report = read_json(path)
            if report.get("candidate_sha256") == freeze:
                matches.append((name, report))
        if len(matches) != 1:
            return dict(accepted=False, reason="One bound memory check report required", evidence=evidence)
        name, report = matches[0]
        expected = dict(schema="physicalrsi.memory-checks/v1", memory_revision=self.revision,
                        parent_sha256=self.parent, context=self.context, producer=self.producer)
        if any(report.get(k) != v for k, v in expected.items()):
            return dict(accepted=False, reason="Memory check binding mismatch", evidence=evidence)
        if set(report.get("lessons", {})) != set(lessons):
            return dict(accepted=False, reason="Applicable lessons missing or outside scope", evidence=evidence)
        failed = []
        for lesson_id, lesson in lessons.items():
            checks = report["lessons"][lesson_id]
            for key in set(lesson["required_checks"] + lesson["counterexample_checks"]):
                check = checks.get(key, {})
                if check.get("passed") is not True:
                    failed.append(lesson_id + ":" + key)
                else:
                    _evidence(check.get("evidence"))
        evidence.update(report=name, report_sha256=self.reports[name], failed_checks=sorted(failed))
        return dict(accepted=not failed, evidence=evidence,
                    reason="Required memory checks failed" if failed else None)

    def policy_identity(self):
        return dict(kind="research-memory-gate-policy/v1", context=deepcopy(self.context),
                    producer=self.producer, implementation=file_digest(Path(__file__)))


class MemoryBoundSuite:
    """Wrap an existing suite, including PreregisteredSuite, before freezing it."""
    def __init__(self, suite, gate):
        self.suite, self.gate = suite, gate

    def identity(self):
        from ..experiments import trial_factory_kind
        return dict(kind="memory-bound-suite/v1", suite=self.suite.identity(), memory=self.gate.identity(),
                    trial_factory=trial_factory_kind(self.suite))

    def policy_identity(self):
        from .bindings import policy_identity
        from ..experiments import trial_factory_kind
        return dict(kind="memory-bound-suite-policy/v1", suite=policy_identity(self.suite),
                    trial_factory=trial_factory_kind(self.suite),
                    memory=self.gate.policy_identity())

    def register_round(self, state, round_identity, *, context=None):
        if context is not None:
            if (self.gate.revision != context["memory_revision"] or self.gate.context != context["scope"]
                    or self.gate.parent != state.read(context["parent_revision"])["freeze_sha256"]):
                raise ValueError("Memory gate differs from the proposer research context")
        return self.suite.register_round(state, round_identity, context=context)

    def claim_test(self, **kwargs):
        return self.suite.claim_test(**kwargs)

    def admit(self, candidate):
        memory = self.gate.check(candidate)
        if not memory["accepted"]:
            return memory
        result = deepcopy(self.suite.admit(candidate))
        result.setdefault("evidence", {})["research_memory"] = memory["evidence"]
        return result

    def development_cases(self, parent):
        if verify_harness(parent) != self.gate.parent:
            raise ValueError("Memory gate differs from current parent")
        return self.suite.development_cases(parent)

    def validation_cases(self, comparison):
        if comparison["candidates"][comparison["parent_id"]] != self.gate.parent:
            raise ValueError("Memory gate differs from comparison parent")
        return self.suite.validation_cases(comparison)

    def environment(self, task):
        return self.suite.environment(task)

    def trial(self, candidate, task):
        from ..experiments import trial_adapters
        return trial_adapters(self.suite, candidate, task)

    def policy(self, candidate, task):
        return self.suite.policy(candidate, task)

    def verifier(self, task):
        return self.suite.verifier(task)
