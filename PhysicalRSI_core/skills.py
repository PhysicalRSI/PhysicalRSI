"""Versioned executable skills bound to an explicitly pinned research memory.

Backends are explicitly supplied, trusted implementations. A memory record
cannot import code or select a backend. Binding permits a candidate experiment;
independent Self-Harness validation still owns deployment selection.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .contracts import Operation
from .infra.owned_files import OwnedFiles
from .infra.storage import digest, file_digest


class SkillUnavailable(ValueError):
    pass


def operation_identity(operation):
    if not isinstance(operation, Operation):
        raise ValueError("A skill backend must return an Operation")
    return dict(name=operation.name, revision=operation.revision, input=asdict(operation.input),
                output=asdict(operation.output), effects=sorted(operation.effects))


class SkillLibrary:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.skills = OwnedFiles(self.root / "skills")
        self.experiences = OwnedFiles(self.root / "experiences")

    def identity(self):
        return dict(schema="physicalrsi.skill-library/v1", root=str(self.root),
                    implementation=file_digest(Path(__file__)),
                    storage=file_digest(Path(__file__).parent / "infra/owned_files.py"))

    def read(self, revision):
        record = self.skills.read(revision)
        value = record["metadata"]
        if value.get("schema") not in {"physicalrsi.compiled-skill/v1", "physicalrsi.compiled-skill/v2"}:
            raise ValueError("Unknown compiled skill")
        if value["schema"] == "physicalrsi.compiled-skill/v2" and (
                record["schema"] != "physicalrsi.owned-files/v2"
                or digest(value.get("bundle")) != digest(dict(files=record["files"], file_modes=record["file_modes"]))):
            raise ValueError("Published skill differs from its checked bundle")
        self.experiences.read(value["experience_revision"])
        if (set(value["checks"]) != set(value["request"]["required_checks"])
                or any(row.get("passed") is not True for row in value["checks"].values())):
            raise ValueError("Skill lacks its required compilation checks")
        return deepcopy(record)

    def lesson(self, revision):
        """Return an experiment-eligible lesson; this is not a promotion."""
        record = self.read(revision)
        value = record["metadata"]
        request = value["request"]
        path = self.skills.folder(revision) / "record.json"
        return dict(conditions=deepcopy(request["conditions"]), hypothesis=value["hypothesis"],
                    required_checks=list(request["required_checks"]), counterexample_checks=[],
                    status="active", evidence={str(path): file_digest(path)},
                    skill=dict(library=self.identity(), revision=revision), qualification=None)

    def materialize(self, revision, output):
        """Copy the checked executable package into a candidate's declared closure."""
        record = self.read(revision)
        owned = OwnedFiles(output)
        if owned.root.is_relative_to(self.root) or self.root.is_relative_to(owned.root):
            raise ValueError("Materialize the candidate package outside the skill library")
        folder = self.skills.folder(revision) / "files"
        copied = owned.capture(record["metadata"],
            {name: dict(path=str(folder / name), sha256=sha) for name, sha in record["files"].items()},
            bind_modes=record['schema'] == 'physicalrsi.owned-files/v2')
        if copied != revision:
            raise ValueError("Materialized skill identity changed")
        return owned.folder(copied)

    def bind(self, *, memory, memory_revision, lesson_id, context, backend, package=None):
        """Check applicability and executable closure before constructing a skill.

        New suspended memory revisions prevent new bindings. Historical pinned
        revisions remain historical; this API is not a global emergency stop.
        """
        context = deepcopy(context)
        lesson = memory.read(memory_revision)["lessons"].get(lesson_id)
        if (not lesson or lesson.get("status", "active") != "active"
                or any(context.get(k) != v for k, v in lesson["conditions"].items())):
            raise SkillUnavailable("Skill lesson is suspended or outside its applicability conditions")
        if lesson.get("counterexamples"):
            raise SkillUnavailable("A contradicted skill needs a new compiled revision and lesson")
        reference = lesson.get("skill", {})
        if reference.get("library") != self.identity():
            raise ValueError("Skill memory refers to a different library or implementation")
        revision = reference["revision"]
        record = self.read(revision)
        value = record["metadata"]
        if (any(lesson["conditions"].get(k) != v for k, v in value["request"]["conditions"].items())
                or lesson["required_checks"] != value["request"]["required_checks"]):
            raise ValueError("Skill memory broadened the compiled conditions or changed required checks")
        if backend.identity() != value["backend"]:
            raise ValueError("Skill backend changed")
        owned = self.skills
        if package is not None:
            package = Path(package)
            if package.name != revision:
                raise ValueError("Candidate package belongs to another skill revision")
            owned = OwnedFiles(package.parent)
            if owned.read(revision) != record:
                raise ValueError("Candidate skill package differs from its checked version")
        root = owned.folder(revision) / "files"
        operation = backend.build(root, value["entrypoint"])
        if operation_identity(operation) != value["operation"]:
            raise ValueError("Bound skill executable differs from the checked operation")
        library_identity = self.identity()

        def execute(request, execution_context):
            # Evidence is audited at binding. Per-call checks retain the pinned
            # memory and executable bytes without re-reading every old rollout.
            current = memory.read(memory_revision)["lessons"].get(lesson_id)
            if current != lesson or self.identity() != library_identity:
                raise ValueError("Pinned skill memory or library changed")
            if owned.read(revision) != record or backend.identity() != value["backend"]:
                raise ValueError("Skill executable changed during execution")
            return operation(request, execution_context)

        return Operation(value["request"]["name"],
                         digest(dict(skill=revision, memory=memory_revision, lesson=lesson_id)),
                         operation.input, operation.output, execute, operation.effects, (operation,))
