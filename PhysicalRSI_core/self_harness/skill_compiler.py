"""System 2 compiles owned development experience into checked skill artifacts.

Compilation, offline checks and memory installation do not select a survivor.
Compilers/checkers/backends are explicit trusted ports. Unknown provider calls
require a matching recovery method; no arbitrary code is loaded from memory.
"""
from copy import deepcopy
from pathlib import Path
import stat

from ..contracts import ReconciliationRequired
from ..infra.owned_files import _file_names, physical
from ..infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json
from ..skills import SkillLibrary, operation_identity
from .evaluation import freeze_json
from .experience import DevelopmentExperience


class CompilationRejected(ValueError):
    pass


class ExperienceArchive(DevelopmentExperience):
    """Compatibility entry point for an executable skill library's experience."""
    def __init__(self, library: SkillLibrary):
        super().__init__(library.experiences)


class ExperienceCompiler:
    def __init__(self, *, library, compiler, backend, checker):
        self.library, self.compiler, self.backend, self.checker = library, compiler, backend, checker
        self.experiences = ExperienceArchive(library)
        self._identity = deepcopy(self.identity())

    def identity(self):
        return dict(schema="physicalrsi.experience-compiler/v1", library=self.library.identity(),
                    compiler=self.compiler.identity(), backend=self.backend.identity(), checker=self.checker.identity(),
                    implementation=file_digest(Path(__file__)),
                    experience=file_digest(Path(__file__).with_name("experience.py")))

    def _stable(self):
        if digest(self.identity()) != digest(self._identity):
            raise ValueError("Frozen skill compiler or execution backend changed")

    def _stage(self, path, inputs, call, recover):
        self._stable()
        sha = digest(inputs)
        if path.exists():
            receipt = read_json(path)
            if receipt["input_sha256"] != sha:
                raise ValueError("Skill compilation stage inputs changed")
            if receipt["state"] == "completed":
                if digest(receipt["output"]) != receipt["output_sha256"]:
                    raise ValueError("Skill compilation stage output changed")
                return deepcopy(receipt["output"])
            if receipt["state"] != "started" or recover is None:
                raise ReconciliationRequired("Reconcile interrupted skill compilation: " + str(path))
            call = recover
        atomic_json(path, dict(state="started", input_sha256=sha))
        result = call()
        self._stable()
        atomic_json(path, dict(state="completed", input_sha256=sha, output=result, output_sha256=digest(result)))
        return deepcopy(result)

    def compile(self, experience_revision, *, name, conditions, required_checks, output):
        identifier(name)
        if (not isinstance(conditions, dict) or not conditions
                or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in conditions.items())
                or not isinstance(required_checks, list) or not required_checks
                or len(set(required_checks)) != len(required_checks)):
            raise ValueError("Declare exact applicability conditions and required checks")
        for check in required_checks:
            identifier(check)
        output = Path(output).resolve()
        if output.is_relative_to(self.library.root) or self.library.root.is_relative_to(output):
            raise ValueError("Compile in a separate workspace outside the skill library")
        request = dict(name=name, conditions=deepcopy(conditions), required_checks=list(required_checks))
        inputs = dict(experience_revision=experience_revision, request=request, compiler=self._identity)
        with locked(output / ".compile.lock"):
            self._stable()
            values = self.experiences.read(experience_revision)
            freeze_json(output / "inputs.json", inputs)
            completed = output / "result.json"
            if completed.exists():
                result = read_json(completed)
                if result["input_sha256"] != digest(inputs):
                    raise ValueError("Compiled skill result inputs changed")
                record = self.library.read(result["skill_revision"])
                if record["metadata"]["input_sha256"] != digest(inputs):
                    raise ValueError("Compiled skill result belongs to another request")
                return result["skill_revision"]
            bundle = output / "bundle"
            bundle.mkdir(exist_ok=True)

            def materialize(method):
                return method(deepcopy(values), deepcopy(request), bundle)

            resume = getattr(self.compiler, "resume_compile", None)
            draft = self._stage(output / "materialize.json", inputs,
                lambda: materialize(self.compiler.compile),
                (lambda: materialize(resume)) if callable(resume) else None)
            if (set(draft) != {"entrypoint", "files", "hypothesis"} or not draft["files"]
                    or draft["entrypoint"] not in draft["files"]
                    or not isinstance(draft["hypothesis"], str) or not draft["hypothesis"].strip()):
                raise ValueError("Compiler must declare an executable bundle and testable hypothesis")

            def verify_bundle():
                actual = _file_names(bundle)
                if actual != set(draft["files"]):
                    raise ValueError("Compiler bundle closure differs from its declaration")
                modes = {}
                for path, sha in draft["files"].items():
                    source = physical(bundle, path)
                    if file_digest(source) != sha:
                        raise ValueError("Compiled executable changed")
                    modes[path] = stat.S_IMODE(source.stat().st_mode)
                    if modes[path] > 0o777:
                        raise ValueError("Compiled bundle requires regular POSIX permission bits")
                return dict(files=deepcopy(draft["files"]), file_modes=modes)

            # Pin the runnable declaration before backend construction or any
            # check. A resumed stage cannot adopt changed modes with old reports.
            checked_bundle = verify_bundle()
            freeze_json(output / "bundle.json", checked_bundle)
            operation = self.backend.build(bundle, draft["entrypoint"])
            if verify_bundle() != checked_bundle:
                raise ValueError("Compiled bundle modes changed during backend construction")
            operation_spec = operation_identity(operation)
            check_inputs = dict(inputs=inputs, draft=draft, bundle=checked_bundle, operation=operation_spec)

            def check(method):
                return method(operation, deepcopy(values), deepcopy(request))

            resume = getattr(self.checker, "resume_check", None)
            reports = self._stage(output / "checks.json", check_inputs,
                lambda: check(self.checker.check), (lambda: check(resume)) if callable(resume) else None)
            self._stable()
            if verify_bundle() != checked_bundle:
                raise ValueError("Compiled bundle modes changed during checks")
            # Verify the complete source again after untrusted proposal/check ports.
            values = self.experiences.read(experience_revision)
            trial_ids = {v["trial_sha256"] for v in values}
            if not isinstance(reports, dict) or set(reports) != set(request["required_checks"]):
                raise ValueError("Compilation checks must cover the exact declared checks")
            for report in reports.values():
                if (type(report.get("passed")) is not bool or not isinstance(report.get("trials"), list)
                        or not report["trials"] or not set(report["trials"]) <= trial_ids):
                    raise ValueError("Compilation check requires bound development trial evidence")
            if not all(report["passed"] for report in reports.values()):
                raise CompilationRejected("Executable skill failed its compilation checks")
            metadata = dict(schema="physicalrsi.compiled-skill/v2", input_sha256=digest(inputs), request=request,
                experience_revision=experience_revision, compiler=self.compiler.identity(), backend=self.backend.identity(),
                checker=self.checker.identity(), operation=operation_spec, checks=reports,
                bundle=checked_bundle, entrypoint=draft["entrypoint"], hypothesis=draft["hypothesis"], qualification=None)
            revision = self.library.skills.capture(metadata,
                {name: dict(path=str(bundle / name), sha256=sha) for name, sha in draft["files"].items()},
                expected_modes=checked_bundle["file_modes"])
            self.library.read(revision)
            atomic_json(completed, dict(input_sha256=digest(inputs), skill_revision=revision))
            return revision

    def install(self, skill_revision, *, memory, memory_revision, lesson_id):
        """Append a skill lesson without overwriting prior lessons or counterexamples."""
        identifier(lesson_id)
        lesson = self.library.lesson(skill_revision)
        lessons = memory.read(memory_revision)["lessons"] if memory_revision is not None else {}
        if lesson_id in lessons:
            if lessons[lesson_id] != lesson:
                raise ValueError("Install a new skill version under a new lesson ID")
            return memory_revision
        lessons[lesson_id] = lesson
        return memory.write(lessons, parent=memory_revision)

    def suspend(self, experience_revision, *, memory, memory_revision, lesson_id, check, explanation):
        """Suspend only from definite failures of this exact skill/memory binding."""
        lesson = memory.read(memory_revision)["lessons"][lesson_id]
        reference = lesson["skill"]
        if reference["library"] != self.library.identity():
            raise ValueError("Counterexample belongs to another skill library")
        skill = self.library.read(reference["revision"])["metadata"]
        values = self.experiences.read(experience_revision)
        required = {("skill", skill["request"]["name"], reference["revision"]),
                    ("memory", "compiled-skill-memory", memory_revision)}
        failures = []
        for value in values:
            binding = value["receipt"].get("binding")
            dependencies = {(d["kind"], d["name"], d["revision"]) for d in
                            binding["system1"]["dependencies"]} if binding else set()
            if value["outcome"] == "failure" and required <= dependencies:
                failures.append(value)
        if not failures:
            raise ValueError("Counterexample must contain a failed development trial using the pinned skill and memory")
        return memory.counterexample(memory_revision, lesson_id, check=check, explanation=explanation,
                                     evidence=self.library.experiences.evidence(experience_revision))
