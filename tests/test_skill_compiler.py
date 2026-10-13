from copy import deepcopy
from pathlib import Path
import os
import shutil
import stat

import pytest

from PhysicalRSI_core.contracts import Context, ReconciliationRequired
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.skills import SkillLibrary, SkillUnavailable
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.skill_compiler import CompilationRejected, ExperienceArchive, ExperienceCompiler
from PhysicalRSI_demos.rollout_evolution import Environment
from PhysicalRSI_demos.skill_transfer import (CONDITIONS, LESSON, InverseReplayCheck, ResponseCompiler,
                                            VectorBackend, prepare, run)


def compile_request(root):
    return dict(name=LESSON, conditions=CONDITIONS, required_checks=["inverse-transition"], output=root)


@pytest.mark.parametrize("gain", [2., .25])
def test_source_only_skill_transfers_then_counterexample_withdraws_it(tmp_path, monkeypatch, gain):
    resets = []
    original = Environment.reset

    def observed(self, case, context):
        resets.append((self.task, case))
        return original(self, case, context)

    monkeypatch.setattr(Environment, "reset", observed)
    result = run(tmp_path, gain=gain)
    library = SkillLibrary(tmp_path / "library")
    source = ExperienceArchive(library).read(result["source_experience"])
    assert {v["trial"]["task"] for v in source} == {"axis"}
    assert all(v["trial"]["system2"]["split"] == "development" for v in source)
    record = library.read(result["skill_revision"])
    program = read_json(library.skills.folder(result["skill_revision"]) / "files/program.json")
    assert program["steps"][1]["factor"] == pytest.approx(1 / gain)
    assert record["metadata"]["checks"]["inverse-transition"]["passed"]
    selected = result["research"]["current"]
    assert read_json(Path(selected["harness"]["root"]) / "control.json") == {"axis": 1., "vector": 1.}
    assert len(selected["harness"]["components"]["skills"]) == 3
    directory = tmp_path / "campaign/rounds/round-0001/execution"
    selection = read_json(directory / "selection.json")["output"]
    assert selection["metrics"]["uncompiled"]["per_task"] == {"axis": 0., "vector": 0.}
    assert selection["metrics"]["compiled-response"]["per_task"] == {"axis": 1., "vector": 1.}
    assert all(m["successes"] == m["planned"] == 1 for m in result["final_test"]["metrics"].values())
    memory = ResearchMemory(tmp_path / "memory")
    assert memory.read(result["memory_revision"])["lessons"][LESSON]["status"] == "active"
    suspended = memory.read(result["suspended_memory_revision"])
    assert suspended["parent"] == result["memory_revision"]
    assert suspended["lessons"][LESSON]["status"] == "suspended"
    assert result["restored"]["harness"]["id"] == "uncompiled"
    assert result["withdrawn_trial"]["outcome"] == "failure"  # Withdrawal is not a new solution.
    assert not result["withdrawn_trial"]["receipt"]["binding"]["system1"]["dependencies"]
    assert len(resets) == result["quota"]["reserved_trials"] == 18
    assert run(tmp_path, gain=gain) == result
    assert len(resets) == 18


def test_owned_source_and_executable_survive_source_workspace_loss(tmp_path, monkeypatch):
    sensor = tmp_path / "sensor.bin"
    sensor.write_bytes(b"\x00\xffdevelopment sensor" * 20)
    sha = file_digest(sensor)
    monkeypatch.setattr(Environment, "export_evidence", lambda self: {
        "sensor.bin": dict(path=str(sensor),sha256=sha)}, raising=False)
    root, _, library, memory, compiler, _, _, experience, skill, revision = prepare(tmp_path / "study")
    sensor.unlink()
    for name in ("source", "seed", "compilation"):
        shutil.rmtree(root / name)
    values = compiler.experiences.read(experience)
    assert all(any(p.endswith("sensor.bin") for p in v["evidence"]) for v in values)
    operation = library.bind(memory=memory,memory_revision=revision,lesson_id=LESSON,context=CONDITIONS,backend=VectorBackend())
    assert operation(dict(position=[0.,0.],target=[2.,-4.]),Context("recovered")) == [1.,-2.]
    package = library.materialize(skill,root / "candidate")
    copied = library.bind(memory=memory,memory_revision=revision,lesson_id=LESSON,context=CONDITIONS,
                          backend=VectorBackend(),package=package)
    assert (package / "files/program.json").stat().st_ino != (library.skills.folder(skill) / "files/program.json").stat().st_ino
    (package / "files/program.json").write_text("changed")
    with pytest.raises(ValueError,match="changed"):
        copied(dict(position=[0.],target=[2.]),Context("tampered-copy"))
    assert operation(dict(position=[0.],target=[2.]),Context("original")) == [1.]


@pytest.mark.parametrize("damage", ["executable", "experience", "extra", "symlink"])
def test_damaged_skill_or_source_is_rejected_before_binding(tmp_path, damage):
    _, _, library, memory, _, _, _, experience, skill, revision = prepare(tmp_path)
    if damage == "experience":
        path = next(p for p in library.experiences.folder(experience).rglob("trajectory.json"))
    else:
        path = library.skills.folder(skill) / "files" / ("extra.json" if damage == "extra" else "program.json")
    if damage == "symlink":
        copy = tmp_path / "same-content.json"
        shutil.copyfile(path,copy)
        path.unlink()
        path.symlink_to(copy)
    else:
        path.write_text("damaged")
    with pytest.raises(ValueError):
        library.bind(memory=memory,memory_revision=revision,lesson_id=LESSON,context=CONDITIONS,backend=VectorBackend())


def test_scope_and_suspended_revision_cannot_be_reactivated_by_prose_refinement(tmp_path):
    result = run(tmp_path)
    library, memory = SkillLibrary(tmp_path / "library"), ResearchMemory(tmp_path / "memory")
    with pytest.raises(SkillUnavailable,match="conditions"):
        library.bind(memory=memory,memory_revision=result["memory_revision"],lesson_id=LESSON,
                     context=dict(CONDITIONS,actuator_family="another-device"),backend=VectorBackend())
    with pytest.raises(SkillUnavailable,match="suspended"):
        library.bind(memory=memory,memory_revision=result["suspended_memory_revision"],lesson_id=LESSON,
                     context=CONDITIONS,backend=VectorBackend())
    refined = memory.refine(result["suspended_memory_revision"],LESSON,conditions=dict(CONDITIONS,regime="narrow"),
        hypothesis="A textual refinement does not recheck the contradicted executable",
        evidence=library.lesson(result["skill_revision"])["evidence"])
    with pytest.raises(SkillUnavailable,match="new compiled revision"):
        library.bind(memory=memory,memory_revision=refined,lesson_id=LESSON,
                     context=dict(CONDITIONS,regime="narrow"),backend=VectorBackend())


def test_failed_actual_replay_check_does_not_publish_a_skill(tmp_path):
    _, _, library, _, _, _, _, experience, _, _ = prepare(tmp_path / "source-study")

    class WrongCompiler(ResponseCompiler):
        def identity(self):
            return dict(kind="deliberately-wrong-test-compiler")

        def compile(self, values, request, output):
            self.calls += 1
            atomic_json(output / "program.json",dict(schema="vector-action/v1",steps=[dict(op="difference"),dict(op="scale",factor=42.)]))
            return dict(entrypoint="program.json",files={"program.json":file_digest(output / "program.json")},hypothesis="Deliberately incorrect response")

    provider = WrongCompiler()
    engine = ExperienceCompiler(library=library,compiler=provider,backend=VectorBackend(),checker=InverseReplayCheck())
    before = set(library.skills.root.iterdir())
    for _ in range(2):
        with pytest.raises(CompilationRejected):
            engine.compile(experience,**compile_request(tmp_path / "bad-compilation"))
    report = read_json(tmp_path / "bad-compilation/checks.json")["output"]["inverse-transition"]
    assert not report["passed"] and any(row["actual"] != row["expected"] for row in report["cases"])
    assert provider.calls == 1 and set(library.skills.root.iterdir()) == before


def test_unknown_compiler_call_requires_explicit_recovery(tmp_path):
    _, _, library, _, _, _, _, experience, _, _ = prepare(tmp_path / "source-study")

    class Uncertain:
        calls = 0

        def identity(self):
            return dict(kind="uncertain-test-compiler")

        def compile(self, values, request, output):
            self.calls += 1
            raise RuntimeError("provider reply lost")

    provider = Uncertain()
    engine = ExperienceCompiler(library=library,compiler=provider,backend=VectorBackend(),checker=InverseReplayCheck())
    with pytest.raises(RuntimeError,match="reply lost"):
        engine.compile(experience,**compile_request(tmp_path / "unknown-compilation"))
    with pytest.raises(ReconciliationRequired,match="Reconcile"):
        engine.compile(experience,**compile_request(tmp_path / "unknown-compilation"))
    assert provider.calls == 1


def test_compiler_recovers_a_persisted_reply_without_repeating_synthesis(tmp_path):
    _, _, library, _, _, _, _, experience, _, _ = prepare(tmp_path / "source-study")

    class Recoverable(ResponseCompiler):
        def identity(self):
            return dict(kind="recoverable-test-compiler")

        def compile(self, values, request, output):
            draft = super().compile(values,request,output)
            atomic_json(output.parent / "provider.json",draft)
            raise RuntimeError("reply publication interrupted")

        def resume_compile(self, values, request, output):
            return read_json(output.parent / "provider.json")

    first = Recoverable()
    engine = ExperienceCompiler(library=library,compiler=first,backend=VectorBackend(),checker=InverseReplayCheck())
    output = tmp_path / "recoverable"
    with pytest.raises(RuntimeError,match="interrupted"):
        engine.compile(experience,**compile_request(output))
    second = Recoverable()
    restarted = ExperienceCompiler(library=library,compiler=second,backend=VectorBackend(),checker=InverseReplayCheck())
    skill = restarted.compile(experience,**compile_request(output))
    assert first.calls == 1 and second.calls == 0
    shutil.rmtree(output / "bundle")
    assert restarted.compile(experience,**compile_request(output)) == skill


@pytest.mark.parametrize("split", ["validation", "test"])
def test_non_development_input_is_rejected_before_compilation(tmp_path, split):
    root, initial, library, _, _, _, _, _, _, _ = prepare(tmp_path)
    feedback = read_json(root / "source/execution/experiments/development/uncompiled/plan.json")
    # Even a caller-supplied summary cannot authorize a validation/test source.
    with pytest.raises(ValueError,match="development"):
        ExperienceArchive(library).capture(initial,dict(feedback,split=split),root / "source/execution")


def test_counterexample_must_bind_the_actual_skill_and_memory(tmp_path):
    _, _, library, memory, compiler, _, _, experience, _, revision = prepare(tmp_path)
    before = set(memory.root.iterdir())
    with pytest.raises(ValueError,match="pinned skill and memory"):
        compiler.suspend(experience,memory=memory,memory_revision=revision,lesson_id=LESSON,
                         check="unrelated-failure",explanation="Source failure did not execute the compiled skill")
    assert set(memory.root.iterdir()) == before


def test_withdrawal_recovers_after_rollback_without_repeating_trials(tmp_path,monkeypatch):
    from PhysicalRSI_core.lineage import HarnessState
    rollback = HarnessState.rollback
    interrupted = False
    resets = []
    reset = Environment.reset

    def observed(self,case,context):
        resets.append(case)
        return reset(self,case,context)

    def interrupt(self,*args,**kwargs):
        nonlocal interrupted
        result = rollback(self,*args,**kwargs)
        if not interrupted:
            interrupted = True
            raise RuntimeError("withdrawal publication interrupted")
        return result

    monkeypatch.setattr(Environment,"reset",observed)
    with monkeypatch.context() as patch:
        patch.setattr(HarnessState,"rollback",interrupt)
        with pytest.raises(RuntimeError,match="withdrawal publication"):
            run(tmp_path)
    assert len(resets) == 17
    result = run(tmp_path)
    assert result["restored"]["harness"]["id"] == "uncompiled"
    assert len(resets) == 18
    assert run(tmp_path) == result and len(resets) == 18
    result_path = tmp_path / "result.json"
    altered = read_json(result_path)
    altered["final_test"]["metrics"]["vector"]["successes"] = 99
    atomic_json(result_path,altered)
    with pytest.raises(ValueError,match="evidence changed"):
        run(tmp_path)
    assert len(resets) == 18


def test_skill_publication_interruption_preserves_checked_compilation(tmp_path,monkeypatch):
    from PhysicalRSI_core.infra import owned_files
    _, _, library, _, compiler, _, _, experience, _, _ = prepare(tmp_path / "source-study")
    copying = owned_files.shutil.copyfileobj
    before = set(library.skills.root.iterdir())
    calls = compiler.compiler.calls

    def interrupt(incoming,outgoing,*args,**kwargs):
        copying(incoming,outgoing,*args,**kwargs)
        if str(library.skills.root / ".pending-") in str(outgoing.name):
            raise RuntimeError("skill publication interrupted")

    kwargs = dict(compile_request(tmp_path / "publication"),name="another-checked-skill")
    with monkeypatch.context() as patch:
        patch.setattr(owned_files.shutil,"copyfileobj",interrupt)
        with pytest.raises(RuntimeError,match="publication interrupted"):
            compiler.compile(experience,**kwargs)
    assert set(library.skills.root.iterdir()) == before
    assert not list(library.skills.root.glob(".pending-*"))
    revision = compiler.compile(experience,**kwargs)
    assert library.read(revision)["metadata"]["checks"]["inverse-transition"]["passed"]
    assert compiler.compiler.calls == calls + 1


def test_counterexample_can_compile_a_new_version_without_reactivating_old_skill(tmp_path):
    result = run(tmp_path)
    library, memory = SkillLibrary(tmp_path / "library"), ResearchMemory(tmp_path / "memory")
    engine = ExperienceCompiler(library=library,compiler=ResponseCompiler(),backend=VectorBackend(),checker=InverseReplayCheck())
    repaired = engine.compile(result["counterexample_experience"],**compile_request(tmp_path / "recompile"))
    assert repaired != result["skill_revision"]
    revised = engine.install(repaired,memory=memory,memory_revision=result["suspended_memory_revision"],lesson_id="inverse-response-v2")
    snapshot = memory.read(revised)
    assert snapshot["parent"] == result["suspended_memory_revision"]
    assert snapshot["lessons"][LESSON]["status"] == "suspended"
    operation = library.bind(memory=memory,memory_revision=revised,lesson_id="inverse-response-v2",
                             context=CONDITIONS,backend=VectorBackend())
    assert operation(dict(position=[0.,0.],target=[3.,-6.]),Context("recompiled-candidate")) == pytest.approx([1.,-2.])
    with pytest.raises(SkillUnavailable):
        library.bind(memory=memory,memory_revision=revised,lesson_id=LESSON,context=CONDITIONS,backend=VectorBackend())


@pytest.mark.parametrize('boundary', ['backend', 'check', 'publication'])
def test_compilation_cannot_publish_a_mode_that_was_not_checked(tmp_path, monkeypatch, boundary):
    _, _, library, _, _, _, _, experience, _, _ = prepare(tmp_path / 'source')
    output = tmp_path / 'compilation'
    before = set(library.skills.root.iterdir())
    calls = []
    class Backend(VectorBackend):
        def build(self, root, entrypoint):
            operation = super().build(root, entrypoint)
            if boundary == 'backend':
                (root / entrypoint).chmod(0o755)
            return operation
    class Checker(InverseReplayCheck):
        def check(self, *args):
            calls.append('check')
            result = super().check(*args)
            if boundary == 'check':
                (output / 'bundle/program.json').chmod(0o755)
            return result
    capture = library.skills.capture
    def changed(*args, **kwargs):
        if boundary == 'publication':
            (output / 'bundle/program.json').chmod(0o755)
        return capture(*args, **kwargs)
    monkeypatch.setattr(library.skills, 'capture', changed)
    engine = ExperienceCompiler(library=library, compiler=ResponseCompiler(), backend=Backend(), checker=Checker())
    with pytest.raises(ValueError, match='modes'):
        engine.compile(experience, **compile_request(output))
    assert calls == ([] if boundary == 'backend' else ['check'])
    assert set(library.skills.root.iterdir()) == before
    assert not (output / 'result.json').exists()


def test_completed_check_cannot_be_reused_after_mode_drift_on_restart(tmp_path, monkeypatch):
    _, _, library, _, _, _, _, experience, _, _ = prepare(tmp_path / 'source')
    provider = ResponseCompiler()
    calls = []
    class Checker(InverseReplayCheck):
        def check(self, *args):
            calls.append('check')
            return super().check(*args)
    engine = ExperienceCompiler(library=library, compiler=provider, backend=VectorBackend(), checker=Checker())
    output = tmp_path / 'compilation'
    def interrupted(*args, **kwargs):
        raise RuntimeError('publication interrupted after completed check')
    with monkeypatch.context() as patch:
        patch.setattr(library.skills, 'capture', interrupted)
        with pytest.raises(RuntimeError, match='publication interrupted'):
            engine.compile(experience, **compile_request(output))
    path = output / 'bundle/program.json'
    expected = read_json(output / 'bundle.json')['file_modes']['program.json']
    path.chmod(0o755)
    with pytest.raises(ValueError):
        engine.compile(experience, **compile_request(output))
    assert calls == ['check'] and provider.calls == 1
    path.chmod(expected)
    revision = engine.compile(experience, **compile_request(output))
    record = library.read(revision)
    assert record['metadata']['schema'] == 'physicalrsi.compiled-skill/v2'
    assert record['metadata']['bundle'] == dict(files=record['files'], file_modes=record['file_modes'])
    assert calls == ['check'] and provider.calls == 1


@pytest.mark.parametrize('kind', ['fifo', 'symlink_directory'])
def test_nonregular_compiler_outputs_are_rejected_before_backend_execution(tmp_path, kind):
    _, _, library, _, _, _, _, experience, _, _ = prepare(tmp_path / 'source')
    class Compiler(ResponseCompiler):
        def compile(self, values, request, output):
            draft = super().compile(values, request, output)
            if kind == 'fifo':
                os.mkfifo(output / 'undeclared.pipe')
            else:
                (output / 'alias').symlink_to(output, target_is_directory=True)
            return draft
    class Backend(VectorBackend):
        def build(self, *args):
            raise AssertionError('Nonregular bundle reached backend')
    engine = ExperienceCompiler(library=library, compiler=Compiler(), backend=Backend(), checker=InverseReplayCheck())
    with pytest.raises(ValueError, match='non-regular|physical'):
        engine.compile(experience, **compile_request(tmp_path / 'compilation'))


def test_v2_skill_read_requires_its_checked_bundle_and_preserves_v1_reading(tmp_path):
    _, _, library, _, _, _, _, _, revision, _ = prepare(tmp_path / 'source')
    record = library.read(revision)
    folder = library.skills.folder(revision) / 'files'
    sources = {name: dict(path=str(folder / name), sha256=sha) for name, sha in record['files'].items()}
    changed = deepcopy(record['metadata'])
    changed['bundle']['file_modes']['program.json'] ^= 0o100
    bad = library.skills.capture(changed, sources)
    with pytest.raises(ValueError, match='checked bundle'):
        library.read(bad)
    legacy = deepcopy(record['metadata'])
    legacy['schema'] = 'physicalrsi.compiled-skill/v1'
    del legacy['bundle']
    old = library.skills.capture(legacy, sources, bind_modes=False)
    assert library.read(old)['metadata'] == legacy


def test_withdrawing_one_skill_preserves_independent_skill_and_historical_memory(tmp_path, monkeypatch):
    from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
    from PhysicalRSI_core.trials import Trial
    from PhysicalRSI_core.self_harness.artifacts import verify_harness
    from PhysicalRSI_demos.rollout_evolution import Verifier
    from PhysicalRSI_demos.skill_transfer import SkillPolicy

    resets = []
    reset = Environment.reset

    def observed(self, case, context):
        resets.append((self.task, case))
        return reset(self, case, context)

    monkeypatch.setattr(Environment, 'reset', observed)

    primary = tmp_path / 'primary'
    _, initial, library, memory, engine, _, _, _, _, _ = prepare(primary, gain=2.)
    _, _, source_library, _, _, _, _, source_experience, _, _ = prepare(tmp_path / 'independent', gain=.25)
    source_record = source_library.experiences.read(source_experience)
    source_files = source_library.experiences.folder(source_experience) / 'files'
    imported = library.experiences.capture(source_record['metadata'],
        {name: dict(path=str(source_files / name), sha256=sha) for name, sha in source_record['files'].items()})
    conditions = dict(CONDITIONS, actuator_family='independent-linear')
    independent = engine.compile(imported, name=LESSON, conditions=conditions,
        required_checks=['inverse-transition'], output=tmp_path / 'independent-compilation')
    # Start the ordinary transfer/withdrawal campaign with both skills pinned.
    install = ExperienceCompiler.install

    def install_both(self, skill_revision, *, memory, memory_revision, lesson_id):
        revision = install(self, skill_revision, memory=memory, memory_revision=memory_revision, lesson_id=lesson_id)
        if self.library.root == library.root and lesson_id == LESSON:
            revision = install(self, independent, memory=memory, memory_revision=revision, lesson_id='independent-response')
        return revision

    monkeypatch.setattr(ExperienceCompiler, 'install', install_both)
    result = run(primary, gain=2.)
    before = memory.read(result['memory_revision'])
    after = memory.read(result['suspended_memory_revision'])
    assert after['parent'] == result['memory_revision']
    assert before['lessons'][LESSON]['status'] == 'active'
    assert after['lessons'][LESSON]['status'] == 'suspended'
    assert after['lessons']['independent-response'] == before['lessons']['independent-response']
    assert memory.retrieve(result['suspended_memory_revision'], conditions).keys() == {'independent-response'}
    assert memory.retrieve(result['suspended_memory_revision'], CONDITIONS) == {}
    # Rebinding under both memory revisions preserves the independent dynamics.
    runtime = ExperimentRuntime(tmp_path / 'retention-trials')
    for name, revision in [('historical', result['memory_revision']), ('current', result['suspended_memory_revision'])]:
        operation = library.bind(memory=memory, memory_revision=revision, lesson_id='independent-response',
                                 context=conditions, backend=VectorBackend())
        policy = SkillPolicy('vector', verify_harness(initial), operation, independent, revision, [])
        evaluation = runtime.evaluate(Trial(name, task='vector', case=dict(initial=[0., 0.], target=[7., -11.]),
            scope='software', budget=Budget(1, 30)), environment=Environment('vector', .25, []),
            policy=policy, verifier=Verifier()).to_dict()
        assert evaluation['outcome'] == 'success'
        assert evaluation['verdict']['measurements']['squared_error'] == 0.
        dependencies = evaluation['receipt']['binding']['system1']['dependencies']
        assert any(d['kind'] == 'skill' and d['revision'] == independent for d in dependencies)
        assert any(d['kind'] == 'memory' and d['revision'] == revision for d in dependencies)
    with pytest.raises(SkillUnavailable):
        library.bind(memory=memory, memory_revision=result['suspended_memory_revision'], lesson_id=LESSON,
                     context=CONDITIONS, backend=VectorBackend())
    with pytest.raises(SkillUnavailable):
        library.bind(memory=memory, memory_revision=result['suspended_memory_revision'], lesson_id='independent-response',
                     context=CONDITIONS, backend=VectorBackend())
    assert memory.read(result['memory_revision']) == before
    assert len(resets) == 22
    assert run(primary, gain=2.) == result
    assert len(resets) == 22


def test_compilation_uses_frozen_checks_when_caller_configuration_changes(tmp_path):
    _, _, library, _, _, _, _, experience, _, _ = prepare(tmp_path / 'source-study')
    request = compile_request(tmp_path / 'changed-caller-settings')
    frozen_checks = list(request['required_checks'])

    class CallerSettingsChanged(InverseReplayCheck):
        def check(self, operation, values, supplied):
            reports = super().check(operation, values, supplied)
            request['required_checks'].append('next-compilation-check')
            return reports

    compiler = ResponseCompiler()
    engine = ExperienceCompiler(library=library, compiler=compiler, backend=VectorBackend(),
                                checker=CallerSettingsChanged())
    revision = engine.compile(experience, **request)
    record = library.read(revision)['metadata']
    assert record['request']['required_checks'] == frozen_checks
    assert set(record['checks']) == set(frozen_checks)
    assert request['required_checks'] == frozen_checks + ['next-compilation-check']
    # Resume the original request; later caller settings belong to a new run.
    restored = dict(request, required_checks=frozen_checks)
    assert engine.compile(experience, **restored) == revision
    assert compiler.calls == 1
    with pytest.raises(ValueError, match='changed|Changed'):
        engine.compile(experience, **request)
