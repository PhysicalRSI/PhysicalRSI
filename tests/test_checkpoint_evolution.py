from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.checkpoints import CheckpointStore
from PhysicalRSI_core.contracts import Context, ReconciliationRequired
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.training import (CheckpointEvolutionSpec, TrainingCoordinator,
                                                   TrainingPending, TrainingRejected)
from PhysicalRSI_demos.checkpoint_evolution import Backend, Provider, model_spec, run, setup


def pending(root):
    campaign, initial, suite, quota, provider = setup(root)
    with pytest.raises(TrainingPending):
        campaign.run(initial)
    directory = root / "campaign/rounds/round-0001/execution/training-proposal/job"
    request = read_json(directory / "request.json")
    coordinator = TrainingCoordinator(store=CheckpointStore(root / "models"),provider=provider,backend=Backend())
    return campaign, initial, suite, quota, provider, coordinator, directory, request


def test_actual_gradient_updates_recurse_preserve_tasks_restart_and_rollback(tmp_path):
    result = run(tmp_path)
    research = result["research"]
    assert [r["outcome"] for r in research["rounds"]] == ["inherit_child", "inherit_child"]
    assert all(m["successes"] == m["planned"] == 2 for m in result["final_test"]["metrics"].values())
    assert result["quota"]["reserved_trials"] == 52
    assert len(result["training_jobs"]["jobs"]) == 2 and result["training_jobs"]["used"] == {"cpu":0}
    store = CheckpointStore(tmp_path / "models")
    chosen = store.binding(research["current"]["harness"],model=model_spec())
    second = chosen["record"]["metadata"]["training"]
    first = store.read(second["request"]["parent_checkpoint"])["metadata"]["training"]
    assert second["request"]["parent_checkpoint"] != first["request"]["parent_checkpoint"]
    for training in (first, second):
        assert training["result"]["costs"]["updates"] == 1
        assert training["result"]["role"] == "training_only"
        values = store.experiences.read(training["request"]["experience_revision"])
        assert len(values) == 16 and all(v["trial"]["system2"]["split"] == "development" for v in values)
    weights = read_json(Path(chosen["root"]) / "weights.json")["logits"]
    assert weights[0] == pytest.approx([.423883115234171, .7880584423829144, -.21194155761708547])
    assert max(range(3),key=weights[1].__getitem__) == 2
    campaign, initial, observed, quota, _ = setup(tmp_path)
    assert campaign.run(initial) == research
    assert campaign.report_test(initial) == result["final_test"] and observed.resets == []
    assert run(tmp_path) == result
    restored = campaign.state.rollback(research["current"]["revision"],reason="checkpoint rollback")
    operation = store.bind(restored["harness"],backend=Backend())
    assert operation.revision == second["request"]["parent_checkpoint"]
    assert max(range(3),key=operation(0,Context("restored-axis")).__getitem__) == 1
    assert max(range(3),key=operation(1,Context("restored-vector")).__getitem__) == 0
    assert campaign.state.rollback(research["current"]["revision"],reason="checkpoint rollback") == restored
    assert quota.status()["reserved_trials"] == 52


def test_default_weights_remain_frozen_and_explicit_spec_roundtrips(tmp_path):
    _, initial, _, _, provider, coordinator, directory, request = pending(tmp_path)
    provider.scheduler.work_once(provider.handler)
    revision = coordinator.advance(request,directory)
    store = coordinator.store
    child = store.install(initial,revision,tmp_path / "child",candidate_id="trained")
    with pytest.raises(ValueError,match="model state stays fixed"):
        EvolutionSpec(("weights",))
    with pytest.raises(ValueError,match="fixed evolution"):
        EvolutionSpec().compare(initial,child)
    spec = CheckpointEvolutionSpec(store,model_spec())
    assert EvolutionSpec.from_identity(spec.identity()).identity() == spec.identity()
    assert set(spec.compare(initial,child)) == {"weights"}
    assert set(spec.compare(child,initial)) == {"weights"}  # Real ancestry permits rollback.
    atomic_json(Path(child["root"]) / "control.json",dict(changed=True))
    child["components"]["control"]["control.json"] = file_digest(Path(child["root"]) / "control.json")
    with pytest.raises(ValueError,match="fixed evolution"):
        spec.compare(initial,child)


def test_lost_submission_reply_reconciles_original_job_without_resubmission(tmp_path):
    _, _, _, _, provider, _, _, request = pending(tmp_path)
    class LostReply(Provider):
        submissions = 0
        def submit(self, request, output):
            self.submissions += 1
            super().submit(request,output)
            raise OSError("lost acknowledgement")
    broken = LostReply(tmp_path)
    engine = TrainingCoordinator(store=CheckpointStore(tmp_path / "models"),provider=broken,backend=Backend())
    output = tmp_path / "lost-reply"
    with pytest.raises(OSError,match="acknowledgement"):
        engine.advance(request,output)
    with pytest.raises(TrainingPending):
        engine.advance(request,output)
    provider.scheduler.work_once(provider.handler)
    revision = engine.advance(request,output)
    assert broken.submissions == 1 and len(provider.scheduler.status()["jobs"]) == 1
    assert engine.advance(request,output) == revision


@pytest.mark.parametrize("recovery", ["absent", "unknown"])
def test_unknown_submission_is_never_automatically_repeated(tmp_path,recovery):
    _, _, _, _, _, _, _, request = pending(tmp_path)
    class Unknown(Provider):
        submissions = 0
        def submit(self, request, output):
            self.submissions += 1
            raise OSError("unknown submission")
        recover_submission = None if recovery == "absent" else lambda *args: None
    provider = Unknown(tmp_path)
    engine = TrainingCoordinator(store=CheckpointStore(tmp_path / "models"),provider=provider,backend=Backend())
    with pytest.raises(OSError):
        engine.advance(request,tmp_path / "unknown")
    for _ in range(2):
        with pytest.raises(ReconciliationRequired):
            engine.advance(request,tmp_path / "unknown")
    assert provider.submissions == 1


@pytest.mark.parametrize("problem", ["unknown", "running", "not-quiescent", "failed", "job", "request"])
def test_incomplete_or_misattributed_training_never_publishes_checkpoint(tmp_path,problem):
    _, _, _, _, provider, _, directory, request = pending(tmp_path)
    def poll(handle):
        status = dict(handle,state="completed",quiescent=True)
        if problem in {"unknown","running","failed"}: status["state"] = problem
        if problem == "not-quiescent": status["quiescent"] = False
        if problem == "job": status["job_id"] = "another-job"
        if problem == "request": status["request_sha256"] = "0"*64
        return status
    provider.poll = poll
    engine = TrainingCoordinator(store=CheckpointStore(tmp_path / "models"),provider=provider,backend=Backend())
    before = set(engine.store.checkpoints.root.iterdir())
    with pytest.raises((ValueError,ReconciliationRequired)):
        engine.advance(request,directory)
    assert set(engine.store.checkpoints.root.iterdir()) == before
    assert not (directory / "checkpoint.json").exists()


@pytest.mark.parametrize("problem", ["budget", "metrics-role", "files", "interface", "source"])
def test_invalid_training_output_or_source_is_rejected(tmp_path,problem):
    _, _, _, _, provider, _, directory, request = pending(tmp_path)
    provider.scheduler.work_once(provider.handler)
    original = provider.collect
    def collect(handle):
        result = deepcopy(original(handle))
        if problem == "budget": result["costs"]["updates"] = 2
        if problem == "metrics-role": result["role"] = "validation"
        if problem == "files": result["files"]["missing.bin"] = "0"*64
        return result
    provider.collect = collect
    if problem == "interface":
        request["model"]["architecture"]["features"] = ["changed"]
    if problem == "source":
        source = next((tmp_path / "models/experience").rglob("trajectory.json"))
        source.write_text("corrupt")
    engine = TrainingCoordinator(store=CheckpointStore(tmp_path / "models"),provider=provider,backend=Backend())
    with pytest.raises((ValueError,FileNotFoundError)):
        engine.advance(request,directory)
    assert not (directory / "checkpoint.json").exists()


def test_owned_checkpoint_and_experience_survive_source_loss(tmp_path):
    _, initial, _, _, provider, engine, directory, request = pending(tmp_path)
    provider.scheduler.work_once(provider.handler)
    revision = engine.advance(request,directory)
    child = engine.store.install(initial,revision,tmp_path / "child",candidate_id="trained")
    for name in ("campaign", "training-jobs", "seed", "seed-input", "seed-model"):
        shutil.rmtree(tmp_path / name)
    assert len(engine.store.experiences.read(request["experience_revision"])) == 16
    operation = engine.store.bind(child,backend=Backend())
    assert operation.revision == revision and operation(0,Context("source-loss"))[1] > .4
    path = Path(engine.store.binding(child)["root"]) / "weights.json"
    original = engine.store.checkpoints.folder(revision) / "files/weights.json"
    assert path.stat().st_ino != original.stat().st_ino
    path.write_text("changed")
    with pytest.raises(ValueError,match="Changed|changed"):
        engine.store.bind(child,backend=Backend())
    # Already loaded backend owns its exact immutable values.
    assert operation(0,Context("loaded-version"))[1] > .4


def test_training_source_cannot_be_rebound_to_another_parent_model(tmp_path):
    _, _, _, _, _, engine, directory, request = pending(tmp_path)
    request["parent_harness_sha256"] = "0"*64
    with pytest.raises(ValueError,match="frozen parent"):
        engine.advance(request,tmp_path / "other-training")
    assert not (tmp_path / "other-training/submission.json").exists()


def test_unknown_worker_effect_holds_resources_until_explicit_reconciliation(tmp_path):
    _, _, _, _, provider, engine, directory, request = pending(tmp_path)
    handler = provider.handler
    execute = handler.execute
    def lost_result(payload):
        execute(payload)
        raise OSError("worker lost after writing its result")
    handler.execute = lost_result
    with pytest.raises(OSError):
        provider.scheduler.work_once(handler)
    status = provider.scheduler.status()
    assert status["used"] == {"cpu":1}
    with pytest.raises(ReconciliationRequired):
        engine.advance(request,directory)
    key, = status["jobs"]
    provider.scheduler.recover_completed(key,handler=handler)
    assert engine.advance(request,directory)
    assert provider.scheduler.status()["used"] == {"cpu":0}


def test_large_training_reward_cannot_promote_an_unimproved_checkpoint(tmp_path):
    campaign, initial, _, _, provider, _, _, request = pending(tmp_path)
    handler = provider.handler
    execute = handler.execute
    def misleading_training(payload):
        result = execute(payload)
        atomic_json(Path(result["root"]) / "weights.json",dict(logits=[[3.,0.,0.],[3.,0.,0.]]))
        result["files"]["weights.json"] = file_digest(Path(result["root"]) / "weights.json")
        result["metrics"]["mean_sample_reward"] = 1000000.
        atomic_json(tmp_path / "training-jobs" / digest(payload) / "result.json",result)
        return result
    handler.execute = misleading_training
    provider.scheduler.work_once(handler)
    with pytest.raises(TrainingPending):  # Next round, after rejecting the first model.
        campaign.run(initial)
    selected = read_json(tmp_path / "campaign/rounds/round-0001/execution/selection.json")["output"]
    assert selected["survivor_id"] == "untrained"
    assert campaign.state.resolve()["harness"]["id"] == "untrained"
    assert all(m["normalized_score"] == 0 for m in selected["metrics"].values())


def test_imported_seed_cannot_bypass_training_transition(tmp_path):
    _, initial, _, _, _ = setup(tmp_path)
    store = CheckpointStore(tmp_path / "models")
    other = tmp_path / "another-model"
    atomic_json(other / "weights.json",dict(logits=[[0.,1.,0.],[0.,0.,1.]]))
    revision = store.publish(model=model_spec(),backend=Backend(),root=other,entrypoint="weights.json",
                             files={"weights.json":file_digest(other / "weights.json")})
    child = store.install(initial,revision,tmp_path / "other-candidate",candidate_id="untrained-import")
    with pytest.raises(ValueError,match="training provenance"):
        CheckpointEvolutionSpec(store,model_spec()).compare(initial,child)


def test_terminal_training_failure_is_durable_and_does_not_collect(tmp_path):
    _, _, _, _, provider, _, directory, request = pending(tmp_path)
    calls = []
    def failed(handle):
        calls.append(handle)
        return dict(handle,state="failed",quiescent=True)
    provider.poll = failed
    provider.collect = lambda handle: pytest.fail("Failed job must not collect a model")
    engine = TrainingCoordinator(store=CheckpointStore(tmp_path / "models"),provider=provider,backend=Backend())
    for _ in range(2):
        with pytest.raises(TrainingRejected):
            engine.advance(request,directory)
    assert len(calls) == 1 and not (directory / "checkpoint.json").exists()


def test_published_candidate_manifest_cannot_change_on_install_resume(tmp_path):
    _, initial, _, _, _ = setup(tmp_path)
    store = CheckpointStore(tmp_path / "models")
    revision = store.binding(initial)["revision"]
    destination = tmp_path / "installed"
    child = store.install(initial,revision,destination,candidate_id="copy")
    assert child['file_modes'] == initial['file_modes']
    assert store.install(initial,revision,destination,candidate_id="copy") == child
    child["id"] = "substituted"
    atomic_json(destination / "manifest.json",child)
    with pytest.raises(ValueError,match="installation request"):
        store.install(initial,revision,destination,candidate_id="copy")


@pytest.mark.parametrize('damage', ['source-package', 'weights-binding', 'fixed-input'])
def test_checkpoint_permission_drift_is_rejected_before_use_or_resume(tmp_path, damage):
    _, initial, _, _, _ = setup(tmp_path)
    store = CheckpointStore(tmp_path / 'models')
    binding = store.binding(initial)
    revision = binding['revision']
    child = store.install(initial, revision, tmp_path / 'copy', candidate_id='copy')
    if damage == 'source-package':
        path = store.checkpoints.folder(revision) / 'files' / binding['record']['metadata']['entrypoint']
    else:
        path = Path(child['root']) / ('weights.json' if damage == 'weights-binding' else 'tools.json')
    path.chmod(0o755)
    with pytest.raises(ValueError, match='file mode'):
        store.binding(child)
    with pytest.raises(ValueError, match='file mode'):
        store.install(initial, revision, tmp_path / 'copy', candidate_id='copy')


@pytest.mark.parametrize('recover', [False, True])
def test_training_provenance_owns_provider_replies_across_callbacks(tmp_path, recover):
    _, _, _, _, original, _, _, request = pending(tmp_path)
    original.scheduler.work_once(original.handler)

    class MutableReplies(Provider):
        submissions = 0

        def submit(self, request, output):
            self.submissions += 1
            self.handle = dict(super().submit(request, output), notes=dict(stage='submitted'))
            if recover:
                raise RuntimeError('lost submission reply')
            return self.handle

        def recover_submission(self, request, output):
            return self.handle

        def poll(self, handle):
            self.status = super().poll(handle)
            self.handle['notes']['stage'] = 'later provider progress'
            return self.status

        def collect(self, handle):
            self.status['notes']['stage'] = 'later collection progress'
            return super().collect(handle)

    provider = MutableReplies(tmp_path)
    coordinator = TrainingCoordinator(store=CheckpointStore(tmp_path / 'models'), provider=provider, backend=Backend())
    output = tmp_path / 'mutable-replies'
    if recover:
        with pytest.raises(RuntimeError, match='lost submission reply'):
            coordinator.advance(request, output)
    revision = coordinator.advance(request, output)
    training = coordinator.store.read(revision)['metadata']['training']
    submission = read_json(output / 'submission.json')
    terminal = read_json(output / 'terminal.json')
    assert training['handle'] == submission['handle']
    assert training['status'] == terminal
    assert training['handle']['notes']['stage'] == 'submitted'
    assert training['status']['notes']['stage'] == 'submitted'
    assert provider.handle['notes']['stage'] == 'later provider progress'
    assert provider.status['notes']['stage'] == 'later collection progress'
    assert coordinator.advance(request, output) == revision
    assert provider.submissions == 1 and len(provider.scheduler.status()['jobs']) == 1


@pytest.mark.parametrize('recover', [False, True])
def test_training_candidate_owns_planner_reply_across_provider_callbacks(tmp_path, monkeypatch, recover):
    from PhysicalRSI_demos.checkpoint_evolution import Planner

    campaign, initial, _, quota, provider = setup(tmp_path)
    original_plan, original_submit, original_collect = Planner.plan, Provider.submit, Provider.collect
    replies, published = [], []
    original_install = CheckpointStore.install
    lost = False

    def plan(self, *args):
        nonlocal lost
        result = original_plan(self, *args)
        replies.append(result)
        if recover and not lost:
            lost = True
            raise RuntimeError('lost planner reply')
        return result

    def submit(self, request, output):
        handle = original_submit(self, request, output)
        self.scheduler.work_once(self.handler)
        return handle

    def collect(self, handle):
        result = original_collect(self, handle)
        replies[-1]['required_checks'].append('later planner diagnostics')
        replies[-1]['hypothesis'] = 'later planner hypothesis'
        return result

    def install(self, parent, revision, output, **kwargs):
        child = original_install(self, parent, revision, output, **kwargs)
        if 'hypothesis' in kwargs:
            frozen = read_json(Path(output).parent / 'plan.json')['output']
            for key in ('hypothesis', 'expected_effect', 'required_checks', 'used_lessons'):
                assert child[key] == frozen[key]
            published.append(child)
        return child

    monkeypatch.setattr(Planner, 'plan', plan)
    monkeypatch.setattr(Planner, 'resume_plan', plan)
    monkeypatch.setattr(Provider, 'submit', submit)
    monkeypatch.setattr(Provider, 'collect', collect)
    monkeypatch.setattr(CheckpointStore, 'install', install)
    if recover:
        with pytest.raises(RuntimeError, match='lost planner reply'):
            campaign.run(initial)
        assert provider.scheduler.status()['jobs'] == {}
    research = campaign.run(initial)
    assert [row['outcome'] for row in research['rounds']] == ['inherit_child', 'inherit_child']
    assert len(published) == 2
    assert replies[-1]['hypothesis'] == 'later planner hypothesis'
    reserved = quota.status()['reserved_trials']
    assert campaign.run(initial) == research
    assert quota.status()['reserved_trials'] == reserved
    assert len(provider.scheduler.status()['jobs']) == 2


@pytest.mark.parametrize('recover', [False, True])
def test_changed_planner_identity_is_rejected_before_training(tmp_path, monkeypatch, recover):
    from PhysicalRSI_demos.checkpoint_evolution import Planner

    identity = dict(Planner().identity(), settings=dict(version=1))
    original_plan = Planner.plan
    lost = False

    def plan(self, *args):
        nonlocal lost
        if recover and not lost:
            lost = True
            raise RuntimeError('lost planner reply')
        result = original_plan(self, *args)
        identity['settings']['version'] = 2
        return result

    monkeypatch.setattr(Planner, 'identity', lambda self: identity)
    monkeypatch.setattr(Planner, 'plan', plan)
    monkeypatch.setattr(Planner, 'resume_plan', plan)
    campaign, initial, _, quota, provider = setup(tmp_path)
    path = tmp_path / 'campaign/rounds/round-0001/execution/training-proposal/plan.json'
    if recover:
        with pytest.raises(RuntimeError, match='lost planner reply'):
            campaign.run(initial)
        started = read_json(path)
    with pytest.raises(ValueError, match='Training planner changed'):
        campaign.run(initial)
    assert read_json(path)['state'] == 'started'
    if recover:
        assert read_json(path) == started
    assert provider.scheduler.status()['jobs'] == {}
    assert quota.status()['reserved_trials'] == 16
    assert not (path.parent / 'job').exists()


@pytest.mark.parametrize('receipt', ['submission.json', 'terminal.json', 'collection.json'])
@pytest.mark.parametrize('damage', ['changed', 'missing'])
def test_completed_training_resume_checks_local_receipts(tmp_path, monkeypatch, receipt, damage):
    _, _, _, _, provider, coordinator, directory, request = pending(tmp_path)
    provider.scheduler.work_once(provider.handler)
    revision = coordinator.advance(request, directory)
    original = coordinator.store.read(revision)
    path = directory / receipt
    saved = read_json(path)

    def unexpected(*args):
        pytest.fail('Completed checkpoint recovery must not invoke the provider')

    for method in ('submit', 'recover_submission', 'poll', 'collect'):
        monkeypatch.setattr(provider, method, unexpected)
    assert coordinator.advance(request, directory) == revision
    if damage == 'missing':
        path.unlink()
    else:
        changed = deepcopy(saved)
        if receipt == 'submission.json':
            changed['handle']['job_id'] = 'another-job'
            changed['handle_sha256'] = digest(changed['handle'])
        elif receipt == 'terminal.json':
            changed['state'] = 'failed'
        else:
            changed['metrics']['unrecorded_reward'] = 100
        atomic_json(path, changed)
    with pytest.raises(ValueError, match='Published training receipt'):
        coordinator.advance(request, directory)
    assert coordinator.store.read(revision) == original
    atomic_json(path, saved)
    assert coordinator.advance(request, directory) == revision
    assert len(provider.scheduler.status()['jobs']) == 1
