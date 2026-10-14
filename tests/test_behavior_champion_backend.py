"""Frozen source and transport checks without model dependencies."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from PhysicalRSI_baselines.behavior_baseline import champion_backend as backend
from PhysicalRSI_baselines.behavior_baseline.openpi_backend import supported_tasks, select_task
from PhysicalRSI_core.infra.storage import file_digest
from PhysicalRSI_core.contracts import Context


def test_champion_execution_mode_is_fixed_per_episode_and_resets(monkeypatch):
    calls = []
    wrapper = SimpleNamespace(config=SimpleNamespace(actions_to_execute=26),
                              reset=lambda: calls.append("reset"))
    wrapper.act = lambda observation: calls.append(wrapper.config.actions_to_execute) or observation
    policy = backend.ChampionBackend.__new__(backend.ChampionBackend)
    policy.wrapper, policy.task, policy._episode_execution_mode = wrapper, "turning_on_radio", None
    monkeypatch.setattr(backend, "champion_observation", lambda observation: observation)
    context = Context("timing-fixture")
    assert policy.act({"public": True}, context) == {"public": True}
    assert calls == [26]
    with pytest.raises(ValueError, match="cannot change"):
        policy.act_with_execution({}, context, execution_mode="uncompressed")
    assert calls == [26]
    policy.reset("turning_on_radio")
    policy.act_with_execution({}, context, execution_mode="uncompressed")
    policy.act_with_execution({}, context, execution_mode="uncompressed")
    assert calls[-2:] == [20, 20]
    with pytest.raises(ValueError, match="cannot change"):
        policy.act({}, context)
    policy.reset("turning_on_radio")
    assert wrapper.config.actions_to_execute == 26
    with pytest.raises(ValueError, match="Unknown"):
        policy.act_with_execution({}, context, execution_mode="arbitrary")
    policy.act({}, context)
    assert calls[-1] == 26


def test_champion_task_binding_cannot_silently_expand():
    spec = dict(kind=backend.KIND, task="turning_on_radio", task_id=0, checkpoint_name="checkpoint_2")
    assert select_task(spec) == "turning_on_radio"
    for changed in (dict(task="picking_up_trash"), dict(task_id=True), dict(checkpoint_name="checkpoint_1")):
        with pytest.raises(ValueError):
            supported_tasks(dict(spec, **changed))


def test_checkpoint_two_tasks_reject_cross_task_ids_and_other_specialists():
    expected = {
        "turning_on_radio": 0, "picking_up_trash": 1, "picking_up_toys": 7,
        "rearranging_kitchen_furniture": 8, "putting_up_Christmas_decorations_inside": 9,
        "preparing_lunch_box": 12, "moving_boxes_to_storage": 16,
        "bringing_water": 17, "tidying_bedroom": 18, "sorting_vegetables": 20,
        "collecting_childrens_toys": 21, "putting_shoes_on_rack": 22,
        "assembling_gift_baskets": 26, "setting_the_fire": 30,
        "slicing_vegetables": 43, "cook_hot_dogs": 45,
    }
    assert backend.CHECKPOINT_2_TASKS == expected
    for task, task_id in expected.items():
        spec = dict(kind=backend.KIND, task=task, task_id=task_id, checkpoint_name="checkpoint_2")
        assert supported_tasks(spec) == frozenset([task])
        assert select_task(spec, task) == task
        for other_id in expected.values():
            if other_id != task_id:
                with pytest.raises(ValueError):
                    supported_tasks(dict(spec, task_id=other_id))
    for task, task_id in (("make_microwave_popcorn", 40), ("new_task", 50)):
        with pytest.raises(ValueError):
            supported_tasks(dict(spec, task=task, task_id=task_id))


def test_seed_rejects_unassigned_task_before_reading_weights(tmp_path):
    from PhysicalRSI_baselines.behavior_baseline.champion_seed import materialize
    with pytest.raises(ValueError, match="not assigned"):
        materialize(tmp_path / "candidate", source=tmp_path, checkpoint=tmp_path,
                    utility=tmp_path, transport=tmp_path, environment_manifest=tmp_path,
                    task="make_microwave_popcorn")
    assert not (tmp_path / "candidate").exists()


def test_verify_detects_changed_weights_transport_and_policy(tmp_path, monkeypatch):
    source = tmp_path / "source"
    revisions = []
    for root in (source, source / "openpi"):
        root.mkdir(parents=True, exist_ok=True)
        (root / "policy.py").write_text("# fixture\n")
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        git("init")
        git("add", "policy.py")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
        revisions.append(git("rev-parse", "HEAD"))
    monkeypatch.setattr(backend, "SOURCE_REVISION", revisions[0])
    monkeypatch.setattr(backend, "OPENPI_REVISION", revisions[1])
    utility, transport = tmp_path / "utility.py", tmp_path / "transport.py"
    utility.write_text("# utility fixture\n")
    transport.write_text("# transport fixture\n")
    monkeypatch.setattr(backend, "UTILITY_SHA256", file_digest(utility))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights"
    weights.write_bytes(b"fixture")
    spec = dict(kind=backend.KIND, task="turning_on_radio", task_id=0, checkpoint_name="checkpoint_2",
                source_root=str(source), source_revision=revisions[0],
                adapter_sha256=file_digest(Path(backend.__file__)),
                projection_sha256=file_digest(Path(backend.__file__).with_name("champion_observation.py")),
                legacy_utility=str(utility), checkpoint_root=str(checkpoint),
                checkpoint_files={"weights": file_digest(weights)},
                transport_source=str(transport), transport_sha256=file_digest(transport))
    assert backend.verify(spec) == (source, checkpoint, utility)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        backend.verify(spec)
    weights.write_bytes(b"fixture")
    transport.write_text("# changed\n")
    with pytest.raises(ValueError, match="transport changed"):
        backend.verify(spec)
    transport.write_text("# transport fixture\n")
    (source / "openpi/policy.py").write_text("# changed\n")
    with pytest.raises(subprocess.CalledProcessError):
        backend.verify(spec)
