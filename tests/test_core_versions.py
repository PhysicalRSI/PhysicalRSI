from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState, StateConflict
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, HarnessSnapshots, verify_harness
from test_improvement_campaign import manifest, setup


def seed(root):
    for kind in CLOSURE:
        atomic_json(root / (kind + ".json"), {"component": kind})
    return manifest(root, "seed")


def test_snapshot_survives_source_loss_and_does_not_alias_files(tmp_path):
    original = seed(tmp_path / "source")
    state = HarnessState(tmp_path / "state")
    revision = state.initialize(original, policy={}, scope="software")
    assert Path(revision["harness"]["root"]) != Path(original["root"])
    (Path(original["root"]) / "control.json").write_text("changed")
    shutil.rmtree(original["root"])
    assert state.resolve() == revision
    assert verify_harness(revision["harness"]) == revision["freeze_sha256"]


def test_damaged_published_snapshot_is_never_silently_overwritten(tmp_path):
    source = seed(tmp_path / "source")
    snapshots = HarnessSnapshots(tmp_path / "snapshots")
    captured = snapshots.capture(source)
    (Path(captured["root"]) / "control.json").write_text("damaged")
    with pytest.raises(ValueError, match="Changed or missing"):
        snapshots.capture(source)


def test_snapshot_interruption_does_not_publish_partial_state(tmp_path, monkeypatch):
    source = seed(tmp_path / "source")
    state = HarnessState(tmp_path / "state")
    real_copy = shutil.copyfile

    def broken(*args, **kwargs):
        real_copy(*args, **kwargs)
        raise RuntimeError("disk interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", broken)
        with pytest.raises(RuntimeError, match="disk interrupted"):
            state.initialize(source, policy={}, scope="software")
    assert not (state.root / "current.json").exists()
    assert not list(state.snapshots.root.glob(".snapshot-*"))
    assert state.initialize(source, policy={}, scope="software")["action"] == "bootstrap"


@pytest.mark.parametrize("damage", ["artifact", "evidence"])
def test_rollback_escapes_damaged_active_version_and_is_idempotent(tmp_path, damage):
    campaign, initial, _, _ = setup(tmp_path, max_rounds=1)
    selected = campaign.run(initial)["current"]
    old = selected["previous"]
    if damage == "artifact":
        path = Path(selected["harness"]["root"]) / "control.json"
    else:
        path = Path(next(iter(selected["evidence"])))
    path.write_text("corrupted")
    with pytest.raises(ValueError):
        campaign.state.resolve()
    restored = campaign.state.rollback(selected["revision"], reason="regression investigation")
    assert restored["rollback_to"] == old and restored["harness"]["id"] == initial["id"]
    assert campaign.state.rollback(selected["revision"], reason="regression investigation") == restored
    with pytest.raises(StateConflict):
        campaign.state.rollback(selected["revision"], reason="another operation")


def test_rollback_rejects_bad_target_and_corrupted_history(tmp_path):
    campaign, initial, _, _ = setup(tmp_path, max_rounds=1)
    selected = campaign.run(initial)["current"]
    state = campaign.state
    other = HarnessState(tmp_path / "other").initialize(initial, policy={"other": True}, scope="software")
    with pytest.raises(ValueError, match="not an ancestor"):
        state.rollback(selected["revision"], target=other["revision"])
    assert state.resolve() == selected
    path = state.root / "history" / (selected["previous"] + ".json")
    value = read_json(path)
    value["scope"] = "changed"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="Lineage record changed"):
        state.rollback(selected["revision"])
    assert state.resolve() == selected


def test_component_boundary_uses_actual_files_and_keeps_model_fixed(tmp_path):
    parent = seed(tmp_path / "seed")
    child = deepcopy(parent)
    child["root"] = str(tmp_path / "child")
    shutil.copytree(parent["root"], child["root"])
    path = Path(child["root"]) / "control.json"
    atomic_json(path, {"gain": 0.5})
    child["components"]["control"]["control.json"] = file_digest(path)
    assert EvolutionSpec(("control",)).compare(parent, child) == {"control": ["control.json"]}
    with pytest.raises(ValueError, match="fixed evolution components"):
        EvolutionSpec(("prompts",)).compare(parent, child)
    with pytest.raises(ValueError, match="model state stays fixed"):
        EvolutionSpec(("weights",))


def test_consecutive_rollbacks_traverse_generations_without_toggling(tmp_path):
    # Build just the history-pointer structure; selection itself is covered by
    # the real campaign tests above. No evaluation result is asserted here.
    state = HarnessState(tmp_path / "state")
    initial = state.initialize(seed(tmp_path / "source"), policy={}, scope="software")
    body = {key: value for key, value in initial.items() if key != "revision"}
    second = state._publish(dict(body, previous=initial["revision"], action="selection"))
    third = state._publish(dict(body, previous=second["revision"], action="selection"))
    undo_third = state.rollback(third["revision"])
    assert undo_third["rollback_to"] == second["revision"]
    undo_second = state.rollback(undo_third["revision"])
    assert undo_second["rollback_to"] == initial["revision"]
    with pytest.raises(ValueError, match="No previous harness"):
        state.rollback(undo_second["revision"])


def test_rollback_after_pointer_write_interruption_reuses_revision(tmp_path, monkeypatch):
    from PhysicalRSI_core import lineage
    campaign, initial, _, _ = setup(tmp_path, max_rounds=1)
    current = campaign.run(initial)["current"]
    real_write = lineage.atomic_json

    def interrupted(path, value):
        real_write(path, value)
        if Path(path) == campaign.state.root / "current.json":
            raise RuntimeError("stopped after pointer publication")

    with monkeypatch.context() as patch:
        patch.setattr(lineage, "atomic_json", interrupted)
        with pytest.raises(RuntimeError, match="pointer publication"):
            campaign.state.rollback(current["revision"], reason="observed regression")
    published = campaign.state.resolve()
    assert campaign.state.rollback(current["revision"], reason="observed regression") == published


def test_self_harness_rejects_fixed_change_before_candidate_evaluation(tmp_path, monkeypatch):
    from test_improvement_campaign import Proposal
    campaign, initial, suite, quota = setup(tmp_path, max_rounds=1)
    propose = Proposal.propose

    def edit_weights(self, *args):
        children = propose(self, *args)
        path = Path(children[0]["root"]) / "weights.json"
        atomic_json(path, {"checkpoint": "unreviewed"})
        children[0]["components"]["weights"]["weights.json"] = file_digest(path)
        return children

    monkeypatch.setattr(Proposal, "propose", edit_weights)
    with pytest.raises(ValueError, match="fixed evolution components"):
        campaign.run(initial)
    assert suite.resets == [3] and quota.status()["reserved_trials"] == 1
    assert campaign.state.resolve()["action"] == "bootstrap"
