from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.infra.storage import digest, file_digest, read_json
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import EvolutionSpec, verify_harness
from PhysicalRSI_demos.branch_evolution import setup
from PhysicalRSI_demos.rollout_evolution import setup as calibration_setup


def archive_entries(root):
    archive = CandidateArchive(root / "archive")
    return archive, [archive.read(p.parent.name) for p in (archive.root / "entries").glob("*/record.json")]


def test_losing_branch_is_reused_with_new_parent_and_fresh_evidence(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path)
    result = campaign.run(initial)
    assert [r["outcome"] for r in result["rounds"]] == ["inherit_child", "inherit_child"]
    archive, entries = archive_entries(tmp_path)
    assert len(entries) == 2
    first = next(e for e in entries if "specialist-axis" in e["candidates"])
    loser = first["candidates"]["specialist-axis"]
    assert loser["disposition"] == "evaluated_not_selected"
    assert loser["metrics"]["per_task"] == {"axis": 1.0, "vector": 0.0}
    assert first["candidates"]["specialist-vector"]["disposition"] == "selected"
    directory = campaign.root / "rounds/round-0002/execution"
    child = read_json(directory / "propose.json")["output"][0]
    parent = read_json(directory / "parent.json")
    assert child["parent_sha256"] == parent["freeze_sha256"]
    assert child["archive_origin"]["freeze_sha256"] == loser["freeze_sha256"]
    assert child["archive_origin"]["candidate_id"] == "specialist-axis"
    assert child["parent_sha256"] != loser["harness"]["parent_sha256"]
    assert read_json(Path(child["root"]) / "control.json") == {"axis": .5, "vector": 2.}
    old_cases = {c for cases in first["cohort"]["layouts"].values() for c in cases}
    new_cases = {c for cases in read_json(directory / "validation.json")["output"]["layouts"].values() for c in cases}
    assert not old_cases & new_cases
    view = read_json(directory / "archive-view.json")
    assert view["revisions"] == [first["revision"]]
    before = archive.summaries(view)
    report = campaign.report_test(initial)
    assert all(m["success_rate"] == 1 for m in report["metrics"].values())
    assert len(suite.resets) == quota.status()["reserved_trials"] == 26
    assert archive.summaries(view) == before
    assert all("test" not in Path(name).parts for e in entries for name in e["files"])
    assert campaign.run(initial) == result
    assert campaign.report_test(initial) == report
    assert len(suite.resets) == 26
    assert len(archive_entries(tmp_path)[1]) == 2


def test_archive_owns_candidate_and_evidence_after_original_round_loss(tmp_path, monkeypatch):
    from PhysicalRSI_demos.rollout_evolution import Environment
    sensor = tmp_path / "adapter-sensor.bin"
    sensor.write_bytes(b"\x00\xffarchived sensor bytes" * 100)
    sensor_sha = file_digest(sensor)
    monkeypatch.setattr(Environment, "export_evidence", lambda self: {
        "sensor.bin": dict(path=str(sensor), sha256=sensor_sha)}, raising=False)
    campaign, initial, _, _ = setup(tmp_path)
    result = campaign.run(initial)
    archive, entries = archive_entries(tmp_path)
    first = next(e for e in entries if "specialist-axis" in e["candidates"])
    row = first["candidates"]["specialist-axis"]
    view = archive.view(scope=first["scope"], policy_sha256=first["policy_sha256"])
    original = Path(first["source_round"]) / "specialist-axis/control.json"
    owned = Path(row["harness"]["root"]) / "control.json"
    assert original.stat().st_ino != owned.stat().st_ino
    sensor.unlink()
    shutil.rmtree(campaign.root)
    assert archive.read(first["revision"]) == first
    retained = [name for name in first["files"] if name.endswith("sensor.bin")]
    assert retained
    assert all(file_digest(archive.root / "entries" / first["revision"] / "evidence" / name) == sensor_sha
               for name in retained)
    child = archive.fork(view, first["revision"], "specialist-axis", parent=result["current"]["harness"],
        output=tmp_path / "recovered-proposal", name="recovered", evolution=EvolutionSpec(("control",)))
    assert verify_harness(child) == row["freeze_sha256"]
    assert child['file_modes'] == row['harness']['file_modes']
    assert child["parent_sha256"] == result["current"]["freeze_sha256"]
    (Path(child["root"]) / "control.json").write_text("edited locally")
    assert archive.read(first["revision"]) == first


@pytest.mark.parametrize("damage", ["candidate", "mode", "evidence", "extra", "symlink"])
def test_archive_damage_is_detected_on_read_and_campaign_resume(tmp_path, damage):
    campaign, initial, _, _ = calibration_setup(tmp_path)
    campaign.run(initial)
    archive, entries = archive_entries(tmp_path)
    entry = entries[0]
    folder = archive.root / "entries" / entry["revision"] / "evidence"
    if damage in {"candidate", "mode"}:
        path = Path(next(iter(entry["candidates"].values()))["harness"]["root"]) / "control.json"
    elif damage == "extra":
        path = folder / "unexpected.bin"
    else:
        path = folder / next(name for name in entry["files"] if name.endswith("trajectory.json"))
    if damage == "mode":
        path.chmod(0o755)
    elif damage == "symlink":
        saved = tmp_path / "same-bytes.json"
        shutil.copyfile(path, saved)
        path.unlink()
        path.symlink_to(saved)
    else:
        path.write_text("corrupted")
    with pytest.raises(ValueError):
        archive.read(entry["revision"])
    with pytest.raises(ValueError):
        campaign.run(initial)


@pytest.mark.parametrize("boundary", ["copy", "receipt"])
def test_archive_interruption_recovers_without_reexecuting_trials(tmp_path, monkeypatch, boundary):
    from PhysicalRSI_core.self_harness import archive as module, evaluation
    campaign, initial, suite, quota = calibration_setup(tmp_path)
    copy = module.shutil.copyfileobj
    freeze = evaluation.freeze_json
    interrupted = False

    def stop_copy(incoming, outgoing, **kwargs):
        nonlocal interrupted
        copy(incoming, outgoing, **kwargs)
        if "/archive/entries/.pending-" in str(outgoing.name) and not interrupted:
            interrupted = True
            raise RuntimeError("archive copy interrupted")

    def stop_receipt(path, value):
        nonlocal interrupted
        result = freeze(path, value)
        if Path(path).name == "archive.json" and not interrupted:
            interrupted = True
            raise RuntimeError("archive receipt interrupted")
        return result

    with monkeypatch.context() as patch:
        if boundary == "copy":
            patch.setattr(module.shutil, "copyfileobj", stop_copy)
        else:
            patch.setattr(evaluation, "freeze_json", stop_receipt)
        with pytest.raises(RuntimeError, match="archive .* interrupted"):
            campaign.run(initial)
    assert len(suite.resets) == quota.status()["reserved_trials"] == 10
    assert campaign.state.resolve()["harness"]["id"] == "calibrated-0"
    assert not list((tmp_path / "archive/entries").glob(".pending-*"))
    result = campaign.run(initial)
    assert result["completed_rounds"] == 2
    assert len(suite.resets) == 12
    assert len(archive_entries(tmp_path)[1]) == 2


@pytest.mark.parametrize("damage", ["missing", "corrupted"])
def test_descendant_requires_intact_search_history(tmp_path, damage):
    campaign, initial, suite, quota = setup(tmp_path)
    campaign.run(initial)
    archive, entries = archive_entries(tmp_path)
    ancestor = next(e for e in entries if "specialist-axis" in e["candidates"])
    descendant = next(e for e in entries if "composed" in e["candidates"])
    view = archive.view(scope=descendant["scope"], policy_sha256=descendant["policy_sha256"])
    folder = archive.root / "entries" / ancestor["revision"]
    if damage == "missing":
        shutil.rmtree(folder)
    else:
        name = next(name for name in ancestor["files"] if name.endswith("trajectory.json"))
        (folder / "evidence" / name).write_text("damaged historical rollout")
    for operation in (
        lambda: archive.read(descendant["revision"]),
        lambda: archive.view(scope=view["scope"], policy_sha256=view["policy_sha256"]),
        lambda: archive.summaries(view),
        lambda: campaign.run(initial),
    ):
        with pytest.raises(ValueError, match="evidence"):
            operation()
    assert len(suite.resets) == quota.status()["reserved_trials"] == 24


def test_scope_view_and_fixed_components_cannot_be_bypassed(tmp_path):
    campaign, initial, _, _ = setup(tmp_path)
    result = campaign.run(initial)
    archive, entries = archive_entries(tmp_path)
    first = next(e for e in entries if "specialist-axis" in e["candidates"])
    view = read_json(campaign.root / "rounds/round-0002/execution/archive-view.json")
    assert archive.view(scope="another-task-scope", policy_sha256=first["policy_sha256"])["revisions"] == []
    assert archive.view(scope=first["scope"], policy_sha256="0"*64)["revisions"] == []
    changed = deepcopy(view)
    changed["scope"] = "another-task-scope"
    with pytest.raises(ValueError, match="scope"):
        archive.summaries(changed)
    second = next(e for e in entries if e != first)
    origin = dict(revision=second["revision"], candidate_id="composed",
                  freeze_sha256=second["candidates"]["composed"]["freeze_sha256"], view_sha256=digest(view))
    with pytest.raises(ValueError, match="outside"):
        archive.verify_origin(view, origin)
    with pytest.raises(ValueError, match="fixed evolution"):
        archive.fork(view, first["revision"], "specialist-axis", parent=result["current"]["harness"],
            output=tmp_path / "not-created", name="bad", evolution=EvolutionSpec(("prompts",)))
    assert not (tmp_path / "not-created").exists()


def test_rejected_candidates_keep_reasons_without_inventing_evaluation_scores(tmp_path):
    campaign, initial, suite, quota = calibration_setup(tmp_path)
    suite.admit = lambda h: dict(accepted=h["id"] == "uncalibrated", reason="review required",
                                evidence=dict(closure=verify_harness(h)))
    result = campaign.run(initial)
    archive, entries = archive_entries(tmp_path)
    assert [r["outcome"] for r in result["rounds"]] == ["all_candidates_rejected"] * 2
    assert len(suite.resets) == quota.status()["reserved_trials"] == 4
    for entry in entries:
        rejected = [r for r in entry["candidates"].values() if r["disposition"] == "admission_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["admission"]["reason"] == "review required"
        assert rejected[0]["metrics"] is None and entry["comparison"] is None
    assert campaign.run(initial) == result
    assert len(suite.resets) == 4
