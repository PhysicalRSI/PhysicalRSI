import pytest

from PhysicalRSI_baselines.behavior_baseline import seed
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness


def test_materialized_seed_is_bound_and_cannot_replace_existing_revision(tmp_path, monkeypatch):
    # Source check is stubbed; this test establishes artifact semantics only.
    monkeypatch.setattr(seed.subprocess, "check_output", lambda *a, **k: seed.OPENPI_REVISION)
    monkeypatch.setattr(seed.subprocess, "run", lambda *a, **k: None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "uv.lock").write_text("fixture lock")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"fixture weights")
    stats = checkpoint / "assets/turning_on_radio/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    environment = tmp_path / "environment.txt"
    environment.write_text("fixture==1")
    output = tmp_path / "seed-001"
    kwargs = dict(source=source, checkpoint=checkpoint, task="turning_on_radio",
                  environment_manifest=environment)
    candidate = seed.materialize(output, **kwargs)
    assert read_json(output / "candidate.json") == candidate
    assert verify_harness(candidate) == read_json(output / "materialization.json")["candidate_sha256"]
    assert read_json(output / "backend.json")["repo_id"] == "turning_on_radio"
    with pytest.raises(ValueError, match="normalization statistics"):
        seed.materialize(tmp_path / "wrong-stats", **dict(kwargs, repo_id="b1k/turning_on_radio"))
    with pytest.raises(FileExistsError):
        seed.materialize(output, **kwargs)
    (output / "control.py").write_text("changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        verify_harness(candidate)


def test_download_partials_and_symlinks_are_not_frozen_as_weights(tmp_path):
    (tmp_path / "weights.incomplete").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        seed.checkpoint_inventory(tmp_path)
    (tmp_path / "weights.incomplete").unlink()
    (tmp_path / "weights").symlink_to("outside")
    with pytest.raises(ValueError, match="symlinks"):
        seed.checkpoint_inventory(tmp_path)
