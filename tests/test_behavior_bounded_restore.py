"""Verify the bounded loader's source boundary without model dependencies."""
import hashlib
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_baselines.behavior_baseline import engine
from PhysicalRSI_core.infra.storage import atomic_json, file_digest


LOADERS = ("champion_chunk_restore.py", "champion_restore_params.py",
           "champion_policy_config.py")


@pytest.mark.parametrize("changed", LOADERS)
def test_loader_changes_invalidate_frozen_engine(tmp_path, monkeypatch, changed):
    source = Path(engine.__file__).resolve().parents[2]
    manifest = engine.manifest()
    assert set(LOADERS) <= {Path(name).name for name in manifest["files"]}
    for name in manifest["files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    monkeypatch.setattr(engine, "__file__", str(
        tmp_path / "PhysicalRSI_baselines/behavior_baseline/engine.py"))
    candidate = tmp_path / "candidate"
    atomic_json(candidate / "engine.json", manifest)
    specification = {"root": str(candidate), "components": {"dependencies": {
        "engine.json": file_digest(candidate / "engine.json")}}}
    assert engine.verify(specification) == manifest
    target = tmp_path / "PhysicalRSI_baselines/behavior_baseline" / changed
    target.write_bytes(target.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="differs from the frozen"):
        engine.verify(specification)


def test_factory_preserves_pinned_model_and_transform_code():
    path = Path(engine.__file__).with_name("champion_policy_config.py")
    original = path.read_text().replace(
        "from .champion_restore_params import restore_params as _bounded_restore\n\n",
        "", 1).replace(
        '_bounded_restore(checkpoint_dir / "params", dtype=jnp.bfloat16)',
        '_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)')
    # b1k/policies/policy_config.py at ca556f74a455cef7987a2be4537b5ac85cc56dd7.
    assert hashlib.sha256(original.encode()).hexdigest() == (
        "e782721a35b668f9305c26cb2ca0bd889b7026aa7dc5705a7f5fa4314c87f45e")
