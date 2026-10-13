from collections import Counter
from pathlib import Path
import os

import pytest

from PhysicalRSI_core.experiments import ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.trials import Evaluation, Trial
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


def recorded(root):
    source = root / "source.bin"
    source.write_bytes(b"binary observation\x00\xff" * 100000)
    class Environment(CounterEnvironment):
        def export_evidence(self):
            return {"sensor.bin":dict(path=str(source),sha256=file_digest(source))}
    runtime = ExperimentRuntime(root / "runtime")
    result = runtime.evaluate(Trial("trial",task="counter",case=dict(initial=0,target=1),scope="software"),
        environment=Environment(),policy=CounterPolicy(),verifier=CounterVerifier())
    return runtime, result


def test_binary_evidence_is_streamed_once_and_json_is_parsed_from_checked_bytes(tmp_path,monkeypatch):
    runtime, expected = recorded(tmp_path)
    counts = Counter()
    original = Path.open
    def opened(path,mode="r",*args,**kwargs):
        if path.is_relative_to(runtime.root) and "r" in mode:
            counts[str(path.relative_to(runtime.root))] += 1
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened)
    assert Evaluation.read(runtime,"trial") == expected
    assert counts["trial/artifacts/sensor.bin"] == 1
    assert counts["trial/trajectory.json"] == counts["trial/experiment.json"] == 1
    assert counts["trial/receipt.json"] == 2  # Publication marker checked again at the end.


def test_new_audit_detects_same_size_same_timestamp_corruption(tmp_path):
    runtime, expected = recorded(tmp_path)
    assert Evaluation.read(runtime,"trial") == expected
    path = runtime.root / "trial/artifacts/sensor.bin"
    before = path.stat()
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError,match="evidence changed"):
        Evaluation.read(runtime,"trial")


def test_audit_never_exposes_unhashed_json_after_a_file_is_replaced(tmp_path,monkeypatch):
    runtime, expected = recorded(tmp_path)
    original = Path.read_bytes
    path = runtime.root / "trial/trajectory.json"
    replacement = read_json(path)
    replacement[-1]["observation"]["position"] = 99
    changed = []
    def read_then_replace(current):
        payload = original(current)
        if current == path and not changed:
            changed.append(True)
            atomic_json(path,replacement)
        return payload
    monkeypatch.setattr(Path,"read_bytes",read_then_replace)
    # This read returns the original verified byte snapshot, never the changed
    # semantic value. A later read must perform a fresh audit and reject it.
    assert Evaluation.read(runtime,"trial") == expected
    with pytest.raises(ValueError,match="evidence changed"):
        Evaluation.read(runtime,"trial")


def test_receipt_replacement_during_audit_is_rejected(tmp_path,monkeypatch):
    runtime, _ = recorded(tmp_path)
    original = Path.read_bytes
    marker = runtime.root / "trial/receipt.json"
    def replace_marker(path):
        payload = original(path)
        if path.name == "trajectory.json":
            marker.write_bytes(marker.read_bytes()+b"\n")
        return payload
    monkeypatch.setattr(Path,"read_bytes",replace_marker)
    with pytest.raises(ValueError,match="during audit"):
        Evaluation.read(runtime,"trial")


def test_returned_documents_do_not_cache_mutable_values_across_reads(tmp_path):
    runtime, expected = recorded(tmp_path)
    snapshot = runtime.read_evidence("trial")
    snapshot["receipt"]["outcome"] = "failure"
    snapshot["documents"]["trajectory.json"].clear()
    snapshot["evidence"].clear()
    assert Evaluation.read(runtime,"trial") == expected


def test_trial_directory_cannot_be_replaced_with_a_symlink(tmp_path):
    runtime, _ = recorded(tmp_path)
    folder = runtime.root / "trial"
    folder.rename(tmp_path / "outside")
    folder.symlink_to(tmp_path / "outside",target_is_directory=True)
    with pytest.raises(ValueError,match="physical directory"):
        Evaluation.read(runtime,"trial")


def test_artifact_manifest_and_receipt_must_agree_on_each_binary_hash(tmp_path):
    runtime, _ = recorded(tmp_path)
    folder = runtime.root / "trial"
    manifest = read_json(folder / "artifacts.json")
    manifest["files"]["artifacts/sensor.bin"] = "0" * 64
    atomic_json(folder / "artifacts.json",manifest)
    receipt = read_json(folder / "receipt.json")
    receipt["evidence"]["artifacts.json"] = file_digest(folder / "artifacts.json")
    atomic_json(folder / "receipt.json",receipt)
    with pytest.raises(ValueError,match="manifest disagrees"):
        runtime.read("trial")


def test_development_audit_still_checks_auxiliary_declared_files(tmp_path):
    from PhysicalRSI_core.self_harness.rollout import development_evaluations, read_development_evaluations
    from PhysicalRSI_demos.rollout_evolution import setup
    campaign, initial, _, _ = setup(tmp_path)
    loop = campaign.build_loop(tmp_path / "probe",campaign._context(0,"0"*64,campaign.initial_memory,None))
    feedback = loop.proposer.develop(initial,loop.root)
    extra = loop.root / "experiments/development/notes.json"
    atomic_json(extra,dict(note="development-only auxiliary evidence"))
    feedback["evidence"][str(extra.relative_to(loop.root))] = file_digest(extra)
    assert development_evaluations(initial,feedback,loop.root) == feedback["evaluations"]
    del feedback["evaluations"]
    assert len(read_development_evaluations(initial,feedback,loop.root)) == 2
    extra.write_text("damaged")
    with pytest.raises(ValueError,match="verified development data"):
        read_development_evaluations(initial,feedback,loop.root)


@pytest.mark.parametrize("alias_kind", ["dot", "symlink"])
def test_duplicate_resolved_evidence_paths_cannot_hide_a_wrong_hash(tmp_path,alias_kind):
    from PhysicalRSI_core.self_harness.rollout import development_evaluations
    from PhysicalRSI_demos.rollout_evolution import setup
    campaign, initial, _, _ = setup(tmp_path)
    loop = campaign.build_loop(tmp_path / "probe",campaign._context(0,"0"*64,campaign.initial_memory,None))
    feedback = loop.proposer.develop(initial,loop.root)
    name = next(name for name in feedback["evidence"] if name.endswith("trajectory.json"))
    sha = feedback["evidence"][name]
    feedback["evidence"][name] = "0" * 64
    alias = name.replace("/trajectory.json", "/./trajectory.json" if alias_kind == "dot" else "/trace-alias.json")
    if alias_kind == "symlink":
        (loop.root / alias).symlink_to(loop.root / name)
    feedback["evidence"][alias] = sha
    with pytest.raises(ValueError):
        development_evaluations(initial,feedback,loop.root)


@pytest.mark.parametrize("damage", ["symlink", "extra", "fifo"])
def test_owned_file_enumeration_rejects_nonclosure_entries(tmp_path,damage):
    from PhysicalRSI_core.infra.owned_files import OwnedFiles
    store = OwnedFiles(tmp_path / "store")
    source = tmp_path / "source"
    source.write_bytes(b"owned bytes")
    revision = store.capture({"kind":"test"},{"nested/file.bin":dict(path=str(source),sha256=file_digest(source))})
    files = store.folder(revision) / "files"
    if damage == "symlink":
        (files / "alias").symlink_to(files / "nested",target_is_directory=True)
    elif damage == "extra":
        (files / "extra.bin").write_bytes(b"undeclared")
    else:
        os.mkfifo(files / "unexpected.fifo")
    with pytest.raises(ValueError):
        store.read(revision)
