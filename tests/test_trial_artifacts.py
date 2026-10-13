from pathlib import Path

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.trials import Trial, Evaluation
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


class RecordedEnvironment(CounterEnvironment):
    def __init__(self, recording, exports=None):
        self.recording, self.exports = recording, exports
        self.calls = 0

    def reset(self, case, context):
        self.calls += 1
        return super().reset(case, context)

    def export_evidence(self):
        return self.exports if self.exports is not None else {
            "sensor/depth.dat": dict(path=str(self.recording), sha256=file_digest(self.recording))}


def execute(root, environment):
    return ExperimentRuntime(root).evaluate(Trial("recorded", task="counter", scope="software",
        case=dict(initial=0, target=1), budget=Budget(1, 30)), environment=environment,
        policy=CounterPolicy(1), verifier=CounterVerifier())


def test_owned_artifacts_survive_source_loss_and_are_verified_on_read(tmp_path):
    source = tmp_path / "sensor.dat"
    source.write_bytes(b"sensor bytes\x00\xff")
    environment = RecordedEnvironment(source)
    result = execute(tmp_path / "runtime", environment)
    copy = tmp_path / "runtime/recorded/artifacts/sensor/depth.dat"
    assert copy.read_bytes() == source.read_bytes()
    assert str(copy) in result.to_dict()["evidence"]
    source.unlink()
    assert execute(tmp_path / "runtime", environment).revision == result.revision
    assert environment.calls == 1
    copy.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        Evaluation.read(ExperimentRuntime(tmp_path / "runtime"), "recorded")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "./aliased", "a/../b"])
def test_artifact_path_escape_never_publishes_completion(tmp_path, name):
    source = tmp_path / "sensor.dat"
    source.write_bytes(b"evidence")
    environment = RecordedEnvironment(source, {name: dict(path=str(source), sha256=file_digest(source))})
    with pytest.raises(ValueError):
        execute(tmp_path / "runtime", environment)
    with pytest.raises(ReconciliationRequired):
        execute(tmp_path / "runtime", environment)
    assert environment.calls == 1


def test_artifact_hash_mismatch_and_symlink_are_not_accepted(tmp_path):
    source = tmp_path / "sensor.dat"
    source.write_bytes(b"evidence")
    environment = RecordedEnvironment(source, {"file": dict(path=str(source), sha256="0" * 64)})
    with pytest.raises(ValueError, match="changed during capture"):
        execute(tmp_path / "mismatch", environment)
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular file"):
        execute(tmp_path / "symlink", RecordedEnvironment(link))


def test_artifact_manifest_cannot_drop_a_recorded_file(tmp_path):
    source = tmp_path / "sensor.dat"
    source.write_bytes(b"evidence")
    execute(tmp_path / "runtime", RecordedEnvironment(source))
    path = tmp_path / "runtime/recorded/artifacts.json"
    value = read_json(path)
    value["files"] = {}
    atomic_json(path, value)
    with pytest.raises(ValueError, match="evidence changed"):
        ExperimentRuntime(tmp_path / "runtime").read("recorded")
    # Even a rewritten manifest hash cannot omit a file still named by the
    # completed receipt. Exercise closure equality separately from byte hashes.
    receipt_path = path.with_name("receipt.json")
    receipt = read_json(receipt_path)
    receipt["evidence"]["artifacts.json"] = file_digest(path)
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        ExperimentRuntime(tmp_path / "runtime").read("recorded")


def test_export_cannot_write_through_an_artifact_directory_symlink(tmp_path):
    from PhysicalRSI_core.infra.trial_artifacts import capture
    folder, outside = tmp_path / "trial", tmp_path / "outside"
    folder.mkdir()
    outside.mkdir()
    (folder / "artifacts").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "sensor"
    source.write_bytes(b"sensor")
    with pytest.raises(ValueError, match="escaped|symlink"):
        capture(folder, {"recording": dict(path=str(source), sha256=file_digest(source))})
    assert not list(outside.iterdir())
