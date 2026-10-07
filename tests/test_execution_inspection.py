from pathlib import Path

import pytest

from PhysicalRSI_core.contracts import Context, Contract, Operation
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json


def execute(root, *, fail=False):
    calls = []

    def run(value, context):
        calls.append(value)
        if fail:
            raise RuntimeError("Operation failed after dispatch")
        return dict(reused=[value, value], other=b"def")

    execution = Execution(root, observe=lambda operation, context: b"abc")
    operation = Operation("fixture", "1", Contract("bytes"), Contract("payload"), run, effects=("device",))
    if fail:
        with pytest.raises(RuntimeError, match="after dispatch"):
            execution(operation, b"abc", Context("episode"))
    else:
        execution(operation, b"abc", Context("episode"))
    path = next((root / "episode").glob("*.json"))
    return execution, path, calls


@pytest.mark.parametrize("fail", [False, True])
def test_inspection_verifies_unique_artifacts_and_never_reexecutes(tmp_path, fail):
    execution, path, calls = execute(tmp_path, fail=fail)
    expected = file_digest(path)
    first = execution.read("episode", path.stem, max_artifact_bytes=6, expected_sha256=expected)
    assert first["record_sha256"] == expected and first["record"] == read_json(path)
    assert first["artifact_bytes"] == (3 if fail else 6)
    assert len(first["artifacts"]) == (1 if fail else 2)
    assert first["record"]["state"] == ("failed" if fail else "completed")
    assert execution.read("episode", path.stem, max_artifact_bytes=6) == first
    assert calls == [b"abc"] and file_digest(path) == expected


def test_inspection_preserves_uncertain_and_started_states(tmp_path):
    execution, path, calls = execute(tmp_path)
    record = read_json(path)
    for state in ("started", "uncertain", "cancelled"):
        atomic_json(path, dict(record, state=state))
        result = execution.read("episode", path.stem)
        assert result["record"]["state"] == state and calls == [b"abc"]


@pytest.mark.parametrize("change", ["budget", "record_budget", "blob", "record", "conflict", "outside", "duplicate", "identity"])
def test_inspection_rejects_changed_or_unbounded_evidence(tmp_path, change):
    execution, path, calls = execute(tmp_path)
    expected = file_digest(path)
    record = read_json(path)
    options = {}
    if change == "budget":
        options["max_artifact_bytes"] = 5
    elif change == "record_budget":
        options["max_record_bytes"] = 1
    elif change == "blob":
        Path(record["input"]["artifact"]).write_bytes(b"xyz")
    elif change == "record":
        atomic_json(path, dict(record, operation="another-operation"))
        options["expected_sha256"] = expected
    elif change == "conflict":
        record["output"]["reused"][0]["bytes"] = 2
        atomic_json(path, record)
    elif change == "outside":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"abc")
        record["input"]["artifact"] = str(outside)
        atomic_json(path, record)
    elif change == "duplicate":
        path.write_text(path.read_text().rstrip()[:-1] + ', "state": "completed"}')
    else:
        atomic_json(path, dict(record, id="wrong"))
    with pytest.raises(ValueError):
        execution.read("episode", path.stem, **options)
    assert calls == [b"abc"]


def test_inspection_does_not_follow_episode_symlinks_or_traversal(tmp_path):
    execution, path, _ = execute(tmp_path / "source")
    other = Execution(tmp_path / "other")
    other.root.mkdir()
    (other.root / "episode").symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="physical"):
        other.read("episode", path.stem)
    with pytest.raises(ValueError, match="identifier"):
        execution.read("../source/episode", path.stem)


def test_root_binding_survives_working_directory_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    execution, path, calls = execute(Path("relative"))
    monkeypatch.chdir(tmp_path.parent)
    result = execution.read("episode", path.stem)
    assert result["record"]["state"] == "completed" and calls == [b"abc"]
