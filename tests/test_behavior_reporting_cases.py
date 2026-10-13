from copy import deepcopy

import pytest

from PhysicalRSI_baselines.behavior_baseline.reporting_cases import register_reporting
from PhysicalRSI_core.infra.storage import atomic_json, digest, read_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol


def inventory(tmp_path):
    tasks = [f"task-{i}" for i in range(100)]
    splits = {
        "development": {t: [{"instance_id": 0}] for t in tasks},
        "validation": {t: [{"instance_id": 200}] for t in tasks},
        "test": {t: [dict(instance_id=i + 301, instance_index=i, mode="public_test")
                     for i in range(20)] for t in tasks},
    }
    pool = CaseProtocol(tmp_path / "source", splits=splits, identity_keys=("instance_id",))
    return pool.root / "protocol.json", pool.revision


def test_prescribed_report_cohort_preserves_source_and_training_splits(tmp_path):
    path, revision = inventory(tmp_path)
    before = path.read_bytes()
    record = register_reporting(path, tmp_path / "report", expected_revision=revision)
    new = read_json(tmp_path / "report/cases/protocol.json")
    old = read_json(path)
    assert path.read_bytes() == before
    for split in ("development", "validation"):
        assert new["splits"][split] == old["splits"][split]
    assert record["report_episode_count"] == 1000
    for task, cases in new["splits"]["test"].items():
        assert [c["instance_index"] for c in cases] == list(range(10))
        assert [c["instance_id"] for c in cases] == list(range(301, 311))
        assert record["excluded_public_cases"][task] == old["splits"]["test"][task][10:]
    with pytest.raises(FileExistsError):
        register_reporting(path, tmp_path / "report", expected_revision=revision)


@pytest.mark.parametrize("mutation", ["task", "instance", "revision"])
def test_partial_or_changed_source_cannot_register_a_report(tmp_path, mutation):
    path, revision = inventory(tmp_path)
    source = deepcopy(read_json(path))
    if mutation == "task":
        for split in source["splits"].values():
            del split["task-0"]
    else:
        source["splits"]["test"]["task-0"][0]["instance_index"] = 10
    atomic_json(path, source)
    if mutation != "revision":
        revision = digest(source)
    with pytest.raises(ValueError):
        register_reporting(path, tmp_path / "report", expected_revision=revision)
    assert not (tmp_path / "report").exists()
