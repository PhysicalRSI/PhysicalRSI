import pytest

from PhysicalRSI_baselines.behavior_baseline.round_cases import RoundCases
from PhysicalRSI_core.infra.storage import digest
from PhysicalRSI_core.self_harness.protocol import CaseProtocol


def test_fresh_rounds_preserve_all_tasks_and_actual_reset_ids(tmp_path):
    splits = {split: {} for split in ("development", "validation", "test")}
    for task, first in (("zero_based", 0), ("one_based", 1)):
        for split, ids in (("development", range(first, first + 4)),
                           ("validation", range(200 + first, 204 + first)), ("test", [301, 302])):
            mode = "public_test" if split == "test" else "train"
            splits[split][task] = [dict(instance_id=i, mode=mode,
                instance_index=i - 301 if mode == "public_test" else i) for i in ids]
    pool = CaseProtocol(tmp_path / "pool", splits=splits, identity_keys=("instance_id",))
    kwargs = dict(expected_pool_revision=pool.revision, max_rounds=2,
                  development_per_task=2, validation_per_task=2)
    schedule = RoundCases(tmp_path / "schedule", pool, **kwargs)
    first, second = schedule.protocol(0), schedule.protocol(1)
    assert first.cases("development")["one_based"][0]["instance_index"] == 1
    a, b = first.reset_identities(), second.reset_identities()
    assert not set(a["validation"]) & set(b["validation"])
    assert a["test"] == b["test"]
    assert all(set(first.cases(split)) == set(splits[split]) for split in ("development", "validation"))
    hashes = [digest(c) for values in first.cases("validation").values() for c in values]
    assert len(hashes) == len(set(hashes))
    assert schedule.protocol(0).revision == first.revision
    with pytest.raises(ValueError, match="final-candidate"):
        first.cases("test")
    with pytest.raises(ValueError, match="Frozen round schedule"):
        RoundCases(tmp_path / "schedule", pool, **dict(kwargs, validation_per_task=1))
    with pytest.raises(ValueError, match="Insufficient fresh"):
        RoundCases(tmp_path / "exhausted", pool, **dict(kwargs, max_rounds=3))
    with pytest.raises(ValueError, match="admitted revision"):
        RoundCases(tmp_path / "changed", pool, **dict(kwargs, expected_pool_revision="0" * 64))
