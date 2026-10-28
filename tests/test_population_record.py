from pathlib import Path

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.population import PopulationRecord
from PhysicalRSI_core.self_harness.skill_mutation import SkillMutation


def harness(root, name, *, mutation=None):
    for kind in CLOSURE:
        atomic_json(root / (kind + ".json"), {"kind": kind, "name": name})
    value = dict(
        id=name, root=str(root),
        components={kind: {kind + ".json": file_digest(root / (kind + ".json"))}
                    for kind in CLOSURE},
    )
    if mutation is not None:
        value["skill_mutation"] = mutation
    return value


def mutation():
    return SkillMutation(
        unit_kind="primitive", operation="replace",
        parent={"kind": "primitive", "id": "track", "revision": "track-v1"},
        child={"kind": "primitive", "id": "track", "revision": "track-v2"},
        operator="skill.replace", changes=({"path": "revision", "before": "track-v1", "after": "track-v2"},),
        hypothesis="h", expected_effect="e",
    ).as_dict()


def test_population_record_binds_candidate_hashes_and_skill_mutation(tmp_path):
    parent = harness(tmp_path / "parent", "parent")
    child = harness(tmp_path / "child", "child", mutation=mutation())
    child["parent_sha256"] = verify_harness(parent)
    record = PopulationRecord.from_candidates(
        round_id="round-0001", generation=1, scope="fixture",
        parent=parent, children=[child],
    )
    path = tmp_path / "population.json"
    record.write(path)
    reopened = PopulationRecord.read(path)
    assert reopened.as_dict() == record.as_dict()
    assert reopened.candidate_ids == ("parent", "child")
    assert reopened.members[0]["skill_mutation"]["unit_kind"] == "primitive"


def test_population_record_is_immutable_and_rejects_parent_drift(tmp_path):
    parent = harness(tmp_path / "parent", "parent")
    child = harness(tmp_path / "child", "child")
    child["parent_sha256"] = verify_harness(parent)
    record = PopulationRecord.from_candidates(
        round_id="round-0001", generation=1, scope="fixture",
        parent=parent, children=[child],
    )
    path = tmp_path / "population.json"
    record.write(path)
    changed = record.as_dict()
    changed["scope"] = "other"
    with pytest.raises(ValueError, match="revision differs"):
        PopulationRecord.from_dict(changed)
    with pytest.raises(ValueError, match="another candidate pool"):
        PopulationRecord.from_candidates(
            round_id="round-0001", generation=1, scope="other",
            parent=parent, children=[],
        ).write(path)
