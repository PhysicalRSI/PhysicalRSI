import pytest

from PhysicalRSI_core.skill_graph.schema import (
    CompositionEdge,
    CompositionNode,
    SCHEMA,
    SkillComposition,
)


def node(name, *, kind="primitive"):
    return CompositionNode(name, kind, "skill." + name, "r1", {"name": "state"}, {"name": "state"})


def composition():
    return SkillComposition(
        "pick", "ontology-r1", "dual-arm-v1", "observe", "release",
        (node("observe"), node("release")),
        (CompositionEdge("observe", "release"),),
    )


def test_composition_is_canonical_and_content_addressed():
    value = composition().as_dict()
    assert value["schema"] == SCHEMA
    assert value["revision"] == composition().revision
    assert SkillComposition.from_dict(value).as_dict() == value


def test_nodes_and_edges_are_sorted_for_stable_identity():
    value = SkillComposition(
        "pick", "ontology-r1", "dual-arm-v1", "observe", "release",
        (node("release"), node("observe")),
        (CompositionEdge("observe", "release"),),
    )
    assert [row.id for row in value.nodes] == ["observe", "release"]


@pytest.mark.parametrize("change", [
    {"entry": "missing"},
    {"edges": [CompositionEdge("observe", "missing").as_dict()]},
])
def test_invalid_graph_is_rejected(change):
    value = composition().as_dict()
    value.update(change)
    value["revision"] = composition().revision
    with pytest.raises(ValueError):
        SkillComposition.from_dict(value)


def test_revision_tampering_is_rejected():
    value = composition().as_dict()
    value["nodes"][0]["revision"] = "changed"
    with pytest.raises(ValueError, match="revision"):
        SkillComposition.from_dict(value)
