import pytest

from PhysicalRSI_core.self_harness.skill_mutation import SkillMutation
from PhysicalRSI_core.skill_graph.schema import CompositionEdge, CompositionNode, SkillComposition
from physicalrsi.skills.protocol import SkillDescriptor


def primitive(name, revision):
    return SkillDescriptor(
        name, revision,
        {"name": "state"}, {"name": "state"},
        metadata={"kind": "primitive"},
    )


def composition(name, revision):
    return SkillComposition(
        name, "ontology-v1", "arm-v1", "observe", "release",
        (
            CompositionNode("observe", "primitive", "observe", revision,
                            {"name": "state"}, {"name": "state"}),
            CompositionNode("release", "primitive", "release", "release-v1",
                            {"name": "state"}, {"name": "state"}),
        ),
        (CompositionEdge("observe", "release"),),
    )


def test_primitive_mutation_is_content_addressed_and_roundtrips():
    mutation = SkillMutation.from_primitives(
        primitive("track", "track-v1"), primitive("track", "track-v2"),
        operator="skill.replace", hypothesis="Reduce tracking lag",
        expected_effect="Improve intercept timing",
    )
    value = mutation.as_dict()
    assert value["unit_kind"] == "primitive"
    assert value["parent"]["revision"] == "track-v1"
    assert value["child"]["revision"] == "track-v2"
    assert value["revision"] == mutation.revision
    assert SkillMutation.from_dict(value).as_dict() == value


def test_composition_mutation_records_node_level_changes():
    mutation = SkillMutation.from_compositions(
        composition("pick", "observe-v1"), composition("pick", "observe-v2"),
        operator="composition.replace", hypothesis="Use the revised observer",
        expected_effect="Preserve the downstream contract",
    )
    assert mutation.unit_kind == "composition"
    assert any(change["path"].startswith("nodes") for change in mutation.changes)


def test_skill_mutation_rejects_tampering_and_unknown_units():
    mutation = SkillMutation.from_primitives(
        primitive("track", "track-v1"), primitive("track", "track-v2"),
        operator="skill.replace", hypothesis="h", expected_effect="e",
    )
    value = mutation.as_dict()
    value["child"]["revision"] = "track-v3"
    with pytest.raises(ValueError, match="revision differs"):
        SkillMutation.from_dict(value)
    value = mutation.as_dict()
    value["unit_kind"] = "task"
    with pytest.raises(ValueError, match="unit_kind"):
        SkillMutation.from_dict(value)
