import pytest

from PhysicalRSI_core.self_harness.skill_binding import (
    validate_candidate_skills,
)
from PhysicalRSI_core.self_harness.skill_mutation import SkillMutation
from physicalrsi import SkillDescriptor, SkillRegistry, CallableSkill
from physicalrsi.skills import attach_skill_mutation


def make_skill(amount):
    return CallableSkill(
        SkillDescriptor(
            "predict",
            f"predict-v{amount}",
            {"name": "estimate"},
            {"name": "target"},
            metadata={"amount": amount},
        ),
        lambda value, context, amount=amount: value + amount,
    )


def test_skill_mutation_carries_executable_registry_bindings(tmp_path):
    parent = make_skill(0)
    child = make_skill(1)
    registry = SkillRegistry(tmp_path)
    parent_key = registry.register(parent)
    child_key = registry.register(child)
    parent_bound = registry.load(parent_key, implementation=parent)
    child_bound = registry.load(child_key, implementation=child)
    mutation = SkillMutation.from_primitives(
        parent.descriptor, child.descriptor,
        operator="skill.replace", hypothesis="h", expected_effect="e",
    )
    candidate = attach_skill_mutation(
        {"id": "candidate"}, mutation,
        parent_skill=parent_bound, child_skill=child_bound,
    )
    value = validate_candidate_skills(candidate)
    assert value["mutation"].revision == mutation.revision
    assert {row["role"] for row in candidate["skill_bindings"]} == {"parent", "child"}
    assert child_bound.execute(2) == 3
    assert registry.load_binding(candidate["skill_bindings"][1]).execute(2) == 3


def test_mutation_without_both_executable_bindings_is_rejected():
    with pytest.raises(ValueError, match="requires parent and child bindings"):
        validate_candidate_skills({"skill_mutation": {"not": "a mutation"}})


def test_unregistered_skill_cannot_create_a_binding():
    from physicalrsi.skills.binding import bind_skill
    with pytest.raises(ValueError, match="Register the skill"):
        bind_skill(make_skill(0), role="parent")


def test_portable_binding_resolves_after_registry_move(tmp_path):
    parent, child = make_skill(0), make_skill(1)
    source = SkillRegistry(tmp_path / "source", registry_id="portable.skills")
    parent_key, child_key = source.register(parent), source.register(child)
    parent_bound = source.load(parent_key, implementation=parent)
    child_bound = source.load(child_key, implementation=child)
    mutation = SkillMutation.from_primitives(
        parent.descriptor, child.descriptor,
        operator="skill.replace", hypothesis="h", expected_effect="e",
    )
    candidate = attach_skill_mutation(
        {"id": "candidate"}, mutation,
        parent_skill=parent_bound, child_skill=child_bound,
    )
    import shutil
    shutil.copytree(tmp_path / "source", tmp_path / "moved")

    def resolve(record, revision):
        amount = record["descriptor"]["metadata"]["amount"]
        return make_skill(amount)

    moved = SkillRegistry(tmp_path / "moved", registry_id="portable.skills", resolver=resolve)
    assert moved.load_binding(candidate["skill_bindings"][1]).execute(2) == 3
