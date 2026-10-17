"""Reuse the historical serial Operation composer; add a portable declaration."""
from PhysicalRSI.Embodied_Harness.skills.composition import sequence
from PhysicalRSI_core.skill_graph.schema import CompositionEdge, CompositionNode, SkillComposition
from .protocol import CallableSkill


def compose_sequence(name, *skills, ontology_revision='unspecified', embodiment_profile='unspecified'):
    if not skills:
        raise ValueError('A skill sequence needs at least one skill')
    operations = [skill.as_operation() for skill in skills]
    operation = sequence(name, *operations)
    result = CallableSkill.from_operation(operation)
    ids = [f'step-{i}' for i in range(len(skills))]
    result.composition = SkillComposition(
        name, ontology_revision, embodiment_profile, ids[0], ids[-1],
        tuple(CompositionNode(ids[i], 'primitive', s.id, s.revision,
            s.descriptor.input, s.descriptor.output, s.descriptor.effects)
            for i, s in enumerate(skills)),
        tuple(CompositionEdge(ids[i], ids[i+1]) for i in range(len(skills)-1)))
    return result
