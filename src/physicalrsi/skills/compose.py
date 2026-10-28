"""Small native composition helpers for the skill core."""
from dataclasses import replace
import json
from PhysicalRSI_core.skill_graph.schema import CompositionEdge, CompositionNode, SkillComposition
from .protocol import CallableSkill, SkillDescriptor


def compose_sequence(name, *skills, ontology_revision='unspecified', embodiment_profile='unspecified'):
    if not skills:
        raise ValueError('A skill sequence needs at least one skill')
    ids = [f'step-{i}' for i in range(len(skills))]
    composition = SkillComposition(
        name, ontology_revision, embodiment_profile, ids[0], ids[-1],
        tuple(CompositionNode(ids[i], 'composite' if s.composition is not None else 'primitive',
            s.id, s.revision, s.descriptor.input, s.descriptor.output, s.descriptor.effects,
            implementation=s.executable_identity(),
            composition=s.composition.as_dict(include_revision=False) if s.composition is not None else None)
            for i, s in enumerate(skills)),
        tuple(CompositionEdge(ids[i], ids[i+1]) for i in range(len(skills)-1)))

    def execute(value, context=None):
        for skill in skills:
            value = skill.execute(value, context)
        return value

    descriptor = SkillDescriptor(
        id=name,
        revision=json.dumps(['sequence', name, [[s.id, s.revision] for s in skills]],
                            ensure_ascii=False, separators=(',', ':')),
        input=skills[0].descriptor.input,
        output=skills[-1].descriptor.output,
        effects=tuple(sorted({effect for skill in skills for effect in skill.descriptor.effects})),
        metadata={"kind": "sequence", "children": [skill.id for skill in skills]},
    )
    executable_identity = {
        "schema": "physicalrsi.executable/v1",
        "kind": "sequence",
        "name": name,
        "children": [skill.executable_identity() for skill in skills],
    }
    return _SequenceSkill(
        descriptor,
        execute,
        composition=composition,
        executable_identity=executable_identity,
        children=skills,
    )


class _SequenceSkill(CallableSkill):
    def as_operation(self):
        operation = super().as_operation()

        def execute(value, context):
            for child in operation.children:
                value = child(value, context)
            return value

        return replace(operation, call=execute)
