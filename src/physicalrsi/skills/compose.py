"""Small native composition helpers for the skill core."""
from dataclasses import replace
import json

from PhysicalRSI_core.contracts import Contract, contract_from, contract_payload, mismatches
from PhysicalRSI_core.skill_graph.schema import CompositionEdge, CompositionNode, SkillComposition
from .protocol import CallableSkill, CompositionTypeError, SkillDescriptor


def check_composition(*skills) -> dict[str, list[str]]:
    def as_contract(value):
        return Contract("__legacy__") if value == {} else contract_from(value)
    problems = {}
    for index, (producer, consumer) in enumerate(zip(skills, skills[1:])):
        errors = mismatches(as_contract(producer.descriptor.output),
                            as_contract(consumer.descriptor.input))
        if errors:
            problems[f"step-{index}->step-{index + 1}"] = errors
    return problems


def compose_sequence(name, *skills, ontology_revision='unspecified',
                     embodiment_profile='unspecified', strict=True):
    if not skills:
        raise ValueError('A skill sequence needs at least one skill')
    if strict:
        problems = check_composition(*skills)
        if problems:
            details = []
            for pair, errors in problems.items():
                index = int(pair.split('-')[1])
                details.append(f"{pair} ({skills[index].id} -> {skills[index + 1].id}): " + '; '.join(errors))
            raise CompositionTypeError(
                f"Composition {name!r} has incompatible adjacent skills: " + ' | '.join(details)
                + ". Insert an explicit adapt(...) skill between incompatible steps.")
    ids = [f'step-{i}' for i in range(len(skills))]
    nodes = []
    for i, skill in enumerate(skills):
        metadata = skill.descriptor.metadata
        if isinstance(metadata, dict) and metadata.get('kind') == 'adapter':
            kind = 'adapter'
        elif skill.composition is not None:
            kind = 'composite'
        else:
            kind = 'primitive'
        nodes.append(CompositionNode(
            ids[i], kind, skill.id, skill.revision, skill.descriptor.input,
            skill.descriptor.output, skill.descriptor.effects,
            implementation=skill.executable_identity(),
            composition=skill.composition.as_dict(include_revision=False)
            if skill.composition is not None else None))
    composition = SkillComposition(
        name, ontology_revision, embodiment_profile, ids[0], ids[-1], tuple(nodes),
        tuple(CompositionEdge(ids[i], ids[i + 1]) for i in range(len(skills) - 1)))

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
        "schema": "physicalrsi.executable/v1", "kind": "sequence", "name": name,
        "children": [skill.executable_identity() for skill in skills],
    }
    return _SequenceSkill(descriptor, execute, composition=composition,
                          executable_identity=executable_identity, children=skills)


def adapt(name, source, target, function, *, revision=None, effects=()):
    source_payload = contract_payload(contract_from(source))
    target_payload = contract_payload(contract_from(target))
    if revision is None:
        revision = json.dumps(["adapt", name, source_payload, target_payload],
                              sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    descriptor = SkillDescriptor(
        name, revision, source_payload, target_payload, effects=effects,
        metadata={"kind": "adapter", "source": source_payload, "target": target_payload})
    return CallableSkill(descriptor, function)


class _SequenceSkill(CallableSkill):
    def as_operation(self):
        operation = super().as_operation()

        def execute(value, context):
            for child in operation.children:
                value = child(value, context)
            return value

        return replace(operation, call=execute)
