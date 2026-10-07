"""Prepare explicit Self-Harness edits; never run or select candidate code."""
import ast
import copy
from pathlib import Path

from .candidate_program import MEMORY, PRIMITIVES


def reference_relation_edits(primitives, memory, *, relation, references,
                             reference_prompts, reference_minimum_score=.4):
    """Replace initial grasp localization, preserving motion and combinations.

    The caller must obtain relations from task language and aliases from public
    perception development. This does not erase the parent's provenance.
    Reject unfamiliar source structure instead of silently leaving a fallback
    that could grasp a different object after relation resolution fails.
    """
    if relation not in ('near', 'between') or len(references) != (1 if relation == 'near' else 2):
        raise ValueError('Invalid relation specification')
    if set(reference_prompts) != set(references):
        raise ValueError('Declare prompts for every reference')
    if (type(reference_minimum_score) not in (int, float)
            or not 0 <= reference_minimum_score <= 1):
        raise ValueError('Invalid reference confidence threshold')
    for prompts in reference_prompts.values():
        if (not isinstance(prompts, list) or not 1 <= len(prompts) <= 3
                or any(not isinstance(p, str) or not p.strip() for p in prompts)
                or len(set(prompts)) != len(prompts)):
            raise ValueError('Invalid reference prompts')
    source = primitives['source']
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'grasp_and_lift']
    if len(functions) != 1:
        raise ValueError('Expected an unmodified grasp primitive')
    helper_source = Path(__file__).with_name('relation_reference.py').read_text()
    helper_names = ('locate_task_relation', 'locate_reference')
    existing = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helper_names]
    if existing:
        reviewed = [n for n in ast.parse(helper_source).body
                    if isinstance(n, ast.FunctionDef) and n.name in helper_names]
        if [ast.dump(n) for n in existing] != [ast.dump(n) for n in reviewed]:
            raise ValueError('Existing relation helpers differ from reviewed source')
        expected_call = ast.parse('located = locate_task_relation(robot, memory)').body[0]
        body = functions[0].body
        matches = [i for i, n in enumerate(body) if ast.dump(n) == ast.dump(expected_call)]
        if (len(matches) != 1 or matches[0]+1 >= len(body)
                or not isinstance(body[matches[0]+1], ast.If)
                or ast.unparse(body[matches[0]+1].test) != "located['status'] != 'estimated_surface'"):
            raise ValueError('Existing relation localization guard not recognized')
        # An inherited relation policy needs only a new memory revision.
        # Retain its exact source bytes and all motion parameters.
        m = copy.deepcopy(memory)
        m['memory'].update(target='black bowl', targets=['black bowl'],
                           relation_kind=relation, relation_references=list(references),
                           reference_prompts=copy.deepcopy(reference_prompts),
                           reference_minimum_score=reference_minimum_score)
        return {PRIMITIVES: copy.deepcopy(primitives), MEMORY: m}
    body = functions[0].body
    expected = ast.parse("located = robot.locate_object(memory['target'], camera=memory['camera'])").body[0]
    matches = [i for i, n in enumerate(body) if ast.dump(n) == ast.dump(expected)]
    if len(matches) != 1:
        raise ValueError('Initial target localization not recognized')
    start = matches[0]
    end = start + 1
    if isinstance(body[end], ast.For):
        loop = body[end]
        if 'target_aliases' not in ast.unparse(loop.iter):
            raise ValueError('Unrecognized localization fallback')
        end += 1
    if not isinstance(body[end], ast.If) or ast.unparse(body[end].test) != "located['status'] != 'estimated_surface'":
        raise ValueError('Missing failed-localization guard')
    lines = source.splitlines(keepends=True)
    first = body[start].lineno - 1
    last = body[end].lineno - 1
    updated = ''.join(lines[:first]) + '    located = locate_task_relation(robot, memory)\n' + ''.join(lines[last:])
    updated += '\n' + Path(__file__).with_name('relation_reference.py').read_text()
    ast.parse(updated)
    p = copy.deepcopy(primitives)
    m = copy.deepcopy(memory)
    p['source'] = updated
    m['memory'].update(target='black bowl', targets=['black bowl'],
                       relation_kind=relation, relation_references=list(references),
                       reference_prompts=copy.deepcopy(reference_prompts),
                       reference_minimum_score=reference_minimum_score)
    return {PRIMITIVES: p, MEMORY: m}
