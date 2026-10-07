import copy

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.candidate_program import MEMORY, PRIMITIVES
from PhysicalRSI_baselines.embodied_goodharts_law.relation_proposal import reference_relation_edits


SOURCE = '''def grasp_and_lift(robot, memory):
    located = robot.locate_object(memory['target'], camera=memory['camera'])
    for alias in memory['target_aliases'].get(memory['target'], []):
        if located['status'] == 'estimated_surface':
            break
        located = robot.locate_object(alias, camera=memory['camera'])
    if located['status'] != 'estimated_surface':
        return {'status': 'target_not_located'}
    robot.move(located['surface_quantiles']['median'])
    return {'status': 'lift_motion_completed'}
'''


def edits(source=SOURCE):
    p = {'source': source}
    m = {'memory': {'camera': 'agentview', 'target_aliases': {'black bowl': ['bowl']}}}
    before = copy.deepcopy((p, m))
    result = reference_relation_edits(p, m, relation='near', references=['ramekin'],
                                      reference_prompts={'ramekin': ['ramekin', 'small bowl']})
    assert (p, m) == before
    assert set(result) == {PRIMITIVES, MEMORY}
    return result


def surface(x):
    return {'status': 'estimated_surface', 'detection_score': .9,
            'touches_image_border': False, 'surface_quantiles': {'median': [x, 0, 0]}}


class Robot:
    def __init__(self, ambiguous=False):
        self.moves = []
        self.ambiguous = ambiguous

    def locate_object(self, *args, **kwargs):
        raise AssertionError('Single-target fallback must not bypass relation checks')

    def locate_objects(self, prompt, **kwargs):
        if prompt == 'ramekin':
            return {'objects': []}
        if prompt == 'small bowl':
            return {'objects': [surface(.1), surface(.4)] if self.ambiguous else [surface(.1)]}
        assert prompt == 'black bowl'
        return {'objects': [surface(.15), surface(.6)]}

    def move(self, target):
        self.moves.append(target)


def test_reviewed_fixture_grasp_uses_relation_after_reference_alias():
    candidate = edits()
    namespace = {}
    exec(candidate[PRIMITIVES]['source'], namespace)
    robot = Robot()
    assert namespace['grasp_and_lift'](robot, candidate[MEMORY]['memory'])['status'] == 'lift_motion_completed'
    assert robot.moves == [[.15, 0, 0]]


def test_ambiguous_reference_never_reaches_motion_or_generic_target_fallback():
    candidate = edits()
    namespace = {}
    exec(candidate[PRIMITIVES]['source'], namespace)
    robot = Robot(ambiguous=True)
    assert namespace['grasp_and_lift'](robot, candidate[MEMORY]['memory'])['status'] == 'target_not_located'
    assert robot.moves == []


def test_unrecognized_primitive_rejected_instead_of_partially_rewritten():
    with pytest.raises(ValueError, match='localization'):
        edits(SOURCE.replace('located = robot.locate_object', 'located = robot.other'))


def test_inherited_relation_policy_accepts_memory_revision_without_source_change():
    original = edits()
    revised = reference_relation_edits(original[PRIMITIVES], original[MEMORY],
        relation='near', references=['ramekin'],
        reference_prompts={'ramekin': ['silver cylindrical container']},
        reference_minimum_score=.45)
    assert revised[PRIMITIVES] == original[PRIMITIVES]
    assert original[MEMORY]['memory']['reference_minimum_score'] == .4
    assert revised[MEMORY]['memory']['reference_minimum_score'] == .45
    broken = copy.deepcopy(original[PRIMITIVES])
    broken['source'] = broken['source'].replace('distance > .25', 'distance > 100')
    with pytest.raises(ValueError, match='reviewed source'):
        reference_relation_edits(broken, original[MEMORY], relation='near',
            references=['ramekin'], reference_prompts={'ramekin': ['ramekin']})
