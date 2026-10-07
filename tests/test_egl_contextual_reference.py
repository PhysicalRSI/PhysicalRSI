from pathlib import Path

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law import contextual_reference, relation_reference


def surface(x, y=0, score=.8):
    return {'status': 'estimated_surface', 'detection_score': score,
            'touches_image_border': False,
            'surface_quantiles': {'median': [x, y, 0]}}


class Robot:
    def __init__(self, references):
        self.references = references

    def locate_objects(self, prompt, **kwargs):
        return {'objects': {'plate': [surface(.72, .2)],
                            'container': self.references,
                            'black bowl': [surface(.52, .2)]}[prompt]}


def test_relation_uses_public_anchor_to_reject_distant_false_positive():
    namespace = {}
    for module in [relation_reference, contextual_reference]:
        exec(Path(module.__file__).read_text(), namespace)
    memory = {'reference_anchor_prompt': 'plate', 'reference_anchor_radius_m': .5,
              'relation_references': ['container'], 'reference_prompts': {'container': ['container']},
              'reference_minimum_score': .45, 'relation_kind': 'near', 'camera': 'agentview'}
    robot = Robot([surface(.05), surface(.45, .2)])
    result = namespace['locate_contextual_task_relation'](robot, memory)
    assert result['status'] == 'estimated_surface'
    assert result['surface_quantiles']['median'] == [.52, .2, 0]
    assert not result['semantic_identity_verified']


def test_multiple_nearby_references_remain_ambiguous():
    result = contextual_reference.locate_reference_near_anchor(
        Robot([surface(.45, .2), surface(.5, .3)]), ['container'],
        camera='agentview', minimum_score=.45, anchor=[.72, .2, 0], radius=.5)
    assert result['status'] == 'reference_ambiguous'


def test_outside_neighborhood_returns_no_position_and_radius_is_bounded():
    kwargs = dict(camera='agentview', minimum_score=.45, anchor=[.72, .2, 0])
    result = contextual_reference.locate_reference_near_anchor(Robot([surface(.05)]),
                                                               ['container'], radius=.5, **kwargs)
    assert result['status'] == 'reference_not_located'
    with pytest.raises(ValueError):
        contextual_reference.locate_reference_near_anchor(Robot([]), ['container'], radius=100, **kwargs)
