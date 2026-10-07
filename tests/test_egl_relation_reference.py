import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.relation_reference import locate_reference


def surface(**changes):
    return dict(status='estimated_surface', detection_score=.9,
                touches_image_border=False,
                surface_quantiles={'median': [.1, .2, .3]}, **changes)


class PublicRobot:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def locate_objects(self, prompt, *, camera):
        assert camera == 'agentview'
        self.calls.append(prompt)
        return {'objects': self.replies[prompt]}


def test_missing_specialized_word_uses_declared_alias_without_semantic_claim():
    robot = PublicRobot({'ramekin': [], 'small bowl': [surface()]})
    result = locate_reference(robot, ['ramekin', 'small bowl'], camera='agentview')
    assert result['status'] == 'estimated_surface'
    assert result['matched_prompt'] == 'small bowl'
    assert not result['semantic_identity_verified']
    assert result['surface_quantiles']['median'] == [.1, .2, .3]


def test_ambiguous_alias_does_not_pick_highest_score_or_try_another_prompt():
    robot = PublicRobot({'ramekin': [], 'bowl': [surface(), surface()]})
    result = locate_reference(robot, ['ramekin', 'bowl', 'dish'], camera='agentview')
    assert result['status'] == 'reference_ambiguous'
    assert robot.calls == ['ramekin', 'bowl']


def test_clipped_or_low_confidence_detections_do_not_become_reference():
    clipped = surface(); clipped['touches_image_border'] = True
    weak = surface(); weak['detection_score'] = .1
    robot = PublicRobot({'ramekin': [clipped, weak]})
    assert locate_reference(robot, ['ramekin'], camera='agentview')['status'] == 'reference_not_located'


def test_declared_stricter_threshold_preserves_ambiguity_at_that_threshold():
    strongest = surface(); strongest['detection_score'] = .81
    weaker = surface(); weaker['detection_score'] = .47
    robot = PublicRobot({'round silver container': [strongest, weaker]})
    result = locate_reference(robot, ['round silver container'], camera='agentview', minimum_score=.6)
    assert result['status'] == 'estimated_surface'
    assert result['detection_score'] == .81
    assert not result['semantic_identity_verified']
    weaker['detection_score'] = .7
    assert locate_reference(robot, ['round silver container'], camera='agentview', minimum_score=.6)['status'] == 'reference_ambiguous'


@pytest.mark.parametrize('prompts', [[], ['a']*2, ['a', 'b', 'c', 'd'], [' ']])
def test_prompt_budget_is_validated_before_perception(prompts):
    robot = PublicRobot({})
    with pytest.raises(ValueError):
        locate_reference(robot, prompts, camera='agentview')
    assert robot.calls == []
