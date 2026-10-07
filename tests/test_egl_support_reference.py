from PhysicalRSI_baselines.embodied_goodharts_law.support_reference import locate_supported_target


def surface(x, z, score=.8, clipped=False):
    return {'status': 'estimated_surface', 'detection_score': score,
            'touches_image_border': clipped, 'surface_quantiles': {
                'lower_05': [x-.04, -.04, z-.02], 'median': [x, 0, z],
                'upper_95': [x+.04, .04, z+.02]}}


MEMORY = {'support_prompts': ['cabinet'], 'support_minimum_score': .4,
          'support_allow_clipped': True, 'supported_target_minimum_score': .3,
          'support_xy_margin_m': .04, 'support_gap_min_m': -.04,
          'support_gap_max_m': .15, 'camera': 'agentview'}


class Robot:
    def __init__(self, targets, supports=None):
        self.targets = targets
        self.supports = [surface(.5, .2, clipped=True)] if supports is None else supports

    def locate_objects(self, prompt, **kwargs):
        return {'objects': self.supports if prompt == 'cabinet' else self.targets}


def test_support_relation_selects_lower_confidence_clipped_target():
    selected = surface(.5, .25, score=.35, clipped=True)
    result = locate_supported_target(Robot([surface(.8, .05, score=.95), selected]), MEMORY)
    assert result['surface_quantiles'] == selected['surface_quantiles']
    assert not result['semantic_identity_verified']
    assert not result['public_relation']['contact_verified']


def test_multiple_valid_targets_or_supports_remain_ambiguous():
    assert locate_supported_target(Robot([surface(.5, .25), surface(.52, .25)]), MEMORY)['status'] == 'supported_target_ambiguous'
    assert locate_supported_target(Robot([], [surface(.5, .2), surface(.8, .2)]), MEMORY)['status'] == 'support_ambiguous'


def test_missing_support_and_vertical_mismatch_do_not_supply_a_target():
    assert locate_supported_target(Robot([], []), MEMORY)['status'] == 'support_not_located'
    assert locate_supported_target(Robot([surface(.5, -.1)]), MEMORY)['status'] == 'supported_target_not_located'
