import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.projected_motion_admission import admit_projected_motion

MEMORY = dict(minimum_reference_points=100, minimum_support_fraction=.2,
              minimum_motion_advantage=.15, ambiguity_margin=.1)


def row(camera, moving, stationary, mask='target', n=1000):
    return dict(camera=camera, mask=mask, reference_points=n, hypotheses={
        'stationary': {'supported': stationary, 'behind_reference': n-stationary},
        'hand_rigid_motion': {'supported': moving, 'outside_view': n-moving}})


def check(rows):
    return admit_projected_motion({'rows': rows}, MEMORY)


def test_partial_visibility_can_support_motion_without_certifying_attachment():
    result = check([row('agentview', 280, 0), row('robot0_eye_in_hand', 970, 0)])
    assert result['admitted'] and result['attachment_verified'] is False
    assert result['qualification'] is None


def test_empty_grasp_is_rejected_even_when_one_camera_supports_motion():
    assert not check([row('agentview', 0, 970), row('robot0_eye_in_hand', 970, 0)])['admitted']


def test_small_motion_does_not_discriminate_stationary_from_attached():
    assert not check([row('agentview', 900, 850), row('robot0_eye_in_hand', 970, 0)])['admitted']


def test_ambiguous_surfaces_and_missing_camera_are_rejected():
    assert check([row('agentview', 800, 0), row('agentview', 750, 0, 'other')])['reason'] == 'ambiguous_moving_surface'
    assert not check([row('agentview', 800, 0)])['admitted']


def test_tiny_or_corrupt_evidence_cannot_pass():
    assert not check([row('agentview', 90, 0, n=99)])['admitted']
    bad = row('agentview', 800, 0)
    bad['hypotheses']['stationary']['supported'] = 10
    with pytest.raises(ValueError, match='partition'):
        check([bad])
