import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.aspire_grasp_frame import contact_grasp_to_model_hand
from PhysicalRSI_baselines.embodied_goodharts_law.aspire_grasp_frame import contact_grasp_hand_variants


def test_upstream_two_offsets_compose_in_local_axis_without_mutation():
    pose = np.array([[1., 0, 0, .7], [0, 0, -1., -.3], [0, 1., 0, .3], [0, 0, 0, 1.]])
    original = pose.copy()
    fingertip = pose.copy()
    fingertip[:3, 3] += pose[:3, 2] * .12
    expected = fingertip.copy()
    expected[:3, 3] -= pose[:3, 2] * .1168
    np.testing.assert_allclose(contact_grasp_to_model_hand(pose), expected)
    np.testing.assert_array_equal(pose, original)
    assert contact_grasp_to_model_hand(pose)[1, 3] == pytest.approx(-.3032)


def test_reflection_is_not_a_grasp_rotation():
    with pytest.raises(ValueError):
        contact_grasp_to_model_hand(np.diag([1., 1., -1., 1.]))


def test_half_turn_preserves_approach_and_exchanges_contact_points():
    pose = np.array([[0., 0, 1., .7], [1., 0, 0, -.3], [0, 1., 0, .3], [0, 0, 0, 1.]])
    original, rotated = contact_grasp_hand_variants(pose)
    np.testing.assert_allclose(original[:3, 3], rotated[:3, 3])
    np.testing.assert_allclose(original[:3, 2], rotated[:3, 2])
    for opening in [.01, .04]:
        np.testing.assert_allclose(original @ [opening, 0, .1168, 1],
                                   rotated @ [-opening, 0, .1168, 1])
    assert np.linalg.det(rotated[:3, :3]) == pytest.approx(1.)
