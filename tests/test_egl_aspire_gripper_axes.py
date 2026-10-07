import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.aspire_gripper_axes import (
    contact_grasp_panda_hand_variants,
)
from PhysicalRSI_baselines.embodied_goodharts_law.aspire_grasp_frame import (
    contact_grasp_to_model_hand,
)


def test_panda_y_contact_pair_matches_cgn_x_contact_pair():
    raw = np.array([[0., 0., 1., .7], [1., 0., 0., -.3],
                    [0., 1., 0., .3], [0., 0., 0., 1.]])
    saved = raw.copy()
    variants = contact_grasp_panda_hand_variants(raw)
    for width in [.02, .06, .08]:
        # ASPIRE's fingertip center is raw CGN local Z + .12 m.
        contacts = [raw @ [sign * width / 2, 0., .12, 1.]
                    for sign in [-1, 1]]
        for hand in variants:
            # Panda URDF finger motion is along hand-local Y, not X.
            actual = [hand @ [0., sign * width / 2, .1168, 1.]
                      for sign in [-1, 1]]
            for point in actual:
                assert min(np.linalg.norm(point - p) for p in contacts) < 1e-12
            np.testing.assert_allclose(hand[:3, 2], raw[:3, 2])
            assert np.linalg.det(hand[:3, :3]) == pytest.approx(1.)
    old = contact_grasp_to_model_hand(raw)
    assert abs(np.dot(old[:3, 1], raw[:3, 0])) < 1e-12
    np.testing.assert_array_equal(raw, saved)


def test_invalid_reflected_grasp_is_rejected_before_axis_mapping():
    with pytest.raises(ValueError):
        contact_grasp_panda_hand_variants(np.diag([1., 1., -1., 1.]))
