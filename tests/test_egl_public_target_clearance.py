import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.public_target_clearance import public_target_clearance


def test_palm_overlap_rejected_even_when_fingers_are_allowed_contact():
    points = [[0., 0., 0.]]
    spheres = {'panda_hand': [[.01, 0., 0., .02]],
               'panda_leftfinger': [[0., 0., 0., .01]]}
    result = public_target_clearance(points, spheres)
    assert result['status'] == 'observed_surface_overlap'
    assert result['links'][0]['minimum_surface_clearance_m'] == pytest.approx(-.01)
    spheres['panda_hand'][0][0] = .1
    result = public_target_clearance(points, spheres)
    assert result['status'] == 'no_observed_surface_overlap'
    assert not result['full_object_clearance_verified']
    assert not result['native_contact_verified']


def test_all_point_chunks_checked_and_rigid_frame_invariance():
    points = np.full((4097, 3), 10.); points[-1] = 0.
    spheres = np.array([[.01, 0., 0., .02]])
    before = public_target_clearance(points, {'panda_link7': spheres})
    offset = np.array([3., -2., 1.])
    spheres[:, :3] += offset
    after = public_target_clearance(points + offset, {'panda_link7': spheres})
    assert before['status'] == after['status'] == 'observed_surface_overlap'
    assert after['links'][0]['minimum_surface_clearance_m'] == pytest.approx(-.01)


@pytest.mark.parametrize('points,spheres', [
    ([], {'panda_hand': [[0, 0, 0, 1]]}),
    ([[0, 0, float('nan')]], {'panda_hand': [[0, 0, 0, 1]]}),
    ([[0, 0, 0]], {'panda_hand': [[0, 0, 0, -1]]}),
    ([[0, 0, 0]], {'panda_leftfinger': [[0, 0, 0, 1]]}),
])
def test_invalid_or_unchecked_geometry_fails_closed(points, spheres):
    with pytest.raises(ValueError):
        public_target_clearance(points, spheres)
