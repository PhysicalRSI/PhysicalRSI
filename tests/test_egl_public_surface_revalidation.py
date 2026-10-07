from copy import deepcopy

import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.public_surface_revalidation import compare_projected_surface


def fixture():
    camera = {'images': {'depth': np.ones((8, 8)), 'rgb': np.zeros((8, 8, 3), np.uint8)},
              'intrinsics': np.array([[8., 0, 3.5], [0, 8., 3.5], [0, 0, 1.]]), 'pose_mat': np.eye(4)}
    mask = np.zeros((8, 8), bool); mask[1:7, 1:7] = True
    return camera, mask


def test_shared_surface_changes_only_count_projected_target_pixels():
    camera, mask = fixture(); after = deepcopy(camera)
    after['images']['depth'][0, 0] = 2.
    result = compare_projected_surface(camera, mask, camera, after)
    assert result['compared_pixels'] == 36 and result['changed_depth_pixels'] == 0
    after['images']['depth'][3, 3] += .01
    after['images']['depth'][4, 4] = np.nan
    result = compare_projected_surface(camera, mask, camera, after)
    assert result['changed_depth_pixels'] == 2 and result['invalid_after_depth_pixels'] == 1
    assert not result['full_object_pose_verified'] and not result['grasp_verified']


def test_occluded_or_out_of_view_surface_cannot_be_declared_unchanged():
    camera, mask = fixture(); before = deepcopy(camera)
    before['images']['depth'][:] = .5
    assert compare_projected_surface(camera, mask, before, before)['status'] == 'insufficient_shared_surface'
    reference = deepcopy(camera); reference['pose_mat'][0, 3] = 100.
    assert compare_projected_surface(reference, mask, camera, camera)['status'] == 'insufficient_shared_surface'


def test_camera_motion_is_not_mistaken_for_object_motion():
    camera, mask = fixture(); after = deepcopy(camera); after['pose_mat'][0, 3] = .01
    with pytest.raises(ValueError, match='unchanged camera'):
        compare_projected_surface(camera, mask, camera, after)
