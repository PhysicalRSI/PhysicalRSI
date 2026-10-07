import time

import numpy as np
import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.public_kinematic_preview import PublicKinematicPreviewMixin
from PhysicalRSI_baselines.embodied_goodharts_law.persistent_aspire_ik import IKNonConvergence


class Base:
    @property
    def handlers(self):
        return {}

    def get_robot_state(self):
        return {'robot_joint_pos': [0.] * 7 + [.7]}


class Preview(PublicKinematicPreviewMixin, Base):
    _joint_limits = np.array([[-2., 2.]] * 7)


POSE = [.5, 0., .2, 1., 0., 0., 0.]


def call(api, poses, deadline=None):
    return api.handlers['preview_poses']([poses], {}, deadline=time.monotonic()+10 if deadline is None else deadline)


def test_preview_seeds_each_waypoint_without_forwarding_solver_fields():
    api = Preview()
    seeds = []
    def solve(**kwargs):
        seeds.append(kwargs['joints'].tolist())
        assert kwargs['gripper_fraction'] == .7
        return {'joints': [len(seeds)*.1]*7, 'private_debug': 'must not leave host'}
    api._pose_solver = solve
    result = call(api, [POSE, POSE])
    assert seeds == [[0.]*7, [.1]*7]
    assert result['kinematically_solved'] and result['solved_waypoints'] == 2
    assert not result['native_effects'] and not result['collision_checked']
    assert 'private_debug' not in result


def test_only_expected_nonconvergence_becomes_failure():
    api = Preview()
    def failed(**kwargs):
        raise IKNonConvergence('IK target did not converge')
    api._pose_solver = failed
    assert call(api, [POSE])['reason'] == 'solver_nonconvergence'
    def broken(**kwargs):
        raise RuntimeError('worker crashed')
    api._pose_solver = broken
    with pytest.raises(RuntimeError, match='worker crashed'):
        call(api, [POSE])


def test_limits_deadlines_and_invalid_inputs():
    api = Preview()
    api._pose_solver = lambda **kwargs: {'joints': [3.]*7}
    assert call(api, [POSE])['reason'] == 'joint_limits'
    with pytest.raises(TimeoutError):
        call(api, [POSE], deadline=time.monotonic()-1)
    for poses in [[], [POSE]*9, [[0.]*7], [[float('nan')]+POSE[1:]]]:
        with pytest.raises(ValueError):
            call(api, poses)
