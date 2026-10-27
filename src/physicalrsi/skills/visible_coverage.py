"""Test planar target coverage using observed points, without filling hidden areas."""
import math
from pathlib import Path
from PhysicalRSI_core.infra.storage import digest, file_digest
from .protocol import CallableSkill, SkillDescriptor


def visible_coverage(*, resolution_m=.005, minimum_points=9):
    if type(resolution_m) not in (int, float) or not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError('Positive finite resolution required')
    if type(minimum_points) is not int or minimum_points < 1:
        raise ValueError('Positive minimum point count required')
    config = dict(resolution_m=resolution_m, minimum_points=minimum_points,
                  measurement='observed_planar_coverage_not_hidden_geometry_or_support_strength')

    def run(value, context):
        import numpy as np
        from scipy.spatial import cKDTree
        if set(value) != {'points_world_m', 'target_xy_m', 'radius_m'}:
            raise ValueError('Expected only observed points, target and uncertainty radius')
        points = np.asarray(value['points_world_m'], dtype=float)
        target = np.asarray(value['target_xy_m'], dtype=float)
        radius = value['radius_m']
        if points.size == 0:
            points = np.empty((0, 3))
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Finite observed xyz points required')
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Finite planar target required')
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
            raise ValueError('Nonnegative finite radius required')
        # Samples approximate the disk; this is explicitly not continuous coverage proof.
        steps = math.ceil(radius / resolution_m)
        if steps > 200:
            raise ValueError('Target exceeds coverage sampling budget')
        offsets = np.array([(x*resolution_m, y*resolution_m)
                           for x in range(-steps, steps+1) for y in range(-steps, steps+1)
                           if math.hypot(x*resolution_m, y*resolution_m) <= radius])
        boundary = (np.array([[math.cos(a), math.sin(a)] for a in
                    np.linspace(0, 2*math.pi, max(8, math.ceil(2*math.pi*radius/resolution_m)), endpoint=False)])*radius)
        queries = np.vstack([offsets, boundary]) + target
        context.check()
        if len(points) < minimum_points:
            return dict(status='unknown', reason='insufficient_observed_points', sampled_coverage=0., samples=len(queries))
        distance, _ = cKDTree(points[:, :2]).query(queries)
        covered = distance <= resolution_m
        return dict(status='observed_coverage' if covered.all() else 'unknown',
                    reason=None if covered.all() else 'unobserved_target_samples',
                    sampled_coverage=float(covered.mean()), samples=len(queries),
                    maximum_nearest_observation_distance_m=float(distance.max()))

    return CallableSkill(SkillDescriptor('visible_coverage',
        digest(dict(source=file_digest(Path(__file__)), parameters=config)),
        {'name': 'physicalrsi.observed-coverage-request/v1', 'unit': 'SI', 'frame': 'world'},
        {'name': 'physicalrsi.observed-coverage/v1'}, metadata=config), run)
