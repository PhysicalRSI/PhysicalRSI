"""Screen non-finger robot geometry against a partial public target surface.

This is a conservative model diagnostic, not a native contact detector. A clear
visible surface cannot establish clearance from unobserved object geometry.
"""
import numpy as np


def public_target_clearance(surface_points, link_spheres):
    """Accept robot-base points and named Nx4 [x,y,z,radius] sphere arrays."""
    points = np.asarray(surface_points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError('Expected nonempty finite public surface points')
    if not isinstance(link_spheres, dict) or not link_spheres:
        raise ValueError('Expected named robot collision spheres')
    rows = []
    for name, value in link_spheres.items():
        spheres = np.asarray(value, dtype=float)
        if (not isinstance(name, str) or not name or spheres.ndim != 2
                or spheres.shape[1] != 4 or not np.isfinite(spheres).all()
                or (spheres[:, 3] <= 0).any()):
            raise ValueError('Expected finite positive-radius named spheres')
        if name in ('panda_leftfinger', 'panda_rightfinger') or not len(spheres):
            continue
        distances = np.full(len(spheres), np.inf)
        for start in range(0, len(points), 2048):
            delta = points[start:start + 2048, None, :] - spheres[None, :, :3]
            distances = np.minimum(distances, np.linalg.norm(delta, axis=-1).min(axis=0))
        clearance = distances - spheres[:, 3]
        rows.append({'link': name, 'minimum_surface_clearance_m': float(clearance.min()),
                     'overlapping_spheres': int((clearance < 0).sum())})
    if not rows:
        raise ValueError('No non-finger geometry checked')
    overlap = any(row['overlapping_spheres'] for row in rows)
    return {'status': 'observed_surface_overlap' if overlap else 'no_observed_surface_overlap',
            'links': rows, 'surface_points': len(points),
            'full_object_clearance_verified': False, 'native_contact_verified': False}
