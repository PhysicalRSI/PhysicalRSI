"""Compare world/base points with an independent public masked depth view.

The caller owns frame agreement and temporal association. Lack of support is
not proof that a point is wrong: occlusion and new visible surfaces are common.
"""
import numpy as np
from .public_geometry import masked_surface_geometry


def cross_view_surface_support(points, camera, mask, *, tolerance_m=.005):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) > 100000
            or not np.isfinite(points).all() or np.any(np.abs(points) > 20)
            or not np.isfinite(tolerance_m) or not 0 < tolerance_m <= .05):
        raise ValueError('Invalid public point support query')
    masked_surface_geometry(camera, mask, depth_range_m=(.015,20.))
    depth = np.asarray(camera['images']['depth'], dtype=float).squeeze()
    mask = np.asarray(mask, dtype=bool)
    k = np.asarray(camera['intrinsics'], dtype=float)
    pose = np.asarray(camera['pose_mat'], dtype=float)
    local = (points-pose[:3,3]) @ pose[:3,:3]
    states = np.full(len(points), 'outside_view', dtype='<U32')
    positive = np.flatnonzero(local[:,2] > .015)
    uv = local[positive,:2]/local[positive,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    h,w = depth.shape
    inside = (uv[:,0]>=0)&(uv[:,0]<w-.5)&(uv[:,1]>=0)&(uv[:,1]<h-.5)
    indices=positive[inside];uv=np.rint(uv[inside]).astype(int)
    measured=depth[uv[:,1],uv[:,0]]
    valid=np.isfinite(measured)&(measured>.015)&(measured<20.)
    states[indices]='invalid_reference_depth'
    indices=indices[valid];uv=uv[valid];measured=measured[valid]
    delta=local[indices,2]-measured
    states[indices]='unmasked_surface'
    states[indices[delta < -tolerance_m]]='in_front_of_reference'
    states[indices[delta > tolerance_m]]='behind_reference'
    supported=(np.abs(delta)<=tolerance_m)&mask[uv[:,1],uv[:,0]]
    states[indices[supported]]='supported'
    return dict(states=states, supported=states=='supported',
                semantic_identity_verified=False, temporal_stability_verified=False,
                tolerance_m=tolerance_m)
