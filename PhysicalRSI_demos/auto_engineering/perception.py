"""Color-labelled sample detection from rendered RGB and metric depth only.

A deliberately inspectable detector for this demo, not a general detector.
No instance segmentation, object IDs or scene transforms are accepted.
"""
import numpy as np
from scipy.ndimage import label


def detect(rgb, depth, unproject):
    image = np.asarray(rgb)[..., :3].astype(float)/255.
    r,g,b = image[...,0],image[...,1],image[...,2]
    masks = {'tube': (b>2*r)&(b>1.6*g)&(b>.18),
             'receiver': (g-r>.04)&(g-b>.025)&(g>.18),
             'obstacle': (r-b>.06)&(g-b>.025)&(r-g>.005)&(r>.3)}
    result = {}
    depth = np.asarray(depth).squeeze()
    for name,mask in masks.items():
        regions,n = label(mask & np.isfinite(depth) & (depth>.05) & (depth<3))
        if not n:
            result[name] = {'status':'not_detected'}
            continue
        counts = np.bincount(regions.ravel());counts[0]=0
        v,u = np.nonzero(regions==int(counts.argmax()))
        if len(u)<12:
            result[name] = {'status':'insufficient_pixels'}
            continue
        points = np.asarray(unproject(np.column_stack([u,v]), depth[v,u]))
        lo,hi = np.quantile(points,[.05,.95],axis=0)
        # The receiving pad is a support surface; minority elevated pixels
        # should not move its estimated landing height. Other objects retain
        # their upper-surface estimate.
        surface_z = np.median(points[:,2]) if name == 'receiver' else hi[2]
        result[name] = dict(status='estimated_surface', center_xy=((lo[:2]+hi[:2])/2).tolist(),
                            top_z=float(surface_z), lower=lo.tolist(), upper=hi.tolist(), pixels=len(u))
    return result
