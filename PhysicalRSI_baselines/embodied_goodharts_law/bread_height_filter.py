"""Filter tabletop bread proposals using visible basket-relative height only.

The basket surface is an uncertain public RGB-D reference. This does not certify
object identity, contact, full geometry, or a successful grasp.
"""
import math


def filter_bread_height(pieces, basket_bounds, maximum_above_m=.08):
    ceiling = basket_bounds['upper_95'][2] + maximum_above_m
    accepted = []
    for item in pieces:
        height = item['surface_quantiles']['median'][2]
        if math.isfinite(height) and height <= ceiling:
            accepted.append(item)
    return accepted
