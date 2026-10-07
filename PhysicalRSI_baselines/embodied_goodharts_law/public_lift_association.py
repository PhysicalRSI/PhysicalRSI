"""Conservative association of visible surfaces during a near-vertical lift."""
import math


def associated_visible_lift(before, after, hand, max_xy_m=.12, minimum_lift_m=.05):
    if any(value.get('status') != 'estimated_surface' for value in [before, after]):
        return False
    old = before['surface_quantiles']['median']
    new = after['surface_quantiles']['median']
    if not all(math.isfinite(x) for x in old + new + hand[:3]):
        return False
    return (new[2]-old[2] >= minimum_lift_m
            and sum((new[i]-old[i])**2 for i in range(2)) <= max_xy_m**2
            and sum((new[i]-hand[i])**2 for i in range(2)) <= max_xy_m**2)
