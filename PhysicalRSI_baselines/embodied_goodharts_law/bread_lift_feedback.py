"""Public surface association after a bread lift; never a grasp certificate."""
import math


def bread_lift_feedback(before, objects, measured_hand, *, minimum_score=.5,
                        stationary_radius_m=.02, minimum_rise_m=.05,
                        association_radius_m=.06):
    """A unique nearby stationary surface can falsify the intended pickup.

    Partial surfaces and segmentation errors limit this heuristic. Uncertain
    or absent detections remain unknown, not a positive grasp observation.
    """
    limits = [minimum_score, stationary_radius_m, minimum_rise_m, association_radius_m]
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in limits)
            or not 0 <= minimum_score <= 1
            or not 0 < stationary_radius_m < minimum_rise_m <= .3
            or not stationary_radius_m < association_radius_m <= .3):
        raise ValueError('Invalid public lift association thresholds')
    def center(item):
        if item.get('status') != 'estimated_surface' or item.get('frame') != 'world':
            return None
        value = item.get('surface_quantiles', {}).get('median')
        if (not isinstance(value, (list, tuple)) or len(value) != 3
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
            return None
        return value
    initial = center(before)
    result = dict(status='unknown', semantic_identity_verified=False, grasp_verified=False)
    if initial is None:
        return result
    if (len(measured_hand) < 3 or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                         for v in measured_hand[:3])):
        return result
    nearby = []
    for item in objects:
        point = center(item)
        score = item.get('detection_score', 0)
        if (point is None or item.get('touches_image_border', True)
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or score < minimum_score):
            continue
        distance = sum((point[i]-initial[i])**2 for i in range(2))**.5
        if distance <= association_radius_m:
            nearby.append((point, distance))
    result['nearby_surface_count'] = len(nearby)
    if len(nearby) != 1:
        return result
    point, distance = nearby[0]
    rise = point[2]-initial[2]
    result.update(surface_rise_m=rise, surface_xy_displacement_m=distance)
    if distance <= stationary_radius_m and abs(rise) <= stationary_radius_m:
        result['status'] = 'stationary_surface_observed'
    elif (rise >= minimum_rise_m
          and sum((point[i]-measured_hand[i])**2 for i in range(2)) <= association_radius_m**2):
        result['status'] = 'raised_surface_near_hand'
    return result
