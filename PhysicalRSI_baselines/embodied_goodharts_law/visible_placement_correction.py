"""Bounded placement hypothesis from public visible surfaces, not object GT."""
import math


def propose_visible_correction(bowl, plate, *, supported_points, reference_points,
                               maximum_translation_m=.05, deadband_m=.005):
    if (type(supported_points) is not int or type(reference_points) is not int
            or not 0 <= supported_points <= reference_points):
        raise ValueError('Invalid public projection counts')
    if (not math.isfinite(maximum_translation_m) or not 0 < maximum_translation_m <= .05
            or not math.isfinite(deadband_m) or not 0 <= deadband_m < maximum_translation_m):
        raise ValueError('Invalid correction limits')
    result = dict(qualification=None, full_object_center_verified=False)
    if reference_points < 100 or supported_points / reference_points < .2:
        return dict(result, status='insufficient_plate_support')
    for geometry in (bowl, plate):
        if geometry['status'] != 'estimated_surface' or geometry['touches_image_border']:
            return dict(result, status='incomplete_visible_surface')
        q = geometry['surface_quantiles']
        for key in ('lower_05', 'median', 'upper_95'):
            if len(q[key]) != 3 or not all(math.isfinite(x) for x in q[key]):
                raise ValueError('Invalid surface quantiles')
        if any(not q['lower_05'][i] <= q['median'][i] <= q['upper_95'][i] for i in range(3)):
            raise ValueError('Unordered surface quantiles')
    q = bowl['surface_quantiles']
    center = [(q['lower_05'][i] + q['upper_95'][i]) / 2 for i in range(2)]
    delta = [plate['surface_quantiles']['median'][i] - center[i] for i in range(2)]
    distance = math.hypot(*delta)
    if distance > maximum_translation_m:
        return dict(result, status='correction_exceeds_limit', distance_m=distance)
    return dict(result, status='not_needed' if distance <= deadband_m else 'correction_proposed',
                translation_m=delta + [0.], distance_m=distance)
