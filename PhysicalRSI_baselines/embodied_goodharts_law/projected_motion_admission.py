"""Development admission heuristic for partial public surface motion evidence."""
import math


def admit_projected_motion(report, memory):
    """Select an unambiguous moving surface in each declared camera.

    This consumes projection counts, not object state. Admission does not
    certify attachment, semantic identity, or complete object geometry.
    """
    cameras = ('agentview', 'robot0_eye_in_hand')
    minimum = memory['minimum_reference_points']
    if type(minimum) is not int or minimum < 100:
        raise ValueError('Declare at least 100 reference samples')
    keys = ('minimum_support_fraction', 'minimum_motion_advantage', 'ambiguity_margin')
    for key in keys:
        value = memory[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Invalid projection admission fraction')
    selected = {}
    for camera in cameras:
        candidates = []
        for row in report['rows']:
            if row['camera'] != camera:
                continue
            n = row['reference_points']
            if type(n) is not int or n <= 0:
                raise ValueError('Invalid reference sample count')
            counts = row['hypotheses']
            for hypothesis in ('stationary', 'hand_rigid_motion'):
                values = counts[hypothesis].values()
                if any(type(x) is not int or x < 0 for x in values) or sum(values) != n:
                    raise ValueError('Projection counts must partition the reference samples')
            if n < minimum:
                continue
            moving = counts['hand_rigid_motion'].get('supported', 0) / n
            stationary = counts['stationary'].get('supported', 0) / n
            candidates.append((moving, stationary, row['mask']))
        candidates.sort(key=lambda x: (-x[0], x[2]))
        if not candidates:
            return dict(admitted=False, reason='insufficient_reference', camera=camera)
        moving, stationary, mask = candidates[0]
        if (moving < memory['minimum_support_fraction'] or
                moving - stationary < memory['minimum_motion_advantage']):
            return dict(admitted=False, reason='insufficient_motion_support', camera=camera)
        if len(candidates) > 1 and moving - candidates[1][0] < memory['ambiguity_margin']:
            return dict(admitted=False, reason='ambiguous_moving_surface', camera=camera)
        selected[camera] = dict(mask=mask, moving_support=moving, stationary_support=stationary)
    return dict(admitted=True, matches=selected, attachment_verified=False,
                semantic_identity_verified=False, qualification=None)
