"""Candidate support-relation grounding from public visible-surface bounds.

Bounds can be incomplete, including image-border clipping. They do not certify
contact, full extent, or object identity. All tolerances are proposal memory.
"""


def locate_supported_target(robot, memory):
    support = None
    for prompt in memory['support_prompts']:
        candidates = [item for item in robot.locate_objects(prompt, camera=memory['camera'])['objects']
                      if item.get('status') == 'estimated_surface'
                      and item.get('detection_score', 0) >= memory['support_minimum_score']
                      and (memory['support_allow_clipped'] or not item.get('touches_image_border', True))]
        if len(candidates) > 1:
            return {'status': 'support_ambiguous', 'semantic_identity_verified': False}
        if candidates:
            support = candidates[0]
            break
    if support is None:
        return {'status': 'support_not_located', 'semantic_identity_verified': False}
    bounds = support['surface_quantiles']
    eligible = []
    for item in robot.locate_objects('black bowl', camera=memory['camera'])['objects']:
        if item.get('status') != 'estimated_surface' or item.get('detection_score', 0) < memory['supported_target_minimum_score']:
            continue
        target = item['surface_quantiles']
        center = target['median']
        if any(center[i] < bounds['lower_05'][i]-memory['support_xy_margin_m']
               or center[i] > bounds['upper_95'][i]+memory['support_xy_margin_m'] for i in range(2)):
            continue
        gap = target['lower_05'][2]-bounds['upper_95'][2]
        if not memory['support_gap_min_m'] <= gap <= memory['support_gap_max_m']:
            continue
        eligible.append(item)
    if len(eligible) != 1:
        return {'status': 'supported_target_ambiguous' if eligible else 'supported_target_not_located',
                'semantic_identity_verified': False}
    return dict(eligible[0], semantic_identity_verified=False,
                public_relation={'kind': 'above_visible_support',
                                 'support_prompt': prompt,
                                 'contact_verified': False})
