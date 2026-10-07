"""Joint public support/target grounding; no contact or identity certification.

Keep this separate from the original helper so running native cohorts retain
their admitted dependencies. Candidate admission must explicitly select it.
"""


def locate_supported_target(robot, memory):
    supports = []
    for prompt in memory['support_prompts']:
        for item in robot.locate_objects(prompt, camera=memory['camera'])['objects']:
            if (item.get('status') == 'estimated_surface'
                    and item.get('detection_score', 0) >= memory['support_minimum_score']
                    and (memory['support_allow_clipped'] or not item.get('touches_image_border', True))):
                supports.append((prompt, item['surface_quantiles']))
    if not supports:
        return {'status': 'support_not_located', 'semantic_identity_verified': False}
    eligible = []
    for item in robot.locate_objects('black bowl', camera=memory['camera'])['objects']:
        if (item.get('status') != 'estimated_surface'
                or item.get('detection_score', 0) < memory['supported_target_minimum_score']):
            continue
        target = item['surface_quantiles']
        matching = []
        for prompt, bounds in supports:
            if any(target['median'][i] < bounds['lower_05'][i] - memory['support_xy_margin_m']
                   or target['median'][i] > bounds['upper_95'][i] + memory['support_xy_margin_m'] for i in range(2)):
                continue
            gap = target['lower_05'][2] - bounds['upper_95'][2]
            if memory['support_gap_min_m'] <= gap <= memory['support_gap_max_m']:
                matching.append(prompt)
        if matching:
            eligible.append((item, matching))
    if len(eligible) != 1:
        return {'status': 'supported_target_ambiguous' if eligible else 'supported_target_not_located',
                'semantic_identity_verified': False}
    item, matching = eligible[0]
    return dict(item, semantic_identity_verified=False,
                public_relation={'kind': 'above_visible_support',
                                 'support_prompts': sorted(set(matching)),
                                 'support_candidates_matched': len(matching),
                                 'support_identity_verified': False,
                                 'contact_verified': False})
