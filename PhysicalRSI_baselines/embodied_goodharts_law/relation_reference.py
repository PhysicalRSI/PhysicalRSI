"""Resolve declared reference descriptions using public RGB-D detections only.

Aliases are proposal memory, not verified semantic labels. This helper makes
no simulator queries and never substitutes a stored position for perception.
"""


def locate_reference(robot, prompts, *, camera, minimum_score=0.4):
    """Try at most three descriptions; reject ambiguous or clipped surfaces.

    A broad alias may recover a missed detection but cannot prove identity.
    Ambiguity stops the search rather than allowing another prompt to hide it.
    Geometry deduplication belongs to the public multi-surface adapter.
    """
    if (not isinstance(prompts, (list, tuple)) or not 1 <= len(prompts) <= 3
            or any(not isinstance(p, str) or not p.strip() for p in prompts)
            or len(set(prompts)) != len(prompts)):
        raise ValueError('Declare one to three distinct reference prompts')
    if not 0 <= minimum_score <= 1:
        raise ValueError('Invalid detection confidence threshold')
    attempts = []
    for prompt in prompts:
        reply = robot.locate_objects(prompt, camera=camera)
        candidates = []
        for item in reply['objects']:
            if (item.get('status') == 'estimated_surface'
                    and item.get('detection_score', 0) >= minimum_score
                    and not item.get('touches_image_border', True)):
                candidates.append(item)
        attempts.append({'prompt': prompt, 'eligible_surfaces': len(candidates)})
        if len(candidates) > 1:
            return {'status': 'reference_ambiguous', 'attempts': attempts,
                    'semantic_identity_verified': False}
        if candidates:
            return dict(candidates[0], matched_prompt=prompt,
                        attempts=attempts, semantic_identity_verified=False)
    return {'status': 'reference_not_located', 'attempts': attempts,
            'semantic_identity_verified': False}


def locate_task_relation(robot, memory):
    references = []
    for prompt in memory['relation_references']:
        value = locate_reference(robot, memory['reference_prompts'].get(prompt, [prompt]),
                                 camera=memory['camera'],
                                 minimum_score=memory.get('reference_minimum_score', .4))
        if value.get('status') != 'estimated_surface' or value.get('detection_score',0) < .4 or value.get('touches_image_border',True):
            return {'status':'reference_not_located','semantic_identity_verified':False}
        references.append(value['surface_quantiles']['median'])
    ranked = []
    for item in robot.locate_objects('black bowl', camera=memory['camera'])['objects']:
        if item.get('detection_score',0) < (.5 if item.get('touches_image_border',True) else .4):
            continue
        center = item['surface_quantiles']['median']
        if memory['relation_kind'] == 'near':
            distance = sum((center[i]-references[0][i])**2 for i in range(2))**.5
            if distance > .25:continue
        elif memory['relation_kind'] == 'between':
            axis = [references[1][i]-references[0][i] for i in range(2)]
            length2 = sum(x*x for x in axis)
            if length2 < .05**2:
                return {'status':'ambiguous_references','semantic_identity_verified':False}
            fraction = sum((center[i]-references[0][i])*axis[i] for i in range(2))/length2
            deviation = sum((center[i]-references[0][i]-fraction*axis[i])**2 for i in range(2))**.5
            if not .1 <= fraction <= .9 or deviation > .10:continue
            distance = sum((center[i]-(references[0][i]+references[1][i])/2)**2 for i in range(2))**.5
        else:
            raise ValueError('Unsupported task relation')
        ranked.append((distance,item))
    ranked.sort(key=lambda x:x[0])
    if not ranked or (len(ranked)>1 and ranked[1][0]-ranked[0][0] < .025):
        return {'status':'relative_target_ambiguous','semantic_identity_verified':False}
    return dict(ranked[0][1],semantic_identity_verified=False,public_relation={'kind':memory['relation_kind'],'reference_prompts':memory['relation_references'],'distance_m':ranked[0][0]})
