"""Candidate relation grounding with a declared public destination neighborhood.

Uses locate_reference from the reviewed relation helper. The radius is a task
memory hypothesis, not a simulator workspace boundary or semantic identity proof.
"""


def locate_reference_near_anchor(robot, prompts, *, camera, minimum_score, anchor, radius):
    if not isinstance(radius, (float, int)) or not .1 <= radius <= 1.0:
        raise ValueError('Invalid reference neighborhood radius')
    attempts = []
    for prompt in prompts:
        eligible = []
        for item in robot.locate_objects(prompt, camera=camera)['objects']:
            if (item.get('status') != 'estimated_surface'
                    or item.get('detection_score', 0) < minimum_score
                    or item.get('touches_image_border', True)):
                continue
            center = item['surface_quantiles']['median']
            distance = sum((center[i]-anchor[i])**2 for i in range(2))**.5
            if distance <= radius:
                eligible.append(item)
        attempts.append({'prompt': prompt, 'eligible_surfaces': len(eligible)})
        if len(eligible) > 1:
            return {'status': 'reference_ambiguous', 'attempts': attempts,
                    'semantic_identity_verified': False}
        if eligible:
            return dict(eligible[0], attempts=attempts, semantic_identity_verified=False)
    return {'status': 'reference_not_located', 'attempts': attempts,
            'semantic_identity_verified': False}


def locate_contextual_task_relation(robot, memory):
    anchor = locate_reference(robot, [memory["reference_anchor_prompt"]], camera=memory["camera"], minimum_score=.4)
    if anchor.get("status") != "estimated_surface":
        return {"status": "anchor_not_located", "semantic_identity_verified": False}
    anchor_center = anchor["surface_quantiles"]["median"]
    references = []
    for prompt in memory['relation_references']:
        value = locate_reference_near_anchor(robot, memory['reference_prompts'].get(prompt, [prompt]),
                                 camera=memory['camera'],
                                 minimum_score=memory.get('reference_minimum_score', .4),
                                 anchor=anchor_center, radius=memory["reference_anchor_radius_m"])
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
