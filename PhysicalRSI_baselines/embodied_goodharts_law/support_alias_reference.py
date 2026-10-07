"""Alias fallback with the same public support relation and ambiguity checks.

Aliases nominate visible surfaces, not certified semantic identities. Keep the
original grounding helper unchanged for already frozen experiments.
"""

from .support_pair_reference import locate_supported_target as original_grounding


def locate_supported_target(robot, memory):
    cache = {}

    class View:
        def __init__(self, target):
            self.target = target

        def locate_objects(self, prompt, *, camera):
            actual = self.target if prompt == 'black bowl' else prompt
            key = (camera, actual)
            if key not in cache:
                cache[key] = robot.locate_objects(actual, camera=camera)
            return cache[key]

    original = original_grounding(View('black bowl'), memory)
    if original['status'] != 'supported_target_not_located':
        return original
    aliases = memory.get('target_aliases', {}).get('black bowl', [])
    if not isinstance(aliases, list) or len(aliases) > 4:
        raise ValueError('Declare at most four target aliases')
    candidates = []
    for prompt in dict.fromkeys(aliases):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Target aliases must be nonempty strings')
        if prompt == 'black bowl':
            continue
        found = original_grounding(View(prompt), memory)
        if found['status'] == 'supported_target_ambiguous':
            return found
        if found['status'] == 'estimated_surface':
            candidates.append((prompt, found))
    if not candidates:
        return original
    # Never resolve conflicting aliases merely by taking the first result.
    centers = [item['surface_quantiles']['median'] for _, item in candidates]
    if any(sum((a[i] - b[i]) ** 2 for i in range(3)) > .03 ** 2
           for a in centers for b in centers):
        return {'status': 'supported_target_ambiguous',
                'semantic_identity_verified': False}
    chosen = max(candidates, key=lambda pair: pair[1]['detection_score'])[1]
    return dict(chosen, semantic_identity_verified=False,
                alias_evidence={'original_status': original['status'],
                                'matched_prompts': [p for p, _ in candidates],
                                'identity_verified': False})
