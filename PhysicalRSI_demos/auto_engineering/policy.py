"""Reviewed code-as-policy: observations -> pick / lift / carry / release.

The controller executes physical joint targets. No teleportation or attach joint.
"""


def plan(observation, memory):
    for name in ('tube','receiver','obstacle'):
        if observation[name]['status']!='estimated_surface':
            raise ValueError('Required visual detection missing: '+name)
    tube,goal = observation['tube'],observation['receiver']
    # Tube dimensions are the declared sample specification, not object state.
    # Default to the upper quarter; bounded search can vary the grasp depth.
    pick = [*tube['center_xy'], tube['top_z']-memory['sample_height_m']*memory.get('grasp_depth_fraction',.25)]
    place = [*goal['center_xy'], goal['top_z']+memory['sample_height_m']*.75]
    return dict(pick=pick, place=place, transit_height_m=memory['transit_height_m'],
                perception=observation, source='RGB-D color detection',
                task_success_claimed=False)


def propose_repair(feedback, parent_memory):
    """Bounded, transparent System 2 rule; not a general autonomous LLM agent."""
    if feedback['success']:
        return None
    obstacle = feedback['public_observation'].get('obstacle', {})
    if obstacle.get('status')!='estimated_surface':
        return None
    clearance = round(max(parent_memory['transit_height_m'],
                          obstacle['top_z']+parent_memory['sample_height_m']+.06), 3)
    if clearance<=parent_memory['transit_height_m'] or clearance>.55:
        return None
    return dict(parent_memory, transit_height_m=clearance)
