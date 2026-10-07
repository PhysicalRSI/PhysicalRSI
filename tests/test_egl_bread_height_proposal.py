from PhysicalRSI_baselines.embodied_goodharts_law.bread_height_proposal import add_bread_height_filter


def test_filter_runs_before_high_candidate_is_consumed():
    source="""def policy(robot, memory):
    bounds = memory['bounds']
    observed = robot.locate_objects('bread', camera='head_camera')
    return observed['objects']
"""
    result, count=add_bread_height_filter(source)
    class Robot:
        def locate_objects(self,*a,**k):return {'objects':[{'surface_quantiles':{'median':[0,0,z]}} for z in [.8,.95]]}
    scope={};exec(result,scope)
    assert count==1
    assert len(scope['policy'](Robot(),{'bounds':{'upper_95':[0,0,.81]}}))==1


def test_existing_reviewed_gate_is_unchanged():
    source="""def policy(robot, memory):
    bounds = memory['bounds']
    observed = robot.locate_objects('bread', camera='head_camera')
    observed = dict(observed)
    observed['objects'] = [item for item in observed['objects'] if item['surface_quantiles']['median'][2] <= bounds['upper_95'][2]+.08]
    return observed['objects']
"""
    assert add_bread_height_filter(source)==(source,0)
