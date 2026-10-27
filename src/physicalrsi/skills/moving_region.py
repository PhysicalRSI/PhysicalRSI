"""Resolve a moving visual region using reciprocal temporal associations.

Requires a stationary or externally motion-compensated camera. This is image
motion evidence, not recognition of a benchmark's privileged target identity.
"""
from copy import deepcopy
import math
from PhysicalRSI_core.infra.storage import digest
from .protocol import CallableSkill, SkillDescriptor
from .visual_motion import estimate_visual_motion


def select_moving_region(*, window=5, minimum_samples=3, minimum_speed_uv_s=.01,
                         maximum_speed_uv_s=1., association_margin_uv=.01,
                         max_gap_s=.3, minimum_confidence=.25):
    if type(minimum_samples) is not int or type(window) is not int or not 2 <= minimum_samples <= window:
        raise ValueError('Invalid temporal window')
    for n in (minimum_speed_uv_s,maximum_speed_uv_s,association_margin_uv,max_gap_s,minimum_confidence):
        if type(n) not in (int,float) or not math.isfinite(n):raise ValueError('Invalid association gate')
    if not 0 < minimum_speed_uv_s < maximum_speed_uv_s or association_margin_uv < 0:
        raise ValueError('Invalid motion gate')
    estimator=estimate_visual_motion(window=window,minimum_confidence=minimum_confidence,max_gap_s=max_gap_s)
    config=dict(window=window,minimum_samples=minimum_samples,minimum_speed_uv_s=minimum_speed_uv_s,
        maximum_speed_uv_s=maximum_speed_uv_s,association_margin_uv=association_margin_uv,
        max_gap_s=max_gap_s,minimum_confidence=minimum_confidence,requires_stationary_camera=True)

    def run(value,context):
        if value.get('stationary_camera') is not True:
            raise ValueError('Declare a stationary camera; moving-camera compensation is not implemented')
        frames=value['observations'][-window:];feature=value['feature']
        # Validate chronology/calibration even when no candidate can be selected.
        estimator.execute(dict(observations=frames,feature=feature),context)
        if not frames:return dict(status='unavailable',reason='no_observations')
        candidate_lists=[]
        for frame in frames:
            row=frame['regions'][feature]
            candidates=row.get('candidates', [row] if row['status']=='visible' else [])
            clean=[]
            for candidate in candidates:
                score=candidate['confidence'];uv=candidate['uv']
                if (type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 1
                    or not isinstance(uv,list) or len(uv)!=2
                    or any(type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1 for x in uv)):
                    raise ValueError('Invalid visual candidate')
                if score>=minimum_confidence:clean.append(candidate)
            candidate_lists.append(clean)
        tracks=[[i] for i in range(len(candidate_lists[-1]))]
        accepted=[]
        def distance(a,b):return math.dist(a['uv'],b['uv'])
        def nearest(point,choices,limit):
            ranks=sorted((distance(point,x),i) for i,x in enumerate(choices))
            if not ranks or ranks[0][0]>limit:return None
            if len(ranks)>1 and ranks[1][0]-ranks[0][0]<=association_margin_uv:return None
            return ranks[0][1]
        for track in tracks:
            for k in range(len(frames)-1,0,-1):
                dt=frames[k]['timestamp_s']-frames[k-1]['timestamp_s']
                if dt>max_gap_s:break
                current=track[0];now=candidate_lists[k];before=candidate_lists[k-1]
                prior=nearest(now[current],before,maximum_speed_uv_s*dt)
                if prior is None or nearest(before[prior],now,maximum_speed_uv_s*dt)!=current:break
                if now[current]['measurement']!=before[prior]['measurement']:break
                track.insert(0,prior)
            if len(track)<minimum_samples:continue
            associated=[]
            start=len(frames)-len(track)
            for offset,index in enumerate(track):
                row=deepcopy(frames[start+offset]);point=candidate_lists[start+offset][index]
                row['regions'][feature]=dict(point,status='visible')
                associated.append(row)
            motion=estimator.execute(dict(observations=associated,feature=feature),context)
            speed=math.hypot(*motion['velocity_uv_s'])
            if minimum_speed_uv_s<=speed<=maximum_speed_uv_s:
                accepted.append(dict(candidate=candidate_lists[-1][track[-1]],motion=motion))
        if len(accepted)!=1:
            return dict(status='unavailable',reason='multiple_moving_regions' if accepted else 'no_unique_moving_region',
                        moving_candidates=len(accepted),observation_sequence=frames[-1]['observation_sequence'])
        return dict(status='selected',observation_sequence=frames[-1]['observation_sequence'],**accepted[0],
                    qualification=None)

    return CallableSkill(SkillDescriptor('select_moving_region',digest(config),
        {'name':'physicalrsi.visual-history/v1'},{'name':'physicalrsi.moving-region/v1'},metadata=config),run)
