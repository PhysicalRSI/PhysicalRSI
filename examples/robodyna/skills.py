"""Reusable skills: inputs contain observations, never task IDs or seeds."""
import math
from pathlib import Path
from PhysicalRSI_core.infra.storage import digest, file_digest
from physicalrsi import CallableSkill, SkillDescriptor


def descriptor(name, input_name, output_name, **parameters):
    return SkillDescriptor(name, digest(dict(source=file_digest(Path(__file__)),
        skill=name, parameters=parameters)), {'name': input_name}, {'name': output_name},
        effects=('robot.position',) if name == 'MoveToTarget' else (), metadata=parameters)


def estimate_motion():
    def run(observation, context):
        dt = observation['sample_dt']
        if dt <= 0:
            raise ValueError('Positive observation interval required')
        return dict(position=observation['position'],
                    velocity=(observation['position']-observation['previous_position'])/dt,
                    horizon=observation['horizon'])
    return CallableSkill(descriptor('EstimateMotion', 'observation', 'motion_estimate',
                                   category='perception'), run)


def predict_intercept(*, gain=1.0):
    if isinstance(gain, bool) or not math.isfinite(gain) or not 0 <= gain <= 2:
        raise ValueError('gain must be finite and in [0, 2]')
    gain = float(gain)
    def run(estimate, context):
        return {'target_position': estimate['position'] + gain * estimate['velocity'] * estimate['horizon']}
    return CallableSkill(descriptor('PredictIntercept', 'motion_estimate', 'intercept_plan',
                                   category='prediction', gain=gain), run)


def move_to_target():
    def run(plan, context):
        return {'target_position': plan['target_position']}
    return CallableSkill(descriptor('MoveToTarget', 'intercept_plan', 'action',
                                   category='manipulation'), run)
