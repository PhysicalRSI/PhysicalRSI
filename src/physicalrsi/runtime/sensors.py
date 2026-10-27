"""Strict RGB/proprioception wire contract for sensor-only policies.

This is a data boundary, not a Python sandbox. Run untrusted candidates in the
existing isolated policy runtime, without environment or verifier capabilities.
The host owns camera acquisition, embodiment calibration and actuator limits.
"""
import base64
from copy import deepcopy
import io
import math

SCHEMA = 'physicalrsi.sensors/v1'
ACTION_SCHEMA = 'physicalrsi.actuator-target/v1'


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('Unexpected or missing fields at sensor boundary')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Expected finite numeric measurement')
    return float(value)


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected nonempty identity')
    return value


def decode_rgb(encoded):
    from PIL import Image
    import numpy as np
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.mode != 'RGB':
            raise ValueError('Camera frame must be RGB PNG')
        if image.width * image.height > 16_777_216:
            raise ValueError('Camera frame exceeds pixel budget')
        return np.array(image)


def encode_rgb(rgb):
    import numpy as np
    from PIL import Image
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('Expected HxWx3 uint8 RGB pixels')
    output = io.BytesIO()
    Image.fromarray(array).save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def validate_observation(value, *, max_camera_skew_s=.1):
    _keys(value, ('schema', 'sequence', 'timestamp_s', 'embodiment_revision',
                  'calibration_revision', 'cameras', 'actuator_position'))
    if value['schema'] != SCHEMA or type(value['sequence']) is not int or value['sequence'] < 0:
        raise ValueError('Invalid sensor schema or sequence')
    timestamp = _number(value['timestamp_s'])
    skew = _number(max_camera_skew_s)
    if timestamp < 0 or skew < 0:
        raise ValueError('Invalid timestamp or camera skew')
    _text(value['embodiment_revision'])
    _text(value['calibration_revision'])
    positions = value['actuator_position']
    if not isinstance(positions, list) or not positions:
        raise ValueError('Measured actuator vector is required')
    for item in positions:
        _number(item)
    cameras = value['cameras']
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError('At least one RGB camera is required')
    for name, camera in cameras.items():
        _text(name)
        _keys(camera, ('captured_at_s', 'rgb_png_b64'))
        captured = _number(camera['captured_at_s'])
        if not 0 <= captured <= timestamp or timestamp - captured > skew:
            raise ValueError('Camera capture is stale or from the future')
        decode_rgb(camera['rgb_png_b64'])
    return deepcopy(value)


def validate_action(action, observation):
    _keys(action, ('schema', 'observation_sequence', 'target'))
    if (action['schema'] != ACTION_SCHEMA
            or type(action['observation_sequence']) is not int
            or action['observation_sequence'] != observation['sequence']):
        raise ValueError('Action is not bound to the current observation')
    target = action['target']
    if not isinstance(target, list) or len(target) != len(observation['actuator_position']):
        raise ValueError('Action does not match the embodiment actuator layout')
    for item in target:
        _number(item)
    return deepcopy(action)


class SensorPolicy:
    """Core Policy adapter. The supplied policy sees only a validated sensor copy.

    The immutable embodiment revision defines actuator order, units and bounds.
    No task case, seed, native object, score or evaluator context is forwarded.
    """
    def __init__(self, policy, *, embodiment_revision, calibration_revision):
        self.policy = policy
        self.embodiment_revision = _text(embodiment_revision)
        self.calibration_revision = _text(calibration_revision)

    def identity(self):
        return dict(adapter=SCHEMA, policy=self.policy.identity(),
                    embodiment_revision=self.embodiment_revision,
                    calibration_revision=self.calibration_revision)

    def act(self, observation, context):
        context.check()
        sensors = validate_observation(observation)
        if (sensors['embodiment_revision'] != self.embodiment_revision
                or sensors['calibration_revision'] != self.calibration_revision):
            raise ValueError('Embodiment or calibration changed')
        # Deliberately no Context.execution callback or live environment handle.
        result = self.policy.act(deepcopy(sensors))
        context.check()
        return validate_action(result, sensors)
