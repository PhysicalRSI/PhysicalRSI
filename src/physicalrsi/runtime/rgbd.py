"""Explicit RGB-D envelope; the RGB-only v1 observation remains unchanged."""
import base64
from copy import deepcopy
import io
from .sensors import _keys, _text, decode_rgb, validate_observation

SCHEMA = 'physicalrsi.rgbd/v1'


def _range(minimum_mm, maximum_mm):
    if (type(minimum_mm) is not int or type(maximum_mm) is not int
            or not 1 <= minimum_mm < maximum_mm <= 65535):
        raise ValueError('Declare a representable positive depth range in millimetres')


def encode_depth_m(metres, *, minimum_mm=100, maximum_mm=3000):
    import numpy as np
    from PIL import Image
    _range(minimum_mm, maximum_mm)
    depth = np.asarray(metres, dtype=float)
    if depth.ndim != 2 or not depth.size or depth.size > 16_777_216:
        raise ValueError('Expected bounded HxW camera depth')
    valid = np.isfinite(depth) & (depth >= minimum_mm/1000.) & (depth <= maximum_mm/1000.)
    quantized = np.zeros(depth.shape, dtype=np.uint16)
    quantized[valid] = np.rint(depth[valid]*1000.).astype(np.uint16)
    output = io.BytesIO()
    Image.fromarray(quantized).save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def decode_depth_mm(encoded):
    import numpy as np
    from PIL import Image
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.mode not in ('I', 'I;16', 'I;16B'):
            raise ValueError('Depth must be a 16-bit grayscale PNG')
        if image.width*image.height > 16_777_216:
            raise ValueError('Depth frame exceeds pixel budget')
        result = np.array(image)
    if result.ndim != 2 or np.any(result < 0) or np.any(result > 65535):
        raise ValueError('Invalid uint16 depth')
    return result.astype(np.uint16)


def validate_rgbd(value):
    import numpy as np
    _keys(value, ('schema', 'rgb', 'depth_model', 'depth'))
    if value['schema'] != SCHEMA:
        raise ValueError('Invalid RGB-D schema')
    rgb = validate_observation(value['rgb'])
    _text(value['depth_model'])
    if not isinstance(value['depth'], dict) or set(value['depth']) != set(rgb['cameras']):
        raise ValueError('Depth cameras must match declared RGB cameras')
    for name, frame in value['depth'].items():
        _keys(frame, ('captured_at_s', 'encoding', 'invalid_value', 'minimum_mm', 'maximum_mm', 'png_b64'))
        _range(frame['minimum_mm'], frame['maximum_mm'])
        if frame['encoding'] != 'png-u16-mm' or type(frame['invalid_value']) is not int or frame['invalid_value'] != 0:
            raise ValueError('Depth units or missing-value convention changed')
        if (type(frame['captured_at_s']) not in (int,float)
                or frame['captured_at_s'] != rgb['cameras'][name]['captured_at_s']):
            raise ValueError('RGB and depth must be synchronized')
        pixels = decode_depth_mm(frame['png_b64'])
        if pixels.shape != decode_rgb(rgb['cameras'][name]['rgb_png_b64']).shape[:2]:
            raise ValueError('Depth must be registered to RGB pixels')
        valid = pixels != 0
        if np.any(pixels[valid] < frame['minimum_mm']) or np.any(pixels[valid] > frame['maximum_mm']):
            raise ValueError('Depth exceeds its declared range')
    return deepcopy(value)
