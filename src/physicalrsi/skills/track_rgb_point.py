"""Short-horizon image tracking without simulator identity or object state."""
import math
from PhysicalRSI_core.infra.storage import digest
from physicalrsi.runtime.sensors import decode_rgb, validate_observation
from .protocol import CallableSkill, SkillDescriptor


def track_rgb_point(*, camera, max_gap_s=.1, max_roundtrip_px=1.,
                    max_patch_error=20., window_px=15):
    """Forward/backward Lucas-Kanade evidence, not persistent object identity."""
    import cv2
    import numpy as np
    if not isinstance(camera, str) or not camera.strip():
        raise ValueError('Declare a camera')
    for v in (max_gap_s, max_roundtrip_px, max_patch_error):
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Tracking gates must be finite and positive')
    if type(window_px) is not int or window_px < 3 or window_px % 2 != 1:
        raise ValueError('Use an odd positive tracking window')
    config = dict(camera=camera, max_gap_s=max_gap_s,
                  max_roundtrip_px=max_roundtrip_px, max_patch_error=max_patch_error,
                  window_px=window_px, levels=2, iterations=30, epsilon=.01,
                  opencv=cv2.__version__, numpy=np.__version__, contract_revision=1)

    def run(value, context):
        if not isinstance(value, dict) or set(value) != {'previous', 'current', 'uv'}:
            raise ValueError('Unexpected tracking inputs')
        old, new = [validate_observation(value[k]) for k in ('previous', 'current')]
        if any(old[k] != new[k] for k in ('calibration_revision', 'embodiment_revision')):
            raise ValueError('Sensor configuration changed')
        if new['sequence'] <= old['sequence']:
            raise ValueError('Tracking requires ordered frames')
        dt = new['cameras'][camera]['captured_at_s'] - old['cameras'][camera]['captured_at_s']
        if not 0 < dt <= max_gap_s:
            raise ValueError('Frame gap exceeds tracking interval')
        uv = value['uv']
        if (not isinstance(uv, (list, tuple)) or len(uv) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       or not 0 <= x < 1 for x in uv)):
            raise ValueError('Invalid image point')
        images = [decode_rgb(o['cameras'][camera]['rgb_png_b64']) for o in (old, new)]
        if images[0].shape != images[1].shape:
            raise ValueError('Image resolution changed')
        h, w = images[0].shape[:2]
        images = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images]
        # Public uv uses pixel centres; OpenCV uses integer pixel indices.
        point = np.array([[[uv[0]*w-.5, uv[1]*h-.5]]], dtype=np.float32)
        options = dict(winSize=(window_px, window_px), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        result = dict(schema='physicalrsi.rgb-point-track/v1',
                      observation_sequence=new['sequence'], previous_sequence=old['sequence'],
                      timestamp_s=new['cameras'][camera]['captured_at_s'],
                      camera=camera, calibration_revision=new['calibration_revision'],
                      status='lost', uv=None, semantic_identity=None,
                      roundtrip_px=None, patch_error=None)
        context.check()
        forward, status, error = cv2.calcOpticalFlowPyrLK(images[0], images[1], point, None, **options)
        if forward is None or not status[0, 0] or not np.isfinite(forward).all():
            return result
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            images[1], images[0], forward, None, **options)
        context.check()
        if backward is None or not reverse_status[0, 0] or not np.isfinite(backward).all():
            return result
        roundtrip = float(np.linalg.norm(backward-point))
        patch_error = float(error[0, 0])
        u, v = (forward[0, 0] + .5) / [w, h]
        result.update(roundtrip_px=roundtrip, patch_error=patch_error)
        if (math.isfinite(patch_error) and patch_error <= max_patch_error
                and roundtrip <= max_roundtrip_px and 0 <= u < 1 and 0 <= v < 1):
            result.update(status='tracked', uv=[float(u), float(v)])
        return result

    return CallableSkill(SkillDescriptor('track_rgb_point', digest(config),
        {'name': 'physicalrsi.rgb-point-track-input/v1'},
        {'name': 'physicalrsi.rgb-point-track/v1'}, metadata=config), run)
