"""Image-relative motion from contiguous visible measurements, without phases."""
import math
from PhysicalRSI_core.infra.storage import digest
from .protocol import CallableSkill, SkillDescriptor


def estimate_visual_motion(*, window=5, minimum_confidence=.25, max_gap_s=.3):
    if type(window) is not int or window < 2:
        raise ValueError('Motion needs at least two measurements')
    for value in (minimum_confidence, max_gap_s):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Invalid observation gate')
    if not 0 <= minimum_confidence <= 1 or max_gap_s <= 0:
        raise ValueError('Invalid observation gate')
    config = dict(window=window, minimum_confidence=minimum_confidence, max_gap_s=max_gap_s)

    def run(value, context):
        frames, feature = value['observations'], value['feature']
        if not isinstance(frames, list) or not isinstance(feature, str) or not feature:
            raise ValueError('Expected visual observations and a feature name')
        history, previous, camera_identity = [], None, None
        for frame in frames:
            context.check()
            if frame['schema'] != 'physicalrsi.visual-regions/v1':
                raise ValueError('Expected image-derived region measurements')
            identity = (frame['camera'], frame['calibration_revision'], tuple(frame['image_size_px']), digest(frame['provider']))
            if camera_identity is not None and identity != camera_identity:
                raise ValueError('Camera, calibration, resolution or detector changed')
            camera_identity = identity
            t, sequence = frame['timestamp_s'], frame['observation_sequence']
            if (type(t) not in (float, int) or not math.isfinite(t) or t < 0
                    or type(sequence) is not int or sequence < 0):
                raise ValueError('Invalid observation time or sequence')
            if previous is not None:
                if t <= previous[0] or sequence <= previous[1]:
                    raise ValueError('Measurements must have increasing time and sequence')
                if t-previous[0] > max_gap_s:
                    history.clear()
            previous = (t, sequence)
            row = frame['regions'][feature]
            if row['status'] != 'visible':
                history.clear()
                continue
            uv, score = row['uv'], row['confidence']
            if (not isinstance(uv, list) or len(uv) != 2
                    or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in uv)
                    or type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Invalid image measurement')
            if score < minimum_confidence:
                history.clear()
                continue
            # Do not mix mask centroids and bounding-box centres across frames.
            measurement = row['measurement']
            if measurement not in ('box_center', 'mask_centroid'):
                raise ValueError('Unknown image measurement')
            if history and measurement != history[-1][4]:
                history.clear()
            history.append((float(t), uv, score, sequence, measurement))
            history = history[-window:]
        result = dict(schema='physicalrsi.visual-motion/v1', feature=feature,
                      coordinates='normalized_image_uv', camera=None if not frames else frames[-1]['camera'],
                      timestamp_s=None if previous is None else previous[0],
                      observation_sequences=[r[3] for r in history], qualification=None)
        if len(history) < 2:
            return dict(result, status='unavailable', reason='insufficient_contiguous_visible_measurements')
        times = [r[0] for r in history]
        mean_t = sum(times)/len(times)
        mean_uv = [sum(r[1][axis] for r in history)/len(history) for axis in range(2)]
        denominator = sum((t-mean_t)**2 for t in times)
        velocity = [sum((r[0]-mean_t)*(r[1][axis]-mean_uv[axis]) for r in history)/denominator for axis in range(2)]
        residual = math.sqrt(sum(sum((r[1][a]-mean_uv[a]-velocity[a]*(r[0]-mean_t))**2
                                         for a in range(2)) for r in history)/len(history))
        return dict(result, status='estimated', uv=history[-1][1], velocity_uv_s=velocity,
                    residual_rms_uv=residual, minimum_detector_score=min(r[2] for r in history),
                    interpretation='Image-relative feature motion; includes camera motion and is not 3D object velocity')

    return CallableSkill(SkillDescriptor('estimate_visual_motion', digest(config),
        {'name': 'physicalrsi.visual-history/v1'}, {'name': 'physicalrsi.visual-motion/v1'}, metadata=config), run)
