"""Image-change proposals from sensor frames, not semantic object identities."""
import math
from PhysicalRSI_core.infra.storage import digest
from physicalrsi.runtime.sensors import decode_rgb, validate_observation
from .protocol import CallableSkill, SkillDescriptor


def locate_visual_changes(*, camera, threshold=30, minimum_area_px=3,
                          max_gap_s=.3, selection='dominant_area', dominance_ratio=2.,
                          self_model=None, self_margin_px=2):
    """Requires a stationary camera. Robot/shadow motion can also create regions.

    A selected change centroid is not a confirmed target, body centre, grasp
    point or contact. Two-frame differencing can include an object's old position.
    """
    import cv2
    import numpy as np
    if not isinstance(camera, str) or not camera.strip():
        raise ValueError('Declare a camera')
    if type(threshold) is not int or not 1 <= threshold <= 255:
        raise ValueError('Invalid pixel threshold')
    if type(minimum_area_px) is not int or minimum_area_px < 1:
        raise ValueError('Invalid minimum area')
    if (type(max_gap_s) not in (int, float) or not math.isfinite(max_gap_s) or max_gap_s <= 0
            or type(dominance_ratio) not in (int, float)
            or not math.isfinite(dominance_ratio) or dominance_ratio <= 1):
        raise ValueError('Invalid temporal or selection gate')
    if selection not in ('unique', 'dominant_area'):
        raise ValueError('Unknown selection rule')
    if type(self_margin_px) is not int or not 0 <= self_margin_px <= 32:
        raise ValueError('Invalid self-projection margin')
    model_identity = self_model.identity() if self_model is not None else None
    model_digest = digest(model_identity)
    config = dict(camera=camera, threshold=threshold, minimum_area_px=minimum_area_px,
                  max_gap_s=max_gap_s, selection=selection, dominance_ratio=dominance_ratio,
                  closing_kernel_px=3, opencv=cv2.__version__, numpy=np.__version__,
                  contract_revision=2, self_model=model_identity, self_margin_px=self_margin_px)

    def run(value, context):
        if set(value) != {'previous', 'current', 'stationary_camera'}:
            raise ValueError('Unexpected image-change inputs')
        if value['stationary_camera'] is not True:
            raise ValueError('A stationary camera is required')
        previous = validate_observation(value['previous'])
        current = validate_observation(value['current'])
        for field in ('embodiment_revision', 'calibration_revision'):
            if previous[field] != current[field]:
                raise ValueError('Sensor configuration changed')
        if current['sequence'] <= previous['sequence']:
            raise ValueError('Observation sequence must increase')
        if current['timestamp_s'] <= previous['timestamp_s']:
            raise ValueError('Observation time must increase')
        old, new = previous['cameras'][camera], current['cameras'][camera]
        dt = new['captured_at_s'] - old['captured_at_s']
        if not 0 < dt <= max_gap_s:
            raise ValueError('Camera frames are unordered or too far apart')
        before, after = decode_rgb(old['rgb_png_b64']), decode_rgb(new['rgb_png_b64'])
        if before.shape != after.shape:
            raise ValueError('Camera resolution changed')
        context.check()
        mask = (np.max(np.abs(after.astype(np.int16)-before.astype(np.int16)), axis=2)
                >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if self_model is not None:
            if digest(self_model.identity()) != model_digest:
                raise ValueError('Self model changed during skill lifetime')
            projected = [self_model.mask(obs, camera=camera) for obs in (previous, current)]
            if any(x.dtype != np.bool_ or x.shape != mask.shape for x in projected):
                raise ValueError('Invalid self-projection mask')
            exclusion = np.logical_or(*projected).astype(np.uint8)
            size = 2*self_margin_px + 1
            exclusion = cv2.dilate(exclusion, np.ones((size, size), np.uint8))
            mask[exclusion != 0] = 0
            if digest(self_model.identity()) != model_digest:
                raise ValueError('Self model changed during inference')
        count, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        height, width = after.shape[:2]
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] >= minimum_area_px:
                x, y = centres[i]
                candidates.append(dict(uv=[float((x+.5)/width), float((y+.5)/height)],
                                       area_px=int(stats[i, cv2.CC_STAT_AREA])))
        ranked = sorted(candidates, key=lambda row: row['area_px'], reverse=True)
        selected = None
        if len(ranked) == 1 or (selection == 'dominant_area' and len(ranked) > 1
                               and ranked[0]['area_px'] >= dominance_ratio*ranked[1]['area_px']):
            selected = ranked[0]
        context.check()
        return dict(schema='physicalrsi.visual-changes/v1',
                    observation_sequence=current['sequence'], previous_sequence=previous['sequence'],
                    timestamp_s=new['captured_at_s'], camera=camera, image_size_px=[width, height],
                    calibration_revision=current['calibration_revision'], candidates=candidates,
                    selected=selected, status='selected' if selected else ('ambiguous' if candidates else 'unobserved'),
                    measurement='image_change_centroid', semantic_identity=None)

    return CallableSkill(SkillDescriptor('locate_visual_changes', digest(config),
        {'name': 'physicalrsi.sensor-pair/v1'}, {'name': 'physicalrsi.visual-changes/v1'},
        metadata=config), run)
