"""RGB category evidence for a candidate; never an object identity certificate."""
import hashlib
import math
from PhysicalRSI_core.infra.storage import digest
from physicalrsi.runtime.sensors import decode_rgb, validate_observation
from .protocol import CallableSkill, SkillDescriptor


def verify_region(provider, feature, *, camera, threshold=.25,
                  minimum_side_px=96, area_scale=6.):
    """Check whether category boxes contain a sensor-derived candidate centroid.

    Input is an RGB observation and an explicitly frame-bound proposal. A positive
    result does not establish persistent identity, contact, or a grasp point.
    """
    if not all(isinstance(x, str) and x.strip() for x in (feature, camera)):
        raise ValueError('Declare a visual category and camera')
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 < threshold <= 1 or type(minimum_side_px) is not int
            or minimum_side_px < 1 or type(area_scale) not in (int, float)
            or not math.isfinite(area_scale) or area_scale <= 0):
        raise ValueError('Invalid verification configuration')
    provider_digest = digest(provider.identity())
    config = dict(provider=provider.identity(), feature=feature, camera=camera,
                  threshold=threshold, minimum_side_px=minimum_side_px,
                  area_scale=area_scale, contract_revision=1)

    def run(value, context):
        if not isinstance(value, dict) or set(value) != {'observation', 'proposal'}:
            raise ValueError('Unexpected verification inputs')
        obs = validate_observation(value['observation'])
        rgb = decode_rgb(obs['cameras'][camera]['rgb_png_b64'])
        height, width = rgb.shape[:2]
        proposal = value['proposal']
        fields = {'observation_sequence', 'calibration_revision', 'camera',
                  'image_size_px', 'selected'}
        if not isinstance(proposal, dict) or set(proposal) != fields:
            raise ValueError('Expected a frame-bound proposal')
        if (type(proposal['observation_sequence']) is not int
                or proposal['observation_sequence'] != obs['sequence']
                or proposal['calibration_revision'] != obs['calibration_revision']
                or proposal['camera'] != camera
                or proposal['image_size_px'] != [width, height]):
            raise ValueError('Proposal belongs to another observation')
        if digest(provider.identity()) != provider_digest:
            raise ValueError('Provider changed')
        selected = proposal['selected']
        result = dict(schema='physicalrsi.region-verification/v1',
                      observation_sequence=obs['sequence'],
                      timestamp_s=obs['cameras'][camera]['captured_at_s'],
                      calibration_revision=obs['calibration_revision'], camera=camera,
                      image_size_px=[width, height], feature=feature,
                      rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                      accepted=False, status='unobserved', semantic_identity=None,
                      crop_xyxy=None, boxes=[], selected=None)
        if selected is None:
            return result
        if not isinstance(selected, dict) or set(selected) != {'uv', 'area_px'}:
            raise ValueError('Unexpected candidate fields')
        uv, area = selected['uv'], selected['area_px']
        if (not isinstance(uv, (list, tuple)) or len(uv) != 2
                or any(type(a) not in (int, float) or not math.isfinite(a)
                       or not 0 <= a < 1 for a in uv)
                or type(area) is not int or not 0 < area <= width*height):
            raise ValueError('Invalid candidate')
        x, y = uv[0]*width, uv[1]*height
        side = min(width, height, max(minimum_side_px, math.ceil(area_scale*math.sqrt(area))))
        left = max(0, min(width-side, int(x-side/2)))
        top = max(0, min(height-side, int(y-side/2)))
        context.check()
        rows = provider.segment(rgb[top:top+side, left:left+side].copy(),
                                feature, threshold=threshold)
        context.check()
        boxes = []
        accepted = False
        for row in rows:
            box, score = row['bbox_xyxy'], row['score']
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(a) not in (int, float) or not math.isfinite(a) for a in box)
                    or not 0 <= box[0] < box[2] <= side
                    or not 0 <= box[1] < box[3] <= side
                    or type(score) not in (int, float) or not math.isfinite(score)
                    or not 0 <= score <= 1):
                raise ValueError('Invalid detection')
            if score < threshold:
                continue
            boxes.append(dict(bbox_xyxy=list(box), score=score))
            accepted |= box[0] <= x-left < box[2] and box[1] <= y-top < box[3]
        if digest(provider.identity()) != provider_digest:
            raise ValueError('Provider changed during inference')
        result.update(accepted=accepted, status='supported' if accepted else 'unsupported',
                      crop_xyxy=[left, top, left+side, top+side], boxes=boxes,
                      selected=dict(uv=list(uv), area_px=area) if accepted else None)
        return result

    return CallableSkill(SkillDescriptor('verify_region', digest(config),
        {'name': 'physicalrsi.region-verification-input/v1'},
        {'name': 'physicalrsi.region-verification/v1'}, metadata=config), run)
