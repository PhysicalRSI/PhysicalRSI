"""Composable image-region localization; no object pose, phase or task oracle."""
import hashlib

from PhysicalRSI_core.infra.storage import digest
from physicalrsi.runtime.sensors import decode_rgb, validate_observation
from .protocol import CallableSkill, SkillDescriptor


def locate_visible_regions(provider, features, *, camera, threshold=.65):
    """Bind an RGB region provider to a versioned primitive.

    Multiple confident regions are ambiguous, not silently assigned an identity.
    Missing measurements remain unknown. A mask centroid is a 2D image feature,
    not a grasp point, body centre, contact measurement or task verdict.
    """
    import math
    if (not isinstance(features, (list, tuple)) or not features
            or any(not isinstance(x, str) or not x.strip() for x in features)
            or len(set(features)) != len(features) or not isinstance(camera, str) or not camera):
        raise ValueError('Declare unique visual features and a camera')
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Invalid detection threshold')
    features = tuple(features)
    identity = provider.identity()
    identity_digest = digest(identity)
    config = dict(provider=identity, features=list(features), camera=camera, threshold=threshold,
                  contract_revision=2)

    def run(observation, context):
        import numpy as np
        if digest(provider.identity()) != identity_digest:
            raise ValueError('Perception provider changed')
        sensors = validate_observation(observation)
        frame = sensors['cameras'][camera]
        pixels = decode_rgb(frame['rgb_png_b64'])
        height, width = pixels.shape[:2]
        measured = {}
        batch = None
        if callable(getattr(provider, 'segment_many', None)):
            context.check()
            batch = provider.segment_many(pixels.copy(), list(features), threshold=threshold)
            context.check()
            if not isinstance(batch, dict) or set(batch) != set(features):
                raise ValueError('Batch regions must cover exactly the requested features')
        for feature in features:
            context.check()
            rows = (batch[feature] if batch is not None
                    else provider.segment(pixels.copy(), feature, threshold=threshold))
            context.check()
            regions = []
            for row in rows:
                score = row['score']
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid perception confidence')
                if 'mask' in row:
                    mask = np.asarray(row['mask'])
                    if mask.dtype != np.bool_ or mask.shape != (height, width):
                        raise ValueError('Mask does not match the RGB observation')
                    if score < threshold or not mask.any():
                        continue
                    y, x = np.nonzero(mask)
                    regions.append(dict(uv=[float((x.mean()+.5)/width), float((y.mean()+.5)/height)],
                                        confidence=score, measurement='mask_centroid'))
                else:
                    box = np.asarray(row['bbox_xyxy'], dtype=float)
                    if (box.shape != (4,) or not np.isfinite(box).all()
                            or not 0 <= box[0] < box[2] <= width
                            or not 0 <= box[1] < box[3] <= height):
                        raise ValueError('Box does not match the RGB observation')
                    if score >= threshold:
                        regions.append(dict(uv=[float((box[0]+box[2])/(2*width)),
                                                float((box[1]+box[3])/(2*height))],
                                            confidence=score, measurement='box_center'))
            measured[feature] = (dict(status='visible', **regions[0]) if len(regions) == 1
                                 else dict(status='ambiguous' if regions else 'unobserved', uv=None,
                                           confidence=None))
            measured[feature]['candidates'] = regions
        if digest(provider.identity()) != identity_digest:
            raise ValueError('Perception provider changed during inference')
        return dict(schema='physicalrsi.visual-regions/v1', observation_sequence=sensors['sequence'],
                    timestamp_s=frame['captured_at_s'], camera=camera,
                    calibration_revision=sensors['calibration_revision'],
                    image_size_px=[width, height],
                    rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    regions=measured, provider=identity)

    return CallableSkill(SkillDescriptor('locate_visible_regions', digest(config),
        {'name': 'physicalrsi.sensors/v1'}, {'name': 'physicalrsi.visual-regions/v1'},
        metadata=config), run)
