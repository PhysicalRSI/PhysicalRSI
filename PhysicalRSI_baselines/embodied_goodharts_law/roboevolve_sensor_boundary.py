"""Host-only conversion of pinned RoboEvolve's flat camera observations.

The caller must collect the observation and sensor frame from the same rendered
state. Native scene handles and this host function are not policy capabilities.
"""
import numpy as np

from .roboevolve_perception import public_world_camera


def named_native_depth_frame(sensor):
    """Read explicitly named Isaac depth channels from one rendered state.

    Some pinned Camera implementations expose attached annotators outside the
    frame dictionary. Only the two declared depth annotators are eligible;
    generic get_depth(), segmentation and scene metadata are never consulted.
    The host must prevent stepping/rendering during this read and RGB capture.
    """
    frame = sensor.get_current_frame()
    if not isinstance(frame, dict):
        raise ValueError('Native camera frame must be a dictionary')
    annotators = getattr(sensor, '_custom_annotators', {})
    for kind in ('distance_to_image_plane', 'distance_to_camera'):
        value = frame.get(kind)
        if value is None and isinstance(annotators, dict):
            annotator = annotators.get(kind)
            if annotator is not None:
                value = annotator.get_data()
        if value is not None:
            return {kind: np.asarray(value).copy()}
    raise ValueError('A named native depth annotator is required')


def public_native_camera(observation, sensor_frame):
    """Use declared depth channels and preserve the task's rendered RGB image.

    RoboEvolve Camera.get_observation(['config', 'rgb', 'depth']) flattens
    calibration fields into the observation. Generic depth, point clouds and
    prim metadata are deliberately ignored. No sensor fallback is inferred.
    """
    for kind in ('distance_to_image_plane', 'distance_to_camera'):
        value = sensor_frame.get(kind)
        if value is not None:
            depth = np.asarray(value, dtype=float)
            break
    else:
        raise ValueError('A named native depth frame channel is required')
    channels = {'rgb': observation['rgb'], 'depth': depth,
                'config': {'intrinsic': observation['intrinsic'],
                           'camera_to_world': observation['camera_to_world']}}
    camera = public_world_camera(channels, depth_kind=kind)
    return {'camera': camera,
            'sensor_protocol': {'depth_source': 'simulator-rendered',
                                'native_depth_channel': kind,
                                'output_depth': 'optical-axis-metres',
                                'output_frame': 'world',
                                'rgb_source': 'task-camera-observation'}}
