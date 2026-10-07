"""Compose public RGB-D segmentation and grasp inference without simulator access."""
import time


class PublicGraspPlanner:
    def __init__(self, *, segmenter, grasp_client):
        self.segmenter, self.grasp_client = segmenter, grasp_client

    def __call__(self, observation, prompt, *, camera='agentview', deadline):
        import numpy as np
        if camera not in ('agentview', 'robot0_eye_in_hand'):
            raise ValueError('Unsupported public camera')
        source = observation[camera]
        transform = np.asarray(source['pose_mat'], dtype=float)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Invalid public camera-to-base transform')
        result = self.segmenter(source['images']['rgb'], prompt, deadline=deadline)
        detections = result['detections']
        if not detections:
            return {'status':'not_detected', 'frame':'robot-base', 'grasps':[],
                    'semantic_identity_verified':False}
        detection = max(detections, key=lambda item:item['score'])
        inferred = self.grasp_client(np.asarray(source['images']['depth']).squeeze(),
            source['intrinsics'], detection['mask'], deadline=deadline)
        if inferred['frame'] != 'camera-optical':
            raise ValueError('Unexpected grasp coordinate frame')
        if time.monotonic() >= deadline:
            raise TimeoutError('Public grasp planning deadline reached')
        return {'status':'estimated' if inferred['grasps'] else 'no_grasps',
                'frame':'robot-base', 'semantic_identity_verified':False,
                'detection_score':float(detection['score']), 'detection_alternatives':len(detections),
                'grasps':[{'matrix':(transform@np.asarray(row['matrix'])).tolist(),
                           'score':row['score'],
                           **({'contact_point':(transform[:3, :3]@np.asarray(row['contact_point'])
                                                + transform[:3, 3]).tolist()}
                              if 'contact_point' in row else {})} for row in inferred['grasps']]}
