"""Bounded loopback client for the pinned ASPIRE SAM3 service."""
import base64
import http.client
import io
import json
import math
import socket
import time
from PhysicalRSI_core.infra.rpc.authenticated_http import SocketDeadline


class Sam3Client:
    def __init__(self, *, port=8114, confidence=.3, max_detections=8):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid local service port')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid segmentation confidence threshold')
        if type(max_detections) is not int or not 1 <= max_detections <= 32:
            raise ValueError('Invalid detection limit')
        self.port, self.confidence, self.max_detections = port, confidence, max_detections

    def __call__(self, rgb, prompt, *, deadline):
        import numpy as np
        from PIL import Image
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256:
            raise ValueError('Expected a nonempty text prompt of at most 256 characters')
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise TimeoutError('Segmentation deadline reached')
        rgb = np.asarray(rgb)
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or min(rgb.shape[:2]) < 1 or max(rgb.shape[:2]) > 512):
            raise ValueError('Expected an RGB uint8 image no larger than 512x512')
        image = io.BytesIO()
        Image.fromarray(rgb).save(image, format='PNG')
        body = json.dumps({'image_base64':base64.b64encode(image.getvalue()).decode(),
                           'text_prompt':prompt}).encode()
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Segmentation deadline reached')
        connection = http.client.HTTPConnection('127.0.0.1',self.port,timeout=remaining)
        # Upstream serializes up to 256 raw uint8 masks before client-side
        # confidence filtering. Bound the body by the admitted image geometry;
        # a fixed 16 MiB cap rejects valid 512x512 responses.
        mask_bytes = 4*((rgb.shape[0]*rgb.shape[1]+2)//3)
        response_limit = 4096+256*(mask_bytes+1024)
        watchdog = None
        try:
            connection.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            connection.sock.settimeout(remaining)
            watchdog = SocketDeadline(connection.sock,remaining)
            connection.sock.connect(('127.0.0.1',self.port))
            connection.request('POST','/segment',body=body,headers={'Content-Type':'application/json'})
            with connection.getresponse() as response:
                data=response.read(response_limit+1)
                if response.status != 200:
                    raise RuntimeError('SAM3 service rejected the request')
            if len(data)>response_limit:
                raise ValueError('SAM3 response exceeds byte allowance')
            if watchdog.expired or time.monotonic()>=deadline:
                raise TimeoutError('Segmentation deadline reached')
        finally:
            if watchdog is not None:
                watchdog.cancel()
            connection.close()
        rows=json.loads(data)['results']
        if not isinstance(rows,list) or len(rows)>256:
            raise ValueError('Invalid SAM3 detection response')
        detections=[]
        for row in rows:
            score=float(row['score']);box=np.asarray(row['box'],dtype=float)
            if (not math.isfinite(score) or not 0<=score<=1 or box.shape!=(4,)
                    or not np.isfinite(box).all() or row['shape']!=list(rgb.shape[:2])):
                raise ValueError('Invalid SAM3 detection geometry')
            if score<self.confidence:
                continue
            encoded=row['mask_base64']
            if not isinstance(encoded,str) or len(encoded)>4*((rgb.shape[0]*rgb.shape[1]+2)//3):
                raise ValueError('Invalid SAM3 mask length')
            mask=base64.b64decode(encoded,validate=True)
            if len(mask)!=rgb.shape[0]*rgb.shape[1]:
                raise ValueError('SAM3 mask shape differs from image')
            array=np.frombuffer(mask,dtype=np.uint8).reshape(rgb.shape[:2])
            if np.any(array>1):
                raise ValueError('SAM3 mask must be binary')
            detections.append({'score':score,'box':box.tolist(),'mask':array.tolist()})
        detections.sort(key=lambda row:row['score'],reverse=True)
        return {'prompt':prompt,'image_shape':list(rgb.shape[:2]),
                'detections':detections[:self.max_detections]}
