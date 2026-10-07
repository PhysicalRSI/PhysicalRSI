import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.sam3_client import Sam3Client


def test_sam3_response_filtering_and_invalid_geometry():
    good={'score':.8,'box':[0,0,2,2],'shape':[2,2],
          'mask_base64':base64.b64encode(bytes([1,0,0,1])).decode()}
    reply={'results':[{**good,'score':.2},good]}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path=='/segment' and request['text_prompt']=='can'
            body=json.dumps(reply).encode();self.send_response(200)
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        client=Sam3Client(port=server.server_port)
        result=client(np.zeros((2,2,3),dtype=np.uint8),'can',deadline=time.monotonic()+5)
        assert len(result['detections'])==1
        assert result['detections'][0]['mask']==[[1,0],[0,1]]
        reply['results']=[{**good,'shape':[1,4]}]
        with pytest.raises(ValueError,match='geometry'):
            client(np.zeros((2,2,3),dtype=np.uint8),'can',deadline=time.monotonic()+5)
    finally:
        server.shutdown();server.server_close();worker.join(2)


def test_sam3_rejects_expired_deadline_and_non_rgb_before_connection():
    client=Sam3Client(port=1)
    with pytest.raises(TimeoutError):client(np.zeros((2,2,3),dtype=np.uint8),'can',deadline=time.monotonic()-1)
    with pytest.raises(ValueError,match='RGB'):client(np.zeros((2,2,3)),'can',deadline=time.monotonic()+5)
