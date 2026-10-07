import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import pytest
from PhysicalRSI_baselines.embodied_goodharts_law.grasp_client import ContactGraspClient, _encode, _decode


@pytest.mark.parametrize('depth_range', [(.2, 2.), (.015, 2.)])
def test_grasp_client_real_http_validates_pose_and_orders_scores(depth_range):
    poses=np.stack([np.eye(4),np.eye(4)])
    reply={'grasps_base64':_encode(poses),'scores_base64':_encode(np.array([.3,.8]))}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path=='/plan' and data['max_retries']==1
            assert data['z_range']==list(depth_range)
            assert _decode(data['depth_base64']).shape==(8,8)
            body=json.dumps(reply).encode();self.send_response(200)
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=ContactGraspClient(port=server.server_port,max_grasps=1,depth_range_m=depth_range)
        args=(np.ones((8,8)),np.eye(3),np.ones((8,8)))
        result=client(*args,deadline=time.monotonic()+5)
        assert len(result['grasps'])==1 and result['grasps'][0]['score']==.8
        reply['contact_pts_base64']=_encode(np.array([[1.,2.,3.],[4.,5.,6.]]))
        result=client(*args,deadline=time.monotonic()+5)
        assert result['grasps'][0]['contact_point']==[4.,5.,6.]
        reply['contact_pts_base64']=_encode(np.zeros((1,3)))
        with pytest.raises(ValueError,match='align'):client(*args,deadline=time.monotonic()+5)
        reply.pop('contact_pts_base64')
        poses[0,0,0]=2;reply['grasps_base64']=_encode(poses)
        with pytest.raises(ValueError,match='rotation'):client(*args,deadline=time.monotonic()+5)
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_grasp_client_rejects_bad_observation_and_expired_deadline():
    client=ContactGraspClient(port=1)
    with pytest.raises(ValueError,match='depth'):
        client(np.zeros((8,8)),np.eye(3),np.ones((8,8)),deadline=time.monotonic()+1)
    with pytest.raises(TimeoutError):
        client(np.ones((8,8)),np.eye(3),np.ones((8,8)),deadline=time.monotonic()-1)


@pytest.mark.parametrize('bounds', [(0, 2), (.2, .1), (.1, float('inf')), (.1,), None])
def test_invalid_grasp_depth_bounds(bounds):
    with pytest.raises(ValueError, match='depth range'):
        ContactGraspClient(depth_range_m=bounds)


def test_point_cloud_request_uses_explicit_frame_and_bounded_arrays(monkeypatch):
    client=ContactGraspClient();requests=[]
    def request(body,path,frame,deadline):
        requests.append((json.loads(body),path,frame))
        return {'frame':frame,'grasps':[]}
    monkeypatch.setattr(client,'_request',request)
    result=client.from_point_clouds(np.ones((40,3)),np.ones((20,3)),deadline=time.monotonic()+5)
    assert result['frame']=='input-point-cloud'
    body,path,frame=requests[0]
    assert path=='/plan_point_clouds' and _decode(body['pc_full_base64']).shape==(40,3)
    assert _decode(body['pc_segment_base64']).shape==(20,3)
    for bad in [np.zeros((19,3)),np.zeros((20,4)),np.full((20,3),np.nan)]:
        with pytest.raises(ValueError,match='point cloud'):
            client.from_point_clouds(np.ones((40,3)),bad,deadline=time.monotonic()+5)
