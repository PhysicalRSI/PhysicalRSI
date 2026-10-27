"""Lift a registered depth pixel to a visible surface point, not an object pose."""
from PhysicalRSI_core.infra.storage import digest
from physicalrsi.runtime.rgbd import validate_rgbd,decode_depth_mm
from .protocol import CallableSkill,SkillDescriptor


def lift_depth_pixel(*, camera, intrinsic_cv, extrinsic_cv, image_size_px,
                     calibration_revision, depth_model):
    import numpy as np
    if not all(isinstance(x,str) and x for x in (camera,calibration_revision,depth_model)):
        raise ValueError('Declare calibrated sensor identities')
    K=np.array(intrinsic_cv,dtype=float,copy=True);E=np.array(extrinsic_cv,dtype=float,copy=True)
    if E.shape==(4,4):
        if not np.allclose(E[3],[0,0,0,1]):raise ValueError('Invalid extrinsics')
        E=E[:3].copy()
    if (K.shape!=(3,3) or E.shape!=(3,4) or not np.isfinite(K).all() or not np.isfinite(E).all()
            or not np.allclose(K[2],[0,0,1]) or K[0,0]<=0 or K[1,1]<=0
            or abs(np.linalg.det(K))<1e-12
            or not np.allclose(E[:,:3].T@E[:,:3],np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(E[:,:3]),1,atol=1e-5)):
        raise ValueError('Invalid camera calibration')
    if len(image_size_px)!=2 or any(type(x) is not int or x<=0 for x in image_size_px):
        raise ValueError('Declare calibrated image dimensions')
    size=tuple(image_size_px);inverse=np.linalg.inv(K)
    config=dict(camera=camera,intrinsic_cv=K.tolist(),extrinsic_cv=E.tolist(),image_size_px=list(size),
                calibration_revision=calibration_revision,depth_model=depth_model,contract_revision=1)
    def run(value,context):
        if set(value)!={'observation','uv'}:raise ValueError('Unexpected depth-point inputs')
        packet=validate_rgbd(value['observation']);rgb=packet['rgb'];uv=value['uv']
        if (rgb['calibration_revision']!=calibration_revision or packet['depth_model']!=depth_model):
            raise ValueError('Depth model or calibration changed')
        if (not isinstance(uv,(list,tuple)) or len(uv)!=2
                or any(type(x) not in (int,float) or not np.isfinite(x) or not 0<=x<1 for x in uv)):
            raise ValueError('Pixel must lie inside the calibrated image')
        context.check();depth=decode_depth_mm(packet['depth'][camera]['png_b64'])
        if (depth.shape[1],depth.shape[0])!=size:raise ValueError('Image size differs from calibration')
        x,y=[int(a*b) for a,b in zip(uv,size)];z=float(depth[y,x])/1000.
        common=dict(observation_sequence=rgb['sequence'],camera=camera,pixel_xy=[x,y],
                    timestamp_s=rgb['cameras'][camera]['captured_at_s'],semantic_identity=None)
        if z==0:return dict(status='unavailable',reason='missing_depth',**common)
        point=inverse@np.array([x+.5,y+.5,1.])*z
        world=E[:,:3].T@(point-E[:,3])
        return dict(status='measured',measurement='visible_surface_point',xyz_camera_m=point.tolist(),
                    xyz_world_m=world.tolist(),depth_m=z,**common)
    return CallableSkill(SkillDescriptor('lift_depth_pixel',digest(config),
        {'name':'physicalrsi.rgbd-pixel/v1'},{'name':'physicalrsi.visible-surface-point/v1'},metadata=config),run)
