"""Authored lab props and a softly lit room; render meshes use rigid collision proxies."""
import math
import os
from pathlib import Path


def asset_identity():
    from PhysicalRSI_core.infra.storage import file_digest
    root=Path(os.environ.get('PHYSICALRSI_LAB_ASSETS','')).resolve()
    names=('studio_small_09_2k.hdr','beaker.usda','drying_oven.usda','lab-assets.json')
    return {name:file_digest(root/name) for name in names if (root/name).is_file()}


def dress(stage, spec, asset_archive=None):
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdLux, Usd, UsdPhysics
    from isaacsim.core.api.materials import OmniGlass
    from PhysicalRSI_core.infra.storage import file_digest
    assets=Path(os.environ.get('PHYSICALRSI_LAB_ASSETS',''))
    if asset_archive is not None:
        import shutil
        asset_archive=Path(asset_archive);asset_archive.mkdir(exist_ok=False)
        for name in asset_identity():shutil.copyfile(assets/name,asset_archive/name)
        assets=asset_archive
    provenance=[]
    stage.GetPrimAtPath('/World/Floor').GetAttribute('xformOp:translate').Set(Gf.Vec3d(0,0,-.87))
    root='/World/Design'
    UsdGeom.Xform.Define(stage,root)
    def material(name,color,rough=.4,metal=0.,opacity=1.):
        p=root+'/Materials/'+name
        mat=UsdShade.Material.Define(stage,p)
        sh=UsdShade.Shader.Define(stage,p+'/Shader');sh.CreateIdAttr('UsdPreviewSurface')
        for k,t,v in [('diffuseColor',Sdf.ValueTypeNames.Color3f,Gf.Vec3f(*color)),('roughness',Sdf.ValueTypeNames.Float,rough),('metallic',Sdf.ValueTypeNames.Float,metal),('opacity',Sdf.ValueTypeNames.Float,opacity)]:
            sh.CreateInput(k,t).Set(v)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
        return mat
    mats={'white':material('WarmCeramic',(.72,.76,.75),.3),
          'steel':material('BrushedAluminum',(.46,.51,.52),.26,.8),
          'dark':material('Graphite',(.025,.04,.045),.48),
          'teal':material('Petrol',(.025,.13,.14),.35),
          'blue':material('SampleBlue',(.015,.075,.63),.24),
          'red':material('SampleRed',(.57,.022,.028),.28),
          'amber':material('AmberMarker',(.65,.35,.055),.3),
          'green':material('ReceiverGreen',(.035,.43,.13),.38),
          'label':material('Label',(.92,.94,.91),.55),
          'hidden':material('CollisionOnly',(0.,0.,0.),1.,0.,0.),
          'wall':material('Wall',(.65,.71,.71),.8),
          'floor':material('Floor',(.3,.36,.36),.55)}
    glass=OmniGlass(root+'/Materials/Glass',color=np.array([.91,.98,.98]),ior=1.47,depth=.001,thin_walled=False)
    mats['glass']=glass.material
    def bind(prim,mat):UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[mat],bindingStrength=UsdShade.Tokens.weakerThanDescendants)
    def pose(prim,pos):UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
    def box(name,pos,size,mat):
        p=UsdGeom.Cube.Define(stage,root+'/'+name);p.CreateSizeAttr(1.)
        pose(p.GetPrim(),pos);UsdGeom.Xformable(p).AddScaleOp().Set(Gf.Vec3f(*size));bind(p.GetPrim(),mat)
        UsdPhysics.CollisionAPI.Apply(p.GetPrim())
        return p.GetPrim()
    def lathe(path,profile,mat,pos=(0,0,0),segments=96):
        pts=[];faces=[]
        for radius,z in profile:
            pts.extend([(radius*math.cos(a*math.tau/segments),radius*math.sin(a*math.tau/segments),z) for a in range(segments)])
        for j in range(len(profile)-1):
            for i in range(segments):
                a=j*segments+i;b=j*segments+(i+1)%segments;faces.extend([a,b,b+segments,a+segments])
        mesh=UsdGeom.Mesh.Define(stage,path);mesh.CreatePointsAttr(pts);mesh.CreateFaceVertexCountsAttr([4]*(len(faces)//4));mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True);mesh.CreatePurposeAttr('default');pose(mesh.GetPrim(),pos);bind(mesh.GetPrim(),mat)
        return mesh.GetPrim()
    def ring(path,radius,width,z,height,mat,pos=(0,0,0)):
        return lathe(path,[(radius-width,z),(radius,z),(radius,z+height),(radius-width,z+height),(radius-width,z)],mat,pos)
    # The physical task objects retain their cylinder proxies. Detailed child meshes
    # share the same rigid transforms; no animation or object attachment is used.

    for name,color in [('BlueSample','blue'),('RedSample','red')]:
        p='/World/'+name
        lathe(p+'/Vial',[(0,-.06),(.011,-.06),(.016,-.055),(.0175,-.048),(.0175,.036),(.014,.043),(0,.043)],'white')
        lathe(p+'/Cap',[(0,.035),(.017,.035),(.018,.038),(.018,.057),(.016,.06),(0,.06)],color)
        ring(p+'/Seal',.0178,.0008,.031,.004,color)
        # Fine cap knurling and a specimen label, all rigidly part of the vial.
        for i in range(32):
            a=math.tau*i/32
            lathe(p+'/Rib'+str(i),[(.0007,.039),(.0007,.055)],color,(.0178*math.cos(a),.0178*math.sin(a),0),segments=8)
        ring(p+'/Label',.0177,.00025,-.025,.039,'label')
        for i in range(8):
            m=UsdGeom.Cube.Define(stage,p+'/Barcode'+str(i));m.CreateSizeAttr(1.)
            m.CreatePurposeAttr('default')
            pose(m.GetPrim(),(-.010+i*.0025,-.0177,-.006));UsdGeom.Xformable(m).AddScaleOp().Set(Gf.Vec3f(.0007,.0003,.017 if i%3 else .022));bind(m.GetPrim(),'dark')
    def imported(path,filename,pos,size,override=None):
        source=assets/filename
        if not source.is_file():return False
        prim=UsdGeom.Xform.Define(stage,path).GetPrim();prim.GetReferences().AddReference(str(source.resolve()))
        pose(prim,pos);UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*size))
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                UsdGeom.Imageable(child).CreatePurposeAttr('default')
                category=child.GetAttribute('physicalRSI:material').Get() or 'white'
                bind(child,override or category)
        provenance.append({'asset':filename,'source':'https://github.com/Rui-li023/LabUtopia','license':'CC BY-NC 4.0','sha256':file_digest(source)})
        return True
    # A hollow borosilicate beaker with a rolled lip and amber rim marker.
    p='/World/Beaker/Visual'
    visual=UsdGeom.Xform.Define(stage,p)
    visual.AddScaleOp().Set(Gf.Vec3f(1.,1.,spec['obstacle_height']/.18))
    if not imported(p+'/Glass','beaker.usda',(0,0,-.09),(.11,.11,.18),'glass'):
        lathe(p+'/Glass',[(0,-.09),(.05,-.09),(.055,-.086),(.055,.081),(.057,.084),(.057,.088),(.054,.09),(.051,.087),(.051,-.082),(0,-.082)],'glass')
    ring(p+'/SafetyRim',.057,.006,.083,.007,'amber')
    for i in range(1,8):
        tick=UsdGeom.Cube.Define(stage,p+'/Graduation'+str(i));tick.CreateSizeAttr(1.)
        tick.CreatePurposeAttr('default')
        pose(tick.GetPrim(),(.01,-.0549,-.074+i*.019));UsdGeom.Xformable(tick).AddScaleOp().Set(Gf.Vec3f(.018 if i%2==0 else .011,.0005,.001));bind(tick.GetPrim(),'white')
    # Structured holders and a machined mounting plate.
    for prim in ['/World/Bench','/World/SourceRack']:
        bind(stage.GetPrimAtPath(prim),'white')
    for prim in ['/World/RackRail0','/World/RackRail1']:bind(stage.GetPrimAtPath(prim),'steel')
    tx,ty,_=spec['tube'];gx,gy,_=spec['receiver']
    for label,x,y in [('Source',tx,ty),('Receiver',gx,gy)]:
        for sx in [-1,1]:
            box(label+'Bracket'+('L' if sx<0 else 'R'),(x+sx*.043,y,.017),(.007,.09,.021),'steel')
        box(label+'Front',(x,y-.05,.012),(.097,.007,.016),'teal')
        for i in [-1,1]:ring(root+'/'+label+'Fastener'+('L' if i<0 else 'R'),.003,.0015,.012,.002,'steel',(x+i*.035,y-.034,0))
    ring(root+'/RobotMount',.105,.065,-.002,.003,'steel')
    # Full-height bench, feet, and a clean laboratory backdrop.
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/Floor')).CreatePurposeAttr('guide')
    box('RoomFloor',(.3,.4,-.85),(6.,6.,.04),'floor')
    box('Worktop',(.37,0,-.045),(1.5,1.05,.07),'white')
    box('FrontApron',(.37,-.507,-.115),(1.47,.025,.075),'teal')
    for x in [-.24,.99]:
        for y in [-.42,.42]:
            box('Leg'+str(x).replace('-','n').replace('.','_')+str(y).replace('-','n').replace('.','_'),(x,y,-.46),(.045,.045,.72),'steel')
    box('Cabinet',(-.19,.05,-.45),(.30,.76,.68),'white')
    for i in range(3):
        box('Drawer'+str(i),(-.19,-.337,-.23-i*.21),(.27,.008,.18),'white')
        box('Handle'+str(i),(-.19,-.351,-.19-i*.21),(.12,.018,.009),'steel')
    box('LeftWall',(-1.7,.4,.6),(.08,3.0,3.0),'wall')
    box('BackWall',(.3,1.85,.6),(5.,.08,3.0),'wall')
    box('WallBand',(.3,1.8,.03),(5.,.018,.15),'teal')
    box('BackCounter',(.3,1.22,-.15),(2.7,.59,.065),'white')
    box('BackCabinets',(.3,1.25,-.51),(2.65,.54,.65),'white')
    for i in range(6):
        x=-.83+i*.45
        box('DoorSeam'+str(i),(x,.974,-.48),(.003,.008,.55),'steel')
        box('DoorPull'+str(i),(x+.36,.966,-.25),(.055,.017,.008),'steel')
    box('WindowFrame',(.0,1.782,.82),(1.6,.055,.84),'steel')
    box('Window',(.0,1.747,.82),(1.53,.012,.77),'white')
    for x in [-.53,0,.53]:box('Mullion'+str(x).replace('-','n').replace('.','_'),(x,1.737,.82),(.013,.022,.77),'steel')
    # Background instrument and flasks are outside the policy camera workspace.
    box('OvenHousing',(.91,1.21,.089),(.43,.37,.41),'white')
    imported(root+'/DryingOven','drying_oven.usda',(.91,1.2,-.116),(.45,.4,.43))
    for i in range(3):
        x=-.55+i*.15
        lathe(root+'/Flask'+str(i),[(0,0),(.042,0),(.047,.009),(.047,.035),(.015,.115),(.015,.15),(.012,.15),(.012,.118),(.043,.035),(.042,.005),(0,.005)],'glass',(x,1.2,-.116))
        ring(root+'/FlaskRim'+str(i),.017,.006,.148,.006,'teal',(x,1.2,-.116))
    # Environment illumination and broad softboxes keep colors saturated without
    # the harsh point-light shadows of the prototype.
    key=stage.GetPrimAtPath('/World/Key');UsdLux.DistantLight(key).GetIntensityAttr().Set(0.)
    dome=UsdLux.DomeLight(stage.GetPrimAtPath('/World/LabLight'))
    hdri=assets/'studio_small_09_2k.hdr'
    if hdri.is_file():
        dome.CreateTextureFileAttr(str(hdri.resolve()));dome.CreateTextureFormatAttr('latlong');dome.GetIntensityAttr().Set(180.)
        provenance.append({'asset':'Studio Small 09','source':'https://polyhaven.com/a/studio_small_09','license':'CC0-1.0','sha256':file_digest(hdri)})
    else:dome.GetIntensityAttr().Set(230.)
    for name,pos,power,size in [('KeySoftbox',(.5,-.3,1.7),1200.,(1.7,1.2)),('FillSoftbox',(-.4,.4,1.4),750.,(1.2,1.2))]:
        lamp=UsdLux.RectLight.Define(stage,root+'/'+name);pose(lamp.GetPrim(),pos)
        lamp.CreateWidthAttr(size[0]);lamp.CreateHeightAttr(size[1]);lamp.CreateIntensityAttr(power)
    return {'design':'authored laboratory v2','assets':provenance,'collision_geometry':'original rigid cylinder proxies; child render meshes add caps, glass walls, labels and trim','fluid_simulation':False}
