import json, numpy as np
import omni.usd, omni.graph.core as og, omni.kit.app, omni.timeline
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from PIL import Image
import omni.replicator.core as rep
stage=omni.usd.get_context().get_stage()
PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath("/physicsScene")).GetMinPositionIterationCountAttr().Set(16)
for link in ["/World/agx_arm/gripper_link1","/World/agx_arm/gripper_link2"]:
    for cp in Usd.PrimRange(stage.GetPrimAtPath(link)):
        if cp.HasAPI(UsdPhysics.MeshCollisionAPI): UsdPhysics.MeshCollisionAPI(cp).GetApproximationAttr().Set("convexHull")
    v=stage.GetPrimAtPath(link+"/visuals")
    if v.HasAPI(UsdPhysics.CollisionAPI): UsdPhysics.CollisionAPI(v).GetCollisionEnabledAttr().Set(False)
for jn in ["gripper_joint1","gripper_joint2"]:
    UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/agx_arm/joints/"+jn),"linear").GetMaxForceAttr().Set(25.0)
try: og.get_node_by_path("/World/ActionGraph/articulation_controller").set_disabled(True)
except Exception: pass

# RESET objects to standing/placed
from pxr import Gf as _Gf
for _nm,_pos,_kin in [("/World/Cyl",(-0.46,-0.14,0.091),False),("/World/BoxT",(-0.42,0.10,0.041),False)]:
    _p=stage.GetPrimAtPath(_nm)
    if _p.IsValid():
        _x=UsdGeom.Xformable(_p)
        for _op in _x.GetOrderedXformOps():
            if _op.GetOpType()==UsdGeom.XformOp.TypeTranslate: _op.Set(_Gf.Vec3d(*_pos))
        rb=UsdPhysics.RigidBodyAPI(_p)
        rb.GetVelocityAttr().Set(_Gf.Vec3f(0,0,0)); rb.GetAngularVelocityAttr().Set(_Gf.Vec3f(0,0,0))
import omni.timeline as _tl
_tl.get_timeline_interface().stop()
for _ in range(6): omni.kit.app.get_app().update()
_tl.get_timeline_interface().play()
for _ in range(40): omni.kit.app.get_app().update()
art=SingleArticulation("/World/agx_arm"); art.initialize()
names=list(art.dof_names); g1=names.index("gripper_joint1"); g2=names.index("gripper_joint2")
TRANSIT=[[0.00019, -0.99865, 0.0003, 1.39837, 0.0002, -0.0004, 1.54911], [0.00353, -0.98909, 0.00098, 1.39956, -0.00843, -0.00331, 1.54908], [0.01251, -0.96344, 0.00282, 1.40273, -0.03155, -0.0111, 1.54899], [0.02711, -0.9217, 0.00586, 1.40779, -0.069, -0.02384, 1.54884], [0.04804, -0.8617, 0.01037, 1.41483, -0.12245, -0.04229, 1.54864], [0.0748, -0.78474, 0.01625, 1.42355, -0.19065, -0.06608, 1.54841], [0.10705, -0.69167, 0.0234, 1.43385, -0.27291, -0.09488, 1.54817], [0.14518, -0.58123, 0.0318, 1.44591, -0.37068, -0.129, 1.54792], [0.18862, -0.45515, 0.04112, 1.4598, -0.48302, -0.16773, 1.54769], [0.23358, -0.32472, 0.05039, 1.47459, -0.60035, -0.20748, 1.54749], [0.27864, -0.19428, 0.05919, 1.49006, -0.71908, -0.2468, 1.54729], [0.32391, -0.06385, 0.0674, 1.50654, -0.8395, -0.28556, 1.54705], [0.3694, 0.06659, 0.07511, 1.52404, -0.96135, -0.32377, 1.54673], [0.41512, 0.19702, 0.08233, 1.54259, -1.08447, -0.36141, 1.54628], [0.46108, 0.32746, 0.08904, 1.56235, -1.20877, -0.39839, 1.54566], [0.50679, 0.45675, 0.09536, 1.58284, -1.33263, -0.43457, 1.54488], [0.54917, 0.57643, 0.10102, 1.60239, -1.44749, -0.46775, 1.54401], [0.58534, 0.67866, 0.10581, 1.6193, -1.54542, -0.49601, 1.54316], [0.61534, 0.76357, 0.10986, 1.63326, -1.62639, -0.51956, 1.54243], [0.63959, 0.83239, 0.11325, 1.64435, -1.69167, -0.53879, 1.54183], [0.65783, 0.88425, 0.11592, 1.65248, -1.74058, -0.55342, 1.5414], [0.66963, 0.91785, 0.11772, 1.65759, -1.77211, -0.56299, 1.54115], [0.67577, 0.93537, 0.11867, 1.66023, -1.78851, -0.56799, 1.54102], [0.67696, 0.93874, 0.11885, 1.66074, -1.79168, -0.56896, 1.54099]]
ADVANCE=[[0.6794565471826527, 0.9308565948123607, 0.11709118877310241, 1.6604151067799542, -1.7862092575818667, -0.5606204934027609, 1.55], [0.6783524577807953, 0.9309773813963985, 0.11774318244508396, 1.6601827521361727, -1.787104552495145, -0.5608599818794705, 1.55], [0.6428398792777194, 0.9350419541670455, 0.13840119209828108, 1.6525038443129132, -1.8159480426574883, -0.5683364854198731, 1.55], [0.5652960965937182, 0.9463426161354263, 0.18465478154214252, 1.6330977343863988, -1.879335528690659, -0.5874980730328894, 1.55], [0.5145856305822154, 0.9554287941547788, 0.2163763668857112, 1.6182467853109392, -1.921395033663349, -0.6022646246026359, 1.55], [0.5009505383399093, 0.9536009745890606, 0.21582183343455216, 1.6131041727149122, -1.9490505827198057, -0.6265806790605924, 1.55], [0.49381288262375556, 0.9499120260661995, 0.21043697717165852, 1.6100444200969903, -1.9738480958274591, -0.651469861197048, 1.5499999739178056]]
RETREAT=[[0.49381288262375556, 0.9499120260661995, 0.21043697717165852, 1.6100444200969903, -1.9738480958274591, -0.651469861197048, 1.5499999739178056], [0.5009505383399093, 0.9536009745890606, 0.21582183343455216, 1.6131041727149122, -1.9490505827198057, -0.6265806790605924, 1.55], [0.5145856305822154, 0.9554287941547788, 0.2163763668857112, 1.6182467853109392, -1.921395033663349, -0.6022646246026359, 1.55], [0.5652960965937182, 0.9463426161354263, 0.18465478154214252, 1.6330977343863988, -1.879335528690659, -0.5874980730328894, 1.55], [0.6428398792777194, 0.9350419541670455, 0.13840119209828108, 1.6525038443129132, -1.8159480426574883, -0.5683364854198731, 1.55], [0.6783524577807953, 0.9309773813963985, 0.11774318244508396, 1.6601827521361727, -1.787104552495145, -0.5608599818794705, 1.55], [0.6794565471826527, 0.9308565948123607, 0.11709118877310241, 1.6604151067799542, -1.7862092575818667, -0.5606204934027609, 1.55]]
GOPEN=0.05
TGT="/World/Cyl"
def tp(p):
    m=UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t=m.ExtractTranslation(); return np.array([float(t[0]),float(t[1]),float(t[2])])
def fmid():
    a=tp("/World/agx_arm/gripper_link1"); b=tp("/World/agx_arm/gripper_link2"); return (a+b)/2
# obs pose, gripper open
obs=[0.0,-0.9991,-0.0008,1.3986,0.0056,0.0012,1.5491]
q=art.get_joint_positions().copy().astype(float)
for i,v in enumerate(obs): q[i]=v
q[g1]=GOPEN; q[g2]=-GOPEN
art.apply_action(ArticulationAction(joint_positions=q))
for _ in range(40): omni.kit.app.get_app().update()
o={"start":tp(TGT).round(4).tolist()}
def go(wps,steps=3):
    for wp in wps:
        q=art.get_joint_positions().copy().astype(float)
        for i,v in enumerate(wp): q[i]=v
        q[g1]=GOPEN; q[g2]=-GOPEN
        art.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(steps): omni.kit.app.get_app().update()
go(TRANSIT); go(ADVANCE)
for _ in range(15): omni.kit.app.get_app().update()
o["at_grasp_tgt"]=tp(TGT).round(4).tolist()
o["fmid"]=fmid().round(4).tolist()
o["fmid_tgt_d"]=round(float(np.linalg.norm(fmid()-tp(TGT))),4)
rp=rep.create.render_product("/OmniverseKit_Persp",(960,600))
aa=rep.AnnotatorRegistry.get_annotator("rgb"); aa.attach(rp)
for _ in range(20): omni.kit.app.get_app().update()
Image.fromarray(np.asarray(aa.get_data())[...,:3].astype(np.uint8)).save("/tmp/bottle_grasp.png")
G=ADVANCE[-1]
for s in range(1,46):
    f=s/45
    q=art.get_joint_positions().copy().astype(float)
    for i,v in enumerate(G): q[i]=v
    q[g1]=GOPEN*(1-f); q[g2]=-GOPEN*(1-f)
    art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(2): omni.kit.app.get_app().update()
for _ in range(25): omni.kit.app.get_app().update()
jp=art.get_joint_positions()
o["gj"]=[round(float(jp[g1]),4),round(float(jp[g2]),4)]
try:
    e=art.get_measured_joint_efforts(); o["geff"]=[round(float(e[g1]),2),round(float(e[g2]),2)]
except Exception: pass
GC=max(float(jp[g1])*0.6,0.0)
for wp in RETREAT:
    q=art.get_joint_positions().copy().astype(float)
    for i,v in enumerate(wp): q[i]=v
    q[g1]=GC; q[g2]=-GC
    art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(4): omni.kit.app.get_app().update()
for _ in range(12): omni.kit.app.get_app().update()
o["after_retreat"]=tp(TGT).round(4).tolist()
q=art.get_joint_positions().copy().astype(float); q[1]-=0.3; q[3]-=0.15
q[g1]=GC; q[g2]=-GC
art.apply_action(ArticulationAction(joint_positions=q))
for _ in range(60): omni.kit.app.get_app().update()
o["after_lift"]=tp(TGT).round(4).tolist()
o["rose"]=round(float(tp(TGT)[2]-o["start"][2]),4)
o["lift_fmid_d"]=round(float(np.linalg.norm(tp(TGT)-fmid())),4)
Image.fromarray(np.asarray(aa.get_data())[...,:3].astype(np.uint8)).save("/tmp/bottle_lift.png")
o["verdict"]="HELD" if (o["rose"]>0.04 and o["lift_fmid_d"]<0.10) else "DROPPED"
with open("/tmp/bottle_res.json","w") as f: json.dump(o,f,indent=1,default=str)
