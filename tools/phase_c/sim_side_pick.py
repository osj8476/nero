import json, numpy as np
import omni.usd, omni.graph.core as og, omni.kit.app, omni.timeline
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.replicator.core as rep
from PIL import Image
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
for nm in ["/World/Cyl","/World/WaterBottle","/World/grasp_attach"]:
    if stage.GetPrimAtPath(nm).IsValid(): stage.RemovePrim(nm)
c=UsdGeom.Cylinder.Define(stage,"/World/Cyl")
c.CreateRadiusAttr(0.025); c.CreateHeightAttr(0.20); c.CreateAxisAttr("Z")
c.CreateExtentAttr([(-0.025,-0.025,-0.1),(0.025,0.025,0.1)]); c.CreateDisplayColorAttr([(0.15,0.35,0.7)])
p=stage.GetPrimAtPath("/World/Cyl"); UsdGeom.Xformable(p).ClearXformOpOrder()
UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(-0.5,0.0,0.101))
UsdPhysics.RigidBodyAPI.Apply(p); UsdPhysics.CollisionAPI.Apply(p); UsdPhysics.MassAPI.Apply(p).CreateMassAttr(0.4)
mp="/World/Materials/BoxHighFriction"
if stage.GetPrimAtPath(mp).IsValid(): p.CreateRelationship("material:binding:physics",False).SetTargets([Sdf.Path(mp)])
for nm,y in [("/World/TestBox",-1.6),("/World/TestBox1",-1.9),("/World/TestBox2",1.8)]:
    for op in UsdGeom.Xformable(stage.GetPrimAtPath(nm)).GetOrderedXformOps():
        if op.GetOpType()==UsdGeom.XformOp.TypeTranslate: op.Set(Gf.Vec3d(-0.5,y,0.05))
omni.timeline.get_timeline_interface().stop()
for _ in range(6): omni.kit.app.get_app().update()
omni.timeline.get_timeline_interface().play()
for _ in range(35): omni.kit.app.get_app().update()
art=SingleArticulation("/World/agx_arm"); art.initialize()
names=list(art.dof_names); g1=names.index("gripper_joint1"); g2=names.index("gripper_joint2")
TRANSIT=[[0.0, -0.9991, 0.0, 1.3986, 0.0, 0.0, 1.5491], [0.00252, -0.9911, -0.00029, 1.39888, -0.00608, -0.00284, 1.54908], [0.00944, -0.9691, -0.00109, 1.39964, -0.02281, -0.01064, 1.54904], [0.02077, -0.9331, -0.0024, 1.40089, -0.05019, -0.02342, 1.54896], [0.03651, -0.8831, -0.00422, 1.40263, -0.08821, -0.04116, 1.54885], [0.05665, -0.8191, -0.00654, 1.40485, -0.13688, -0.06387, 1.54871], [0.08182, -0.7391, -0.00945, 1.40763, -0.19772, -0.09225, 1.54854], [0.11141, -0.6451, -0.01287, 1.41089, -0.2692, -0.12561, 1.54834], [0.1454, -0.5371, -0.01679, 1.41464, -0.35132, -0.16393, 1.54811], [0.18253, -0.4191, -0.02108, 1.41874, -0.44106, -0.2058, 1.54785], [0.2203, -0.2991, -0.02544, 1.4229, -0.53231, -0.24838, 1.5476], [0.25806, -0.1791, -0.0298, 1.42707, -0.62356, -0.29095, 1.54734], [0.29583, -0.0591, -0.03416, 1.43124, -0.71482, -0.33353, 1.54708], [0.33359, 0.0609, -0.03852, 1.4354, -0.80607, -0.37611, 1.54682], [0.37136, 0.1809, -0.04288, 1.43957, -0.89732, -0.41869, 1.54657], [0.40912, 0.3009, -0.04725, 1.44374, -0.98858, -0.46127, 1.54631], [0.44688, 0.42087, -0.05161, 1.4479, -1.07981, -0.50384, 1.54605], [0.48375, 0.53804, -0.05586, 1.45197, -1.16891, -0.54541, 1.5458], [0.51708, 0.64394, -0.05971, 1.45565, -1.24944, -0.58299, 1.54557], [0.546, 0.73583, -0.06305, 1.45884, -1.31932, -0.61559, 1.54537], [0.57051, 0.81372, -0.06588, 1.46154, -1.37855, -0.64323, 1.54521], [0.58999, 0.87562, -0.06813, 1.46369, -1.42562, -0.66519, 1.54507], [0.60506, 0.92351, -0.06987, 1.46535, -1.46204, -0.68218, 1.54497], [0.61573, 0.9574, -0.0711, 1.46653, -1.48781, -0.69421, 1.5449], [0.62199, 0.9773, -0.07183, 1.46722, -1.50294, -0.70127, 1.54485], [0.62389, 0.98334, -0.07205, 1.46743, -1.50754, -0.70341, 1.54484]]
ADVANCE=[[0.6163469936787398, 0.9813799154007875, -0.07535807188905451, 1.466412389047884, -1.505153657554745, -0.6969808234448907, 1.55], [0.6119566839232939, 0.9812152430362588, -0.07249956324181298, 1.4666914677890261, -1.5088314897274673, -0.6967133683919692, 1.55], [0.5659897756812357, 0.979716184348965, -0.04287771675591243, 1.4684075992736372, -1.5472340254957513, -0.6944865890500882, 1.55], [0.4714587544644924, 0.9800731087558506, 0.018050935406010032, 1.466546365151975, -1.6261646464408852, -0.6951374362806917, 1.55], [0.36373491310122097, 0.9862801960366316, 0.08884162328609314, 1.4555665138565792, -1.71672461607974, -0.7046376797093467, 1.55], [0.3014961655892975, 0.9928477011612807, 0.130880906028813, 1.4454269427553328, -1.7697099524140436, -0.7141921948087004, 1.55], [0.2772353039473655, 0.9936475258805774, 0.13940658514031054, 1.4395585440519316, -1.7823289547785008, -0.7097825458499631, 1.55], [0.25375741166550053, 0.9920659747654917, 0.1375421031506931, 1.4303915681423267, -1.7831994004355758, -0.6945753962124681, 1.5499999807952611]]
RETREAT=[[0.25375741166550053, 0.9920659747654917, 0.1375421031506931, 1.4303915681423267, -1.7831994004355758, -0.6945753962124681, 1.5499999807952611], [0.2772353039473655, 0.9936475258805774, 0.13940658514031054, 1.4395585440519316, -1.7823289547785008, -0.7097825458499631, 1.55], [0.3014961655892975, 0.9928477011612807, 0.130880906028813, 1.4454269427553328, -1.7697099524140436, -0.7141921948087004, 1.55], [0.36373491310122097, 0.9862801960366316, 0.08884162328609314, 1.4555665138565792, -1.71672461607974, -0.7046376797093467, 1.55], [0.4714587544644924, 0.9800731087558506, 0.018050935406010032, 1.466546365151975, -1.6261646464408852, -0.6951374362806917, 1.55], [0.5659897756812357, 0.979716184348965, -0.04287771675591243, 1.4684075992736372, -1.5472340254957513, -0.6944865890500882, 1.55], [0.6119566839232939, 0.9812152430362588, -0.07249956324181298, 1.4666914677890261, -1.5088314897274673, -0.6967133683919692, 1.55], [0.6163469936787398, 0.9813799154007875, -0.07535807188905451, 1.466412389047884, -1.505153657554745, -0.6969808234448907, 1.55]]
GOPEN=0.05
def cyl():
    m=UsdGeom.Xformable(stage.GetPrimAtPath("/World/Cyl")).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t=m.ExtractTranslation(); return np.array([float(t[0]),float(t[1]),float(t[2])])
def fmid():
    a=UsdGeom.Xformable(stage.GetPrimAtPath("/World/agx_arm/gripper_link1")).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    b=UsdGeom.Xformable(stage.GetPrimAtPath("/World/agx_arm/gripper_link2")).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    return (np.array([a[0],a[1],a[2]])+np.array([b[0],b[1],b[2]]))/2
q=art.get_joint_positions().copy().astype(float)
for i in range(7): q[i]=0.0
q[g1]=GOPEN; q[g2]=-GOPEN
art.apply_action(ArticulationAction(joint_positions=q))
for _ in range(45): omni.kit.app.get_app().update()
o={"cyl_start":cyl().round(4).tolist()}
def go(wps,steps=3):
    for wp in wps:
        q=art.get_joint_positions().copy().astype(float)
        for i,v in enumerate(wp): q[i]=v
        q[g1]=GOPEN; q[g2]=-GOPEN
        art.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(steps): omni.kit.app.get_app().update()
go(TRANSIT); go(ADVANCE)
for _ in range(15): omni.kit.app.get_app().update()
o["after_advance_cyl"]=cyl().round(4).tolist()
o["fmid_grasp"]=fmid().round(4).tolist()
o["fmid_cyl_xy"]=round(float(np.hypot(fmid()[0]-cyl()[0],fmid()[1]-cyl()[1])),4)
o["fmid_cyl_z"]=round(float(fmid()[2]-cyl()[2]),4)
# render at grasp pose
rp=rep.create.render_product("/OmniverseKit_Persp",(960,600))
aa=rep.AnnotatorRegistry.get_annotator("rgb"); aa.attach(rp)
for _ in range(25): omni.kit.app.get_app().update()
Image.fromarray(np.asarray(aa.get_data())[...,:3].astype(np.uint8)).save("/tmp/ci_grasp.png")
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
o["after_close_cyl"]=cyl().round(4).tolist()
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
o["after_retreat_cyl"]=cyl().round(4).tolist()
q=art.get_joint_positions().copy().astype(float); q[1]-=0.35; q[3]-=0.2
q[g1]=GC; q[g2]=-GC
art.apply_action(ArticulationAction(joint_positions=q))
for _ in range(60): omni.kit.app.get_app().update()
o["after_lift_cyl"]=cyl().round(4).tolist()
o["lift_rose"]=round(float(cyl()[2]-o["cyl_start"][2]),4)
o["lift_fmid_cyl"]=round(float(np.linalg.norm(cyl()-fmid())),4)
Image.fromarray(np.asarray(aa.get_data())[...,:3].astype(np.uint8)).save("/tmp/ci_lift.png")
o["verdict"]="HELD" if (cyl()[2]-o["cyl_start"][2]>0.05 and np.linalg.norm(cyl()-fmid())<0.10) else "DROPPED"
with open("/tmp/ci_2.json","w") as f: json.dump(o,f,indent=1,default=str)
