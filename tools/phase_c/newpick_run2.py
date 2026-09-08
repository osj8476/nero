import json, time, numpy as np
import omni.usd, omni.graph.core as og, omni.kit.app, omni.timeline
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.replicator.core as rep
from PIL import Image

stage = omni.usd.get_context().get_stage()
CYLXY = (-0.44, 0.070)          # NEW position
CYLH  = 0.20
CYLR  = 0.025
t_exec0 = time.time()

# ---------- FULL physics setup (same as the run that HELD at -0.5,0) ----------
PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath("/physicsScene")).GetMinPositionIterationCountAttr().Set(16)
for link in ["/World/agx_arm/gripper_link1", "/World/agx_arm/gripper_link2"]:
    for cp in Usd.PrimRange(stage.GetPrimAtPath(link)):
        if cp.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(cp).GetApproximationAttr().Set("convexHull")
    v = stage.GetPrimAtPath(link + "/visuals")
    if v.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(v).GetCollisionEnabledAttr().Set(False)
for jn in ["gripper_joint1", "gripper_joint2"]:
    UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/agx_arm/joints/" + jn), "linear").GetMaxForceAttr().Set(25.0)
try:
    og.get_node_by_path("/World/ActionGraph/articulation_controller").set_disabled(True)
except Exception:
    pass

# ---------- fresh cylinder at the new position ----------
for nm in ["/World/Cyl", "/World/WaterBottle", "/World/grasp_attach"]:
    if stage.GetPrimAtPath(nm).IsValid():
        stage.RemovePrim(nm)
c = UsdGeom.Cylinder.Define(stage, "/World/Cyl")
c.CreateRadiusAttr(CYLR); c.CreateHeightAttr(CYLH); c.CreateAxisAttr("Z")
c.CreateExtentAttr([(-CYLR, -CYLR, -CYLH / 2), (CYLR, CYLR, CYLH / 2)])
c.CreateDisplayColorAttr([(0.15, 0.35, 0.7)])
p = stage.GetPrimAtPath("/World/Cyl")
UsdGeom.Xformable(p).ClearXformOpOrder()
UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(CYLXY[0], CYLXY[1], CYLH / 2 + 0.001))
UsdPhysics.RigidBodyAPI.Apply(p); UsdPhysics.CollisionAPI.Apply(p)
UsdPhysics.MassAPI.Apply(p).CreateMassAttr(0.4)
mp = "/World/Materials/BoxHighFriction"
if stage.GetPrimAtPath(mp).IsValid():
    p.CreateRelationship("material:binding:physics", False).SetTargets([Sdf.Path(mp)])

omni.timeline.get_timeline_interface().stop()
for _ in range(6): omni.kit.app.get_app().update()
omni.timeline.get_timeline_interface().play()
for _ in range(30): omni.kit.app.get_app().update()

# stop/play snaps the cylinder to its authored (-0.5,0) transform -> force it to the
# real target position as a physics body (the reliable path)
srp = SingleRigidPrim("/World/Cyl"); srp.initialize()
srp.set_world_pose(position=np.array([CYLXY[0], CYLXY[1], CYLH / 2 + 0.001]),
                   orientation=np.array([1.0, 0.0, 0.0, 0.0]))
srp.set_linear_velocity(np.zeros(3)); srp.set_angular_velocity(np.zeros(3))
for _ in range(30): omni.kit.app.get_app().update()

art = SingleArticulation("/World/agx_arm"); art.initialize()
names = list(art.dof_names); g1 = names.index("gripper_joint1"); g2 = names.index("gripper_joint2")

D = json.loads(open("/tmp/step6_pick_v2.json").read())
TRANSIT = D["transit"]; ADVANCE = D["advance"]; RETREAT = D["retreat"]
GOPEN = 0.05

def tp(path):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    x = m.ExtractTranslation(); return np.array([float(x[0]), float(x[1]), float(x[2])])
def fmid():
    return (tp("/World/agx_arm/gripper_link1") + tp("/World/agx_arm/gripper_link2")) / 2

obs = [0.0, -0.9991, -0.0008, 1.3986, 0.0056, 0.0012, 1.5491]
q = art.get_joint_positions().copy().astype(float)
for i, v in enumerate(obs): q[i] = v
q[g1] = GOPEN; q[g2] = -GOPEN
art.apply_action(ArticulationAction(joint_positions=q))
for _ in range(40): omni.kit.app.get_app().update()

res = {"cyl_start": tp("/World/Cyl").round(4).tolist()}
tA = time.time()

def go(wps, steps=3):
    for wp in wps:
        q = art.get_joint_positions().copy().astype(float)
        for i, v in enumerate(wp): q[i] = v
        q[g1] = GOPEN; q[g2] = -GOPEN
        art.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(steps): omni.kit.app.get_app().update()

go(TRANSIT)
res["t_transit"] = round(time.time() - tA, 2); tB = time.time()
res["cyl_after_transit"] = tp("/World/Cyl").round(4).tolist()
go(ADVANCE, steps=5)                       # slower advance
for _ in range(15): omni.kit.app.get_app().update()
res["t_advance"] = round(time.time() - tB, 2); tC = time.time()
res["cyl_after_approach"] = tp("/World/Cyl").round(4).tolist()
res["fmid_cyl_d"] = round(float(np.linalg.norm(fmid() - tp("/World/Cyl"))), 4)

G = ADVANCE[-1]
for s in range(1, 46):
    f = s / 45
    q = art.get_joint_positions().copy().astype(float)
    for i, v in enumerate(G): q[i] = v
    q[g1] = GOPEN * (1 - f); q[g2] = -GOPEN * (1 - f)
    art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(2): omni.kit.app.get_app().update()
for _ in range(25): omni.kit.app.get_app().update()
res["t_close"] = round(time.time() - tC, 2); tD = time.time()
jp = art.get_joint_positions()
res["gj"] = [round(float(jp[g1]), 4), round(float(jp[g2]), 4)]
try:
    e = art.get_measured_joint_efforts(); res["geff"] = [round(float(e[g1]), 2), round(float(e[g2]), 2)]
except Exception:
    pass
res["cyl_after_close"] = tp("/World/Cyl").round(4).tolist()

GC = max(float(jp[g1]) * 0.6, 0.0)
for wp in RETREAT:
    q = art.get_joint_positions().copy().astype(float)
    for i, v in enumerate(wp): q[i] = v
    q[g1] = GC; q[g2] = -GC
    art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(4): omni.kit.app.get_app().update()
q = art.get_joint_positions().copy().astype(float); q[1] -= 0.30; q[3] -= 0.15
q[g1] = GC; q[g2] = -GC
art.apply_action(ArticulationAction(joint_positions=q))
for _ in range(60): omni.kit.app.get_app().update()
res["t_retreat_lift"] = round(time.time() - tD, 2)
res["t_total_exec"] = round(time.time() - t_exec0, 2)
res["cyl_final"] = tp("/World/Cyl").round(4).tolist()
res["rose"] = round(float(tp("/World/Cyl")[2] - res["cyl_start"][2]), 4)
res["lift_fmid_d"] = round(float(np.linalg.norm(tp("/World/Cyl") - fmid())), 4)
res["verdict"] = "HELD" if (res["rose"] > 0.05 and res["lift_fmid_d"] < 0.10) else "DROPPED"

rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
aa = rep.AnnotatorRegistry.get_annotator("rgb"); aa.attach(rp)
for _ in range(20): omni.kit.app.get_app().update()
Image.fromarray(np.asarray(aa.get_data())[..., :3].astype(np.uint8)).save("/tmp/newpick2_lift.png")
with open("/tmp/newpick2_res.json", "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
