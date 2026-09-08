"""Config-consistent staged pick: for each ranked candidate, STOMP to the GRASP pose
(one config) -> Cartesian retreat (grasp->pre) from that config; take the first candidate
whose Cartesian retreat fraction >= 0.9. advance = reversed retreat -> single arm config
through advance+close+retreat (no elbow flip). Then STOMP transit seed->pre with the
target obstacle present so the approach doesn't sweep the object."""
import json, numpy as np, rclpy, os, time
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import (RobotState, Constraints, MotionPlanRequest, WorkspaceParameters,
                             PositionConstraint, OrientationConstraint, BoundingVolume,
                             JointConstraint, CollisionObject, PlanningScene)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState

D = json.load(open("/home/bpdl/grasp/side_grasp.json"))
CANDS = D.get("candidates", [D])
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
SEED = [0.0, -0.9991, 0.0, 1.3986, 0.0, 0.0, 1.5491]

rclpy.init(); n = rclpy.create_node("step6")
ss = RobotState(); ss.joint_state = JointState(name=ARM, position=SEED)
ac = ActionClient(n, MoveGroup, "/move_action"); ac.wait_for_server(timeout_sec=10)
cart = n.create_client(GetCartesianPath, "/compute_cartesian_path"); cart.wait_for_service(timeout_sec=10)

WS = WorkspaceParameters(); WS.header.frame_id = "base_link"
WS.min_corner.x = WS.min_corner.y = WS.min_corner.z = -2.0
WS.max_corner.x = WS.max_corner.y = WS.max_corner.z = 2.0

# --- planning-scene collision object: present for the TRANSIT plan, absent for the
#     grasp-pose + Cartesian plans (which must reach the object). ---
_pub = n.create_publisher(PlanningScene, "/planning_scene", 10)
_scn = os.path.expanduser("~/grasp/scene_objects.json")
_obs = json.load(open(_scn)) if os.path.exists(_scn) else []

def _scene(op):
    if not _obs:
        return
    ps = PlanningScene(); ps.is_diff = True
    for ob in _obs:
        co = CollisionObject(); co.header.frame_id = "base_link"; co.id = ob["id"]
        pr = SolidPrimitive(); pr.type = SolidPrimitive.BOX; pr.dimensions = [float(x) for x in ob["dims"]]
        po = Pose(); po.position = Point(x=float(ob["xyz"][0]), y=float(ob["xyz"][1]), z=float(ob["xyz"][2]))
        po.orientation.w = 1.0
        co.primitives.append(pr); co.primitive_poses.append(po)
        co.operation = CollisionObject.ADD if op == "add" else CollisionObject.REMOVE
        ps.world.collision_objects.append(co)
    for _ in range(6):
        _pub.publish(ps); rclpy.spin_once(n, timeout_sec=0.2)
    time.sleep(0.6)
    print("[step6] scene %s %d objects" % (op, len(_obs)))

def mkpose(p, q):
    P = Pose(); P.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
    P.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
    return P

def stomp_to_pose(flange, q, attempts=4, tsec=6.0):
    g = MoveGroup.Goal(); r = MotionPlanRequest()
    r.group_name = "arm"; r.pipeline_id = "stomp"; r.num_planning_attempts = attempts
    r.allowed_planning_time = tsec; r.max_velocity_scaling_factor = 0.2
    r.max_acceleration_scaling_factor = 0.2; r.start_state = ss; r.workspace_parameters = WS
    c = Constraints()
    pc = PositionConstraint(); pc.header.frame_id = "base_link"; pc.link_name = "gripper_flange"
    bv = BoundingVolume(); sp = SolidPrimitive(); sp.type = SolidPrimitive.SPHERE; sp.dimensions = [0.005]
    bv.primitives.append(sp); bv.primitive_poses.append(mkpose(flange, q))
    pc.constraint_region = bv; pc.weight = 1.0
    oc = OrientationConstraint(); oc.header.frame_id = "base_link"; oc.link_name = "gripper_flange"
    oc.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
    oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.05
    oc.weight = 1.0
    c.position_constraints.append(pc); c.orientation_constraints.append(oc)
    r.goal_constraints.append(c); g.request = r; g.planning_options.plan_only = True
    f = ac.send_goal_async(g); rclpy.spin_until_future_complete(n, f, timeout_sec=15); gh = f.result()
    if gh is None:
        return None
    rf = gh.get_result_async(); rclpy.spin_until_future_complete(n, rf, timeout_sec=45)
    tr = rf.result().result.planned_trajectory.joint_trajectory
    return list(tr.points[-1].positions) if tr.points else None

def cart_retreat(grasp_cfg, flange, pre, q):
    cr = GetCartesianPath.Request(); cr.header.frame_id = "base_link"; cr.group_name = "arm"
    st = RobotState(); st.joint_state = JointState(name=ARM, position=grasp_cfg); cr.start_state = st
    cr.waypoints = [mkpose(flange + t * (pre - flange), q) for t in np.linspace(0.15, 1.0, 8)]
    cr.max_step = 0.004; cr.jump_threshold = 0.0; cr.avoid_collisions = True
    f = cart.call_async(cr); rclpy.spin_until_future_complete(n, f, timeout_sec=15); c2 = f.result()
    return c2.fraction, [list(p.positions) for p in c2.solution.joint_trajectory.points]

_scene("remove")
best = None                       # (fraction, dict)
for i, cd in enumerate(CANDS):
    q = cd["quat"]; a = np.array(cd["approach"]); flange = np.array(cd["flange"])
    pre = flange - cd["standoff"] * a
    # STOMP-to-pose + Cartesian is stochastic (pick_ik lands on different branches).
    # Retry a few times per candidate and keep the best fraction before moving on.
    cbest = None
    for attempt in range(2):
        gcfg = stomp_to_pose(flange, q)
        if gcfg is None:
            continue
        frac, retreat = cart_retreat(gcfg, flange, pre, q)
        print("[cand %d.%d %s] approach %s  grasp j1=%.2f  Cartesian fraction=%.2f"
              % (i, attempt, cd.get("twin", "?"), np.round(a, 2), gcfg[0], frac))
        rec = dict(cd=cd, q=q, a=a, flange=flange, pre=pre, grasp_cfg=gcfg,
                   retreat=retreat, pre_cfg=(retreat[-1] if retreat else gcfg))
        if cbest is None or frac > cbest[0]:
            cbest = (frac, rec)
        if frac >= 0.9:
            break
    if cbest is None:
        print("[cand %d] no STOMP to grasp pose" % i); continue
    if best is None or cbest[0] > best[0]:
        best = cbest
    if best[0] >= 0.9:
        break

if best is None:
    print("no candidate planned"); rclpy.shutdown(); raise SystemExit
frac, rec = best
q = rec["q"]; a = rec["a"]; flange = rec["flange"]; pre = rec["pre"]
grasp_cfg = rec["grasp_cfg"]; retreat = rec["retreat"]; pre_cfg = rec["pre_cfg"]
advance = retreat[::-1]
cd = rec["cd"]
print("[step6] picked approach %s  twin=%s  fraction %.2f" % (np.round(a, 2), cd.get("twin", "?"), frac))

# STOMP transit seed -> pre_cfg with the target obstacle present
_scene("add")
g2 = MoveGroup.Goal(); r2 = MotionPlanRequest()
r2.group_name = "arm"; r2.pipeline_id = "stomp"; r2.num_planning_attempts = 6; r2.allowed_planning_time = 12.0
r2.max_velocity_scaling_factor = 0.2; r2.max_acceleration_scaling_factor = 0.2
r2.start_state = ss; r2.workspace_parameters = WS
cc = Constraints()
for jn, jv in zip(ARM, pre_cfg):
    jc = JointConstraint(); jc.joint_name = jn; jc.position = float(jv)
    jc.tolerance_above = 0.01; jc.tolerance_below = 0.01; jc.weight = 1.0
    cc.joint_constraints.append(jc)
r2.goal_constraints.append(cc); g2.request = r2; g2.planning_options.plan_only = True
f = ac.send_goal_async(g2); rclpy.spin_until_future_complete(n, f, timeout_sec=15); gh2 = f.result()
rf2 = gh2.get_result_async(); rclpy.spin_until_future_complete(n, rf2, timeout_sec=45)
tr2 = rf2.result().result.planned_trajectory.joint_trajectory
transit = [list(p.positions) for p in tr2.points]
print("STOMP transit seed->pre: wp=%d" % len(transit))

json.dump(dict(quat=q, approach=a.tolist(), contact=cd["contact"], flange=flange.tolist(),
               transit=transit, advance=advance, retreat=retreat,
               grasp_cfg=grasp_cfg, pre_cfg=pre_cfg, opening=cd["opening"],
               radius=cd.get("radius", 0.025), cartesian_fraction=frac),
          open("/home/bpdl/grasp/step6_pick.json", "w"), indent=1)
print("-> step6_pick.json  transit %d + advance %d + retreat %d" % (len(transit), len(advance), len(retreat)))
rclpy.shutdown()
