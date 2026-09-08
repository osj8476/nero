"""Config-consistent staged pick: STOMP to GRASP pose (one config) -> Cartesian retreat
(grasp->pre) from that config -> advance = reversed retreat. Guarantees a single arm config
through advance+close+retreat (no elbow flip)."""
import json, numpy as np, rclpy
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import (RobotState, Constraints, MotionPlanRequest, WorkspaceParameters,
                             PositionConstraint, OrientationConstraint, BoundingVolume)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from sensor_msgs.msg import JointState

D = json.load(open("/home/bpdl/grasp/side_grasp.json"))
ARM = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED = [0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]
q = D["quat"]; a = np.array(D["approach"]); flange = np.array(D["flange"]); SO = D["standoff"]
pre = flange - SO * a

def mkpose(p):
    P = Pose(); P.position = Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
    P.orientation = Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3]))
    return P

rclpy.init(); n = rclpy.create_node("step6")
ss = RobotState(); ss.joint_state = JointState(name=ARM, position=SEED)
ac = ActionClient(n, MoveGroup, "/move_action"); ac.wait_for_server(timeout_sec=10)

# --- planning-scene collision objects so STOMP transit avoids the target + obstacles ---
import os, time
from moveit_msgs.msg import CollisionObject, PlanningScene
_pub = n.create_publisher(PlanningScene, "/planning_scene", 10)
_scn = os.path.expanduser("~/grasp/scene_objects.json")
if os.path.exists(_scn):
    ps = PlanningScene(); ps.is_diff = True
    for ob in json.load(open(_scn)):
        co = CollisionObject(); co.header.frame_id = "base_link"; co.id = ob["id"]
        pr = SolidPrimitive(); pr.type = SolidPrimitive.BOX; pr.dimensions = [float(x) for x in ob["dims"]]
        po = Pose(); po.position = Point(x=float(ob["xyz"][0]), y=float(ob["xyz"][1]), z=float(ob["xyz"][2]))
        po.orientation.w = 1.0
        co.primitives.append(pr); co.primitive_poses.append(po); co.operation = CollisionObject.ADD
        ps.world.collision_objects.append(co)
    for _ in range(6):
        _pub.publish(ps); rclpy.spin_once(n, timeout_sec=0.2)
    time.sleep(0.6)
    print("[step6] +%d collision objects" % len(ps.world.collision_objects))

# STOMP to the GRASP *pose* (pick_ik picks the config, STOMP plans the path)
g = MoveGroup.Goal(); r = MotionPlanRequest()
r.group_name = "arm"; r.pipeline_id = "stomp"; r.num_planning_attempts = 8; r.allowed_planning_time = 15.0
r.max_velocity_scaling_factor = 0.2; r.max_acceleration_scaling_factor = 0.2; r.start_state = ss
r.workspace_parameters = WorkspaceParameters(); r.workspace_parameters.header.frame_id = "base_link"
r.workspace_parameters.min_corner.x = -2.; r.workspace_parameters.min_corner.y = -2.; r.workspace_parameters.min_corner.z = -2.
r.workspace_parameters.max_corner.x = 2.; r.workspace_parameters.max_corner.y = 2.; r.workspace_parameters.max_corner.z = 2.
c = Constraints()
pc = PositionConstraint(); pc.header.frame_id = "base_link"; pc.link_name = "gripper_flange"
bv = BoundingVolume(); sp = SolidPrimitive(); sp.type = SolidPrimitive.SPHERE; sp.dimensions = [0.005]
bv.primitives.append(sp); bv.primitive_poses.append(mkpose(flange))
pc.constraint_region = bv; pc.weight = 1.0
oc = OrientationConstraint(); oc.header.frame_id = "base_link"; oc.link_name = "gripper_flange"
oc.orientation = Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3]))
oc.absolute_x_axis_tolerance = 0.05; oc.absolute_y_axis_tolerance = 0.05; oc.absolute_z_axis_tolerance = 0.05
oc.weight = 1.0
c.position_constraints.append(pc); c.orientation_constraints.append(oc)
r.goal_constraints.append(c); g.request = r; g.planning_options.plan_only = True
f = ac.send_goal_async(g); rclpy.spin_until_future_complete(n, f, timeout_sec=15); gh = f.result()
rf = gh.get_result_async(); rclpy.spin_until_future_complete(n, rf, timeout_sec=45)
res = rf.result().result; tr = res.planned_trajectory.joint_trajectory
print("STOMP seed->GRASP pose: code=%d wp=%d" % (res.error_code.val, len(tr.points)))
if not tr.points:
    print("no plan"); rclpy.shutdown(); raise SystemExit
grasp_cfg = list(tr.points[-1].positions)
print("  grasp config:", [round(x,3) for x in grasp_cfg])

# Cartesian RETREAT from grasp config: grasp -> pre (straight along -a)
cart = n.create_client(GetCartesianPath, "/compute_cartesian_path"); cart.wait_for_service(timeout_sec=10)
cr = GetCartesianPath.Request(); cr.header.frame_id = "base_link"; cr.group_name = "arm"
st = RobotState(); st.joint_state = JointState(name=ARM, position=grasp_cfg); cr.start_state = st
cr.waypoints = [mkpose(flange + t*(pre-flange)) for t in np.linspace(0.15, 1.0, 8)]
cr.max_step = 0.004; cr.jump_threshold = 0.0; cr.avoid_collisions = True
f = cart.call_async(cr); rclpy.spin_until_future_complete(n, f, timeout_sec=15); c2 = f.result()
print("Cartesian retreat (grasp->pre): fraction=%.2f wp=%d" % (c2.fraction, len(c2.solution.joint_trajectory.points)))
retreat = [list(p.positions) for p in c2.solution.joint_trajectory.points]
pre_cfg = retreat[-1] if retreat else grasp_cfg
advance = retreat[::-1]   # pre -> grasp

# STOMP transit seed -> pre_cfg (so the run doesn't jump)
g2 = MoveGroup.Goal(); r2 = MotionPlanRequest()
r2.group_name = "arm"; r2.pipeline_id = "stomp"; r2.num_planning_attempts = 6; r2.allowed_planning_time = 12.0
r2.max_velocity_scaling_factor = 0.2; r2.max_acceleration_scaling_factor = 0.2; r2.start_state = ss
r2.workspace_parameters = r.workspace_parameters
from moveit_msgs.msg import JointConstraint
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

json.dump(dict(quat=q, approach=a.tolist(), contact=D["contact"], flange=flange.tolist(),
               transit=transit, advance=advance, retreat=retreat,
               grasp_cfg=grasp_cfg, pre_cfg=pre_cfg, opening=D["opening"], radius=D.get("radius", 0.025)),
          open("/home/bpdl/grasp/step6_pick.json", "w"), indent=1)
print("-> step6_pick.json  transit %d + advance %d + retreat %d" % (len(transit), len(advance), len(retreat)))
rclpy.shutdown()
