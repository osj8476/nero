"""Staged pick: transit -> approach(advance along +a) -> [close] -> retreat(along -a) -> [lift].
Cartesian for advance/retreat keeps one arm config (no elbow flip)."""
import json, numpy as np, rclpy
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import RobotState, Constraints, MotionPlanRequest, WorkspaceParameters, JointConstraint
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState

D=json.load(open("/home/bpdl/grasp/side_grasp.json"))
ARM=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED=[0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]
q=D["quat"]; a=np.array(D["approach"]); flange=np.array(D["flange"]); SO=D["standoff"]
pre=flange - SO*a
def mkpose(p):
    P=Pose(); P.position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
    P.orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3])); return P

rclpy.init(); n=rclpy.create_node("step5")
ss=RobotState(); ss.joint_state=JointState(name=ARM,position=SEED)

# 1. STOMP seed -> grasp (pose goal via joint goal from provided IK)
ac=ActionClient(n,MoveGroup,"/move_action"); ac.wait_for_server(timeout_sec=10)
def stomp_to(jsol):
    g=MoveGroup.Goal(); r=MotionPlanRequest()
    r.group_name="arm"; r.pipeline_id="stomp"; r.num_planning_attempts=6; r.allowed_planning_time=12.0
    r.max_velocity_scaling_factor=0.25; r.max_acceleration_scaling_factor=0.25; r.start_state=ss
    r.workspace_parameters=WorkspaceParameters(); r.workspace_parameters.header.frame_id="base_link"
    r.workspace_parameters.min_corner.x=-2.;r.workspace_parameters.min_corner.y=-2.;r.workspace_parameters.min_corner.z=-2.
    r.workspace_parameters.max_corner.x=2.;r.workspace_parameters.max_corner.y=2.;r.workspace_parameters.max_corner.z=2.
    c=Constraints()
    for jn,jv in zip(ARM,jsol):
        jc=JointConstraint(); jc.joint_name=jn; jc.position=float(jv); jc.tolerance_above=0.01; jc.tolerance_below=0.01; jc.weight=1.
        c.joint_constraints.append(jc)
    r.goal_constraints.append(c); g.request=r; g.planning_options.plan_only=True
    f=ac.send_goal_async(g); rclpy.spin_until_future_complete(n,f,timeout_sec=15); gh=f.result()
    rf=gh.get_result_async(); rclpy.spin_until_future_complete(n,rf,timeout_sec=40)
    res=rf.result().result; return res.error_code.val, res.planned_trajectory.joint_trajectory

# transit to PRE-grasp config
code1,tr_transit = stomp_to(D["pre_sol"])
print(f"STOMP seed->pre: code={code1} wp={len(tr_transit.points)}")

# 2. Cartesian pre -> grasp  (advance, +a).  start from pre_sol
cart=n.create_client(GetCartesianPath,"/compute_cartesian_path"); cart.wait_for_service(timeout_sec=10)
def cartesian(start_sol, p0, p1, tag):
    cr=GetCartesianPath.Request(); cr.header.frame_id="base_link"; cr.group_name="arm"
    st=RobotState(); st.joint_state=JointState(name=ARM,position=[float(x) for x in start_sol]); cr.start_state=st
    cr.waypoints=[mkpose(p0 + t*(p1-p0)) for t in np.linspace(0.15,1.0,8)]
    cr.max_step=0.004; cr.jump_threshold=0.0; cr.avoid_collisions=True
    f=cart.call_async(cr); rclpy.spin_until_future_complete(n,f,timeout_sec=15); r=f.result()
    print(f"Cartesian {tag}: fraction={r.fraction:.2f} wp={len(r.solution.joint_trajectory.points)}")
    return r.fraction, r.solution.joint_trajectory

fa, tr_adv = cartesian(D["pre_sol"], pre, flange, "advance (pre->grasp)")
grasp_sol_c = list(tr_adv.points[-1].positions) if tr_adv.points else D["grasp_sol"]
# 3. Cartesian grasp -> pre  (retreat, -a). start from end-of-advance config
fr, tr_ret = cartesian(grasp_sol_c, flange, pre, "retreat (grasp->pre)")

out=dict(quat=q, approach=a.tolist(), contact=D["contact"], standoff=SO,
         transit=[{"positions":list(p.positions),"time":p.time_from_start.sec+p.time_from_start.nanosec*1e-9} for p in tr_transit.points],
         advance=[{"positions":list(p.positions),"time":p.time_from_start.sec+p.time_from_start.nanosec*1e-9} for p in tr_adv.points],
         retreat=[{"positions":list(p.positions),"time":p.time_from_start.sec+p.time_from_start.nanosec*1e-9} for p in tr_ret.points],
         grasp_sol=grasp_sol_c, pre_sol=D["pre_sol"])
json.dump(out, open("/home/bpdl/grasp/step5_pick.json","w"), indent=1)
print(f"-> step5_pick.json  transit {len(out['transit'])} + advance {len(out['advance'])} + retreat {len(out['retreat'])} wp")
rclpy.shutdown()
