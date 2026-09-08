import json, numpy as np, rclpy
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import RobotState, Constraints, MotionPlanRequest, WorkspaceParameters, JointConstraint
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState

D=json.load(open("/home/bpdl/grasp/rank2_best.json")); b=D["best"]; box=np.array(D["box_world"])
ARM=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED=[0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]
pos=np.array(b["pos"]); a=np.array(b["approach"]); q=b["quat"]
STANDOFF=0.05; pre=pos-STANDOFF*a
def mkpose(p):
    P=Pose(); P.position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
    P.orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3])); return P
rclpy.init(); n=rclpy.create_node("s4b")
ss=RobotState(); ss.joint_state=JointState(name=ARM,position=SEED)
# STOMP seed -> pre-grasp (joint goal from rank2)
ac=ActionClient(n,MoveGroup,"/move_action"); ac.wait_for_server(timeout_sec=10)
g=MoveGroup.Goal(); mp=MotionPlanRequest()
mp.group_name="arm"; mp.pipeline_id="stomp"; mp.num_planning_attempts=6; mp.allowed_planning_time=12.0
mp.max_velocity_scaling_factor=0.25; mp.max_acceleration_scaling_factor=0.25; mp.start_state=ss
mp.workspace_parameters=WorkspaceParameters(); mp.workspace_parameters.header.frame_id="base_link"
for a_,v in [("min",-2.),("max",2.)]:
    setattr(getattr(mp.workspace_parameters,a_+"_corner"),"x",v); setattr(getattr(mp.workspace_parameters,a_+"_corner"),"y",v); setattr(getattr(mp.workspace_parameters,a_+"_corner"),"z",v)
cc=Constraints()
for jn,jv in zip(ARM,b["pre_sol"]):
    jc=JointConstraint(); jc.joint_name=jn; jc.position=jv; jc.tolerance_above=0.01; jc.tolerance_below=0.01; jc.weight=1.
    cc.joint_constraints.append(jc)
mp.goal_constraints.append(cc); g.request=mp; g.planning_options.plan_only=True
f=ac.send_goal_async(g); rclpy.spin_until_future_complete(n,f,timeout_sec=15); gh=f.result()
rf=gh.get_result_async(); rclpy.spin_until_future_complete(n,rf,timeout_sec=40)
res=rf.result().result; tr1=res.planned_trajectory.joint_trajectory
print(f"STOMP->pre: code={res.error_code.val} wp={len(tr1.points)}")
# Cartesian pre -> grasp
cart=n.create_client(GetCartesianPath,"/compute_cartesian_path"); cart.wait_for_service(timeout_sec=10)
cr=GetCartesianPath.Request(); cr.header.frame_id="base_link"; cr.group_name="arm"
st=RobotState(); st.joint_state=JointState(name=ARM,position=b["pre_sol"]); cr.start_state=st
cr.waypoints=[mkpose(pre+t*(pos-pre)) for t in np.linspace(0.2,1.0,8)]
cr.max_step=0.004; cr.jump_threshold=0.0; cr.avoid_collisions=False
f=cart.call_async(cr); rclpy.spin_until_future_complete(n,f,timeout_sec=15); c=f.result()
print(f"Cartesian: fraction={c.fraction:.2f} wp={len(c.solution.joint_trajectory.points)}")
tr2=c.solution.joint_trajectory
if tr1.points and c.fraction>0.85:
    pts=[{"positions":list(p.positions),"time":p.time_from_start.sec+p.time_from_start.nanosec*1e-9} for p in tr1.points]
    off=pts[-1]["time"]+0.4
    for p in tr2.points:
        pts.append({"positions":list(p.positions),"time":off+p.time_from_start.sec+p.time_from_start.nanosec*1e-9})
    json.dump(dict(joint_names=list(tr1.joint_names),points=pts,best=b,box_world=box.tolist(),
                   pre_sol=b["pre_sol"],grasp_sol=list(tr2.points[-1].positions),
                   fingertip=(pos+0.1358*a).tolist(),opening=b["opening"],
                   n_stomp=len(tr1.points),n_cart=len(tr2.points)),
              open("/home/bpdl/grasp/c4_trajectory.json","w"),indent=1)
    print(f"-> c4_trajectory.json  {len(pts)} pts (STOMP {len(tr1.points)} + Cart {len(tr2.points)})")
    print(f"   grasp joints: {[round(x,3) for x in tr2.points[-1].positions]}  fingertip {np.round(pos+0.1358*a,3)} box {np.round(box,3)}")
rclpy.shutdown()
