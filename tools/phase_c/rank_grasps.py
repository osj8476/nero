"""Phase 3a+3c: CGN grasp + 180deg twin, filter by pick_ik, rank by score - w*theta^4."""
import json, sys, numpy as np, rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import RobotState, Constraints, MotionPlanRequest, WorkspaceParameters, JointConstraint
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

CGN = sys.argv[1] if len(sys.argv)>1 else "/home/bpdl/grasp/s5_grasps_all.json"
W_THETA = float(sys.argv[2]) if len(sys.argv)>2 else 0.6
ARM=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED=[0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]

def q_to_R(q):  # xyzw
    x,y,z,w=q
    return np.array([
      [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
      [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
      [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
def R_to_q(R):
    t=np.trace(R)
    if t>0: s=np.sqrt(t+1)*2;w=.25*s;x=(R[2,1]-R[1,2])/s;y=(R[0,2]-R[2,0])/s;z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2;w=(R[2,1]-R[1,2])/s;x=.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]: s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2;w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=.25*s;z=(R[1,2]+R[2,1])/s
    else: s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2;w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=.25*s
    v=np.array([x,y,z,w]);return v/np.linalg.norm(v)

data=json.load(open(CGN)); box=np.array(data["box_world"])
cand=[]
for g in data["grasps"]:
    R=q_to_R(g["quat_xyzw"]); pos=np.array(g["position_xyz"])
    for twin in (False,True):
        Rr = R@np.diag([-1,-1,1.]) if twin else R   # 180 about approach(z col)
        a=Rr[:,2]
        theta=np.arccos(np.clip(-a[2],-1,1))        # angle of approach from straight-down
        cand.append(dict(score=g["score"],theta=float(theta),twin=twin,
                         pos=pos.tolist(),quat=R_to_q(Rr).tolist(),approach=a.tolist(),
                         opening=g["opening_m"]))

rclpy.init(); n=rclpy.create_node("rank")
ik=n.create_client(GetPositionIK,"/compute_ik"); ik.wait_for_service(timeout_sec=10)
ss=RobotState(); ss.joint_state=JointState(name=ARM,position=SEED)
def check(c):
    req=GetPositionIK.Request(); req.ik_request.group_name="arm"; req.ik_request.robot_state=ss
    ps=PoseStamped(); ps.header.frame_id="base_link"
    ps.pose.position=Point(x=c["pos"][0],y=c["pos"][1],z=c["pos"][2])
    q=c["quat"]; ps.pose.orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3])
    req.ik_request.pose_stamped=ps; req.ik_request.timeout.sec=1; req.ik_request.avoid_collisions=True
    f=ik.call_async(req); rclpy.spin_until_future_complete(n,f,timeout_sec=6)
    r=f.result()
    if r and r.error_code.val==1:
        js=r.solution.joint_state
        return [js.position[js.name.index(j)] for j in ARM]
    return None

reach=[]
for c in cand:
    sol=check(c)
    if sol is not None:
        c["sol"]=sol; c["rank_cost"]=c["score"] - W_THETA*c["theta"]**4
        reach.append(c)
print(f"candidates: {len(cand)} ({len(data['grasps'])} grasps x2 twin)  reachable: {len(reach)}")
reach.sort(key=lambda c:-c["rank_cost"])
for c in reach[:6]:
    tip=np.array(c["pos"])+0.1358*np.array(c["approach"])
    print(f"  s={c['score']:.3f} theta={np.degrees(c['theta']):.0f}deg twin={c['twin']} cost={c['rank_cost']:.3f}  "
          f"tip={np.round(tip,3)} d_box_xy={np.hypot(tip[0]-box[0],tip[1]-box[1]):.3f} open={c['opening']*100:.1f}cm")

if reach:
    best=reach[0]
    ac=ActionClient(n,MoveGroup,"/move_action"); ac.wait_for_server(timeout_sec=10)
    goal=MoveGroup.Goal(); r=MotionPlanRequest()
    r.group_name="arm"; r.pipeline_id="stomp"; r.num_planning_attempts=4; r.allowed_planning_time=10.0
    r.max_velocity_scaling_factor=0.3; r.max_acceleration_scaling_factor=0.3; r.start_state=ss
    r.workspace_parameters=WorkspaceParameters(); r.workspace_parameters.header.frame_id="base_link"
    r.workspace_parameters.min_corner.x=-2.;r.workspace_parameters.min_corner.y=-2.;r.workspace_parameters.min_corner.z=-2.
    r.workspace_parameters.max_corner.x=2.;r.workspace_parameters.max_corner.y=2.;r.workspace_parameters.max_corner.z=2.
    c=Constraints()
    for jn,jv in zip(ARM,best["sol"]):
        jc=JointConstraint(); jc.joint_name=jn; jc.position=jv; jc.tolerance_above=0.01; jc.tolerance_below=0.01; jc.weight=1.
        c.joint_constraints.append(jc)
    r.goal_constraints.append(c); goal.request=r; goal.planning_options.plan_only=True
    f=ac.send_goal_async(goal); rclpy.spin_until_future_complete(n,f,timeout_sec=15); gh=f.result()
    rf=gh.get_result_async(); rclpy.spin_until_future_complete(n,rf,timeout_sec=30)
    res=rf.result().result; tr=res.planned_trajectory.joint_trajectory
    print(f"\nSTOMP best: code={res.error_code.val} wp={len(tr.points)}")
    if tr.points:
        json.dump(dict(joint_names=list(tr.joint_names),
                       points=[dict(positions=list(p.positions),time=p.time_from_start.sec+p.time_from_start.nanosec*1e-9) for p in tr.points],
                       best=best, box_world=box.tolist(),
                       fingertip=(np.array(best["pos"])+0.1358*np.array(best["approach"])).tolist()),
                  open("/home/bpdl/grasp/c3_trajectory.json","w"),indent=1)
        print("  end joints:",[round(x,3) for x in tr.points[-1].positions])
rclpy.shutdown()
