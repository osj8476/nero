"""Phase 3a+3c v2: twin + theta^4 + pre-grasp reachable + box-axis alignment."""
import json, sys, numpy as np, rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

CGN=sys.argv[1]; W_THETA=0.5; W_ALIGN=0.15; STANDOFF=0.05
ARM=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED=[0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]
def q2R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def R2q(R):
    t=np.trace(R)
    if t>0: s=np.sqrt(t+1)*2;w=.25*s;x=(R[2,1]-R[1,2])/s;y=(R[0,2]-R[2,0])/s;z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2;w=(R[2,1]-R[1,2])/s;x=.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]: s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2;w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=.25*s;z=(R[1,2]+R[2,1])/s
    else: s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2;w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=.25*s
    v=np.array([x,y,z,w]);return v/np.linalg.norm(v)

data=json.load(open(CGN)); box=np.array(data["box_world"])
cand=[]
for g in data["grasps"]:
    R=q2R(g["quat_xyzw"]); pos=np.array(g["position_xyz"])
    for tw in (False,True):
        Rr=R@np.diag([-1,-1,1.]) if tw else R
        b,a=Rr[:,0],Rr[:,2]
        theta=np.arccos(np.clip(-a[2],-1,1))
        # box axis alignment: closing axis b should align w/ world X or Y (box is axis-aligned). penalty = 1 - max(|b.x|,|b.y|) projected horizontally
        bh=b.copy(); bh[2]=0; bh=bh/(np.linalg.norm(bh)+1e-9)
        align=max(abs(bh[0]),abs(bh[1]))          # 1.0 = aligned to an axis, 0.707 = 45deg diagonal
        cand.append(dict(score=g["score"],theta=float(theta),twin=tw,align=float(align),
                         pos=pos.tolist(),quat=R2q(Rr).tolist(),approach=a.tolist(),opening=g["opening_m"]))
rclpy.init(); n=rclpy.create_node("rank2")
ik=n.create_client(GetPositionIK,"/compute_ik"); ik.wait_for_service(timeout_sec=10)
ss=RobotState(); ss.joint_state=JointState(name=ARM,position=SEED)
def ck(p,q):
    req=GetPositionIK.Request(); req.ik_request.group_name="arm"; req.ik_request.robot_state=ss
    ps=PoseStamped(); ps.header.frame_id="base_link"
    ps.pose.position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
    ps.pose.orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3]))
    req.ik_request.pose_stamped=ps; req.ik_request.timeout.sec=1; req.ik_request.avoid_collisions=True
    f=ik.call_async(req); rclpy.spin_until_future_complete(n,f,timeout_sec=6); r=f.result()
    if r and r.error_code.val==1:
        js=r.solution.joint_state; return [js.position[js.name.index(j)] for j in ARM]
    return None
good=[]
for c in cand:
    g_sol=ck(c["pos"],c["quat"])
    if g_sol is None: continue
    pre=np.array(c["pos"])-STANDOFF*np.array(c["approach"])
    p_sol=ck(pre,c["quat"])
    if p_sol is None: continue
    c["grasp_sol"]=g_sol; c["pre_sol"]=p_sol; c["pre"]=pre.tolist()
    c["cost"]=c["score"] - W_THETA*c["theta"]**4 - W_ALIGN*(1-c["align"])
    good.append(c)
print(f"{len(cand)} cand -> {len(good)} with BOTH grasp+pre-grasp reachable")
good.sort(key=lambda c:-c["cost"])
for c in good[:6]:
    tip=np.array(c["pos"])+0.1358*np.array(c["approach"])
    print(f"  s={c['score']:.3f} th={np.degrees(c['theta']):.0f} align={c['align']:.2f} twin={c['twin']} cost={c['cost']:.3f} "
          f"tip={np.round(tip,3)} dxy={np.hypot(tip[0]-box[0],tip[1]-box[1]):.3f} open={c['opening']*100:.1f}")
if good:
    json.dump(dict(best=good[0],box_world=box.tolist(),candidates_reachable=len(good)),
              open("/home/bpdl/grasp/rank2_best.json","w"),indent=1)
    print("\n-> rank2_best.json")
rclpy.shutdown()
