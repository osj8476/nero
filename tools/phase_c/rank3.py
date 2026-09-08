"""Phase 3c fast ranking: geometric prefilter -> score - wθ·θ⁴ - wcom·d_com -> fast IK on top-K.
Cuts 400 IK calls -> ~25. Times each phase."""
import json, sys, time, numpy as np, rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

CGN = sys.argv[1]
W_THETA = 0.5; W_COM = 1.2; K_IK = 24; STANDOFF = 0.06
THETA_MAX = np.radians(35)          # prefilter: drop grasps > 35deg off vertical
WS = dict(x=(-0.62, -0.32), y=(-0.20, 0.20), z=(0.05, 0.45))   # coarse reachable box
ARM = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED = [0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]
D_TCP = 0.1358

def q2R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def R2q(R):
    t=np.trace(R)
    if t>0: s=np.sqrt(t+1)*2;w=.25*s;x=(R[2,1]-R[1,2])/s;y=(R[0,2]-R[2,0])/s;z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2;w=(R[2,1]-R[1,2])/s;x=.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]: s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2;w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=.25*s;z=(R[1,2]+R[2,1])/s
    else: s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2;w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=.25*s
    v=np.array([x,y,z,w]);return v/np.linalg.norm(v)

t0=time.time()
data=json.load(open(CGN)); obj=np.array(data["box_world"])
com = obj.copy()                    # object centroid proxy (world). refine w/ seg pc if available.
raw=data["grasps"]

# ---- geometric prefilter + cost (NO IK) ----
cand=[]
for g in raw:
    R=q2R(g["quat_xyzw"]); pos=np.array(g["position_xyz"]); w=g["opening_m"]
    if not (0.005 < w <= 0.102): continue
    for twin in (False, True):
        Rr = R@np.diag([-1,-1,1.]) if twin else R
        a = Rr[:,2]
        theta = np.arccos(np.clip(-a[2], -1, 1))
        if theta > THETA_MAX: continue
        contact = pos + D_TCP*a
        if not (WS["x"][0]<contact[0]<WS["x"][1] and WS["y"][0]<contact[1]<WS["y"][1] and WS["z"][0]<contact[2]<WS["z"][1]):
            continue
        d_com = np.linalg.norm(contact - com)
        cost = g["score"] - W_THETA*theta**4 - W_COM*d_com
        cand.append(dict(score=g["score"], theta=float(theta), twin=twin, d_com=float(d_com),
                         pos=pos.tolist(), quat=R2q(Rr).tolist(), approach=a.tolist(),
                         contact=contact.tolist(), opening=w, cost=float(cost)))
cand.sort(key=lambda c:-c["cost"])
t1=time.time()
print(f"[prefilter] {len(raw)} grasps x2 -> {len(cand)} pass geometry  ({1e3*(t1-t0):.0f} ms)")

# ---- fast IK on top-K ----
rclpy.init(); n=rclpy.create_node("rank3")
ik=n.create_client(GetPositionIK,"/compute_ik"); ik.wait_for_service(timeout_sec=10)
ss=RobotState(); ss.joint_state=JointState(name=ARM,position=SEED)
def fast_ik(pos,quat):
    req=GetPositionIK.Request(); req.ik_request.group_name="arm"; req.ik_request.robot_state=ss
    ps=PoseStamped(); ps.header.frame_id="base_link"
    ps.pose.position=Point(x=float(pos[0]),y=float(pos[1]),z=float(pos[2]))
    ps.pose.orientation=Quaternion(x=float(quat[0]),y=float(quat[1]),z=float(quat[2]),w=float(quat[3]))
    req.ik_request.pose_stamped=ps
    req.ik_request.timeout.sec=0; req.ik_request.timeout.nanosec=120_000_000
    req.ik_request.avoid_collisions=True
    f=ik.call_async(req); rclpy.spin_until_future_complete(n,f,timeout_sec=1.5)
    r=f.result()
    if r and r.error_code.val==1:
        js=r.solution.joint_state; return [js.position[js.name.index(j)] for j in ARM]
    return None

picked=None; nik=0
for c in cand[:K_IK]:
    nik+=1
    g_sol=fast_ik(c["pos"], c["quat"])
    if g_sol is None: continue
    pre = np.array(c["pos"]) - STANDOFF*np.array(c["approach"])
    nik+=1
    p_sol=fast_ik(pre, c["quat"])
    if p_sol is None: continue
    c["grasp_sol"]=g_sol; c["pre_sol"]=p_sol; c["pre"]=pre.tolist()
    picked=c; break
t2=time.time()
print(f"[ik] {nik} calls -> {'PICKED' if picked else 'none'}  ({1e3*(t2-t1):.0f} ms)")
if picked:
    p=picked
    print(f"  score={p['score']:.3f} theta={np.degrees(p['theta']):.0f}deg d_com={p['d_com']*100:.1f}cm twin={p['twin']}")
    tip=np.array(p['pos'])+D_TCP*np.array(p['approach'])
    print(f"  fingertip={np.round(tip,3)}  object={np.round(obj,3)}  d_xy={np.hypot(tip[0]-obj[0],tip[1]-obj[1])*100:.1f}cm")
    json.dump(dict(best=p, box_world=obj.tolist(),
                   timing_ms=dict(prefilter=round(1e3*(t1-t0)), ik=round(1e3*(t2-t1)), n_ik=nik)),
              open("/home/bpdl/grasp/rank3_best.json","w"), indent=1)
print(f"[total] {1e3*(t2-t0):.0f} ms")
rclpy.shutdown()
