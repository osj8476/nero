import json, numpy as np, rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
ARM=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED=[0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]
D=0.1358
contact=np.array([-0.5,0.0,0.11])
def R2q(R):
    t=np.trace(R)
    if t>0: s=np.sqrt(t+1)*2;w=.25*s;x=(R[2,1]-R[1,2])/s;y=(R[0,2]-R[2,0])/s;z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2;w=(R[2,1]-R[1,2])/s;x=.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]: s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2;w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=.25*s;z=(R[1,2]+R[2,1])/s
    else: s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2;w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=.25*s
    v=np.array([x,y,z,w]);return v/np.linalg.norm(v)
rclpy.init(); n=rclpy.create_node("side_ik2")
ik=n.create_client(GetPositionIK,"/compute_ik"); ik.wait_for_service(timeout_sec=10)
ss=RobotState(); ss.joint_state=JointState(name=ARM,position=SEED)
def solve(pos,q):
    req=GetPositionIK.Request(); req.ik_request.group_name="arm"; req.ik_request.robot_state=ss
    ps=PoseStamped(); ps.header.frame_id="base_link"
    ps.pose.position=Point(x=float(pos[0]),y=float(pos[1]),z=float(pos[2]))
    ps.pose.orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3]))
    req.ik_request.pose_stamped=ps; req.ik_request.timeout.sec=1; req.ik_request.avoid_collisions=True
    f=ik.call_async(req); rclpy.spin_until_future_complete(n,f,timeout_sec=6)
    r=f.result(); c=r.error_code.val if r else None
    if c==1:
        js=r.solution.joint_state; return [round(js.position[js.name.index(j)],4) for j in ARM]
    return None
best=None
for name,a in [("from +x",np.array([-1,0,0.])),("from -y",np.array([0,1,0.])),("from +y",np.array([0,-1,0.])),
               ("from +x-y",np.array([-.7,.7,0.])),("from +x+y",np.array([-.7,-.7,0.]))]:
    a=a/np.linalg.norm(a)
    # closing axis b: horizontal, perp to a and to world-z (fingers close across the cylinder diameter)
    b=np.cross(np.array([0,0,1.]),a); b/=np.linalg.norm(b)
    R=np.column_stack([b,np.cross(a,b),a]); q=R2q(R)
    flange=contact-D*a
    for so in [0.06,0.09]:
        pre=flange-so*a
        gs=solve(flange,q); psol=solve(pre,q)
        ok = gs is not None and psol is not None
        print(f"  {name:9s} standoff {so}: grasp {'OK' if gs else 'X'}  pre {'OK' if psol else 'X'}")
        if ok and best is None:
            best=dict(name=name,quat=q.tolist(),approach=a.tolist(),flange=flange.tolist(),pre=pre.tolist(),
                      contact=contact.tolist(),standoff=so,grasp_sol=gs,pre_sol=psol)
if best:
    json.dump(best,open("/home/bpdl/grasp/side_grasp.json","w"),indent=1)
    print(f"\n-> side_grasp.json  ({best['name']}, standoff {best['standoff']})")
    print("  grasp_sol",best['grasp_sol'])
rclpy.shutdown()
