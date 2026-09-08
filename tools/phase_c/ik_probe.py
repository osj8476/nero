import json, numpy as np, rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

ARM=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED=[0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]

def quat_from_R(R):
    t=np.trace(R)
    if t>0: s=np.sqrt(t+1)*2;w=.25*s;x=(R[2,1]-R[1,2])/s;y=(R[0,2]-R[2,0])/s;z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2;w=(R[2,1]-R[1,2])/s;x=.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]: s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2;w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=.25*s;z=(R[1,2]+R[2,1])/s
    else: s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2;w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=.25*s
    v=np.array([x,y,z,w]);return v/np.linalg.norm(v)

# top-down: gripper approach axis (assume flange +Z is approach) points -Z world.
# try approach = flange local Z -> world -Z, with yaw sweep about vertical
rclpy.init(); n=rclpy.create_node("ik_probe")
ik=n.create_client(GetPositionIK,"/compute_ik"); ik.wait_for_service(timeout_sec=10)
ss=RobotState(); ss.joint_state=JointState(name=ARM,position=SEED)

def try_pose(pos,quat,tag):
    req=GetPositionIK.Request(); req.ik_request.group_name="arm"; req.ik_request.robot_state=ss
    ps=PoseStamped(); ps.header.frame_id="base_link"
    ps.pose.position=Point(x=float(pos[0]),y=float(pos[1]),z=float(pos[2]))
    ps.pose.orientation=Quaternion(x=float(quat[0]),y=float(quat[1]),z=float(quat[2]),w=float(quat[3]))
    req.ik_request.pose_stamped=ps; req.ik_request.timeout.sec=1; req.ik_request.avoid_collisions=True
    f=ik.call_async(req); rclpy.spin_until_future_complete(n,f,timeout_sec=6)
    r=f.result(); code=r.error_code.val if r else None
    print(f"  {tag:40s} IK={code}")
    return code==1

P=[-0.5,0.0,0.20]
# flange approach axis unknown -> try approach along each of flange local +Z, -Z, +X etc, mapped to world -Z (down)
# build R so that chosen local axis -> world down (0,0,-1)
for local_axis,name in [((0,0,1),"+Zdown"),((0,0,-1),"-Zdown"),((1,0,0),"+Xdown"),((0,1,0),"+Ydown")]:
    la=np.array(local_axis,float); down=np.array([0,0,-1.0])
    # rotation aligning la->down, minimal
    v=np.cross(la,down); s=np.linalg.norm(v); c=np.dot(la,down)
    if s<1e-8: R=np.eye(3) if c>0 else np.diag([1,-1,-1.])
    else:
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R=np.eye(3)+vx+vx@vx*((1-c)/s**2)
    for yaw in [0,45,90,135]:
        cy,sy=np.cos(np.radians(yaw)),np.sin(np.radians(yaw))
        Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]])
        Rf=Rz@R
        if try_pose(P,quat_from_R(Rf),f"topdown local{name} yaw{yaw}"):
            pass
# also try reaching lower/closer
for z in [0.15,0.10,0.25,0.30]:
    la=np.array([0,0,1.]);  R=np.diag([1,-1,-1.])
    try_pose([-0.5,0,z],quat_from_R(R),f"local+Zdown z={z}")
for x in [-0.4,-0.45,-0.55,-0.6]:
    try_pose([x,0,0.2],quat_from_R(np.diag([1,-1,-1.])),f"local+Zdown x={x}")
rclpy.shutdown()
