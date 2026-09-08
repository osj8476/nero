"""step6_pick.json 궤적을 MoveIt /execute_trajectory 액션으로 재생 (move_group 실행 경로).
Isaac 스크립트/Isaac MCP 안 씀. transit -> advance -> 그리퍼 close -> retreat -> lift."""
import json, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
GRIP = ["gripper_joint1", "gripper_joint2"]
D = json.load(open("/home/bpdl/grasp/step6_pick.json"))

rclpy.init(); n = Node("exec_pick")
st = {}
n.create_subscription(JointState, "/joint_states",
                      lambda m: [st.__setitem__(a, b) for a, b in zip(m.name, m.position)], 10)
ex = ActionClient(n, ExecuteTrajectory, "/execute_trajectory")
gc = ActionClient(n, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
ex.wait_for_server(timeout_sec=10); gc.wait_for_server(timeout_sec=10)
for _ in range(20): rclpy.spin_once(n, timeout_sec=0.1)


def jl(j): return ", ".join("%.2f" % st.get(x, 0) for x in ARM)


def exec_traj(wps, dt, label, grip=None):
    rt = RobotTrajectory()
    jt = JointTrajectory(); jt.joint_names = list(ARM)
    wps = [list(st.get(j, 0) for j in ARM)] + [list(w) for w in wps]   # anchor at current
    prev = np.array(wps[0]); t = 0.0
    for i, w in enumerate(wps):
        w = np.array(w, float)
        if i > 0:
            t += dt
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in w]
        v = (w - prev) / dt if i > 0 else np.zeros(7)
        if i == len(wps) - 1:
            v = np.zeros(7)
        p.velocities = [float(x) for x in v]
        p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t % 1) * 1e9)
        jt.points.append(p); prev = w
    rt.joint_trajectory = jt
    g = ExecuteTrajectory.Goal(); g.trajectory = rt
    fut = ex.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
    gh = fut.result()
    if not gh or not gh.accepted:
        print(f"[{label}] REJECTED"); return
    rf = gh.get_result_async()
    tend = time.time() + t + 1.5          # settle 5 -> 1.5s
    while time.time() < tend and not rf.done():
        rclpy.spin_once(n, timeout_sec=0.1)
    code = rf.result().result.error_code.val if rf.done() else None
    print(f"[{label}] err={code}  j=({jl(None)})")


def grip(pos, label, hold=2.0):
    jt = JointTrajectory(); jt.joint_names = list(GRIP)
    p = JointTrajectoryPoint(); p.positions = [float(x) for x in pos]
    p.time_from_start.sec = 1
    jt.points = [p]
    g = FollowJointTrajectory.Goal(); g.trajectory = jt
    fut = gc.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
    gh = fut.result(); rf = gh.get_result_async()
    t0 = time.time()
    while time.time() - t0 < hold and not rf.done():
        rclpy.spin_once(n, timeout_sec=0.1)
    print(f"[{label}] grip=({st.get('gripper_joint1',0):.3f}, {st.get('gripper_joint2',0):.3f})")


res = {"start": {j: round(st.get(j, 0), 3) for j in ARM}}
GOPEN = [0.05, -0.05]; GCLOSE = [0.014, -0.014]

grip(GOPEN, "grip-open", hold=0.5)
exec_traj(D["transit"], dt=0.13, label="transit")
exec_traj(D["advance"], dt=0.20, label="advance")
res["at_grasp"] = {j: round(st.get(j, 0), 3) for j in ARM}
grip(GCLOSE, "grip-close", hold=2.5)
res["grip_after_close"] = [round(st.get("gripper_joint1", 0), 4), round(st.get("gripper_joint2", 0), 4)]
exec_traj(D["retreat"], dt=0.20, label="retreat")
cur = [st.get(j, 0.0) for j in ARM]
lift = cur[:]; lift[1] -= 0.35; lift[3] -= 0.20
exec_traj([lift], dt=2.0, label="lift")
grip(GCLOSE, "grip-hold", hold=0.5)

res["final"] = {j: round(st.get(j, 0), 3) for j in ARM}
gf1 = st.get("gripper_joint1", 0.0)
res["grip_final"] = [round(gf1, 4), round(st.get("gripper_joint2", 0), 4)]
gc1 = res["grip_after_close"][0]
# 성공 판정: 그리퍼가 완전히 안 닫힘(= 손끝 사이에 물체) -> HELD. (씬 캡처 안 함)
CLOSED = 0.020                       # 이 이하면 사실상 완전히 닫힘 = 허공
res["held"] = bool(gc1 > CLOSED)
res["verdict"] = "HELD" if res["held"] else "MISSED"
open("/tmp/exec_pick_res.json", "w").write(json.dumps(res, indent=1))
print(f"[verdict] grip_after_close={gc1:.4f}  ->  {res['verdict']}  "
      f"(> {CLOSED} = 물체 잡힘)")
rclpy.shutdown()
