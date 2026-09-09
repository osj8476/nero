"""step6_pick.json 궤적을 MoveIt /execute_trajectory 액션으로 재생 (move_group 실행 경로).
Isaac 스크립트/Isaac MCP 안 씀. transit -> advance -> 그리퍼 close -> retreat -> lift."""
import json, time, os, subprocess
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
    tend = time.time() + t + 0.8          # settle 1.5 -> 0.8s
    while time.time() < tend and not rf.done():
        rclpy.spin_once(n, timeout_sec=0.1)
    code = rf.result().result.error_code.val if rf.done() else None
    print(f"[{label}] err={code}  j=({jl(None)})")
    return code


def reached(target, tol=0.12):
    """현재 관절이 target(7-list) 근처인가 — advance 가 실제로 grasp pose 에 갔나 검증."""
    e = max(abs(st.get(j, 0.0) - t) for j, t in zip(ARM, target))
    return e, e < tol


def grip(pos, label, hold=2.0, settle=None):
    jt = JointTrajectory(); jt.joint_names = list(GRIP)
    p = JointTrajectoryPoint(); p.positions = [float(x) for x in pos]
    p.time_from_start.sec = 1
    jt.points = [p]
    g = FollowJointTrajectory.Goal(); g.trajectory = jt
    fut = gc.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
    gh = fut.result(); rf = gh.get_result_async()
    t0 = time.time()
    # settle: 목표 근처(±settle)에 실제 도달할 때까지 대기 (hold 는 하한/상한)
    while time.time() - t0 < hold:
        rclpy.spin_once(n, timeout_sec=0.1)
        if rf.done():
            break
        if settle is not None and abs(st.get("gripper_joint1", 0) - pos[0]) < settle and time.time() - t0 > 0.6:
            break
    print(f"[{label}] grip=({st.get('gripper_joint1',0):.3f}, {st.get('gripper_joint2',0):.3f})")


START_Q = [st.get(j, 0.0) for j in ARM]
res = {"start": {j: round(x, 3) for j, x in zip(ARM, START_Q)}}
GOPEN = [0.05, -0.05]; GCLOSE = [0.014, -0.014]
PLANS = D.get("plans", [D])

grip(GOPEN, "grip-open", hold=3.0, settle=0.004)   # 0.014->0.05 이동에 시간 필요
if st.get("gripper_joint1", 0) < 0.040:
    print(f"[warn] 그리퍼가 덜 열림 ({st.get('gripper_joint1',0):.3f}) — 재시도")
    grip(GOPEN, "grip-open2", hold=3.0, settle=0.004)

# A-2 수정: plan 을 순서대로 시도. advance 가 실제로 grasp_cfg 에 도달해야 close.
# 도달 못 하면 START 로 후퇴 후 다음 plan (허공에 대고 close 하는 false SLIPPED 방지).
P = None
for pi, cand in enumerate(PLANS):
    exec_traj(cand["transit"], dt=0.13, label=f"p{pi}-transit")
    exec_traj(cand["advance"], dt=0.20, label=f"p{pi}-advance")
    err, ok = reached(cand["grasp_cfg"])
    print(f"[p{pi}] grasp_cfg 도달 오차 {err:.3f} rad -> {'OK' if ok else '미도달'}")
    if ok:
        P = cand; res["plan_used"] = pi; break
    if pi < len(PLANS) - 1:
        exec_traj([START_Q], dt=2.5, label=f"p{pi}-backoff")

if P is None:
    res["plan_used"] = None
    res["grip_after_close"] = res["grip_final"] = [0.0, 0.0]
    res["verdict"] = "NO_REACH"
    res["run_ts"] = D.get("run_ts") or time.strftime("%Y%m%d_%H%M%S")
    open("/tmp/exec_pick_res.json", "w").write(json.dumps(res, indent=1))
    try:
        _p = f"/tmp/{res['run_ts']}.exec.json"; open(_p, "w").write(json.dumps(res))
        subprocess.run(["rsync", "-q", _p, f"thor:grasp/runs/{res['run_ts']}.exec.json"], timeout=15)
    except Exception:
        pass
    print("[verdict] 모든 plan 이 grasp pose 도달 실패 -> NO_REACH")
    rclpy.shutdown(); raise SystemExit

res["at_grasp"] = {j: round(st.get(j, 0), 3) for j in ARM}
_g_before = st.get("gripper_joint1", 0.05)
grip(GCLOSE, "grip-close", hold=2.5)
if st.get("gripper_joint1", 0) > _g_before - 0.004:      # 안 움직임 -> goal 씹힘, 재전송
    print(f"[warn] 그리퍼가 안 닫힘 ({st.get('gripper_joint1',0):.3f}) — 재전송")
    grip(GCLOSE, "grip-close2", hold=2.5)
res["grip_after_close"] = [round(st.get("gripper_joint1", 0), 4), round(st.get("gripper_joint2", 0), 4)]
exec_traj(P["retreat"], dt=0.20, label="retreat")
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
held_close = bool(gc1 > CLOSED)
held_final = bool(gf1 > 0.018)       # lift 후에도 유지됐나 (얇은 벽 핀치는 여기서 빠짐)
res["held"] = held_close and held_final
if held_close and held_final:
    res["verdict"] = "HELD"
elif held_close and not held_final:
    res["verdict"] = "SLIPPED"       # 닫힐 땐 물렸는데 lift 중 놓침
else:
    res["verdict"] = "MISSED"
res["run_ts"] = D.get("run_ts") or time.strftime("%Y%m%d_%H%M%S")
open("/tmp/exec_pick_res.json", "w").write(json.dumps(res, indent=1))
# 계측: <ts>.exec.json 사이드카를 Thor runs/ 로 올림 (show.py 가 plan 레코드와 join)
try:
    _p = f"/tmp/{res['run_ts']}.exec.json"
    open(_p, "w").write(json.dumps(res))
    subprocess.run(["rsync", "-q", _p, f"thor:grasp/runs/{res['run_ts']}.exec.json"], timeout=15)
except Exception as e:
    print("[runlog] 사이드카 업로드 실패:", e)
print(f"[verdict] after_close={gc1:.4f}  final={gf1:.4f}  ->  {res['verdict']}  "
      f"(HELD = 둘 다 > ~0.02; SLIPPED = 닫힘OK 후 lift 중 놓침)")
rclpy.shutdown()
