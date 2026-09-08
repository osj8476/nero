"""Phase 3c v3 — object-shape-aware ranking.
World-frame aspect: standing elongated (bottle/can) -> SIDE grasp (approach & closing axis
perpendicular to the vertical axis) near the mid-height centroid. Compact/flat (box) -> top-down.
Then geometric prefilter + fast IK (grasp + pre-grasp reachable).
Needs `seg_pc_pca` (aspect_w, centroid_est, major_axis) in the grasps json — see bottle_all.py.
"""
import json, sys, time
import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

CGN = sys.argv[1]
K_IK = 30
STANDOFF = 0.10
D_TCP = 0.1358
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
SEED = [0.0, -0.9991, 0.0, 1.3986, 0.0, 0.0, 1.5491]


def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


data = json.load(open(CGN))
obj = np.array(data["box_world"])
pca = data.get("seg_pc_pca")
if pca:
    e0 = np.array(pca["major_axis"])
    com = np.array(pca["centroid_est"])
    aspect = float(pca["aspect_w"])
    standing = aspect > 1.6
else:
    e0 = np.array([0, 0, 1.0])
    com = obj.copy()
    aspect = 1.0
    standing = False
print("[obj] aspect_w=%.2f  major=%s  centroid=%s  %s"
      % (aspect, np.round(e0, 2), np.round(com, 3),
         "STANDING ELONGATED -> SIDE grasp" if standing else "compact/flat -> TOP-DOWN"))

cand = []
for g in data["grasps"]:
    R = q2R(g["quat_xyzw"])
    pos = np.array(g["position_xyz"])
    w = g["opening_m"]
    if not (0.005 < w <= 0.102):
        continue
    a = R[:, 2]
    b = R[:, 0]
    contact = pos + D_TCP * a
    if pos[2] < 0.02 or contact[2] < 0.015:
        continue
    d_com = np.linalg.norm(contact - com)
    if standing:
        cost = g["score"] - 1.5 * abs(a @ e0) - 0.8 * abs(b @ e0) - 1.5 * d_com
    else:
        theta = np.arccos(np.clip(-a[2], -1, 1))
        cost = g["score"] - 0.5 * theta**4 - 1.2 * d_com
    cand.append(dict(score=g["score"], d_com=float(d_com), cost=float(cost),
                     pos=pos.tolist(), quat=g["quat_xyzw"], approach=a.tolist(), opening=w,
                     ang_to_axis=float(np.degrees(np.arccos(np.clip(abs(a @ e0), 0, 1))))))
cand.sort(key=lambda c: -c["cost"])
t1 = time.time()
print("[prefilter] %d -> %d pass  (top: ang_to_axis=%.0f deg  d_com=%.1f cm)"
      % (len(data["grasps"]), len(cand), cand[0]["ang_to_axis"], cand[0]["d_com"] * 100))

rclpy.init()
n = rclpy.create_node("rank4")
ik = n.create_client(GetPositionIK, "/compute_ik")
ik.wait_for_service(timeout_sec=10)
ss = RobotState()
ss.joint_state = JointState(name=ARM, position=SEED)


def fik(pos, quat):
    req = GetPositionIK.Request()
    req.ik_request.group_name = "arm"
    req.ik_request.robot_state = ss
    ps = PoseStamped()
    ps.header.frame_id = "base_link"
    ps.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    ps.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
    req.ik_request.pose_stamped = ps
    req.ik_request.timeout.sec = 0
    req.ik_request.timeout.nanosec = 120_000_000
    req.ik_request.avoid_collisions = True
    f = ik.call_async(req)
    rclpy.spin_until_future_complete(n, f, timeout_sec=1.5)
    r = f.result()
    if r and r.error_code.val == 1:
        js = r.solution.joint_state
        return [js.position[js.name.index(j)] for j in ARM]
    return None


picked = None
nik = 0
for c in cand[:K_IK]:
    nik += 1
    gs = fik(c["pos"], c["quat"])
    if gs is None:
        continue
    pre = np.array(c["pos"]) - STANDOFF * np.array(c["approach"])
    nik += 1
    psl = fik(pre, c["quat"])
    if psl is None:
        continue
    c["grasp_sol"] = gs
    c["pre_sol"] = psl
    c["pre"] = pre.tolist()
    picked = c
    break
t2 = time.time()
print("[ik] %d calls -> %s  (%.0f ms)" % (nik, "PICKED" if picked else "NONE", 1e3 * (t2 - t1)))
if picked:
    p = picked
    tip = np.array(p["pos"]) + D_TCP * np.array(p["approach"])
    print("  score=%.3f  ang_to_axis=%.0f deg  d_com=%.1f cm  open=%.1f cm"
          % (p["score"], p["ang_to_axis"], p["d_com"] * 100, p["opening"] * 100))
    print("  approach=%s  fingertip=%s  obj=%s" % (np.round(p["approach"], 2), np.round(tip, 3), np.round(obj, 3)))
    json.dump(dict(best=p, box_world=obj.tolist(), standing=bool(standing), major_axis=e0.tolist()),
              open("/home/bpdl/grasp/rank4_best.json", "w"), indent=1)
rclpy.shutdown()
