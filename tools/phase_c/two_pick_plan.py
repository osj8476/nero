"""Stage B: rank CGN grasps (already in gripper_flange convention) + IK, per object.
Elongated -> CGN side grasp re-targeted to the fitted cylinder axis at mid-height.
Compact  -> CGN top-down grasp by score - w*theta^4 - w*d_com.
Writes <name>_pick.json (quat, approach, flange, pre, standoff, grasp_sol, pre_sol) for step6.
"""
import json, sys, os
import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

NAMES = sys.argv[1:] or ["bottle", "box"]
D_TCP = 0.1358
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
SEED = [0.0, -0.9991, 0.0, 1.3986, 0.0, 0.0, 1.5491]

rclpy.init()
n = rclpy.create_node("two_pick")
ik = n.create_client(GetPositionIK, "/compute_ik")
ik.wait_for_service(timeout_sec=10)
ss = RobotState(); ss.joint_state = JointState(name=ARM, position=SEED)


def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fik(pos, quat, tight=True):
    req = GetPositionIK.Request(); req.ik_request.group_name = "arm"; req.ik_request.robot_state = ss
    ps = PoseStamped(); ps.header.frame_id = "base_link"
    ps.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    ps.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
    req.ik_request.pose_stamped = ps
    req.ik_request.timeout.sec = 1 if tight else 0
    req.ik_request.timeout.nanosec = 0 if tight else 150_000_000
    req.ik_request.avoid_collisions = True
    f = ik.call_async(req); rclpy.spin_until_future_complete(n, f, timeout_sec=6); r = f.result()
    if r and r.error_code.val == 1:
        js = r.solution.joint_state
        return [round(js.position[js.name.index(j)], 4) for j in ARM]
    return None


for name in NAMES:
    p = os.path.expanduser(f"~/grasp/{name}_grasps.json")
    if not os.path.exists(p):
        print(f"[{name}] no grasps file"); continue
    data = json.load(open(p))
    grasps = data["grasps"]
    cyl = data.get("cyl_fit")

    cand = []
    for g in grasps:
        Rf = q2R(g["quat_xyzw"])
        a = np.array(g["approach"])        # toward object
        pos = np.array(g["position_xyz"])
        w = g["opening_m"]
        if not (0.005 < w <= 0.102):
            continue
        theta = np.degrees(np.arccos(np.clip(-a[2], -1, 1)))   # from straight-down
        if cyl:
            # SIDE: want approach ~horizontal; re-target so gripper centre = axis at this height
            if theta < 55:
                continue
            ch = float(np.clip(pos[2] + D_TCP * a[2], cyl["base_z"] + 0.03, cyl["top_z"] - 0.03))
            grip_centre = np.array([cyl["axis"][0], cyl["axis"][1], ch])
            ah = a.copy(); ah[2] = 0
            if np.linalg.norm(ah) < 1e-3:
                continue
            ah /= np.linalg.norm(ah)                            # force horizontal approach
            flange = grip_centre - D_TCP * ah
            # rebuild quat for the horizontalised approach, keep closing axis horizontal _|_ ah
            b = np.cross([0, 0, 1.0], ah); b /= np.linalg.norm(b)
            Rf = np.column_stack([-np.cross(ah, b), b, ah])
            opening = float(min(2 * cyl["radius"] + 0.018, 0.10))
            score = g["score"] - 0.02 * abs(ch - cyl["centre"][2]) * 100    # prefer mid-height
            cand.append((score, flange, Rf, ah, opening, "side"))
        else:
            pass   # top-down handled below via box_fit synthesis (yaw sweep)

    # box: synthesize top-down candidates directly from the fitted bbox, sweeping yaw
    if not cyl and data.get("box_fit"):
        bf = data["box_fit"]; c = np.array(bf["centre"]); ext = np.array(bf["ext"])
        gz = float(bf["bbmax"][2] - 0.02)
        grip_centre = np.array([c[0], c[1], gz])
        ad = np.array([0, 0, -1.0])
        best_score = max((g["score"] for g in grasps), default=0.2)
        opening = float(min(min(ext[0], ext[1]) + 0.02, 0.10))
        cand = []
        for yaw in (30, 45, 60, 90, 20, 75, 15, 0):
            b = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
            Rf = np.column_stack([-np.cross(ad, b), b, ad])
            flange = grip_centre - D_TCP * ad
            cand.append((best_score - abs(yaw - 45) * 1e-4, flange, Rf, ad, opening, "top"))

    cand.sort(key=lambda t: -t[0])
    print(f"[{name}] {len(cand)} candidates ({cand[0][5] if cand else '-'})")

    def R2q(Rm):
        t = np.trace(Rm)
        if t > 0:
            u = np.sqrt(t + 1) * 2; w = .25 * u
            x = (Rm[2, 1] - Rm[1, 2]) / u; y = (Rm[0, 2] - Rm[2, 0]) / u; z = (Rm[1, 0] - Rm[0, 1]) / u
        elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
            u = np.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
            w = (Rm[2, 1] - Rm[1, 2]) / u; x = .25 * u; y = (Rm[0, 1] + Rm[1, 0]) / u; z = (Rm[0, 2] + Rm[2, 0]) / u
        elif Rm[1, 1] > Rm[2, 2]:
            u = np.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
            w = (Rm[0, 2] - Rm[2, 0]) / u; x = (Rm[0, 1] + Rm[1, 0]) / u; y = .25 * u; z = (Rm[1, 2] + Rm[2, 1]) / u
        else:
            u = np.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
            w = (Rm[1, 0] - Rm[0, 1]) / u; x = (Rm[0, 2] + Rm[2, 0]) / u; y = (Rm[1, 2] + Rm[2, 1]) / u; z = .25 * u
        v = np.array([x, y, z, w]); return (v / np.linalg.norm(v)).tolist()

    picked = None
    for score, flange, Rf, a, opening, kind in cand[:30]:
        q = R2q(Rf)
        gs = fik(flange, q)
        if gs is None:
            continue
        for so in ((0.05,0.08) if picked is None and False else ((0.05,0.07) if True else (0.10,0.13))):
            pre = np.array(flange) - so * a
            psl = fik(pre, q)
            if psl:
                picked = dict(object=name, kind=kind, quat=q, approach=[float(x) for x in a],
                              flange=[float(x) for x in flange], pre=[float(x) for x in pre],
                              standoff=so, contact=[float(x) for x in (np.array(flange) + D_TCP * a)],
                              grasp_sol=gs, pre_sol=psl, opening=opening,
                              radius=cyl["radius"] if cyl else None)
                break
        if picked:
            break
    if picked:
        json.dump(picked, open(os.path.expanduser(f"~/grasp/{name}_pick.json"), "w"), indent=1)
        print(f"[{name}] {picked['kind']} grasp  approach {np.round(picked['approach'], 2)}  "
              f"contact {np.round(picked['contact'], 3)}  -> {name}_pick.json")
    else:
        print(f"[{name}] no reachable grasp")

rclpy.shutdown()
