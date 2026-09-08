"""Fit a cylinder to the segmented point cloud, synthesize a SIDE grasp on the true axis,
IK-check (grasp + pre-grasp), output side_grasp.json for step5.

Cylinder fit for a ~standing object: circle fit (Taubin) on the xy projection -> axis (cx,cy)
+ radius r; z range from the points. Grasp = approach horizontally toward the axis, gripper
centre ON the axis at mid-height, opening = 2r + margin.
"""
import json, sys, os
import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

NPZ = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/grasp/bottle_cgn.npz")
D_TCP = 0.1358
MARGIN = 0.018          # gripper opening = diameter + this
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
SEED = [0.0, -0.9991, 0.0, 1.3986, 0.0, 0.0, 1.5491]

d = np.load(NPZ, allow_pickle=True)
depth = d["depth_m"].astype(np.float32)
k = d["K"].reshape(-1)
fx, fy, cx0, cy0 = k
seg = d["seg"].astype(np.int32)
W_T_cam = np.array(d["W_T_cam"])

# deproject masked depth -> camera-optical point cloud (simple pinhole), then -> world
ys, xs = np.where((seg > 0) & (depth > 0) & (depth < 1.5))
z = depth[ys, xs]
X = (xs - cx0) * z / fx
Y = (ys - cy0) * z / fy
sp_cam = np.stack([X, Y, z], -1).astype(np.float64)
flip = np.diag([1., -1., -1., 1.])
WT = W_T_cam @ flip
sp = (WT[:3, :3] @ sp_cam.T).T + WT[:3, 3]        # segment in world / base_link
print("[seg] %d pts  z %.3f..%.3f" % (len(sp), sp[:, 2].min(), sp[:, 2].max()))


def circle_fit_taubin(x, y):
    """Taubin algebraic circle fit -> (cx, cy, r)."""
    x = x - x.mean(); y = y - y.mean()
    z = x * x + y * y
    zm = z.mean()
    Z = np.c_[z - zm, x, y]
    _, _, V = np.linalg.svd(Z, full_matrices=False)
    A = V[-1]
    a = A[0]; b = A[1]; c = A[2]
    cx = -b / (2 * a)
    cy = -c / (2 * a)
    return cx, cy


# fit on xy projection (standing cylinder -> vertical axis)
x0, y0 = sp[:, 0].mean(), sp[:, 1].mean()
cx, cy = circle_fit_taubin(sp[:, 0], sp[:, 1])
cx += x0; cy += y0
z_min, z_max = sp[:, 2].min(), sp[:, 2].max()
# radius = median point-to-axis distance (robust to arc-only coverage)
r = float(np.median(np.hypot(sp[:, 0] - cx, sp[:, 1] - cy)))
r = float(np.clip(r, 0.012, 0.06))
axis_pt = np.array([cx, cy, 0.5 * (z_min + z_max)])
print("[cyl fit] axis (x,y)=(%.3f, %.3f)  r=%.3f  z=[%.3f, %.3f]  -> grip centre %s"
      % (cx, cy, r, z_min, z_max, np.round(axis_pt, 3)))

opening = float(min(2 * r + MARGIN, 0.10))


def R2q(Rm):
    t = np.trace(Rm)
    if t > 0:
        s = np.sqrt(t + 1) * 2; w = .25 * s
        x = (Rm[2, 1] - Rm[1, 2]) / s; y = (Rm[0, 2] - Rm[2, 0]) / s; z = (Rm[1, 0] - Rm[0, 1]) / s
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        s = np.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w = (Rm[2, 1] - Rm[1, 2]) / s; x = .25 * s; y = (Rm[0, 1] + Rm[1, 0]) / s; z = (Rm[0, 2] + Rm[2, 0]) / s
    elif Rm[1, 1] > Rm[2, 2]:
        s = np.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w = (Rm[0, 2] - Rm[2, 0]) / s; x = (Rm[0, 1] + Rm[1, 0]) / s; y = .25 * s; z = (Rm[1, 2] + Rm[2, 1]) / s
    else:
        s = np.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w = (Rm[1, 0] - Rm[0, 1]) / s; x = (Rm[0, 2] + Rm[2, 0]) / s; y = (Rm[1, 2] + Rm[2, 1]) / s; z = .25 * s
    v = np.array([x, y, z, w]); return v / np.linalg.norm(v)


rclpy.init()
n = rclpy.create_node("cyl_side")
ik = n.create_client(GetPositionIK, "/compute_ik")
ik.wait_for_service(timeout_sec=10)
ss = RobotState(); ss.joint_state = JointState(name=ARM, position=SEED)


def solve(pos, q):
    req = GetPositionIK.Request(); req.ik_request.group_name = "arm"; req.ik_request.robot_state = ss
    ps = PoseStamped(); ps.header.frame_id = "base_link"
    ps.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    ps.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
    req.ik_request.pose_stamped = ps; req.ik_request.timeout.sec = 1; req.ik_request.avoid_collisions = True
    f = ik.call_async(req); rclpy.spin_until_future_complete(n, f, timeout_sec=6); rr = f.result()
    if rr and rr.error_code.val == 1:
        js = rr.solution.joint_state
        return [round(js.position[js.name.index(j)], 4) for j in ARM]
    return None


# sweep horizontal approach directions toward the axis; pick first reachable (grasp + pre-grasp)
# NERO gripper_flange convention (URDF gripper_flange_joint rpy=(-pi/2,0,-pi/2)):
#   local +Z = approach (toward object),  local +-Y = gripper closing axis.
#   CGN R_g = [b | axb | a]  ->  R_flange = [-(axb) | b | a]
best = None
for deg in range(0, 360, 20):
    a = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])  # points FROM gripper TO axis
    b = np.cross(np.array([0, 0, 1.0]), a); b /= np.linalg.norm(b)          # closing axis, horizontal
    Rm = np.column_stack([-np.cross(a, b), b, a])
    q = R2q(Rm)
    grip_centre = axis_pt.copy()
    flange = grip_centre - D_TCP * a
    for so in (0.10, 0.13):
        pre = flange - so * a
        gs = solve(flange, q); psl = solve(pre, q)
        if gs and psl:
            best = dict(deg=deg, quat=q.tolist(), approach=a.tolist(),
                        flange=flange.tolist(), pre=pre.tolist(), standoff=so,
                        contact=grip_centre.tolist(), grasp_sol=gs, pre_sol=psl,
                        opening=opening, axis=[cx, cy], radius=r)
            break
    if best:
        break

if best:
    json.dump(best, open(os.path.expanduser("~/grasp/side_grasp.json"), "w"), indent=1)
    print("[grasp] approach %d deg  standoff %.2f  opening %.1f cm  -> side_grasp.json"
          % (best["deg"], best["standoff"], best["opening"] * 100))
    print("  grasp_sol", best["grasp_sol"])
else:
    print("[grasp] no reachable side grasp on the fitted axis")
rclpy.shutdown()
