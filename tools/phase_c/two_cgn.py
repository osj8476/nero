"""Stage A: CGN grasps for each GT-segmented object in a multi-object render.
Per object: extract PC -> world frame -> CGN -> world-frame aspect + (for elongated) Taubin
cylinder fit -> write <name>_grasps.json  (grasps in gripper_flange convention already:
R_flange = [-(a x b) | b | a],  a = approach toward object, b = closing axis)."""
import os, sys, json
import numpy as np

NPZ = os.path.expanduser(sys.argv[1])
OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/grasp")
R = os.path.expanduser("~/grasp/contact_graspnet_pytorch")
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "Pointnet_Pointnet2_pytorch"))
os.chdir(R)
import torch
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO

d = np.load(NPZ, allow_pickle=True)
depth = d["depth_m"].astype(np.float32)
k = d["K"].reshape(-1)
K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
seg = d["seg"].astype(np.int32)
W_T_cam = np.array(d["W_T_cam"])
flip = np.diag([1., -1., -1., 1.])
WT = W_T_cam @ flip

cfg = config_utils.load_config(os.path.join(R, "checkpoints", "contact_graspnet"), batch_size=1)
est = GraspEstimator(cfg)
CheckpointIO(checkpoint_dir=os.path.join(R, "checkpoints", "contact_graspnet", "checkpoints"),
             model=est.model).load("model.pt")
est.model.eval()

targets = []
for name, key in [("bottle", "bottle_id"), ("box", "box_id")]:
    if key in d.files and int(d[key][0]) > 0:
        targets.append((name, int(d[key][0])))

# full scene PC + per-id segments (need the scene for local_regions)
full_seg = np.zeros_like(seg)
for i, (_, sid) in enumerate(targets, start=1):
    full_seg[seg == sid] = i
pc_full, pc_segs, _ = est.extract_point_clouds(depth, K, segmap=full_seg, z_range=[0.2, 1.2])


def circle_fit(x, y):
    xm, ym = x.mean(), y.mean()
    xr, yr = x - xm, y - ym
    z = xr * xr + yr * yr
    Z = np.c_[z - z.mean(), xr, yr]
    _, _, V = np.linalg.svd(Z, full_matrices=False)
    A = V[-1]
    return xm - A[1] / (2 * A[0]), ym - A[2] / (2 * A[0])


for idx, (name, sid) in enumerate(targets, start=1):
    if idx not in pc_segs or len(np.asarray(pc_segs[idx])) < 50:
        print(f"[{name}] too few points"); continue
    sp_cam = np.asarray(pc_segs[idx])
    sp = (WT[:3, :3] @ sp_cam.T).T + WT[:3, 3]           # world / base_link
    zext = float(sp[:, 2].max() - sp[:, 2].min())
    xyext = float(max(sp[:, 0].max() - sp[:, 0].min(), sp[:, 1].max() - sp[:, 1].min()))
    aspect = zext / max(xyext, 1e-3)
    base_z, top_z = float(sp[:, 2].min()), float(sp[:, 2].max())

    with torch.no_grad():
        g, s, c, o = est.predict_scene_grasps(
            pc_full, pc_segments={idx: sp_cam}, local_regions=True, filter_grasps=True)
    if idx not in g or np.asarray(g[idx]).size == 0:
        print(f"[{name}] no CGN grasps"); continue
    G = np.asarray(g[idx]).reshape(-1, 4, 4)
    S = np.asarray(s[idx]).reshape(-1)
    O = np.asarray(o[idx]).reshape(-1)
    print(f"[{name}] {len(G)} grasps  score {S.min():.3f}-{S.max():.3f}  aspect_w {aspect:.2f}")

    # world-frame bbox (both shapes)
    bbmin=[float(sp[:,j].min()) for j in range(3)]; bbmax=[float(sp[:,j].max()) for j in range(3)]
    box_fit=dict(bbmin=bbmin,bbmax=bbmax,centre=[0.5*(bbmin[j]+bbmax[j]) for j in range(3)],
                 ext=[bbmax[j]-bbmin[j] for j in range(3)])
    cyl = None
    if aspect > 1.6:
        cx, cy = circle_fit(sp[:, 0], sp[:, 1])
        r = float(np.clip(np.median(np.hypot(sp[:, 0] - cx, sp[:, 1] - cy)) * 1.2, 0.015, 0.06))
        cyl = dict(axis=[float(cx), float(cy)], radius=r, base_z=base_z, top_z=top_z,
                   centre=[float(cx), float(cy), 0.5 * (base_z + top_z)])
        print(f"[{name}] cyl fit axis ({cx:.3f},{cy:.3f}) r {r:.3f} z [{base_z:.3f},{top_z:.3f}]")

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

    out = []
    for i in range(len(G)):
        gw = WT @ G[i]                              # CGN grasp in world
        b = gw[:3, 0]; a = gw[:3, 2]                # closing axis, approach (toward object)
        Rf = np.column_stack([-np.cross(a, b), b, a])   # gripper_flange convention
        out.append(dict(rank=i, score=float(S[i]), opening_m=float(O[i]),
                        position_xyz=[round(float(x), 4) for x in gw[:3, 3]],
                        quat_xyzw=[round(x, 5) for x in R2q(Rf)],
                        approach=[round(float(x), 3) for x in a],
                        closing=[round(float(x), 3) for x in b]))
    obj_world = d[name.replace("bottle", "cyl").replace("box", "box") + "_world"].tolist() \
        if (name.replace("bottle", "cyl") + "_world") in d.files or (name + "_world") in d.files else [0, 0, 0]
    obj_world = d["cyl_world"].tolist() if (name == "bottle" and "cyl_world" in d.files) else \
        (d["box_world"].tolist() if (name == "box" and "box_world" in d.files) else [0, 0, 0])
    json.dump(dict(frame="base_link", object=name, obj_world=obj_world, aspect_w=aspect,
                   cyl_fit=cyl, box_fit=box_fit, grasps=out),
              open(os.path.join(OUT, f"{name}_grasps.json"), "w"), indent=1)
    print(f"[{name}] -> {name}_grasps.json")
