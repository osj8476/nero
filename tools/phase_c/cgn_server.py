"""CGN 상주 추론 서버 (Thor). 모델 1회 로드 -> POST /infer 로 npz 처리.
two_cgn.py 와 동일 로직 (robust circle_fit + depth-bias 보정 포함)."""
import os, sys, json
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

R = os.path.expanduser("~/grasp/contact_graspnet_pytorch")
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "Pointnet_Pointnet2_pytorch"))
os.chdir(R)
import torch
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO

print("[cgn] loading model ...", flush=True)
cfg = config_utils.load_config(os.path.join(R, "checkpoints", "contact_graspnet"), batch_size=1)
EST = GraspEstimator(cfg)
CheckpointIO(checkpoint_dir=os.path.join(R, "checkpoints", "contact_graspnet", "checkpoints"),
             model=EST.model).load("model.pt")
EST.model.eval()
print("[cgn] ready", flush=True)


def _taubin(x, y):
    xm, ym = x.mean(), y.mean()
    xr, yr = x - xm, y - ym
    z = xr * xr + yr * yr
    Z = np.c_[z - z.mean(), xr, yr]
    _, _, V = np.linalg.svd(Z, full_matrices=False)
    A = V[-1]
    return xm - A[1] / (2 * A[0]), ym - A[2] / (2 * A[0])


def circle_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    cx, cy = _taubin(x, y)
    for _ in range(3):
        d = np.hypot(x - cx, y - cy); r = np.median(d)
        keep = np.abs(d - r) < np.percentile(np.abs(d - r), 75) + 1e-6
        if keep.sum() < 8:
            break
        cx, cy = _taubin(x[keep], y[keep]); x, y = x[keep], y[keep]
    return cx, cy


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


def infer(npz_path, out_dir):
    d = np.load(os.path.expanduser(npz_path), allow_pickle=True)
    out_dir = os.path.expanduser(out_dir)
    depth = d["depth_m"].astype(np.float32)
    k = d["K"].reshape(-1)
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
    seg = d["seg"].astype(np.int32)
    WT = np.array(d["W_T_cam"]) @ np.diag([1., -1., -1., 1.])

    targets = []
    for name, key in [("bottle", "bottle_id"), ("box", "box_id")]:
        if key in d.files and int(d[key][0]) > 0:
            targets.append((name, int(d[key][0])))
    full_seg = np.zeros_like(seg)
    for i, (_, sid) in enumerate(targets, start=1):
        full_seg[seg == sid] = i
    pc_full, pc_segs, _ = EST.extract_point_clouds(depth, K, segmap=full_seg, z_range=[0.2, 1.2])

    res = []
    for idx, (name, sid) in enumerate(targets, start=1):
        if idx not in pc_segs or len(np.asarray(pc_segs[idx])) < 50:
            res.append({name: "too few points"}); continue
        sp_cam = np.asarray(pc_segs[idx])
        sp = (WT[:3, :3] @ sp_cam.T).T + WT[:3, 3]
        zext = float(np.percentile(sp[:, 2], 97) - np.percentile(sp[:, 2], 3))
        xw = float(np.percentile(sp[:, 0], 95) - np.percentile(sp[:, 0], 5))
        yw = float(np.percentile(sp[:, 1], 95) - np.percentile(sp[:, 1], 5))
        aspect = zext / max(max(xw, yw), 1e-3)
        base_z, top_z = float(sp[:, 2].min()), float(sp[:, 2].max())

        with torch.no_grad():
            g, s, c, o = EST.predict_scene_grasps(
                pc_full, pc_segments={idx: sp_cam}, local_regions=True, filter_grasps=True)
        if idx not in g or np.asarray(g[idx]).size == 0:
            res.append({name: "no CGN grasps"}); continue
        G = np.asarray(g[idx]).reshape(-1, 4, 4)
        S = np.asarray(s[idx]).reshape(-1); O = np.asarray(o[idx]).reshape(-1)

        bbmin = [float(sp[:, j].min()) for j in range(3)]
        bbmax = [float(sp[:, j].max()) for j in range(3)]
        box_fit = dict(bbmin=bbmin, bbmax=bbmax,
                       centre=[0.5 * (bbmin[j] + bbmax[j]) for j in range(3)],
                       ext=[bbmax[j] - bbmin[j] for j in range(3)])
        cyl = None
        if aspect > 1.6:
            cx, cy = circle_fit(sp[:, 0], sp[:, 1])
            r = float(np.clip(np.median(np.hypot(sp[:, 0] - cx, sp[:, 1] - cy)) * 1.1, 0.015, 0.06))
            cam_xy = np.array([WT[0, 3], WT[1, 3]])
            v = np.array([cx, cy]) - cam_xy; nv = np.linalg.norm(v)
            if nv > 1e-3:
                cx, cy = (np.array([cx, cy]) + (v / nv) * r * 0.35).tolist()
            cyl = dict(axis=[float(cx), float(cy)], radius=r, base_z=base_z, top_z=top_z,
                       centre=[float(cx), float(cy), 0.5 * (base_z + top_z)])

        grasps = []
        for i in range(len(G)):
            gw = WT @ G[i]
            b = gw[:3, 0]; a = gw[:3, 2]
            Rf = np.column_stack([-np.cross(a, b), b, a])
            grasps.append(dict(rank=i, score=float(S[i]), opening_m=float(O[i]),
                               position_xyz=[round(float(x), 4) for x in gw[:3, 3]],
                               quat_xyzw=[round(x, 5) for x in R2q(Rf)],
                               approach=[round(float(x), 3) for x in a],
                               closing=[round(float(x), 3) for x in b]))
        ow = d["cyl_world"].tolist() if (name == "bottle" and "cyl_world" in d.files) else \
            (d["box_world"].tolist() if (name == "box" and "box_world" in d.files) else [0, 0, 0])
        json.dump(dict(frame="base_link", object=name, obj_world=ow, aspect_w=aspect,
                       cyl_fit=cyl, box_fit=box_fit, grasps=grasps),
                  open(os.path.join(out_dir, f"{name}_grasps.json"), "w"), indent=1)
        res.append({name: f"{len(G)} grasps score {S.max():.3f} aspect {aspect:.2f}"
                          + (f" cyl axis ({cyl['axis'][0]:.3f},{cyl['axis'][1]:.3f})" if cyl else "")})
    return res


app = FastAPI()


class Req(BaseModel):
    npz_path: str
    out_dir: str = "~/grasp"


@app.get("/health")
def health():
    return {"status": "ok", "model": "contact_graspnet", "device": str(next(EST.model.parameters()).device)}


@app.post("/infer")
def _infer(r: Req):
    import time
    t0 = time.time()
    out = infer(r.npz_path, r.out_dir)
    return {"ms": round((time.time() - t0) * 1000), "results": out}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
