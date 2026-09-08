#!/usr/bin/env python3
"""Render pipeline panels for pc_spike6/A_box_001 (real matte-stage box)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

R = os.path.expanduser("~/grasp/contact_graspnet_pytorch")
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "Pointnet_Pointnet2_pytorch"))
os.chdir(R)
import torch
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
from ultralytics import SAM

NPZ = os.path.expanduser(sys.argv[1] if len(sys.argv)>1 else "~/grasp/pc_spike6/A_box_001.npz")
BBJ = os.path.expanduser(sys.argv[2] if len(sys.argv)>2 else "~/grasp/pc_spike6/A_box_001.bbox.json")
LABEL = sys.argv[3] if len(sys.argv)>3 else "box"
OUT = os.environ.get("PANEL_OUT","/tmp/panels"); os.makedirs(OUT, exist_ok=True)

INK = "#1a1a1a"; SCENE = "#a9a49a"; OBJ = "#2f6690"; SEL = "#d8532c"; FADE = "#b7b2a8"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": "#333", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
                     "axes.titlesize": 10.5})

d = np.load(NPZ, allow_pickle=True)
depth = d["depth_m"].astype(np.float32)
color = d["color"] if "color" in d.files else d["rgb"]
k = d["K"].reshape(-1); K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
bb = json.loads(open(BBJ).read())["bboxes"][0]
H, W = depth.shape

if "seg" in d.files and d["seg"].shape == depth.shape and d["seg"].max() > 0:
    mask = d["seg"].astype(bool)
else:
    sam = SAM(os.path.expanduser("~/grasp/mobile_sam.pt"))
    res = sam.predict(color[:, :, ::-1], bboxes=[bb], verbose=False)[0]
    mask = res.masks.data[0].cpu().numpy().astype(bool)
    if mask.shape != (H, W):
        import cv2
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
segmap = mask.astype(np.int32)

cfg = config_utils.load_config(os.path.join(R, "checkpoints", "contact_graspnet"), batch_size=1)
est = GraspEstimator(cfg)
CheckpointIO(checkpoint_dir=os.path.join(R, "checkpoints", "contact_graspnet", "checkpoints"),
             model=est.model).load("model.pt")
est.model.eval()
pc_full, pc_seg, _ = est.extract_point_clouds(depth, K, segmap=segmap, z_range=[0.2, 1.2])
with torch.no_grad():
    g, s, c, o = est.predict_scene_grasps(pc_full, pc_segments=pc_seg,
                                          local_regions=True, filter_grasps=True)
G = np.asarray(g[1]).reshape(-1, 4, 4)
S = np.asarray(s[1]).reshape(-1)
O = np.clip(np.asarray(o[1]).reshape(-1), 0.01, 0.10)
seg_pc = np.asarray(pc_seg[1])
print(f"{len(G)} grasps, score {S.min():.3f}-{S.max():.3f}")

D_TCP = 0.1358
down = np.array([0, 1.0, 0.0])  # optical +y ~ toward table
appr = G[:, :3, 2]
theta = np.arccos(np.clip(np.abs(appr @ down), -1, 1))
contacts = G[:, :3, 3] + D_TCP * appr
com = seg_pc.mean(0)
d_com = np.linalg.norm(contacts - com, axis=1)     # 3c: prefer grasps near object centroid
cost = S - 0.5 * theta**4 - 1.4 * d_com
sel = int(np.argmax(cost))
print(f"selected #{sel}: score {S[sel]:.3f}, theta {np.degrees(theta[sel]):.0f}, d_com {d_com[sel]*100:.1f}cm, opening {O[sel]*100:.1f}cm")

def proj(P):
    P = np.atleast_2d(P)
    z = np.clip(P[:, 2], 1e-3, None)
    return np.stack([k[0]*P[:, 0]/z + k[2], k[1]*P[:, 1]/z + k[3]], -1)

FINGER_LEN = 0.055

def grip_polylines(gm, w):
    """Proper Π-shaped 2-finger gripper as a list of 3D polylines.
    CGN: trans t = gripper base (above object); a = z-col approach (toward object, ~down).
    contact plane ~ t + d*a. Palm bar spans w along b at the base end; fingers run
    DOWN (+a) to the fingertips near the object; fingertips are NOT connected."""
    t = gm[:3, 3]; a = gm[:3, 2] / np.linalg.norm(gm[:3, 2]); b = gm[:3, 0] / np.linalg.norm(gm[:3, 0])
    cc = t + D_TCP * a                       # contact-plane centre (on object)
    ft1 = cc + (w/2) * b; ft2 = cc - (w/2) * b        # the two fingertips
    kn1 = ft1 - FINGER_LEN * a; kn2 = ft2 - FINGER_LEN * a   # finger knuckles (base end)
    wrist = 0.5 * (kn1 + kn2)
    stem = wrist - 0.03 * a                  # short wrist stem pointing further up
    return [
        [kn1, kn2],          # palm bar
        [kn1, ft1],          # finger 1 (down)
        [kn2, ft2],          # finger 2 (down)
        [wrist, stem],       # wrist stem (up, away from object)
    ]

rgb = color[:, :, ::-1] if color.ndim == 3 and color.shape[-1] == 3 else color

# ---------- PANEL A ----------
fig, ax = plt.subplots(figsize=(5.8, 3.5), dpi=155)
ax.imshow(rgb)
ov = np.zeros((H, W, 4)); ov[mask] = [0.18, 0.40, 0.56, 0.42]; ax.imshow(ov)
ax.contour(mask, [0.5], colors=[OBJ], linewidths=1.5)
ax.add_patch(Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, ec=SEL, lw=1.7))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("A · RGB → YOLO bbox → SAM mask", loc="left", pad=6)
ax.text(0.015, 0.05, f"mask {mask.sum():,} px", transform=ax.transAxes, fontsize=8,
        color="white", bbox=dict(fc="#000000aa", ec="none", pad=2.5))
fig.tight_layout(); fig.savefig(f"{OUT}/A.png", bbox_inches="tight"); plt.close(fig)

# ---------- PANEL B : gravity-aligned side elevation + selected grasp ----------
_BFIG=(5.8,3.0)
# RANSAC dominant plane (table) -> build frame where table normal = up
rng = np.random.default_rng(0)
pf_s = pc_full[rng.choice(len(pc_full), 8000, replace=False)]
best_n, best_in = None, 0
for _ in range(200):
    tri = pf_s[rng.choice(len(pf_s), 3, replace=False)]
    nrm = np.cross(tri[1]-tri[0], tri[2]-tri[0]); nn = np.linalg.norm(nrm)
    if nn < 1e-9: continue
    nrm /= nn; dd = -nrm @ tri[0]
    inl = int((np.abs(pf_s @ nrm + dd) < 0.006).sum())
    if inl > best_in: best_in, best_n, best_d = inl, nrm, dd
up = best_n if (best_n @ (seg_pc.mean(0) - (-best_d*best_n))) > 0 else -best_n  # points toward object
ex = np.array([1.0, 0, 0]); ex = ex - (ex@up)*up; ex /= np.linalg.norm(ex)
ey = np.cross(up, ex)
def to_elev(P):  # -> (horizontal along ex, height along up)
    P = np.atleast_2d(P); return np.stack([P@ex, P@up], -1)
fig, ax = plt.subplots(figsize=(5.8, 2.9), dpi=155)
oc = seg_pc.mean(0)
near = pc_full[np.linalg.norm(pc_full[:, [0, 2]] - oc[[0, 2]], axis=1) < 0.28]
NE = to_elev(near); SE = to_elev(seg_pc[::max(1, len(seg_pc)//6000)])
h0 = np.median(to_elev(pf_s[np.abs(pf_s@best_n + best_d) < 0.01])[:, 1])
ax.scatter(NE[:, 0], NE[:, 1]-h0, s=2.0, c=[SCENE], lw=0, alpha=0.5)
ax.scatter(SE[:, 0], SE[:, 1]-h0, s=3.4, c=[OBJ], lw=0, alpha=0.95)
for seg in grip_polylines(G[sel], O[sel]):
    E = to_elev(np.array(seg)); ax.plot(E[:, 0], E[:, 1]-h0, "-", c=SEL, lw=3.0, solid_capstyle="round")
gm = G[sel]; a = gm[:3, 2]/np.linalg.norm(gm[:3, 2]); cc3 = gm[:3, 3] + D_TCP*a
e_hi = to_elev(cc3 - 0.13*a)[0]; e_lo = to_elev(cc3 - 0.03*a)[0]
ax.annotate("", xy=(e_lo[0], e_lo[1]-h0), xytext=(e_hi[0], e_hi[1]-h0),
            arrowprops=dict(arrowstyle="-|>", color=SEL, lw=1.7, alpha=0.85))
ax.text(e_hi[0]+0.012, e_hi[1]-h0-0.012, "approach", fontsize=7.5, color=SEL)
ax.axhline(0, color="#00000022", lw=0.8)
ax.set_aspect("equal"); ax.set_facecolor("#faf9f7")
ax.set_xlabel("along table  (m)", fontsize=8.5); ax.set_ylabel("height above table  (m)", fontsize=8.5)
ax.tick_params(labelsize=7.5)
ax.set_title(f"B · side view (gravity-aligned)  ·  grasp descends onto the {LABEL}", loc="left", pad=6)
ax.margins(y=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/B.png", bbox_inches="tight"); plt.close(fig)

# ---------- PANEL C : all grasps projected on RGB ----------
fig, ax = plt.subplots(figsize=(5.8, 3.5), dpi=155)
ax.imshow(rgb, alpha=0.92)
cmap = plt.cm.viridis
norm = plt.Normalize(S.min(), S.max())
for i in np.argsort(S):
    col = cmap(norm(S[i]))
    for seg in grip_polylines(G[i], O[i]):
        P = proj(np.array(seg))
        ax.plot(P[:, 0], P[:, 1], "-", c=col, lw=1.15, alpha=0.7, solid_capstyle="round")
ax.set_xlim(bb[0]-70, bb[2]+70); ax.set_ylim(bb[3]+70, bb[1]-70)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"C · {len(G)} Contact-GraspNet candidates on the {LABEL}  ·  color = graspness", loc="left", pad=6)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6.5)
cb.set_label("score", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/C.png", bbox_inches="tight"); plt.close(fig)

# ---------- PANEL D : selected grasp ----------
fig, ax = plt.subplots(figsize=(5.8, 3.5), dpi=155)
ax.imshow(rgb, alpha=0.9)
for i in range(len(G)):
    if i == sel: continue
    for seg in grip_polylines(G[i], O[i]):
        P = proj(np.array(seg))
        ax.plot(P[:, 0], P[:, 1], "-", c=FADE, lw=0.9, alpha=0.45, solid_capstyle="round")
for seg in grip_polylines(G[sel], O[sel]):
    P = proj(np.array(seg))
    ax.plot(P[:, 0], P[:, 1], "-", c=SEL, lw=3.4, solid_capstyle="round")
# approach arrow: gripper descends along -a from above
gm = G[sel]; a = gm[:3, 2] / np.linalg.norm(gm[:3, 2])
cc3 = gm[:3, 3] + D_TCP * a
a0 = proj(cc3 - 0.11 * a)[0]; a1 = proj(cc3 - 0.03 * a)[0]
ax.annotate("", xy=(a1[0], a1[1]), xytext=(a0[0], a0[1]),
            arrowprops=dict(arrowstyle="-|>", color=SEL, lw=1.6, alpha=0.8))
ax.text(a0[0]+4, a0[1]-4, "approach", fontsize=7.5, color=SEL, alpha=0.9)
cc = proj(cc3)[0]
ax.plot(cc[0], cc[1], "o", c=SEL, ms=6, mec="white", mew=1)
ax.set_xlim(bb[0]-70, bb[2]+70); ax.set_ylim(bb[3]+70, bb[1]-70)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("D · selected grasp  ·  score − w·θ⁴", loc="left", pad=6)
ax.text(0.015, 0.05,
        f"score {S[sel]:.3f}   opening {O[sel]*100:.1f} cm   θ {np.degrees(theta[sel]):.0f}°   {d_com[sel]*100:.0f} cm from centroid",
        transform=ax.transAxes, fontsize=8.5, color=SEL, weight="bold",
        bbox=dict(fc="#ffffffcc", ec="none", pad=2.5))
fig.tight_layout(); fig.savefig(f"{OUT}/D.png", bbox_inches="tight"); plt.close(fig)

stats = dict(n_grasps=int(len(G)), score_max=round(float(S.max()), 3), score_min=round(float(S.min()), 3),
             sel_idx=sel, sel_score=round(float(S[sel]), 3), sel_opening_cm=round(float(O[sel]*100), 1),
             sel_theta_deg=round(float(np.degrees(theta[sel])), 0), sel_dcom_cm=round(float(d_com[sel]*100), 1),
             obj_pts=int(len(seg_pc)), scene_pts=int(len(pc_full)), mask_px=int(mask.sum()),
             depth_valid_pct=round(100*float((depth > 0).mean()), 1))
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
