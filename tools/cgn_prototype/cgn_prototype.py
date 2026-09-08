#!/usr/bin/env python3
"""
cgn_prototype.py  —  Phase 2b/2c: Contact-GraspNet backend (프로토타입)

"파이프를 다 만들고, 모델이 꽂힐 구멍만 남긴다" → 2026-09-07 그 구멍을 채움.
기존 sj_pickplace/ 는 안 건드림. Phase 4 검증 후 learned_grasp_backend.py 로 이관.

━━━ 2026-09-07 수정: masked-only(segonly) → local_regions 경로로 ━━━━━━━━━━━━━━━━
Phase 2c 에서 확인: CGN 에 "물체 마스크 안의 점만" 넣으면(segonly) score ~0.19,
edge-pinch 만 나옴. CGN 이 필요로 하는 국소 씬 컨텍스트(물체 주변 테이블면)를
버리기 때문. **올바른 방식 = 전체 씬 PC + 물체 segment → predict_scene_grasps(
local_regions=True, filter_grasps=True)** (NVIDIA 릴리스의 --local_regions).
이러면 box/cup/bottle/clutter 전부 score ~0.29, 몸통 감싸는 grasp (opening ~5cm).

파이프라인:
  scene depth+K (+ SAM segmap)
    → deproject → pc_full (N,3) + pc_segment (M,3)   (camera_color_optical_frame, OpenCV)
    → ContactGraspNetBackend.predict(pc_full, pc_segment)
        → _run_model(): CGN predict_scene_grasps(local_regions=True, filter_grasps=True)
                        → per-grasp (contact c, R_cgn, opening w, score s)
        → TCP 재계산 (CGN 은 Franka baseline d≈0.10, NERO d=0.1358):
            b = R_cgn[:,0]   (baseline / 손끝 닫히는 축)
            a = R_cgn[:,2]   (approach, contact→gripper 방향)
            b̂,â = Gram-Schmidt(b, a)                       (Contact-GraspNet Eq 6)
            R_g = [ b̂ | â×b̂ | â ]                          (Eq 2)
            t_g = c + (w/2)·b̂ + d_nero·â                   (Eq 1)
          → CGN 이 준 4x4 의 translation 은 버리고 c + NERO d 로 다시 세움.
            이게 CLAUDE.md "TOP/SIDE TCP offset 합치지 마라" 문제를 없앤다 —
            grasp 마다 자기 approach 축 하나로 offset 통일.
    → 필터: w ≤ w_max,  s ≥ s_threshold
    → 180° twin (approach축 대칭, ADD-S min_u) → 둘 다 pick_ik 로 (reachability 2배)
    → GraspOut (position, quaternion xyzw, approach, axis, width, score)
    → 시각화: 그리퍼 와이어프레임 .ply (+ --ros 면 RViz MarkerArray)

CONFIG (NERO AGX 그리퍼):
  d      = 0.1358 m   planning_node.TOP_TCP_OFFSET / URDF gripper_joint1 z (2026-07 실측 확정)
  w_max  = 0.10  m    URDF: gripper_joint1/2 prismatic limit 0.05 × 2
  frame  : camera_color_optical_frame → (호출부에서 _cam_to_base 로 base_link)

Phase 4 매핑 (learned_grasp_backend.LearnedGraspBackend — 이 파일에서 건드리지 않음):
  현재:    predict(pc_full, pc_segment)                       -> List[GraspOut]
  Phase 4: predict(point_cloud, geometry_features, grasp_intent) -> List[LearnedGraspOutput]
  - point_cloud  = pc_full (전체 씬 클라우드).  pc_segment 는 segmentation_backend
    마스크에서 만들어 추가 인자로 넘기거나 geometry_features 에 실어 전달.
  - GraspOut ↔ LearnedGraspOutput 은 필드 동형 (position/approach_vector/grasp_axis/
    score/width).

실행:
  source ~/grasp/cgn_venv/bin/activate
  export CGN_REPO=~/grasp/contact_graspnet_pytorch
  # scene npz (+ 옆에 <name>.bbox.json) → CGN grasp:
  python3 cgn_prototype.py ~/grasp/pc_spike6/A_box_001.npz --model --ply-out /tmp/g
  # 모델 없이 파이프/viz 만 (geometric fallback, masked .ply 도 가능):
  python3 cgn_prototype.py <scene.npz|masked.ply> --fallback --ply-out /tmp/g
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
D_FLANGE_TO_FINGERTIP = 0.1358   # m — NERO AGX (planning_node.TOP_TCP_OFFSET)
W_MAX = 0.10                     # m — URDF prismatic 0.05 × 2
S_THRESHOLD_DEFAULT = 0.15       # CGN pytorch 포트 score 천장 ~0.29 (TF 원본은 더 높음).
                                 # 포트 config first/second_thres 가 이미 0.15 → 그 밑은 안 나옴.
Z_RANGE_DEFAULT = (0.2, 1.5)     # m — workspace depth crop
CGN_REPO = os.environ.get("CGN_REPO", str(Path.home() / "grasp" / "contact_graspnet_pytorch"))


# ── 출력 타입 (learned_grasp_backend.LearnedGraspOutput 미러) ──────────────────
@dataclass
class GraspOut:
    position: np.ndarray            # (3,) t_g, 입력 point cloud 와 같은 프레임
    quaternion: np.ndarray          # (4,) xyzw,  R_g = [b, a×b, a]
    approach: np.ndarray            # (3,) â (단위, contact→gripper)
    axis: np.ndarray                # (3,) b̂ (단위, 손끝 닫히는 축)
    width: float                    # w (m)
    score: float                    # ŝ
    contact: np.ndarray = field(default=None)  # c (디버그)
    is_twin: bool = False


# ── point cloud / capture I/O ────────────────────────────────────────────────
def load_ply_xyz(path: Path) -> np.ndarray:
    """ascii PLY → (N,3). property 순서 무관하게 x,y,z 만."""
    with open(path) as f:
        head, props = [], []
        while True:
            ln = f.readline().strip()
            head.append(ln)
            if ln.startswith("property"):
                props.append(ln.split()[-1])
            if ln == "end_header":
                break
        n = next(int(h.split()[-1]) for h in head if h.startswith("element vertex"))
        xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
        rows = np.array([[float(v) for v in f.readline().split()] for _ in range(n)])
    return rows[:, [xi, yi, zi]]


def load_capture(npz_path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """pc_spike 캡처 npz → (depth_m (H,W), K (3,3), color (H,W,3) or None).
    npz 키: depth_m, K=[fx,fy,cx,cy], (옵션) color."""
    d = np.load(npz_path, allow_pickle=True)
    depth = np.asarray(d["depth_m"], dtype=np.float32)
    k = np.asarray(d["K"], dtype=np.float64).reshape(-1)
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], dtype=np.float64)
    color = np.asarray(d["color"]) if "color" in d.files else None
    return depth, K, color


def load_bboxes(src_path) -> list:
    """<name>.bbox.json (키: bboxes=[[x0,y0,x1,y1],...]) → list. 없으면 []."""
    p = Path(src_path)
    for cand in (Path(str(p.with_suffix("").with_suffix("")) + ".bbox.json"),
                 p.with_suffix(".bbox.json")):
        if cand.exists():
            return json.loads(cand.read_text()).get("bboxes", [])
    return []


def sam_masks(color_rgb, bboxes, sam_ckpt) -> Tuple[list, str]:
    """bbox 프롬프트 → 타이트 마스크 리스트. ultralytics SAM 실패 시 사각 마스크."""
    H, W = color_rgb.shape[:2]
    masks = []
    try:
        from ultralytics import SAM
        m = SAM(sam_ckpt)
        for bb in bboxes:
            r = m.predict(color_rgb[:, :, ::-1], bboxes=[bb], verbose=False)[0]
            mk = r.masks.data[0].cpu().numpy().astype(bool)
            if mk.shape != (H, W):
                import cv2
                mk = cv2.resize(mk.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            masks.append(mk)
        return masks, "sam"
    except Exception as e:
        print(f"[cgn] SAM 실패 ({e}) → 사각 마스크")
        for x0, y0, x1, y1 in bboxes:
            mk = np.zeros((H, W), bool)
            mk[int(y0):int(y1), int(x0):int(x1)] = True
            masks.append(mk)
        return masks, "rect"


def deproject(depth: np.ndarray, K: np.ndarray, mask: Optional[np.ndarray] = None,
              z_range=Z_RANGE_DEFAULT) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """depth(H,W, m) + K(3,3) → pc_full (N,3).  mask 주면 pc_segment (M,3) 도.
    프레임: camera_color_optical_frame (OpenCV: x우, y하, z전) — CGN predict_grasps
    가 기대하는 convention, NERO point_cloud.mask_depth_to_pointcloud 와 동일."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = (z > z_range[0]) & (z < z_range[1]) & np.isfinite(z)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pc_full = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)
    pc_seg = None
    if mask is not None:
        m = valid & mask.astype(bool)
        pc_seg = np.stack([x[m], y[m], z[m]], axis=1).astype(np.float32)
    return pc_full, pc_seg


def normalize(pc: np.ndarray, n_points: int = 20000):
    """mean centering (6-DOF GraspNet §3) + 다운샘플. returns (pc_centered, mean).
    ── CGN predict_scene_grasps 는 내부에서 centering 하므로 그 경로엔 안 씀.
       geometric fallback / 다른 backend 용으로만 유지."""
    mean = pc.mean(axis=0)
    pc = pc - mean
    if len(pc) > n_points:
        idx = np.random.default_rng(0).choice(len(pc), n_points, replace=False)
        pc = pc[idx]
    return pc, mean


# ── 공식 (Contact-GraspNet Eq 1/2/6) ─────────────────────────────────────────
def gram_schmidt(z1: np.ndarray, z2: np.ndarray):
    """→ (b̂, â) 직교정규. Contact-GraspNet Eq (6).
    b̂ = z1/|z1| (baseline),  â = z2 에서 b̂ 성분 뺀 것 정규화 (approach)."""
    b = z1 / (np.linalg.norm(z1) + 1e-9)
    a = z2 - np.dot(b, z2) * b
    a = a / (np.linalg.norm(a) + 1e-9)
    return b, a


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 회전행렬 → 쿼터니언 xyzw."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-9)


def build_grasp(c, z1, z2, w, s, d=D_FLANGE_TO_FINGERTIP) -> GraspOut:
    """per-grasp 예측 → GraspOut. Contact-GraspNet Eq 1/2/6.
    z1 = baseline 방향(b), z2 = approach 방향(a, contact→gripper).
    CGN 4x4 에서는 z1=R[:,0], z2=R[:,2]."""
    b, a = gram_schmidt(np.asarray(z1, float), np.asarray(z2, float))
    R = np.column_stack([b, np.cross(a, b), a])          # Eq 2: [b | a×b | a]
    t = np.asarray(c, float) + (w / 2.0) * b + d * a      # Eq 1
    return GraspOut(position=t, quaternion=rot_to_quat(R),
                    approach=a, axis=b, width=float(w), score=float(s),
                    contact=np.asarray(c, float))


def twin_180(g: GraspOut) -> GraspOut:
    """approach 축 중심 180° 회전 = 동등 grasp (Contact-GraspNet ADD-S min_u).
    b̂ → −b̂ 로 뒤집으면 됨. t_g 는 (w/2)b 항 때문에 재계산."""
    t = build_grasp(g.contact, -g.axis, g.approach, g.width, g.score * 0.999)
    t.is_twin = True
    return t


# ── Backend ───────────────────────────────────────────────────────────────────
class ContactGraspNetBackend:
    """learned_grasp_backend.LearnedGraspBackend 인터페이스 미러.
    predict(pc_full, pc_segment) → List[GraspOut].

    모델(elchun/contact_graspnet_pytorch)은 lazy 로드 (첫 predict 때 ~10s).
    로드 실패하거나 use_fallback 이면 geometric fallback 으로 자동 위임."""

    name = "contact_graspnet"

    def __init__(self, s_threshold=S_THRESHOLD_DEFAULT, w_max=W_MAX,
                 d=D_FLANGE_TO_FINGERTIP, use_fallback=False,
                 cgn_repo=CGN_REPO, local_regions=True, filter_grasps=True,
                 forward_passes=1):
        self.s_threshold = s_threshold
        self.w_max = w_max
        self.d = d
        self.use_fallback = use_fallback
        self.cgn_repo = cgn_repo
        self.local_regions = local_regions
        self.filter_grasps = filter_grasps
        self.forward_passes = forward_passes
        self._est = None
        self._torch = None
        self._cfg = None

    # ── CGN 런타임 로드 (한 번) ──────────────────────────────────────────────
    def _load(self):
        if self._est is not None:
            return
        repo = os.path.expanduser(self.cgn_repo)
        for p in (repo, os.path.join(repo, "Pointnet_Pointnet2_pytorch")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
        from contact_graspnet_pytorch import config_utils
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        ckpt_dir = os.path.join(repo, "checkpoints", "contact_graspnet")
        cwd0 = os.getcwd()
        os.chdir(repo)                       # 포트가 상대경로 checkpoint 를 씀
        try:
            cfg = config_utils.load_config(ckpt_dir, batch_size=1)
            est = GraspEstimator(cfg)
            CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, "checkpoints"),
                         model=est.model).load("model.pt")
            est.model.eval()
        finally:
            os.chdir(cwd0)
        self._torch, self._est, self._cfg = torch, est, cfg
        print(f"[cgn] model loaded  device={est.device}  "
              f"gripper_width(config)={cfg['DATA']['gripper_width']}  "
              f"local_regions={self.local_regions} filter_grasps={self.filter_grasps}")

    # ── CGN forward → per-grasp (c, b, a, w, s) ─────────────────────────────
    def _run_model(self, pc_full: np.ndarray, pc_segment: np.ndarray):
        self._load()
        torch, est = self._torch, self._est
        with torch.no_grad():
            g, s, c, o = est.predict_scene_grasps(
                pc_full.astype(np.float32),
                pc_segments={1: pc_segment.astype(np.float32)},
                local_regions=self.local_regions,
                filter_grasps=self.filter_grasps,
                forward_passes=self.forward_passes)
        raw = []
        for k in g:                                     # local_regions → key = segment id (1)
            G = np.asarray(g[k]).reshape(-1, 4, 4)
            S = np.asarray(s[k]).reshape(-1)
            C = np.asarray(c[k]).reshape(-1, 3)
            O = np.asarray(o[k]).reshape(-1)
            for i in range(len(G)):
                R = G[i][:3, :3]
                raw.append((C[i], R[:, 0].copy(), R[:, 2].copy(), float(O[i]), float(S[i])))
        return raw

    def _fallback_geometric(self, pc: np.ndarray):
        """CGN 없이 파이프/viz 검증용. box류 물체에 top-down + side 후보 생성.
        (품질은 CGN 아님 — 좌표 공식·필터·시각화만 검증)"""
        out = []
        plane = _ransac_plane(pc)
        rng = np.random.default_rng(0)
        if plane is not None:
            n, dd = plane
            if n[2] > 0:      # 카메라에서 볼 때 위쪽 향하도록 (z fwd 광학계)
                n = -n
            inl = pc[np.abs(pc @ n + dd) < 0.01]
            if len(inl) > 30:
                tangent = _perp(n)
                for c in inl[rng.choice(len(inl), min(40, len(inl)), replace=False)]:
                    for ang in (0.0, np.pi / 2):
                        b = np.cos(ang) * tangent + np.sin(ang) * np.cross(n, tangent)
                        w = _width_along(pc, c, b)
                        if 0.005 < w <= self.w_max:
                            out.append((c, b, -n, w, 0.6))
        cen = pc.mean(0)
        for yaw in np.linspace(0, np.pi, 4, endpoint=False):
            a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            b = np.array([0.0, 0.0, 1.0])
            c = cen - a * 0.02
            w = _width_along(pc, c, b)
            if 0.005 < w <= self.w_max:
                out.append((c, b, a, w, 0.4))
        return out

    def predict(self, pc_full: np.ndarray,
                pc_segment: Optional[np.ndarray] = None) -> List[GraspOut]:
        """pc_full (N,3) 전체 씬 + pc_segment (M,3) 타겟 → GraspOut 리스트 (score desc).
        pc_segment 없거나 use_fallback 이면 geometric fallback."""
        if self.use_fallback or pc_segment is None or len(pc_segment) < 20:
            raw = self._fallback_geometric(
                pc_segment if (pc_segment is not None and len(pc_segment) >= 20) else pc_full)
        else:
            try:
                raw = self._run_model(pc_full, pc_segment)
            except Exception as e:
                print(f"[cgn] _run_model 실패 → geometric fallback: {e}")
                raw = self._fallback_geometric(pc_segment)

        grasps: List[GraspOut] = []
        for c, z1, z2, w, s in raw:
            if s < self.s_threshold or not (0.0 < w <= self.w_max):
                continue
            g = build_grasp(c, z1, z2, w, s, d=self.d)
            grasps.append(g)
            grasps.append(twin_180(g))   # ADD-S 대칭 → 둘 다 pick_ik 로 (reachability 2배)
        grasps.sort(key=lambda x: -x.score)
        return grasps


# ── geometric fallback helpers ───────────────────────────────────────────────
def _ransac_plane(pts, iters=120, thr=0.008, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(pts) < 50:
        return None
    best_in, best = 0, None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        na = np.linalg.norm(nrm)
        if na < 1e-9:
            continue
        nrm /= na
        d = -nrm @ s[0]
        ninl = int((np.abs(pts @ nrm + d) < thr).sum())
        if ninl > best_in:
            best_in, best = ninl, (nrm, d)
    return best


def _perp(v):
    v = v / (np.linalg.norm(v) + 1e-9)
    ref = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    p = ref - np.dot(ref, v) * v
    return p / (np.linalg.norm(p) + 1e-9)


def _width_along(pc, c, b):
    """c 근방에서 b 방향으로 잰 물체 폭 (예측 w 대신 기하 측정, CGN 나쁜소식 B4)."""
    b = b / (np.linalg.norm(b) + 1e-9)
    near = pc[np.linalg.norm(pc - c, axis=1) < 0.03]
    if len(near) < 5:
        return 0.0
    proj = (near - c) @ b
    return float(proj.max() - proj.min())


# ── 시각화: 그리퍼 와이어프레임 PLY ──────────────────────────────────────────
def _gripper_wire(g: GraspOut, w_open=None):
    """grasp pose 에 놓인 간이 2지 그리퍼 (선분 vertices). 프레임: R_g, t_g.
    로컬: base(원점) → 손목 뒤로 d, 두 손끝은 ±w/2 (b축), approach 는 +a."""
    w = (w_open or g.width) / 2.0
    a, b = g.approach, g.axis
    tip = g.position - D_FLANGE_TO_FINGERTIP * a        # fingertip 중심 (= c 근방)
    f1, f2 = tip + w * b, tip - w * b
    base = g.position                                    # flange
    pts = [base, tip, f1, f2, f1 + 0.03 * a, f2 + 0.03 * a]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
    return np.array(pts), edges


def write_grasp_ply(path, grasps: List[GraspOut], top_k=40, scene_pc: Optional[np.ndarray] = None):
    """그리퍼 와이어프레임 (초록=원본, 짙은초록=twin). scene_pc 주면 회색 점 같이.
    MeshLab / CloudCompare 로 원본 물체 .ply 와 함께 로드."""
    V, E, C = [], [], []
    if scene_pc is not None and len(scene_pc):
        step = max(1, len(scene_pc) // 60000)
        for p in scene_pc[::step]:
            V.append(p)
            C.append((110, 110, 110))
    for g in grasps[:top_k]:
        pts, edges = _gripper_wire(g)
        off = len(V)
        V.extend(pts)
        E.extend([(off + u, off + v) for u, v in edges])
        col = (0, 200, 0) if not g.is_twin else (0, 110, 0)
        C.extend([col] * len(pts))
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(E)}\n")
        f.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for p, c in zip(V, C):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for u, v in E:
            f.write(f"{u} {v}\n")


def publish_ros_markers(grasps: List[GraspOut], frame, top_k=40):
    import rclpy
    from rclpy.node import Node
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point

    rclpy.init()
    node = Node("cgn_prototype_markers")
    pub = node.create_publisher(MarkerArray, "/cgn_grasps", 1)
    ma = MarkerArray()
    for i, g in enumerate(grasps[:top_k]):
        pts, edges = _gripper_wire(g)
        m = Marker()
        m.header.frame_id = frame
        m.ns = "cgn"
        m.id = i
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.003
        m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 1.0, 0.0, 1.0)
        for u, v in edges:
            for k in (u, v):
                m.points.append(Point(x=float(pts[k][0]), y=float(pts[k][1]), z=float(pts[k][2])))
        ma.markers.append(m)
    node.get_logger().info(f"publishing {len(ma.markers)} grasp markers on /cgn_grasps ({frame})")
    for _ in range(20):
        pub.publish(ma)
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


# ── 입력 로드 (scene npz | masked ply) ───────────────────────────────────────
def load_scene(src: Path, sam_ckpt: str, target: int, z_range):
    """→ (pc_full, pc_segment, viz_scene_pc, tag).
    .npz : depth+K (+ bbox.json → SAM segmap) → deproject.  local_regions 경로.
    .ply : masked point cloud (레거시 segonly).  pc_full=pc_segment, fallback 전용."""
    if src.suffix == ".ply":
        pc = load_ply_xyz(src).astype(np.float32)
        print(f"[cgn] {src.name}  (masked .ply — 레거시 segonly 경로, --fallback 전용) "
              f"{len(pc)} pts")
        return pc, pc, pc, "ply"

    depth, K, color = load_capture(src)
    bboxes = load_bboxes(src)
    mask = None
    if bboxes and color is not None:
        masks, how = sam_masks(color, bboxes, sam_ckpt)
        if 0 <= target < len(masks):
            mask = masks[target]
        print(f"[cgn] {src.name}  segmap={how}  bbox {len(bboxes)}개  "
              f"target#{target} 픽셀 {int(mask.sum()) if mask is not None else 0}")
    else:
        print(f"[cgn] {src.name}  bbox.json 없음 → segment 없이 (fallback 만 가능)")
    pc_full, pc_seg = deproject(depth, K, mask, z_range=z_range)
    print(f"[cgn] pc_full {pc_full.shape}"
          + (f"  pc_segment {pc_seg.shape}  "
             f"extent {np.round((pc_seg.max(0)-pc_seg.min(0))*100,1)}cm" if pc_seg is not None else ""))
    return pc_full, pc_seg, pc_full, "npz"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="scene .npz (+ 옆 <name>.bbox.json)  또는  masked .ply (레거시)")
    ap.add_argument("--model", action="store_true", help="CGN 모델 사용 (기본: .npz 면 자동)")
    ap.add_argument("--fallback", action="store_true", help="CGN 없이 geometric 후보 (파이프/viz 검증)")
    ap.add_argument("--s-threshold", type=float, default=S_THRESHOLD_DEFAULT)
    ap.add_argument("--sam", default=str(Path.home() / "grasp" / "mobile_sam.pt"))
    ap.add_argument("--target", type=int, default=0, help="bbox 인덱스")
    ap.add_argument("--z-range", type=float, nargs=2, default=list(Z_RANGE_DEFAULT))
    ap.add_argument("--no-filter-grasps", action="store_true",
                    help="local_regions 는 쓰되 contact-in-segment 필터 끔")
    ap.add_argument("--forward-passes", type=int, default=1)
    ap.add_argument("--ply-out", default=None, help="그리퍼 와이어프레임 .ply (폴더 or 파일)")
    ap.add_argument("--ros", action="store_true", help="RViz /cgn_grasps MarkerArray 발행")
    ap.add_argument("--frame", default="camera_color_optical_frame")
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    use_fallback = args.fallback and not args.model
    pc_full, pc_seg, viz_pc, tag = load_scene(src, args.sam, args.target, tuple(args.z_range))

    if tag == "ply" and not use_fallback:
        print("[cgn] masked .ply 는 scene context 가 없어 CGN local_regions 불가.\n"
              "      전체 씬 .npz 를 쓰거나 --fallback 을 붙이세요.", file=sys.stderr)
        sys.exit(2)
    if pc_seg is None and not use_fallback:
        print("[cgn] segment 없음(bbox.json 필요) → --fallback 아니면 진행 불가.", file=sys.stderr)
        sys.exit(2)

    be = ContactGraspNetBackend(
        s_threshold=args.s_threshold, use_fallback=use_fallback,
        filter_grasps=not args.no_filter_grasps, forward_passes=args.forward_passes)
    grasps = be.predict(pc_full, pc_seg)

    if not grasps:
        print("[cgn] grasp 0개")
        return
    print(f"[cgn] {len(grasps)} grasp (twin 포함)  score {grasps[0].score:.3f}~{grasps[-1].score:.3f}")
    for g in grasps[:8]:
        # 검산: t_g 는 contact 에서 approach 방향으로 ~d 뒤 (|t-c| ≈ √(d²+(w/2)²))
        tc = g.position - g.contact
        along = float(tc @ g.approach)
        print(f"  t={np.round(g.position,3)}  a={np.round(g.approach,2)}  b={np.round(g.axis,2)}  "
              f"w={g.width*100:.1f}cm  s={g.score:.3f}  |t-c|={np.linalg.norm(tc)*100:.1f}cm "
              f"(along a {along*100:+.1f}cm){'  (twin)' if g.is_twin else ''}")

    if args.ply_out:
        p = Path(args.ply_out).expanduser()
        if p.is_dir() or not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            p = p / (src.stem + "__grasps.ply")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_grasp_ply(p, grasps, top_k=args.top_k, scene_pc=viz_pc)
        print(f"[cgn] → {p}  (씬 회색 점 + 그리퍼 와이어. 원본 물체 .ply 와 같이 로드)")

    if args.ros:
        publish_ros_markers(grasps, args.frame, top_k=args.top_k)


if __name__ == "__main__":
    main()
