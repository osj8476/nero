#!/usr/bin/env python3
"""
cgn_prototype.py  —  Phase 2b: Contact-GraspNet backend 스캐폴딩 (프로토타입)

"파이프를 다 만들고, 모델이 꽂힐 구멍만 남긴다."
기존 sj_pickplace/ 는 안 건드림. Phase 4 검증 후 learned_grasp_backend.py 로 이관.

파이프라인:
  masked point cloud (.ply)
    → 정규화 (mean centering, 6-DOF GraspNet §3)
    → ContactGraspNetBackend.predict()
        → _run_model()   ← ★ Phase 2a (CGN 런타임) 완료 시 이 함수만 채움
        → per-point (s, a, b, w) + contact point c
    → 공식 변환 (Contact-GraspNet Eq 1/2/6):
        b̂ = z1/|z1|,  â = (z2 − ⟨b̂,z2⟩b̂)/|·|          (Gram-Schmidt, Eq 6)
        R_g = [ b̂ | â×b̂ | â ]                          (Eq 2)
        t_g = c + (w/2)·b̂ + d·â                         (Eq 1)   d = flange→fingertip
    → 필터: w ≤ w_max,  s ≥ s_threshold
    → 180° twin (approach축 대칭, Contact-GraspNet ADD-S min_u)
    → GraspOut (position, quaternion xyzw, approach, axis, width, score)
    → 시각화: 그리퍼 와이어프레임 .ply  (+ --ros 면 RViz MarkerArray)

CONFIG (NERO AGX 그리퍼):
  d      = 0.1358 m   planning_node.TOP_TCP_OFFSET / URDF gripper_joint1 z (2026-07 실측 확정)
  w_max  = 0.10  m    URDF: gripper_joint1/2 prismatic limit 0.05 × 2
  frame  : camera_color_optical_frame → (호출부에서 _cam_to_base 로 base_link)

실행 (2a 없이, geometric fallback 로 파이프/viz 검증):
  python3 cgn_prototype.py ~/pc_spike/seg/A_box_001__mobile_sam.ply --fallback --ply-out /tmp/g

2a 완료 후:
  _run_model() 에 CGN 호출 연결 → python3 cgn_prototype.py <ply> --model
"""
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
D_FLANGE_TO_FINGERTIP = 0.1358   # m — NERO AGX (planning_node.TOP_TCP_OFFSET)
W_MAX = 0.10                     # m — URDF prismatic 0.05 × 2
S_THRESHOLD_DEFAULT = 0.30
N_INPUT_POINTS = 20000           # Contact-GraspNet 입력 (FPS 전)


# ── 출력 타입 (learned_grasp_backend.LearnedGraspOutput 미러) ──────────────────
@dataclass
class GraspOut:
    position: np.ndarray            # (3,) t_g, 입력 point cloud 와 같은 프레임
    quaternion: np.ndarray          # (4,) xyzw,  R_g = [b, a×b, a]
    approach: np.ndarray            # (3,) â (단위)
    axis: np.ndarray                # (3,) b̂ (단위, 손끝 닫히는 축)
    width: float                    # w (m)
    score: float                    # ŝ
    contact: np.ndarray = field(default=None)  # c (디버그)
    is_twin: bool = False


# ── point cloud I/O ───────────────────────────────────────────────────────────
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


def normalize(pc: np.ndarray, n_points: int = N_INPUT_POINTS):
    """mean centering (6-DOF GraspNet §3) + 다운/업샘플. returns (pc_centered, mean)."""
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
    """per-point 예측 → GraspOut. Contact-GraspNet Eq 1/2/6."""
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
    predict(point_cloud) → List[GraspOut]. _run_model() 만 Phase 2a 대기."""

    name = "contact_graspnet"

    def __init__(self, s_threshold=S_THRESHOLD_DEFAULT, w_max=W_MAX,
                 d=D_FLANGE_TO_FINGERTIP, use_fallback=False):
        self.s_threshold = s_threshold
        self.w_max = w_max
        self.d = d
        self.use_fallback = use_fallback

    # ★★★ Phase 2a 완료 시 여기만 채움 ★★★
    def _run_model(self, pc_centered: np.ndarray):
        """CGN 모델 호출. → list of (c, z1, z2, w, s) per FPS point.
        c: contact point (pc_centered 프레임), z1/z2: raw baseline/approach 벡터
        (build_grasp 안에서 Gram-Schmidt), w: 폭(m), s: contact confidence.

        2a 옵션:
          A) contact_graspnet_pytorch 를 import (같은 env) — 직접 forward
          B) CGN 을 별도 프로세스/HTTP 서비스로 → 여기서 pc 를 POST, 결과 받음
        릴리스 코드 플래그: --local_regions --filter_grasps --forward_passes N
        """
        raise NotImplementedError(
            "Phase 2a (CGN 런타임) 완료 후 연결. 지금은 --fallback 사용.")

    def _fallback_geometric(self, pc: np.ndarray):
        """CGN 없이 파이프/viz 검증용. box류 물체에 top-down + side 후보 생성.
        (품질은 CGN 아님 — 좌표 공식·필터·시각화만 검증)"""
        out = []
        # 지배 평면(윗면) RANSAC
        plane = _ransac_plane(pc)
        rng = np.random.default_rng(0)
        if plane is not None:
            n, dd = plane
            if n[2] > 0:      # 카메라에서 볼 때 위쪽 향하도록 (z fwd 광학계)
                n = -n
            inl = pc[np.abs(pc @ n + dd) < 0.01]
            if len(inl) > 30:
                # top-down: approach = -n (면으로 내려꽂음), baseline = 면 위 방향들
                tangent = _perp(n)
                for c in inl[rng.choice(len(inl), min(40, len(inl)), replace=False)]:
                    for ang in (0.0, np.pi / 2):
                        b = np.cos(ang) * tangent + np.sin(ang) * np.cross(n, tangent)
                        w = _width_along(pc, c, b)
                        if 0.005 < w <= self.w_max:
                            out.append((c, b, -n, w, 0.6))
        # side: 수평 approach 몇 방향
        cen = pc.mean(0)
        for yaw in np.linspace(0, np.pi, 4, endpoint=False):
            a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            b = np.array([0.0, 0.0, 1.0])
            c = cen - a * 0.02
            w = _width_along(pc, c, b)
            if 0.005 < w <= self.w_max:
                out.append((c, b, a, w, 0.4))
        return out

    def predict(self, point_cloud: np.ndarray) -> List[GraspOut]:
        pc, _ = normalize(point_cloud)
        raw = self._fallback_geometric(pc) if self.use_fallback else self._run_model(pc)
        grasps: List[GraspOut] = []
        for c, z1, z2, w, s in raw:
            if s < self.s_threshold or not (0.0 < w <= self.w_max):
                continue
            g = build_grasp(c, z1, z2, w, s, d=self.d)
            grasps.append(g)
            grasps.append(twin_180(g))  # ADD-S 대칭 → 둘 다 pick_ik 로 (reachability 2배)
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
    # 선분: base→tip, tip→f1, tip→f2, f1→(f1 + 0.02a), f2→(f2 + 0.02a)
    pts = [base, tip, f1, f2, f1 + 0.03 * a, f2 + 0.03 * a]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
    return np.array(pts), edges


def write_grasp_ply(path, grasps: List[GraspOut], top_k=40):
    V, E, C = [], [], []
    for i, g in enumerate(grasps[:top_k]):
        pts, edges = _gripper_wire(g)
        off = len(V)
        V.extend(pts)
        E.extend([(off + u, off + v) for u, v in edges])
        col = (0, 200, 0) if not g.is_twin else (0, 120, 0)
        C.extend([col] * len(pts))
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(E)}\n")
        f.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for p, c in zip(V, C):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", help="masked object point cloud (.ply, seg_bench 출력)")
    ap.add_argument("--fallback", action="store_true", help="CGN 없이 geometric 후보 (파이프 검증)")
    ap.add_argument("--model", action="store_true", help="CGN _run_model 사용 (2a 완료 후)")
    ap.add_argument("--s-threshold", type=float, default=S_THRESHOLD_DEFAULT)
    ap.add_argument("--ply-out", default=None, help="그리퍼 와이어프레임 .ply 저장 경로(폴더 or 파일)")
    ap.add_argument("--ros", action="store_true", help="RViz /cgn_grasps MarkerArray 발행")
    ap.add_argument("--frame", default="camera_color_optical_frame")
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    if not (args.fallback or args.model):
        print("--fallback (지금) 또는 --model (2a 후) 중 하나 필요", file=sys.stderr)
        sys.exit(1)

    pc = load_ply_xyz(Path(args.ply).expanduser())
    print(f"[cgn_prototype] {Path(args.ply).name}  {len(pc)} pts  "
          f"extent {(pc.max(0)-pc.min(0))*100} cm  dist {pc[:,2].mean():.2f}m")

    be = ContactGraspNetBackend(s_threshold=args.s_threshold, use_fallback=args.fallback)
    grasps = be.predict(pc)
    print(f"[cgn_prototype] {len(grasps)} grasp (twin 포함), score {grasps[0].score:.2f}~{grasps[-1].score:.2f}"
          if grasps else "[cgn_prototype] grasp 0개")
    for g in grasps[:8]:
        print(f"  t={np.round(g.position,3)}  a={np.round(g.approach,2)}  "
              f"b={np.round(g.axis,2)}  w={g.width*100:.1f}cm  s={g.score:.2f}"
              f"{'  (twin)' if g.is_twin else ''}")

    if args.ply_out and grasps:
        p = Path(args.ply_out).expanduser()
        if p.is_dir() or not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            p = p / (Path(args.ply).stem + "__grasps.ply")
        write_grasp_ply(p, grasps, top_k=args.top_k)
        print(f"[cgn_prototype] → {p}  (물체 .ply 와 같이 MeshLab/CloudCompare 로)")

    if args.ros and grasps:
        publish_ros_markers(grasps, args.frame, top_k=args.top_k)


if __name__ == "__main__":
    main()
