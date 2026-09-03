#!/usr/bin/env python3
"""
pc_spike_report.py  — grasp generation 착수 전 point cloud sanity spike (2/2)

pc_spike_capture.py 가 남긴 .npz(depth_m + color + K) 를 읽어서:
  1. 카메라 좌표계 point cloud 로 역투영 (point_cloud.py 와 동일 핀홀 공식)
  2. .ply 로 저장 (MeshLab / CloudCompare / Isaac Sim 에서 육안 확인)
  3. 품질 리포트 출력 — grasp net 입력으로 쓸 만한지 판단 근거

리포트 항목 & 왜:
  - hole_ratio (전체 / 중앙 ROI)  : depth 구멍. 무지 골판지 box 는 stereo
    depth 가 잘 뚫려서 여기서 걸린다. ROI 20%+ 면 그 물체엔 grasp net 이
    후보를 거의 못 냄.
  - depth p05/p50/p95 (ROI)       : 작업 거리. 스펙(0.3~1.0m)에 있나.
  - near-cluster extent            : ROI 에서 최근접 depth 대역(±0.12m) 점들의
    bbox 크기. 실제 물체(~10~20cm)보다 훨씬 크면 배경(테이블)이 섞인 것
    (grasp_geometry_pipeline.md 의 "extents 0.27~0.44m" 증상과 동일).
  - bimodality                     : ROI depth 히스토그램이 물체/테이블 두
    봉우리로 갈리나. 안 갈리고 뭉개지면 물체를 depth 로 분리 불가 →
    segmentation(SAM) 의존도가 높아짐.

실행:
    python3 pc_spike_report.py ~/pc_spike/scene_000.npz
    python3 pc_spike_report.py ~/pc_spike/           # 폴더면 전부
    python3 pc_spike_report.py ~/pc_spike/scene_000.npz --roi 0.3 --ply-out /tmp
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def deproject(depth_m, K, stride=1, dmin=0.05, dmax=3.0):
    """(H,W) depth(m) → (N,3) 카메라 광학 좌표계(REP-103: Z fwd, X right, Y down).
    point_cloud.mask_depth_to_pointcloud 와 같은 공식, 마스크 대신 전체."""
    fx, fy, cx, cy = K
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[::stride, ::stride].astype(np.float64)
    valid = np.isfinite(z) & (z > dmin) & (z < dmax)
    z = z[valid]
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def write_ply(path, xyz, rgb=None, max_pts=300_000):
    if len(xyz) > max_pts:
        sel = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz = xyz[sel]
        rgb = rgb[sel] if rgb is not None else None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is not None:
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[2])} {int(c[1])} {int(c[0])}\n")
        else:
            for p in xyz:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def bimodality_coeff(x):
    """Sarle's bimodality coefficient. >0.555 면 bimodal 경향."""
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return float("nan")
    n = len(x)
    m = x.mean()
    s = x.std() + 1e-12
    g = ((x - m) ** 3).mean() / s ** 3            # skew
    k = ((x - m) ** 4).mean() / s ** 4 - 3.0       # excess kurtosis
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def report_one(npz_path: Path, roi_frac: float, ply_dir: Path | None):
    d = np.load(npz_path, allow_pickle=True)
    depth_m = d["depth_m"].astype(np.float32)
    K = d["K"]
    color = d["color"] if "color" in d else None
    H, W = depth_m.shape

    # ROI = 중앙 사각형 (관찰자세에서 대상 물체가 대략 프레임 중앙 가정)
    rh, rw = int(H * roi_frac / 2), int(W * roi_frac / 2)
    cy0, cx0 = H // 2, W // 2
    roi = depth_m[cy0 - rh:cy0 + rh, cx0 - rw:cx0 + rw]

    hole_all = float((depth_m <= 0.0).mean())
    hole_roi = float((roi <= 0.0).mean())
    roi_valid = roi[(roi > 0.05) & (roi < 3.0)]

    print(f"\n=== {npz_path.name}  ({W}x{H})  K=fx{K[0]:.1f} fy{K[1]:.1f} cx{K[2]:.1f} cy{K[3]:.1f}")
    print(f"  hole_ratio   전체 {hole_all*100:5.1f}%   중앙ROI {hole_roi*100:5.1f}%"
          + ("   <- ROI 20%+ : 그 물체 grasp 어려움" if hole_roi > 0.20 else ""))
    if roi_valid.size < 100:
        print("  ROI 유효 depth 거의 없음 — 물체가 중앙에 없거나 depth 전멸")
        return
    p05, p50, p95 = np.percentile(roi_valid, [5, 50, 95])
    print(f"  ROI depth    p05 {p05:.3f}  p50 {p50:.3f}  p95 {p95:.3f} m"
          + ("" if 0.25 <= p50 <= 1.2 else "   <- 작업거리 스펙(0.3~1.0m) 벗어남?"))

    # near-cluster: ROI 최근접 대역 ±0.12m
    band = roi_valid[(roi_valid >= p05) & (roi_valid <= p05 + 0.24)]
    full = deproject(depth_m, K, stride=2)
    # ROI 픽셀에 대응하는 3D 점만 근사 재추출
    roi_pts = deproject(roi, K, stride=1)
    if len(roi_pts):
        near = roi_pts[np.abs(roi_pts[:, 2] - p05) < 0.12]
        if len(near) > 50:
            ext = near.max(0) - near.min(0)
            print(f"  near-cluster extent  dx{ext[0]*100:.1f} dy{ext[1]*100:.1f} "
                  f"dz{ext[2]*100:.1f} cm  (n={len(near)})"
                  + ("   <- 20cm+ : 배경 섞임 의심" if max(ext[0], ext[1]) > 0.25 else ""))

    bc = bimodality_coeff(roi_valid)
    print(f"  ROI depth bimodality  {bc:.3f}  "
          + ("(물체/테이블 분리 가능)" if bc > 0.555 else "(단봉 — depth 만으론 물체 분리 어려움, SAM 필요)"))

    if ply_dir is not None:
        ply_dir.mkdir(parents=True, exist_ok=True)
        # 컬러 입히기 (stride=2 로 뽑았으니 컬러도 동일 stride)
        rgb = None
        if color is not None:
            fx, fy, cx, cy = K
            csub = color[::2, ::2]
            zsub = depth_m[::2, ::2].astype(np.float64)
            vmask = np.isfinite(zsub) & (zsub > 0.05) & (zsub < 3.0)
            rgb = csub[vmask]
        out = ply_dir / (npz_path.stem + ".ply")
        write_ply(out, full, rgb)
        print(f"  → {out}  ({len(full)} pts)  MeshLab/CloudCompare 로 육안 확인")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".npz 파일 또는 폴더")
    ap.add_argument("--roi", type=float, default=0.35, help="중앙 ROI 비율(0~1)")
    ap.add_argument("--ply-out", default=None, help=".ply 저장 폴더 (생략시 안 저장)")
    args = ap.parse_args()

    p = Path(args.path).expanduser()
    files = sorted(p.glob("*.npz")) if p.is_dir() else [p]
    if not files:
        print("npz 파일 없음", file=sys.stderr)
        sys.exit(1)
    ply_dir = Path(args.ply_out).expanduser() if args.ply_out else None
    for f in files:
        report_one(f, args.roi, ply_dir)

    print("\n[판단 기준]")
    print("  - 중앙ROI hole < ~10%, near-cluster extent 물체크기와 비슷, bimodality > 0.555")
    print("    → depth 그대로 grasp net(Contact-GraspNet/VGN) 입력 OK")
    print("  - hole 큼 / extent 과대 / 단봉 → (1) 카메라 자세·거리 조정 (2) SAM mask 로")
    print("    물체만 crop 후 입력 (3) 다중뷰 TSDF 융합(VGN) 로 대응")


if __name__ == "__main__":
    main()
