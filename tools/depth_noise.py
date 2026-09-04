#!/usr/bin/env python3
"""
depth_noise.py  —  Phase 1a-①: RealSense 국소 depth 노이즈 실측

Contact-GraspNet 은 contact 할당 반경 r=5mm (논문 Part B). NERO 국소 depth
노이즈가 이보다 크면 grasp contact 예측이 흔들린다. 이 스크립트는 평평한
면(벽/테이블)을 여러 거리에서 찍어 patch 단위 depth 표준편차를 잰다.

측정 방식:
  - 평면에 RANSAC 평면 적합 → 각 픽셀의 평면까지 거리(residual)
  - patch(기본 20x20px) 단위 residual std = 국소 노이즈
  - temporal std: 같은 픽셀을 N프레임에 걸쳐 본 std (프레임간 흔들림)

실행 (카메라 물린 머신, 평평한 벽/테이블을 정면으로):
    python3 depth_noise.py --distances 0.5 0.65 0.8 --frames 30

pc_spike_capture.py 와 같은 스트림 설정(848x480 + High Accuracy + spatial/temporal
필터)을 기본으로 쓴다. raw 비교는 --no-filters --preset none.
"""
import argparse
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pyrealsense2 미설치: pip install pyrealsense2")


def fit_plane_ransac(pts, iters=200, thresh=0.005, rng=None):
    """pts (N,3) → (normal(3,), d) s.t. n·x + d = 0. inlier thresh = 5mm 기본."""
    rng = rng or np.random.default_rng(0)
    best_in, best = 0, None
    n = len(pts)
    if n < 50:
        return None
    for _ in range(iters):
        s = pts[rng.choice(n, 3, replace=False)]
        v1, v2 = s[1] - s[0], s[2] - s[0]
        nrm = np.cross(v1, v2)
        na = np.linalg.norm(nrm)
        if na < 1e-9:
            continue
        nrm = nrm / na
        d = -nrm @ s[0]
        dist = np.abs(pts @ nrm + d)
        ninl = int((dist < thresh).sum())
        if ninl > best_in:
            best_in, best = ninl, (nrm, d)
    return best


def deproject(depth_m, K):
    fx, fy, cx, cy = K
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H, 0:W]
    z = depth_m.astype(np.float64)
    valid = np.isfinite(z) & (z > 0.1) & (z < 3.0)
    return valid, np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=-1)


def measure(pipe, align, K, depth_scale, filters, n_frames, patch):
    frames_depth = []
    for _ in range(n_frames):
        fr = align.process(pipe.wait_for_frames())
        df = fr.get_depth_frame()
        if not df:
            continue
        for flt in filters:
            df = flt.process(df)
        frames_depth.append(np.asanyarray(df.as_depth_frame().get_data()).astype(np.float32) * depth_scale)
    if not frames_depth:
        return None
    stack = np.stack(frames_depth)                 # (T,H,W)
    depth_mean = np.nanmean(np.where(stack > 0.1, stack, np.nan), axis=0)

    valid, pts3 = deproject(depth_mean, K)
    P = pts3[valid]
    plane = fit_plane_ransac(P)
    if plane is None:
        return None
    nrm, d = plane
    resid = pts3 @ nrm + d                          # (H,W) 부호있는 평면거리
    H, W = depth_mean.shape

    # patch 단위 spatial std (평면 residual)
    spat = []
    for y in range(0, H - patch, patch):
        for x in range(0, W - patch, patch):
            m = valid[y:y+patch, x:x+patch]
            if m.sum() < patch * patch * 0.6:
                continue
            spat.append(np.std(resid[y:y+patch, x:x+patch][m]))
    spat = np.array(spat) if spat else np.array([np.nan])

    # temporal std (프레임간, 유효 픽셀만)
    tvalid = np.all(stack > 0.1, axis=0)
    temp = np.std(stack[:, tvalid], axis=0) if tvalid.sum() else np.array([np.nan])

    return {
        "dist_p50": float(np.nanmedian(depth_mean[valid])),
        "spatial_std_mm_p50": float(np.nanmedian(spat) * 1000),
        "spatial_std_mm_p90": float(np.nanpercentile(spat, 90) * 1000),
        "temporal_std_mm_p50": float(np.nanmedian(temp) * 1000),
        "plane_rms_mm": float(np.sqrt(np.nanmean(resid[valid] ** 2)) * 1000),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=float, nargs="+", default=[0.5, 0.65, 0.8],
                    help="측정할 대략 거리(m). 각 거리에서 Enter 누르면 측정")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--patch", type=int, default=20)
    ap.add_argument("--cam-w", type=int, default=848)
    ap.add_argument("--cam-h", type=int, default=480)
    ap.add_argument("--preset", default="High Accuracy")
    ap.add_argument("--no-filters", action="store_true")
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.cam_w, args.cam_h, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, args.cam_w, args.cam_h, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    ds = profile.get_device().first_depth_sensor()
    depth_scale = ds.get_depth_scale()
    align = rs.align(rs.stream.color)

    if args.preset.lower() != "none" and ds.supports(rs.option.visual_preset):
        rng_max = int(ds.get_option_range(rs.option.visual_preset).max) + 1
        for i in range(rng_max):
            if ds.get_option_value_description(rs.option.visual_preset, i).lower() == args.preset.lower():
                ds.set_option(rs.option.visual_preset, i)
                break

    filters = []
    if not args.no_filters:
        sp = rs.spatial_filter(); tp = rs.temporal_filter()
        filters = [rs.disparity_transform(True), sp, tp, rs.disparity_transform(False)]

    for _ in range(15):
        pipe.wait_for_frames()

    cf = align.process(pipe.wait_for_frames()).get_color_frame()
    intr = cf.profile.as_video_stream_profile().get_intrinsics()
    K = [intr.fx, intr.fy, intr.ppx, intr.ppy]
    print(f"[depth_noise] {args.cam_w}x{args.cam_h} preset={args.preset} "
          f"filters={'off' if args.no_filters else 'spatial+temporal'}")
    print(f"[depth_noise] 평평한 면을 정면으로. Contact-GraspNet contact 반경 = 5mm 기준\n")

    rows = []
    try:
        for dist in args.distances:
            input(f"  카메라를 평면에서 ~{dist:.2f}m 거리로 두고 Enter... ")
            r = measure(pipe, align, K, depth_scale, filters, args.frames, args.patch)
            if r is None:
                print("    측정 실패 (유효 depth 부족)")
                continue
            rows.append(r)
            print(f"    거리 {r['dist_p50']:.3f}m | spatial std p50 {r['spatial_std_mm_p50']:.2f}mm "
                  f"p90 {r['spatial_std_mm_p90']:.2f}mm | temporal std {r['temporal_std_mm_p50']:.2f}mm "
                  f"| plane RMS {r['plane_rms_mm']:.2f}mm")
    finally:
        pipe.stop()

    print("\n[판단]")
    print("  spatial std p50 < ~2mm, p90 < ~5mm → Contact-GraspNet 5mm 반경에 OK")
    print("  p90 > ~8mm → N프레임 평균 / 필터 강화 필요, 아니면 grasp contact 흔들림")
    if rows:
        worst = max(r["spatial_std_mm_p90"] for r in rows)
        print(f"  이번 측정 최악 spatial std p90 = {worst:.2f}mm "
              + ("→ OK" if worst < 5 else "→ 대응 필요" if worst < 8 else "→ 심각"))


if __name__ == "__main__":
    main()
