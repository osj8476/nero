#!/usr/bin/env python3
"""
test_grasp_geometry_pipeline.py

[신설, 2026-08] Segmentation(NoOp) → Point Cloud → 3D Geometry(RANSAC+PCA) →
Learned Grasp Detector(FallbackGeometryBackend) → Semantic Filtering →
GraspCandidate 전체를 로봇/ROS 없이 synthetic 데이터로 검증하는 오프라인
테스트 (architecture 문서 24번 "독립 테스트" 요구사항 대응).

실제 카메라/depth 대신 numpy로 만든 가짜 depth map + bbox를 쓴다 -- 이 세션
환경(GPU/ROS 없음)에서도 전체 파이프라인의 배선과 각 단계 출력을 확인할 수
있게 하기 위함. 실제 RealSense/Isaac Sim depth로 재현하는 건 로봇 쪽 세션의
몫으로 남는다(README 참고).

실행:
    python3 tools/test_grasp_geometry_pipeline.py
    python3 tools/test_grasp_geometry_pipeline.py --debug   # 각 단계 상세 로그
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sj_pickplace.segmentation_backend import NoOpSegmentationBackend
from sj_pickplace.point_cloud import Intrinsics, mask_depth_to_pointcloud
from sj_pickplace import geometry_3d as g3d
from sj_pickplace.learned_grasp_backend import FallbackGeometryBackend
from sj_pickplace.grasp_types import GraspIntent
from sj_pickplace import grasp_pose_generator as gpg


def make_synthetic_scene(handle_yaw_deg=35.0, W=640, H=480):
    """가짜 depth map 하나 생성 -- 화면 중앙에 가늘고 긴 손잡이 모양 물체
    (원통 옆면 일부)가 놓인 상황을 흉내낸다. 카메라 intrinsics는 RealSense
    640x480 color stream 전형값(fx=fy~600, cx=320, cy=240 근사) 사용."""
    intr = Intrinsics(fx=605.0, fy=605.0, cx=320.0, cy=240.0)
    depth = np.full((H, W), 0.0, dtype=np.float32)   # 0 = invalid(배경)

    length, radius, dist = 0.12, 0.015, 0.35
    yaw = math.radians(handle_yaw_deg)
    n = 4000
    rng = np.random.default_rng(7)
    t = rng.uniform(-length / 2, length / 2, size=n)
    theta = rng.uniform(-math.pi / 2, math.pi / 2, size=n)  # 카메라에서 보이는 절반만
    lx = t * math.cos(yaw) - (radius * np.cos(theta)) * math.sin(yaw)
    ly = t * math.sin(yaw) + (radius * np.cos(theta)) * math.cos(yaw)
    lz = radius * np.sin(theta) + dist   # depth(카메라 Z) 기준 중심 dist

    u = intr.cx + lx * intr.fx / lz
    v = intr.cy + ly * intr.fy / lz
    for uu, vv, zz in zip(u, v, lz):
        px, py = int(round(uu)), int(round(vv))
        if 0 <= px < W and 0 <= py < H:
            depth[py, px] = zz

    xs = u[(u >= 0) & (u < W)]
    ys = v[(v >= 0) & (v < H)]
    bbox_px = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return depth, intr, bbox_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--handle-yaw-deg", type=float, default=35.0,
                    help="synthetic 손잡이의 실제 회전각(도) -- 파이프라인이 이 값을 복원하는지 확인")
    args = ap.parse_args()

    t_start = time.time()
    timings = {}

    print(f"{'='*60}\n합성 시나리오: 손잡이(handle) yaw={args.handle_yaw_deg}도\n{'='*60}")

    t0 = time.time()
    depth, intr, bbox_px = make_synthetic_scene(args.handle_yaw_deg)
    fake_image = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    timings["scene_setup_ms"] = (time.time() - t0) * 1000

    # ── 1. Segmentation (NoOp -- bbox를 그대로 마스크로) ──────────────────
    t0 = time.time()
    seg_backend = NoOpSegmentationBackend()
    seg_result = seg_backend.segment(fake_image, bbox_px, target_part="handle")
    timings["segmentation_ms"] = (time.time() - t0) * 1000
    assert seg_result is not None, "segmentation 실패"
    print(f"[1] Segmentation: bbox_px={seg_result.bbox_px} source={seg_result.source} "
          f"({timings['segmentation_ms']:.2f}ms)")

    # ── 2. Point Cloud ────────────────────────────────────────────────────
    t0 = time.time()
    points = mask_depth_to_pointcloud(depth, seg_result.mask, intr, max_points=1500)
    timings["pointcloud_ms"] = (time.time() - t0) * 1000
    print(f"[2] Point Cloud: {len(points)}점 ({timings['pointcloud_ms']:.2f}ms)")
    assert len(points) > 30, f"point cloud 점 부족: {len(points)}"

    # ── 3. 3D Geometry (RANSAC + PCA) ────────────────────────────────────
    t0 = time.time()
    geometry = g3d.compute_geometry(points, min_inliers=20)
    timings["geometry_ms"] = (time.time() - t0) * 1000
    print(f"[3] Geometry: valid={geometry.valid} confidence={geometry.geometry_confidence} "
          f"major_axis_yaw_deg={geometry.major_axis_yaw_deg} "
          f"plane_inlier_ratio={geometry.plane_inlier_ratio:.2f} "
          f"({timings['geometry_ms']:.2f}ms)")
    if args.debug:
        print(f"     centroid={tuple(round(v,3) for v in geometry.centroid)}")
        print(f"     eigenvalues={tuple(round(v,6) for v in geometry.eigenvalues)}")
        print(f"     extents={tuple(round(v,3) for v in geometry.extents)}")

    expected_mod180 = args.handle_yaw_deg % 180.0
    yaw_err = None
    if geometry.major_axis_yaw_deg is not None:
        diff = abs(geometry.major_axis_yaw_deg - expected_mod180)
        yaw_err = min(diff, 180 - diff)
        status = "OK" if yaw_err < 8.0 else "WARN(오차 큼)"
        print(f"     [검증] 기대각={expected_mod180:.1f}도, 복원각={geometry.major_axis_yaw_deg}도, "
              f"오차={yaw_err:.1f}도  [{status}]")

    # ── 4. Learned Grasp Detector (FallbackGeometryBackend) ──────────────
    t0 = time.time()
    detector = FallbackGeometryBackend()
    intent = GraspIntent(grasp_type="SIDE", orientation="HORIZONTAL", confidence=0.9,
                         target_part="handle", grasp_relation="perpendicular_to_long_axis",
                         action="PULL", action_direction="opposite_approach", source="vlm")
    learned_outputs = detector.predict(points, geometry_features=geometry, grasp_intent=intent)
    timings["learned_grasp_ms"] = (time.time() - t0) * 1000
    print(f"[4] Learned Grasp Detector({detector.name}): {len(learned_outputs)}개 후보 "
          f"({timings['learned_grasp_ms']:.2f}ms)")
    assert len(learned_outputs) > 0, "learned grasp 후보 0개"

    # ── 5. Candidate 생성 + Semantic Filtering ───────────────────────────
    t0 = time.time()
    pos = {'x': 0.30, 'y': 0.10, 'z': 0.30}
    candidates = gpg.generate_candidates(learned_outputs, geometry, intent, pos, use_moveit2=True)
    timings["candidate_filter_ms"] = (time.time() - t0) * 1000
    print(f"[5] Candidate 생성+랭킹: {len(candidates)}개 ({timings['candidate_filter_ms']:.2f}ms)")
    for i, c in enumerate(candidates):
        print(f"     #{i} total={c.total_score:.3f} (geo={c.geometry_score:.3f} "
              f"sem={c.semantic_score:.3f}) is_side={c.is_side} dir={c.grasp_dir_hint} "
              f"quat={[round(v,3) for v in c.quaternion]}")
    assert candidates[0].total_score >= candidates[-1].total_score, "랭킹 정렬 오류"

    best = candidates[0]
    print(f"     [선택] source={best.source} approach_vector="
          f"{tuple(round(v,3) for v in best.approach_vector)}")

    # ── 6. Action Vector (PULL 방향) ──────────────────────────────────────
    t0 = time.time()
    action_vec = gpg.generate_action_vector(intent, geometry, best)
    timings["action_vector_ms"] = (time.time() - t0) * 1000
    print(f"[6] Action Vector(action={intent.action}, direction={intent.action_direction}): "
          f"{tuple(round(v,3) for v in action_vec) if action_vec else None} "
          f"({timings['action_vector_ms']:.2f}ms)")
    assert action_vec is not None

    total_ms = (time.time() - t_start) * 1000
    print(f"\n{'='*60}\n전체 파이프라인: {total_ms:.2f}ms")
    for k, v in timings.items():
        print(f"  {k:22s}: {v:7.2f}ms")
    print(f"{'='*60}")
    print("\n[결론] segmentation(NoOp) -> point_cloud -> geometry_3d(RANSAC+PCA) -> "
          "learned_grasp_backend(fallback) -> grasp_pose_generator(candidate+semantic+action) "
          "전 구간 배선 정상 동작 확인 (synthetic 데이터, 실기 미검증).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
