#!/usr/bin/env python3
"""
test_geometry_3d.py

[신설, 2026-08] synthetic point cloud로 geometry_3d.py(RANSAC 평면적합 +
PCA/OBB + confidence)를 검증한다. ROS/GPU 없이 이 세션에서 직접 실행
가능(numpy만 필요) -- architecture 문서 24번 "독립 테스트" 요구사항의
geometry 단계 전용 회귀 테스트.

pytest 정식 스위트는 아님(repo test/ 디렉토리는 ament 컨벤션을 따름) --
순수 assert 스크립트로 빠르게 확인하는 용도. 실행:
    python3 tools/test_geometry_3d.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sj_pickplace import geometry_3d as g3d

rng = np.random.default_rng(42)


def make_box_points(size=(0.20, 0.08, 0.06), center=(0.0, 0.0, 0.30),
                     yaw_deg=0.0, n=600, noise=0.001):
    """직육면체(책/상자) 표면 점군 -- 6면 중 카메라에서 보이는 3면(top+front+side)만
    생성 (실제 depth 센서가 한쪽에서만 보는 상황 재현)."""
    L, W, H = size
    yaw = math.radians(yaw_deg)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw),  math.cos(yaw), 0],
                  [0, 0, 1]])
    pts = []
    n_per_face = n // 3
    for _ in range(n_per_face):
        pts.append(np.array([rng.uniform(-L/2, L/2), rng.uniform(-W/2, W/2), H/2]))
    for _ in range(n_per_face):
        pts.append(np.array([rng.uniform(-L/2, L/2), -W/2, rng.uniform(-H/2, H/2)]))
    for _ in range(n - 2 * n_per_face):
        pts.append(np.array([L/2, rng.uniform(-W/2, W/2), rng.uniform(-H/2, H/2)]))
    pts = np.array(pts)
    pts = pts @ R.T + np.array(center) + rng.normal(0, noise, size=pts.shape)
    return pts


def make_handle_points(length=0.12, radius=0.01, center=(0.0, 0.0, 0.30),
                        yaw_deg=0.0, n=300, noise=0.0008):
    """가늘고 긴 원통형 손잡이 점군 -- major axis가 손잡이 긴 방향과 일치해야 함."""
    yaw = math.radians(yaw_deg)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw),  math.cos(yaw), 0],
                  [0, 0, 1]])
    t = rng.uniform(-length/2, length/2, size=n)
    theta = rng.uniform(-math.pi/2, math.pi/2, size=n)  # 카메라에서 보이는 절반만
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    pts = np.stack([t, y, z], axis=1)
    pts = pts @ R.T + np.array(center) + rng.normal(0, noise, size=pts.shape)
    return pts


passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {detail}")


def main():
    print("=== Case 1: 박스(직육면체), yaw=0deg ===")
    pts = make_box_points(yaw_deg=0.0)
    geo = g3d.compute_geometry(pts)
    check("valid", geo.valid)
    check("point_count == 600", geo.point_count == 600, geo.point_count)
    check("geometry_confidence > 0.5", geo.geometry_confidence > 0.5, geo.geometry_confidence)
    check("plane_inlier_ratio > 0.25 (한 면이 전체의 1/3 근처)", geo.plane_inlier_ratio > 0.25,
          geo.plane_inlier_ratio)
    maj_yaw = geo.major_axis_yaw_deg
    check("major_axis_yaw_deg near 0 or 180", maj_yaw is not None and (maj_yaw < 10 or maj_yaw > 170),
          maj_yaw)

    print("\n=== Case 2: 박스, yaw=40deg 회전 ===")
    pts2 = make_box_points(yaw_deg=40.0)
    geo2 = g3d.compute_geometry(pts2)
    check("valid", geo2.valid)
    maj_yaw2 = geo2.major_axis_yaw_deg
    check("major_axis_yaw_deg near 40 (mod180)", maj_yaw2 is not None and abs(maj_yaw2 - 40.0) < 10.0,
          maj_yaw2)

    print("\n=== Case 3: 가늘고 긴 손잡이(handle), yaw=110deg ===")
    pts3 = make_handle_points(yaw_deg=110.0)
    geo3 = g3d.compute_geometry(pts3, min_inliers=20)
    check("valid", geo3.valid)
    maj_yaw3 = geo3.major_axis_yaw_deg
    check("major_axis_yaw_deg near 110 (mod180)", maj_yaw3 is not None and abs(maj_yaw3 - 110.0) < 10.0,
          maj_yaw3)
    check("eigenvalue separation 큼(원통형이라 major >> minor,third)",
          geo3.eigenvalues[0] > geo3.eigenvalues[1] * 5, geo3.eigenvalues)

    print("\n=== Case 4: 점 부족 (min_inliers 미달) ===")
    few_pts = rng.normal(0, 0.01, size=(10, 3)) + np.array([0, 0, 0.3])
    geo4 = g3d.compute_geometry(few_pts, min_inliers=30)
    check("valid == False", geo4.valid == False)
    check("point_count == 10", geo4.point_count == 10)

    print("\n=== Case 5: 완전 랜덤 블롭(비평면) -- confidence 낮아야 함 ===")
    blob = rng.normal(0, 0.05, size=(300, 3)) + np.array([0, 0, 0.3])
    geo5 = g3d.compute_geometry(blob, min_inliers=30)
    check("valid", geo5.valid)
    check("geometry_confidence < 0.5 (구형 블롭)", geo5.geometry_confidence < 0.5, geo5.geometry_confidence)

    print("\n=== Case 6: fit_dominant_planes 다중 평면 (책 2권 서로 다른 각도로 인접) ===")
    box_a = make_box_points(center=(0.0, 0.0, 0.30), yaw_deg=0.0, n=300)
    box_b = make_box_points(center=(0.25, 0.0, 0.30), yaw_deg=60.0, n=300)
    combined = np.vstack([box_a, box_b])
    planes = g3d.fit_dominant_planes(combined, distance_threshold=0.005, min_inliers=30, max_planes=3)
    check("최소 2개 평면 검출", len(planes) >= 2, len(planes))
    check("평면들이 inlier_count 내림차순",
          all(planes[i].inlier_count >= planes[i+1].inlier_count for i in range(len(planes)-1)))

    print(f"\n{'='*50}\n결과: {passed} passed, {failed} failed\n{'='*50}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
