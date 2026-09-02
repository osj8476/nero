#!/usr/bin/env python3
"""
test_vlm_feature_additions.py

[2026-09-02 신설] 이번 라운드에 추가한 순수 함수/backend 배선 검증
(ROS·vLLM·로봇 불필요, synthetic 데이터). pytest 없이 단독 실행 가능.

    python tools/test_vlm_feature_additions.py

커버:
  - grasp_types.resolve_grasp_dir : CLAUDE.md 변환표 + z 대역 보정
  - segmentation_backend.DepthPlaneSegmentationBackend : 배경 평면 제외
  - segmentation_backend.get_default_backend : SEG_BACKEND 분기
실기 미검증(실제 카메라/로봇 경로는 별도).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sj_pickplace.grasp_types import resolve_grasp_dir, WAIST_Z_MIN_M, WAIST_Z_MAX_M
from sj_pickplace.segmentation_backend import (
    DepthPlaneSegmentationBackend, NoOpSegmentationBackend,
)

_fail = 0


def check(name, cond):
    global _fail
    print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        _fail += 1


def test_resolve_grasp_dir():
    print("[resolve_grasp_dir]")
    r = resolve_grasp_dir({"grasp_type": "TOP", "orientation": "HORIZONTAL"}, None)
    check("TOP/no-z → top", r.grasp_dir == "top" and not r.top_downgraded_to_side)

    z_mid = (WAIST_Z_MIN_M + WAIST_Z_MAX_M) / 2
    r = resolve_grasp_dir({"grasp_type": "TOP"}, z_mid)
    check("TOP/waist-z → side (downgraded)",
          r.grasp_dir == "side" and r.top_downgraded_to_side)

    r = resolve_grasp_dir({"grasp_type": "TOP"}, WAIST_Z_MAX_M + 0.2)
    check("TOP/high-z → top", r.grasp_dir == "top")

    r = resolve_grasp_dir({"grasp_type": "PINCH", "orientation": "VERTICAL"}, 0.1)
    check("PINCH/VERTICAL → side (세로 원통)", r.grasp_dir == "side")

    r = resolve_grasp_dir({"grasp_type": "PINCH", "orientation": "HORIZONTAL"}, 0.1)
    check("PINCH/HORIZONTAL → pinch", r.grasp_dir == "pinch")

    r = resolve_grasp_dir(
        {"grasp_type": "SIDE", "orientation": "VERTICAL",
         "suggested_side_approach_deg": 90.0}, 0.2)
    check("SIDE + suggested_side_approach_deg 전달", r.grasp_dir == "side"
          and abs(r.side_approach_deg - 90.0) < 1e-6)

    r = resolve_grasp_dir({"grasp_type": "SIDE", "approach_direction": "RIGHT"}, 0.2)
    check("approach_direction=RIGHT → -90°", abs(r.side_approach_deg + 90.0) < 1e-6)

    r = resolve_grasp_dir({"grasp_type": ""}, None)
    check("빈 grasp_type → top 기본값", r.grasp_dir == "top")


def test_depth_plane_backend():
    print("[DepthPlaneSegmentationBackend]")
    h, w = 120, 160
    img = np.zeros((h, w, 3), np.uint8)
    depth = np.full((h, w), 1.6)          # 테이블면: 멀다
    depth[40:90, 50:110] = 0.85           # 물체: 가깝다
    bbox = [45, 35, 115, 95]              # 물체보다 약간 큰 bbox (배경 포함)

    seg = DepthPlaneSegmentationBackend().segment(img, bbox, depth_image=depth)
    noop = NoOpSegmentationBackend().segment(img, bbox)
    check("depth_plane source", seg.source == "depth_plane")
    check("depth_plane < noop (배경 제외)", seg.mask.sum() < noop.mask.sum())
    # 물체 영역(50:110 x 40:90 = 3000)에 가깝고, bbox 전체(70x60=4200)보단 작아야
    check("물체 픽셀 수 ~3000", 2000 <= int(seg.mask.sum()) <= 3400)

    # depth 없으면 NoOp 폴백
    seg2 = DepthPlaneSegmentationBackend().segment(img, bbox, depth_image=None)
    check("depth 없음 → bbox 폴백", seg2.source.startswith("bbox"))


def test_default_backend_env():
    print("[get_default_backend / SEG_BACKEND]")
    # 캐시 때문에 서브프로세스로 각 분기 확인
    import subprocess
    for env, expect in [("noop", "NoOp"), ("depth_plane", "DepthPlane")]:
        out = subprocess.run(
            [sys.executable, "-c",
             "from sj_pickplace.segmentation_backend import get_default_backend;"
             "print(type(get_default_backend()).__name__)"],
            capture_output=True, text=True,
            env={**os.environ, "SEG_BACKEND": env,
                 "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))},
        ).stdout
        check(f"SEG_BACKEND={env} → {expect}*", expect in out)


if __name__ == "__main__":
    test_resolve_grasp_dir()
    test_depth_plane_backend()
    test_default_backend_env()
    print()
    if _fail:
        print(f"❌ {_fail} 실패")
        sys.exit(1)
    print("✅ 전부 통과 (synthetic, 실기 미검증)")
