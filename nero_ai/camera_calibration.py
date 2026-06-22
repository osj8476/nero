#!/usr/bin/env python3
"""
camera_calibration.py
RGB-D 카메라 + 호모그래피 기반 좌표 변환.

[변경점 — RGB-D 버전]
- 단안 평면 가정을 버리고, depth 픽셀값(미터)을 직접 받아 z 결정
- 호모그래피는 여전히 (px, py) → (robot_x, robot_y) 매핑에 사용
- z 는 depth 값에서 카메라-테이블 거리를 빼서 계산
  (TABLE_TO_CAMERA_DIST 환경변수로 보정)

[왜 호모그래피가 여전히 필요한가]
- depth 가 있어도 픽셀 좌표는 카메라 좌표계.
- 로봇 베이스 좌표계로 가려면 (1) 카메라 외부파라미터 또는
  (2) 호모그래피 + depth 의 z 보정 — 후자가 훨씬 간단.
"""

import os
import numpy as np


HOMOGRAPHY_PATH = os.environ.get(
    "HOMOGRAPHY_PATH",
    os.path.expanduser("~/nero_calib.npy"),
)

# 카메라가 정면 부착됐을 때 카메라 ↔ 테이블 평면 거리 (미터).
# 캘리브레이션 시 카메라 위치가 정해지면 이 값을 고정.
TABLE_TO_CAMERA_DIST = float(
    os.environ.get("TABLE_TO_CAMERA_DIST", "0.50"))

# depth 픽셀이 무효한 경우 fallback z (테이블 표면 가정)
TABLE_Z_FALLBACK = float(os.environ.get("TABLE_Z", "0.05"))


# ──────────────────────────────────────────────
# 호모그래피 로드
# ──────────────────────────────────────────────
_H: "np.ndarray | None" = None


def _load_homography() -> bool:
    global _H
    if not os.path.exists(HOMOGRAPHY_PATH):
        print(f"[calib] ⚠️ 호모그래피 파일 없음: {HOMOGRAPHY_PATH}")
        print(f"[calib]   → scripts/calibrate_homography.py 먼저 실행")
        return False
    try:
        _H = np.load(HOMOGRAPHY_PATH)
        if _H.shape != (3, 3):
            print(f"[calib] ⚠️ 잘못된 shape {_H.shape}")
            _H = None
            return False
        print(f"[calib] ✅ 호모그래피 로드: {HOMOGRAPHY_PATH}")
        return True
    except Exception as e:
        print(f"[calib] ⚠️ 로드 실패: {e}")
        _H = None
        return False


_loaded = _load_homography()


# ──────────────────────────────────────────────
# 메인 변환 함수 (RGB-D 버전)
# ──────────────────────────────────────────────
def pixel_to_robot_xyz(
    px: float, py: float,
    img_w: int, img_h: int,
    depth_m: float = None,
) -> dict:
    """
    픽셀 좌표 (px, py) + depth(미터) → 로봇 베이스 좌표 (x, y, z) [미터].

    Args:
        px, py: 이미지 픽셀 좌표
        img_w, img_h: 이미지 크기
        depth_m: 해당 픽셀의 depth 값 (미터). None 또는 0 이면 평면 fallback.

    Returns:
        {"x": ..., "y": ..., "z": ...}  단위: 미터
    """
    # ── x, y: 호모그래피로 매핑 ──
    if _H is not None:
        pt = np.array([px, py, 1.0], dtype=np.float64).reshape(3, 1)
        world = _H @ pt
        world /= world[2, 0]
        x = float(world[0, 0])
        y = float(world[1, 0])
    else:
        # 더미 (캘리브레이션 안 된 상태)
        nx = px / img_w
        ny = py / img_h
        x = 0.30 + (nx - 0.5) * 0.4
        y = (0.5 - ny) * 0.3

    # ── z: depth 값 활용 ──
    if depth_m is not None and 0.05 < depth_m < 2.0:
        # depth 가 정상 범위(5cm ~ 2m) 이내인 경우
        # 카메라가 테이블 위에서 아래를 보고 있다는 가정으로
        # 물체 윗면 높이 = (카메라-테이블 거리) - depth
        # 단, 카메라가 비스듬히 보고 있으면 이 식은 근사값
        z = max(0.0, TABLE_TO_CAMERA_DIST - depth_m)
    else:
        # depth 없으면 평면 가정으로 fallback
        z = TABLE_Z_FALLBACK

    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "z": round(z, 3),
    }


def reload_homography():
    return _load_homography()


def is_calibrated() -> bool:
    return _H is not None
