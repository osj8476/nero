#!/usr/bin/env python3
"""
charuco_hand_eye_calib.py  — ChArUco 기반 Eye-in-Hand 캘리브레이션 시스템
==========================================================================

목적:
    NERO 7-DoF Eye-in-Hand RGB-D 카메라의 Camera ↔ Gripper 변환(T_cam2gripper)을
    ChArUco board로 정확하게 구하고, URDF d435_camera_joint를 자동 업데이트하여
    TF2 파이프라인이 실제 장착 기하에 맞도록 한다.

    목표 정확도(독립 검증 기준):
        X STD ≤ 5 mm,  Y STD ≤ 5 mm,  Z STD ≤ 5 mm

사용법:
    # 캘리브레이션 (board 바닥 수평 고정, robot 자동 이동)
    python3 -m sj_pickplace.scripts.charuco_hand_eye_calib --mode calibrate

    # dry-run (robot 이동 없이 계획 자세만 출력)
    python3 -m sj_pickplace.scripts.charuco_hand_eye_calib --mode calibrate --dry-run

    # 독립 검증
    python3 -m sj_pickplace.scripts.charuco_hand_eye_calib --mode validate

    # ChArUco board PNG 생성
    python3 -m sj_pickplace.scripts.charuco_hand_eye_calib --mode generate-board

    # RViz 시각화 (live detection preview)
    python3 -m sj_pickplace.scripts.charuco_hand_eye_calib --mode visualize

프레임 규약 (Frame Convention):
─────────────────────────────────────────────────────────────────────────
    T_A_B : A 프레임에서 바라본 B 프레임의 위치/자세
            (점 변환: p_A = T_A_B @ p_B)

    T_gripper_base : gripper_flange 프레임이 base_link에서 표현된 변환
                     (= /feedback/tcp_pose PoseStamped)
    T_target_cam   : ChArUco target 프레임이 camera_color_optical_frame에서 표현된 변환
                     (= cv2.estimatePoseCharucoBoard 결과)
    T_cam_gripper  : camera 프레임이 gripper_flange에서 표현된 변환
                     (= cv2.calibrateHandEye 출력 R_cam2gripper, t_cam2gripper)
                     JSON 파일의 "T_flange_camera" 키에 저장 (하위호환)

    OpenCV calibrateHandEye 규약:
        Inputs:
            R_gripper2base, t_gripper2base  →  T_gripper_base
            R_target2cam,   t_target2cam    →  T_target_cam
        Output:
            R_cam2gripper, t_cam2gripper    →  T_cam_gripper
        수식: T_gripper_base @ T_cam_gripper = T_cam_gripper @ T_target_cam

    내부 검증 (base 좌표계 target 위치):
        T_target_in_base_est =
            T_gripper_base          # gripper→base
            @ T_cam_gripper         # camera→gripper
            @ T_target_cam          # target→camera

    독립 검증 오차:
        error_xyz = T_target_in_base_est - T_target_in_base_gt

URDF d435_camera_joint 업데이트:
─────────────────────────────────────────────────────────────────────────
    체인: gripper_flange → gripper_base → camera_stand_link
          → [d435_camera_joint] → d435_camera_link
          → camera_bottom_screw → camera_link → camera_color_frame
          → camera_color_optical_frame

    T_cam_gripper = A @ T_X @ B
        A = T_gripper_base_in_flange @ T_camera_stand_in_gripper_base   (고정)
        B = T_d435_bottom@T_camera_link@T_color_frame@T_optical         (고정)
        T_X = T_d435_in_camera_stand  →  업데이트 대상 (d435_camera_joint origin)

    해: T_X = inv(A) @ T_cam_gripper @ inv(B)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as RosDuration
from std_msgs.msg import ColorRGBA

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    from cv_bridge import CvBridge
    _bridge = CvBridge()
except ImportError:
    _bridge = None

import json as _json

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
CALIB_DIR   = os.path.expanduser(os.environ.get("NERO_CALIB_DIR", "~/.nero_calib"))
OUTPUT_JSON = os.path.join(CALIB_DIR, "flange_to_camera.json")

# URDF 경로 (자동 업데이트 시 사용)
URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nero_with_camera.urdf"
)

# ─── 카메라 설정 ─────────────────────────────────────────────────────────────
CAM_W, CAM_H, CAM_FPS = 640, 480, 30

# ─── 로봇 관절 이름 ──────────────────────────────────────────────────────────
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']

# ─── 이동/안정화 타이밍 ───────────────────────────────────────────────────────
SETTLE_SEC    = 2.0     # 관절 도착 후 진동 안정화 대기 (초)
MOVE_TIMEOUT  = 20.0    # 단일 이동 최대 대기 (초)

# ─── 캘리브레이션 최소 요건 ──────────────────────────────────────────────────
MIN_CALIB_SAMPLES = 8   # 캘리브레이션 최소 샘플 수

# ─── Hand-eye solver 방법 ────────────────────────────────────────────────────
SOLVER_METHODS = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ChArUco Board 설정
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoardConfig:
    """ChArUco board 파라미터 — 실제 출력 물리 크기와 반드시 일치해야 한다."""
    squares_x:     int   = 10       # 가로 칸 수
    squares_y:     int   = 7        # 세로 칸 수
    square_length: float = 0.040    # 체스보드 칸 크기 (m) - 실측값으로 수정
    marker_length: float = 0.028    # ArUco marker 크기 (m) - 실측값으로 수정
    dictionary:    str   = "DICT_4X4_50"  # ArUco dictionary 이름

    def to_cv2_dict(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))

    def create_board(self) -> cv2.aruco.CharucoBoard:
        """OpenCV 4.6 호환 CharucoBoard_create API 사용."""
        d = self.to_cv2_dict()
        return cv2.aruco.CharucoBoard_create(
            self.squares_x, self.squares_y,
            self.square_length, self.marker_length, d
        )

    def total_inner_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def board_physical_size_mm(self) -> Tuple[float, float]:
        return (self.squares_x * self.square_length * 1000,
                self.squares_y * self.square_length * 1000)


@dataclass
class QualityGate:
    """샘플 품질 기준."""
    min_charuco_corners:  int   = 12    # 최소 ChArUco 코너 수 (전체의 약 20%)
    max_reproj_error:     float = 1.5   # 최대 재투영 오차 (픽셀)
    min_board_area_ratio: float = 0.05  # 화면 대비 board 최소 면적 비율
    max_board_area_ratio: float = 0.95  # board가 너무 화면을 차지하면 기각
    margin_fraction:      float = 0.05  # 화면 가장자리 margin (board 잘림 감지)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 행렬 유틸리티 (URDF 규약 명확화)
# ══════════════════════════════════════════════════════════════════════════════

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF RPY → 회전 행렬.  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """회전 행렬 → (roll, pitch, yaw) in URDF 규약."""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


def urdf_joint_T(xyz: List[float], rpy: List[float]) -> np.ndarray:
    """URDF <origin xyz rpy> → 4x4 T_child_in_parent.
    T_child_in_parent: 점을 child 좌표계 → parent 좌표계로 변환.
    """
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy[0], rpy[1], rpy[2])
    T[:3, 3]  = xyz
    return T


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    """단위 쿼터니언 → 회전 행렬."""
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def R_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """회전 행렬 → (x, y, z, w) 쿼터니언."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def quat_mul(a: Tuple, b: Tuple) -> Tuple:
    """쿼터니언 곱 (ax,ay,az,aw) * (bx,by,bz,bw)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. URDF d435_camera_joint 업데이트 (체인 역계산)
# ══════════════════════════════════════════════════════════════════════════════

def compute_d435_joint_origin(T_cam_gripper: np.ndarray
                               ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    T_cam_gripper(camera→gripper_flange) 로부터 d435_camera_joint의 올바른
    origin (xyz, rpy)를 계산한다.

    URDF 체인:
        gripper_flange
          → gripper_base_joint (rpy=0,0,0  xyz=0,0,0.0055)   → gripper_base
          → camera_stand_joint (rpy=0,0,π  xyz=0.034,0.001,0.022) → camera_stand_link
          → [d435_camera_joint]  ← UPDATE THIS                → d435_camera_link
          → camera_joint        (rpy=0,0,0  xyz=0,0,0)       → camera_bottom_screw
          → camera_link_joint   (rpy=0,0,0  xyz=0.0106,0.0175,0.0125) → camera_link
          → camera_color_joint  (rpy=0,0,0  xyz=0,0.015,0)   → camera_color_frame
          → camera_color_optical_joint (rpy=-π/2,0,-π/2 xyz=0,0,0) → camera_color_optical_frame

    T_cam_gripper = A @ T_X @ B
    where T_X = d435_camera_joint 변환, A / B = 고정 변환

    → T_X = inv(A) @ T_cam_gripper @ inv(B)
    """
    # A: gripper_flange → camera_stand_link (고정)
    T_gb_in_gf = urdf_joint_T([0, 0, 0.0055], [0, 0, 0])
    T_cs_in_gb = urdf_joint_T([0.034, 0.001, 0.022], [0, 0, math.pi])
    A = T_gb_in_gf @ T_cs_in_gb

    # B: d435_camera_link → camera_color_optical_frame (고정)
    T_bs_in_d4 = urdf_joint_T([0, 0, 0], [0, 0, 0])            # camera_joint
    T_cl_in_bs = urdf_joint_T([0.0106, 0.0175, 0.0125], [0, 0, 0])  # camera_link_joint
    T_cf_in_cl = urdf_joint_T([0, 0.015, 0], [0, 0, 0])         # camera_color_joint
    T_co_in_cf = urdf_joint_T([0, 0, 0], [-math.pi/2, 0, -math.pi/2])  # camera_color_optical_joint
    B = T_bs_in_d4 @ T_cl_in_bs @ T_cf_in_cl @ T_co_in_cf

    T_X = np.linalg.inv(A) @ T_cam_gripper @ np.linalg.inv(B)

    xyz = T_X[:3, 3]
    rpy = R_to_rpy(T_X[:3, :3])
    return xyz, rpy


def update_urdf_d435_joint(urdf_path: str,
                            xyz: np.ndarray,
                            rpy: Tuple[float, float, float]) -> bool:
    """
    nero_with_camera.urdf 의 d435_camera_joint origin 을 업데이트한다.
    성공하면 True, 실패하면 False.
    """
    if not os.path.exists(urdf_path):
        print(f'[URDF] 파일 없음: {urdf_path}')
        return False

    with open(urdf_path, 'r') as f:
        content = f.read()

    xyz_str = f'{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}'
    rpy_str = f'{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}'
    new_origin = f'xyz="{xyz_str}" rpy="{rpy_str}"'

    # d435_camera_joint 블록 내 origin 교체
    pattern = (
        r'(joint\s+name="d435_camera_joint"[^>]*>.*?<origin\s+)'
        r'(?:xyz="[^"]*"\s+rpy="[^"]*"|rpy="[^"]*"\s+xyz="[^"]*")'
    )
    new_content, count = re.subn(
        pattern, r'\g<1>' + new_origin, content, count=1, flags=re.DOTALL
    )
    if count == 0:
        print('[URDF] d435_camera_joint origin 패턴을 찾지 못했습니다. 수동으로 반영하세요:')
        print(f'       <origin xyz="{xyz_str}" rpy="{rpy_str}" />')
        return False

    # 백업 저장
    backup = urdf_path + '.bak'
    with open(backup, 'w') as f:
        f.write(content)

    with open(urdf_path, 'w') as f:
        f.write(new_content)

    print(f'[URDF] d435_camera_joint 업데이트 완료 (백업: {backup})')
    print(f'       xyz = {xyz_str}')
    print(f'       rpy = {rpy_str}')
    print('       → robot_state_publisher 재시작 후 TF2 적용됩니다.')
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 4. ChArUco 검출기
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """단일 프레임 ChArUco 검출 결과."""
    marker_corners: list        # ArUco marker corners
    marker_ids:     Optional[np.ndarray]
    charuco_corners: Optional[np.ndarray]  # (N, 1, 2) float32
    charuco_ids:     Optional[np.ndarray]  # (N, 1) int32
    n_markers:  int = 0
    n_corners:  int = 0
    rvec:        Optional[np.ndarray] = None  # (3,) 또는 (3,1)
    tvec:        Optional[np.ndarray] = None  # (3,)
    reproj_err:  float = float('inf')
    pose_valid:  bool = False
    vis_img:     Optional[np.ndarray] = None  # 시각화용 overlay


class CharucoDetector:
    """OpenCV 4.6 호환 ChArUco 검출 + pose estimation."""

    def __init__(self, board_cfg: BoardConfig,
                 K: np.ndarray, dist: np.ndarray,
                 quality: QualityGate):
        self.cfg     = board_cfg
        self.K       = K
        self.dist    = dist
        self.quality = quality
        self.board   = board_cfg.create_board()
        self.aruco_dict = board_cfg.to_cv2_dict()
        self.det_params = cv2.aruco.DetectorParameters_create()
        # 서브픽셀 정확도 향상을 위한 파라미터
        self.det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def detect(self, img: np.ndarray) -> DetectionResult:
        """이미지에서 ChArUco marker와 corner를 검출하고 pose를 추정한다."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # 1. ArUco marker 검출
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.det_params)

        n_markers = len(corners) if corners is not None else 0
        result = DetectionResult(
            marker_corners=corners or [],
            marker_ids=ids,
            charuco_corners=None,
            charuco_ids=None,
            n_markers=n_markers,
        )

        if n_markers < 4:
            result.vis_img = self._draw_status(img.copy(), result)
            return result

        # 2. ChArUco corner interpolation
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board,
            cameraMatrix=self.K, distCoeffs=self.dist
        )

        result.charuco_corners = ch_corners
        result.charuco_ids = ch_ids
        result.n_corners = retval if retval is not None else 0

        if result.n_corners < self.quality.min_charuco_corners:
            result.vis_img = self._draw_status(img.copy(), result)
            return result

        # 3. Pose estimation
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, self.board, self.K, self.dist, None, None)

        if not ok or rvec is None or tvec is None:
            result.vis_img = self._draw_status(img.copy(), result)
            return result

        result.rvec = rvec.reshape(3)
        result.tvec = tvec.reshape(3)

        # 4. 재투영 오차 계산
        result.reproj_err = self._compute_reproj_error(ch_corners, ch_ids, rvec, tvec)
        result.pose_valid = (
            result.reproj_err < self.quality.max_reproj_error and
            result.n_corners >= self.quality.min_charuco_corners and
            self._board_not_clipped(ch_corners, img.shape)
        )

        result.vis_img = self._draw_status(img.copy(), result)
        return result

    def _compute_reproj_error(self, ch_corners, ch_ids, rvec, tvec) -> float:
        """ChArUco corner 재투영 오차 (픽셀 RMS)."""
        if ch_corners is None or ch_ids is None:
            return float('inf')
        obj_pts = np.array(
            [self.board.chessboardCorners[cid[0]] for cid in ch_ids], dtype=np.float32)
        img_pts_proj, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, self.K, self.dist)
        errors = np.linalg.norm(
            ch_corners.reshape(-1, 2) - img_pts_proj.reshape(-1, 2), axis=1)
        return float(np.sqrt(np.mean(errors**2)))

    def _board_not_clipped(self, ch_corners: np.ndarray, shape: tuple) -> bool:
        """board가 화면에서 지나치게 잘리지 않았는지 확인."""
        if ch_corners is None:
            return False
        h, w = shape[:2]
        m = self.quality.margin_fraction
        pts = ch_corners.reshape(-1, 2)
        return bool(
            np.all(pts[:, 0] > w * m) and np.all(pts[:, 0] < w * (1 - m)) and
            np.all(pts[:, 1] > h * m) and np.all(pts[:, 1] < h * (1 - m))
        )

    def _draw_status(self, img: np.ndarray, r: DetectionResult) -> np.ndarray:
        """검출 상태 overlay 이미지 생성."""
        # ArUco marker 표시
        if r.n_markers > 0:
            cv2.aruco.drawDetectedMarkers(img, r.marker_corners, r.marker_ids)

        # ChArUco corner 표시
        if r.n_corners > 0 and r.charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                img, r.charuco_corners, r.charuco_ids)

        # Pose axis 표시
        if r.pose_valid and r.rvec is not None:
            cv2.drawFrameAxes(img, self.K, self.dist,
                              r.rvec, r.tvec, self.cfg.square_length * 2)

        # 상태 텍스트
        if r.pose_valid:
            status = (f"[VALID] corners:{r.n_corners}  "
                      f"reproj:{r.reproj_err:.2f}px")
            bg_color = (0, 160, 0)
        elif r.n_corners >= self.quality.min_charuco_corners:
            status = (f"[POOR QUALITY] corners:{r.n_corners}  "
                      f"reproj:{r.reproj_err:.2f}px (>{self.quality.max_reproj_error})")
            bg_color = (0, 120, 200)
        elif r.n_markers >= 4:
            status = f"[PARTIAL] markers:{r.n_markers}  corners:{r.n_corners}"
            bg_color = (0, 100, 180)
        else:
            status = f"[NOT DETECTED] markers:{r.n_markers}"
            bg_color = (0, 0, 160)

        cv2.rectangle(img, (0, 0), (img.shape[1], 36), bg_color, -1)
        cv2.putText(img, status, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return img


# ══════════════════════════════════════════════════════════════════════════════
# 5. 캘리브레이션 자세 정의 (Cartesian 커버리지 중심)
# ══════════════════════════════════════════════════════════════════════════════

# [캘리브레이션 자세 설계 원칙]
# 1. 목적: joint angle diversity가 아니라 Camera/TCP Cartesian 위치 다양성
# 2. 기준: NERO DEFAULT_SCAN_BASE_JOINTS (top-down 스캔 기본 자세)
#          j1=0, j2=-0.8026, j3=0, j4=1.8273, j5=0, j6=0, j7=1.3
# 3. XY 커버리지: joint1 변화 (-0.35, 0, 0.35 rad → 좌/중/우 ≈ -20°/0°/+20°)
# 4. Z 커버리지: joint2 변화 (-0.70/-0.80/-0.90 → 높은/중간/낮은 높이)
# 5. 자세 다양성: joint7 변화 (0.9/1.3/1.7 → wrist 회전 -23°/0°/+23°)
# 6. 추가 orientation: joint3 소폭 변화 (reach 변화)
# 7. 총 27 calibration + 12 validation = 39 poses
#
# ⚠️ 안전 주의사항:
# - 모든 자세는 실제 NERO에서 top-down 구성으로 검증된 값 기반
# - ChArUco board는 robot 정면 작업 영역 바닥에 수평으로 고정
# - 첫 실행 시 반드시 --dry-run으로 계획 자세 확인 후 실행
# - board가 모든 자세에서 camera FOV 내에 있어야 함
# - singularity 방지: joint2 = -1.0 이하 주의

_D = math.radians

def _make_pose(j1=0, j2=-0.8026, j3=0, j4=1.8273, j5=0, j6=0, j7=1.3) -> list:
    return [j1, j2, j3, j4, j5, j6, j7]

# ── A 그룹: 중간 높이 (j2=-0.80), 3x3 XY grid, 3 wrist ──────────────────────
# joint1: -0.35=좌, 0=중, 0.35=우 / joint7: 0.9/1.3/1.7 wrist 회전
_A = [
    _make_pose(j1=-0.35, j2=-0.80, j7=0.9),   # A01
    _make_pose(j1=-0.35, j2=-0.80, j7=1.3),   # A02
    _make_pose(j1=-0.35, j2=-0.80, j7=1.7),   # A03
    _make_pose(j1= 0.00, j2=-0.80, j7=0.9),   # A04
    _make_pose(j1= 0.00, j2=-0.80, j7=1.3),   # A05 (기본 자세)
    _make_pose(j1= 0.00, j2=-0.80, j7=1.7),   # A06
    _make_pose(j1= 0.35, j2=-0.80, j7=0.9),   # A07
    _make_pose(j1= 0.35, j2=-0.80, j7=1.3),   # A08
    _make_pose(j1= 0.35, j2=-0.80, j7=1.7),   # A09
]

# ── B 그룹: 낮은 높이 (j2=-0.90), 3x3 XY grid, 3 wrist ──────────────────────
_B = [
    _make_pose(j1=-0.35, j2=-0.90, j7=0.9),   # B01
    _make_pose(j1=-0.35, j2=-0.90, j7=1.3),   # B02
    _make_pose(j1=-0.35, j2=-0.90, j7=1.7),   # B03
    _make_pose(j1= 0.00, j2=-0.90, j7=0.9),   # B04
    _make_pose(j1= 0.00, j2=-0.90, j7=1.3),   # B05
    _make_pose(j1= 0.00, j2=-0.90, j7=1.7),   # B06
    _make_pose(j1= 0.35, j2=-0.90, j7=0.9),   # B07
    _make_pose(j1= 0.35, j2=-0.90, j7=1.3),   # B08
    _make_pose(j1= 0.35, j2=-0.90, j7=1.7),   # B09
]

# ── C 그룹: reach 변화 (j3=-0.20), 중간 높이, 3x3 XY grid ───────────────────
_C = [
    _make_pose(j1=-0.35, j2=-0.80, j3=-0.20, j7=0.9),  # C01
    _make_pose(j1=-0.35, j2=-0.80, j3=-0.20, j7=1.3),  # C02
    _make_pose(j1=-0.35, j2=-0.80, j3=-0.20, j7=1.7),  # C03
    _make_pose(j1= 0.00, j2=-0.80, j3=-0.20, j7=0.9),  # C04
    _make_pose(j1= 0.00, j2=-0.80, j3=-0.20, j7=1.3),  # C05
    _make_pose(j1= 0.00, j2=-0.80, j3=-0.20, j7=1.7),  # C06
    _make_pose(j1= 0.35, j2=-0.80, j3=-0.20, j7=0.9),  # C07
    _make_pose(j1= 0.35, j2=-0.80, j3=-0.20, j7=1.3),  # C08
    _make_pose(j1= 0.35, j2=-0.80, j3=-0.20, j7=1.7),  # C09
]

CALIB_POSES: List[List[float]] = _A + _B + _C  # 27 poses total

# ── 독립 검증용 자세 (캘리브레이션에 사용하지 않은 새로운 위치/자세) ──────────
# 중간 높이에서 calibration과 다른 XY/wrist 조합
VALIDATION_POSES: List[List[float]] = [
    _make_pose(j1=-0.20, j2=-0.85, j7=1.1),   # V01
    _make_pose(j1=-0.20, j2=-0.85, j7=1.5),   # V02
    _make_pose(j1= 0.00, j2=-0.85, j7=1.1),   # V03
    _make_pose(j1= 0.00, j2=-0.85, j7=1.5),   # V04
    _make_pose(j1= 0.20, j2=-0.85, j7=1.1),   # V05
    _make_pose(j1= 0.20, j2=-0.85, j7=1.5),   # V06
    _make_pose(j1=-0.35, j2=-0.75, j7=1.3),   # V07 (높은 위치)
    _make_pose(j1= 0.00, j2=-0.75, j7=1.0),   # V08 (높은 위치)
    _make_pose(j1= 0.35, j2=-0.75, j7=1.6),   # V09 (높은 위치)
    _make_pose(j1=-0.20, j2=-0.95, j7=1.3),   # V10 (낮은 위치)
    _make_pose(j1= 0.20, j2=-0.95, j7=1.0),   # V11 (낮은 위치)
    _make_pose(j1= 0.00, j2=-0.95, j7=1.6),   # V12 (낮은 위치)
]


# ══════════════════════════════════════════════════════════════════════════════
# 6. 캘리브레이션 샘플
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibSample:
    """단일 캘리브레이션 샘플."""
    idx:           int
    timestamp:     float
    joint_positions: dict
    gripper_pos:   Tuple[float, float, float]   # base_link 기준 (m)
    # T_gripper_base: gripper_flange frame → base_link (from /feedback/tcp_pose)
    R_gripper_base: np.ndarray  # 3x3
    t_gripper_base: np.ndarray  # (3,)
    # T_target_cam: ChArUco target → camera_color_optical_frame (from solvePnP)
    R_target_cam:   np.ndarray  # 3x3
    t_target_cam:   np.ndarray  # (3,)
    n_charuco_corners: int
    reproj_error:   float
    is_validation:  bool = False

    def to_dict(self) -> dict:
        return {
            'idx':            self.idx,
            'timestamp':      self.timestamp,
            'is_validation':  self.is_validation,
            'joint_positions': self.joint_positions,
            'gripper_pos_m':  list(self.gripper_pos),
            'R_gripper_base': self.R_gripper_base.tolist(),
            't_gripper_base': self.t_gripper_base.tolist(),
            'R_target_cam':   self.R_target_cam.tolist(),
            't_target_cam':   self.t_target_cam.tolist(),
            'n_charuco_corners': self.n_charuco_corners,
            'reproj_error_px':   round(self.reproj_error, 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7. 손-눈 Solver
# ══════════════════════════════════════════════════════════════════════════════

def solve_hand_eye(
    samples: List[CalibSample],
    method: str = "TSAI"
) -> Tuple[np.ndarray, str]:
    """
    Hand-eye calibration을 수행하고 T_cam_gripper를 반환한다.

    반환: (T_cam_gripper 4x4 numpy 배열, 사용한 method 이름)

    프레임 규약:
        T_cam_gripper: 점을 camera_color_optical_frame → gripper_flange 로 변환
        p_gripper = T_cam_gripper @ p_camera
    """
    if len(samples) < MIN_CALIB_SAMPLES:
        raise ValueError(f"샘플 {len(samples)}개 — 최소 {MIN_CALIB_SAMPLES}개 필요")

    R_g2b = [s.R_gripper_base for s in samples]
    t_g2b = [s.t_gripper_base for s in samples]
    R_t2c = [s.R_target_cam for s in samples]
    t_t2c = [s.t_target_cam for s in samples]

    cv2_method = SOLVER_METHODS.get(method.upper(), cv2.CALIB_HAND_EYE_TSAI)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2_method
    )

    T = np.eye(4)
    T[:3, :3] = R_cam2gripper
    T[:3, 3]  = t_cam2gripper.reshape(3)
    return T, method.upper()


def compare_methods(samples: List[CalibSample]) -> Dict[str, np.ndarray]:
    """여러 hand-eye method를 동일한 dataset에서 비교."""
    results = {}
    for name in SOLVER_METHODS:
        try:
            T, _ = solve_hand_eye(samples, method=name)
            results[name] = T
        except Exception as e:
            print(f'  [{name}] 실패: {e}')
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8. 내부 일관성 검증 (Internal Consistency)
# ══════════════════════════════════════════════════════════════════════════════

def internal_validation(samples: List[CalibSample],
                         T_cam_gripper: np.ndarray) -> dict:
    """
    각 샘플에서 base_link 기준 target 위치를 추정하고 spread를 계산한다.

    T_target_in_base_est = T_gripper_base @ T_cam_gripper @ T_target_cam

    board가 고정이면 모든 샘플의 target 위치가 동일해야 한다.
    STD가 작을수록 내부 일관성이 높다.

    주의: 이 값은 INTERNAL CONSISTENCY일 뿐, 실제 accuracy와 다르다.
    """
    positions = []
    for s in samples:
        T_gb = np.eye(4)
        T_gb[:3, :3] = s.R_gripper_base
        T_gb[:3, 3]  = s.t_gripper_base

        T_tc = np.eye(4)
        T_tc[:3, :3] = s.R_target_cam
        T_tc[:3, 3]  = s.t_target_cam

        # T_target_in_base = T_gripper_base @ T_cam_gripper @ T_target_cam
        T_est = T_gb @ T_cam_gripper @ T_tc
        positions.append(T_est[:3, 3])

    pos = np.array(positions)
    mean  = pos.mean(axis=0)
    std   = pos.std(axis=0)
    deviations = np.linalg.norm(pos - mean, axis=1)

    return {
        'n_samples':    len(samples),
        'mean_m':       mean.tolist(),
        'std_xyz_mm':   (std * 1000).tolist(),
        'std_x_mm':     float(std[0] * 1000),
        'std_y_mm':     float(std[1] * 1000),
        'std_z_mm':     float(std[2] * 1000),
        'max_dev_mm':   float(deviations.max() * 1000),
        'rms_mm':       float(np.sqrt(np.mean(deviations**2)) * 1000),
        'pass_5mm':     bool(std.max() * 1000 <= 5.0),
        'note': '내부 일관성 — 독립 검증과 다름. STD가 작아도 bias가 클 수 있음.',
    }


def print_internal_validation(result: dict):
    print('\n─── 내부 일관성 검증 (Internal Consistency) ───────────────────')
    print(f"  샘플 수: {result['n_samples']}")
    print(f"  base 좌표계 target 위치 STD:")
    print(f"    X: {result['std_x_mm']:6.2f} mm  {'✓' if result['std_x_mm'] <= 5.0 else '⚠'}")
    print(f"    Y: {result['std_y_mm']:6.2f} mm  {'✓' if result['std_y_mm'] <= 5.0 else '⚠'}")
    print(f"    Z: {result['std_z_mm']:6.2f} mm  {'✓' if result['std_z_mm'] <= 5.0 else '⚠'}")
    print(f"  Max deviation: {result['max_dev_mm']:.2f} mm")
    print(f"  3D RMS: {result['rms_mm']:.2f} mm")
    if result['pass_5mm']:
        print('  ✅ 내부 일관성: 5 mm 이하 달성')
    else:
        print('  ⚠️ 내부 일관성: 5 mm 초과 — 자세 다양성 부족 또는 board 이동 의심')
    print(f"  [{result['note']}]")


# ══════════════════════════════════════════════════════════════════════════════
# 9. 독립 검증 (Independent Validation)
# ══════════════════════════════════════════════════════════════════════════════

def independent_validation(
    val_samples: List[CalibSample],
    T_cam_gripper: np.ndarray,
    T_board_base_gt: Optional[np.ndarray] = None,
    calib_samples: Optional[List[CalibSample]] = None,
) -> dict:
    """
    캘리브레이션에 사용하지 않은 새 자세에서 추정된 board 위치와
    ground truth를 비교하여 실제 정확도를 평가한다.

    Ground truth 전략:
        1. T_board_base_gt 가 주어지면 그것을 사용 (가장 신뢰도 높음)
        2. 없으면 calib_samples의 평균 위치를 GT로 사용 (근사값)
           이 경우 GT uncertainty를 명시.

    최종 목표: X STD ≤ 5mm, Y STD ≤ 5mm, Z STD ≤ 5mm
    """
    # GT 결정
    if T_board_base_gt is not None:
        gt_pos = T_board_base_gt[:3, 3]
        gt_source = 'external'
    elif calib_samples is not None:
        calib_positions = []
        for s in calib_samples:
            T_gb = np.eye(4); T_gb[:3, :3] = s.R_gripper_base; T_gb[:3, 3] = s.t_gripper_base
            T_tc = np.eye(4); T_tc[:3, :3] = s.R_target_cam;   T_tc[:3, 3] = s.t_target_cam
            T_est = T_gb @ T_cam_gripper @ T_tc
            calib_positions.append(T_est[:3, 3])
        gt_pos = np.mean(calib_positions, axis=0)
        gt_source = 'calib_mean'
    else:
        raise ValueError("T_board_base_gt 또는 calib_samples 중 하나는 있어야 합니다")

    # 검증 자세별 추정
    errors = []
    per_sample = []
    for s in val_samples:
        T_gb = np.eye(4); T_gb[:3, :3] = s.R_gripper_base; T_gb[:3, 3] = s.t_gripper_base
        T_tc = np.eye(4); T_tc[:3, :3] = s.R_target_cam;   T_tc[:3, 3] = s.t_target_cam
        T_est = T_gb @ T_cam_gripper @ T_tc
        est_pos = T_est[:3, 3]
        err = est_pos - gt_pos
        err3d = float(np.linalg.norm(err))
        errors.append(err)
        per_sample.append({
            'idx':         s.idx,
            'error_x_mm':  float(err[0] * 1000),
            'error_y_mm':  float(err[1] * 1000),
            'error_z_mm':  float(err[2] * 1000),
            'error_3d_mm': round(err3d * 1000, 2),
            'n_corners':   s.n_charuco_corners,
            'reproj_px':   s.reproj_error,
        })

    errs = np.array(errors)
    std_xyz = errs.std(axis=0)
    rmse    = np.sqrt(np.mean(errs**2, axis=0))
    mae     = np.abs(errs).mean(axis=0)
    max_abs = np.abs(errs).max(axis=0)
    errs_3d = np.linalg.norm(errs, axis=1)

    return {
        'n_samples':       len(val_samples),
        'gt_source':       gt_source,
        'gt_pos_m':        gt_pos.tolist(),
        # RMSE
        'rmse_x_mm': float(rmse[0] * 1000),
        'rmse_y_mm': float(rmse[1] * 1000),
        'rmse_z_mm': float(rmse[2] * 1000),
        # STD
        'std_x_mm':  float(std_xyz[0] * 1000),
        'std_y_mm':  float(std_xyz[1] * 1000),
        'std_z_mm':  float(std_xyz[2] * 1000),
        # MAE
        'mae_x_mm':  float(mae[0] * 1000),
        'mae_y_mm':  float(mae[1] * 1000),
        'mae_z_mm':  float(mae[2] * 1000),
        # Max
        'max_abs_x_mm': float(max_abs[0] * 1000),
        'max_abs_y_mm': float(max_abs[1] * 1000),
        'max_abs_z_mm': float(max_abs[2] * 1000),
        'max_3d_mm':    float(errs_3d.max() * 1000),
        'rms_3d_mm':    float(np.sqrt(np.mean(errs_3d**2)) * 1000),
        # PASS/FAIL (STD ≤ 5mm 기준)
        'pass_x': bool(std_xyz[0] * 1000 <= 5.0),
        'pass_y': bool(std_xyz[1] * 1000 <= 5.0),
        'pass_z': bool(std_xyz[2] * 1000 <= 5.0),
        'pass_all': bool(std_xyz.max() * 1000 <= 5.0),
        'per_sample': per_sample,
    }


def print_independent_validation(result: dict):
    print('\n══════════════════════════════════════════════════════════════')
    print('  독립 검증 결과 (Independent Validation)')
    print('══════════════════════════════════════════════════════════════')
    print(f"  GT 소스: {result['gt_source']}")
    if result['gt_source'] == 'calib_mean':
        print('  ⚠️  GT = 캘리브레이션 샘플 평균 (근사값). 실제 독립 GT보다 낙관적일 수 있음.')
    print(f"  검증 샘플: {result['n_samples']}개")
    print()
    print('                X         Y         Z')
    print(f"  RMSE:    {result['rmse_x_mm']:6.2f}mm   {result['rmse_y_mm']:6.2f}mm   {result['rmse_z_mm']:6.2f}mm")
    print(f"  STD:     {result['std_x_mm']:6.2f}mm   {result['std_y_mm']:6.2f}mm   {result['std_z_mm']:6.2f}mm")
    print(f"  MAE:     {result['mae_x_mm']:6.2f}mm   {result['mae_y_mm']:6.2f}mm   {result['mae_z_mm']:6.2f}mm")
    print(f"  Max|err|:{result['max_abs_x_mm']:6.2f}mm   {result['max_abs_y_mm']:6.2f}mm   {result['max_abs_z_mm']:6.2f}mm")
    print(f"  Max 3D:  {result['max_3d_mm']:.2f} mm   |   RMS 3D: {result['rms_3d_mm']:.2f} mm")
    print()
    xp = '✓ PASS' if result['pass_x'] else '✗ FAIL'
    yp = '✓ PASS' if result['pass_y'] else '✗ FAIL'
    zp = '✓ PASS' if result['pass_z'] else '✗ FAIL'
    print(f"  목표 STD ≤ 5mm:  X {xp}   Y {yp}   Z {zp}")
    if result['pass_all']:
        print('  🎯 최종: PASS — 5mm 목표 달성')
    else:
        print('  ❌ 최종: FAIL — 추가 최적화 필요')
    print('══════════════════════════════════════════════════════════════')


# ══════════════════════════════════════════════════════════════════════════════
# 10. JSON 저장 / 로드 (기존 포맷과 하위호환)
# ══════════════════════════════════════════════════════════════════════════════

def save_calibration(
    T_cam_gripper: np.ndarray,
    board_cfg: BoardConfig,
    solver: str,
    calib_samples: List[CalibSample],
    internal_val: dict,
    val_result: Optional[dict] = None,
    output_path: str = OUTPUT_JSON,
) -> str:
    """
    캘리브레이션 결과를 JSON으로 저장한다.
    기존 'T_flange_camera' 키를 그대로 사용하여 하위호환성을 보장한다.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    w, h = board_cfg.board_physical_size_mm()

    data = {
        # ── 기존 포맷 (하위호환) ──
        'note':           'tcp_link→camera_color_optical_frame (eye-in-hand, ChArUco)',
        'T_flange_camera': T_cam_gripper.tolist(),   # T_cam_gripper (camera→gripper_flange)

        # ── 확장 포맷 ──
        'calibration_type': 'eye_in_hand_charuco',
        'camera_frame':     'camera_color_optical_frame',
        'gripper_frame':    'gripper_flange',   # /feedback/tcp_pose frame
        'base_frame':       'base_link',

        'board_config': {
            'squares_x':       board_cfg.squares_x,
            'squares_y':       board_cfg.squares_y,
            'square_length_m': board_cfg.square_length,
            'marker_length_m': board_cfg.marker_length,
            'dictionary':      board_cfg.dictionary,
            'physical_width_mm':  round(w, 1),
            'physical_height_mm': round(h, 1),
        },

        'hand_eye': {
            'T_cam_gripper':   T_cam_gripper.tolist(),
            'T_gripper_cam':   np.linalg.inv(T_cam_gripper).tolist(),
            'description':     'T_cam_gripper: p_gripper = T @ p_camera',
        },

        'solver': solver,
        'num_samples': len(calib_samples),

        'internal_validation': internal_val,
        'independent_validation': val_result,

        'samples': [s.to_dict() for s in calib_samples],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path


def load_calibration(path: str = OUTPUT_JSON) -> Optional[np.ndarray]:
    """저장된 JSON에서 T_cam_gripper를 로드한다."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return np.array(data['T_flange_camera'], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 11. ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════

class ChArUcoCalibNode(Node):
    """캘리브레이션 ROS2 노드.

    - /feedback/tcp_pose  (PoseStamped, BEST_EFFORT) 구독
    - /feedback/joint_states (JointState) 구독
    - /arm_command (String) 발행 — 로봇 이동 명령
    - /pick_result (String) 구독 — 이동 결과
    - /calib/image (Image) 발행 — RViz 시각화
    - /calib/markers (MarkerArray) 발행 — 캡처 자세 마커
    """

    def __init__(self, board_cfg: BoardConfig, quality: QualityGate):
        super().__init__('charuco_hand_eye_calib')

        qos_be  = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)

        self._lock = threading.Lock()
        self.latest_pose: Optional[PoseStamped] = None
        self.latest_joints: Optional[dict] = None
        self._latest_result: Optional[dict] = None

        self.create_subscription(PoseStamped, '/feedback/tcp_pose',
                                  self._pose_cb, qos_be)
        self.create_subscription(JointState, '/feedback/joint_states',
                                  self._joint_cb, qos_be)
        self.create_subscription(String, '/pick_result',
                                  self._result_cb, qos_rel)

        self.pub_cmd     = self.create_publisher(String,      '/arm_command',  qos_rel)
        self.pub_image   = self.create_publisher(Image,       '/calib/image',  qos_rel)
        self.pub_markers = self.create_publisher(MarkerArray, '/calib/markers', qos_rel)

        # ── RealSense 초기화 ──
        if not HAS_REALSENSE:
            raise RuntimeError('pyrealsense2 미설치')

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        profile = self.pipe.start(cfg)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.K    = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(intr.coeffs[:5], dtype=np.float64)
        print(f'[카메라] fx={intr.fx:.2f} fy={intr.fy:.2f} '
              f'cx={intr.ppx:.2f} cy={intr.ppy:.2f}')

        self.detector = CharucoDetector(board_cfg, self.K, self.dist, quality)

        # 백그라운드 카메라 스레드 상태
        self._cam_running = True
        self._det_result: Optional[DetectionResult] = None
        self._last_img: Optional[np.ndarray] = None
        self._calib_status = ''    # 표시할 상태 텍스트
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        with self._lock:
            self.latest_pose = msg

    def _joint_cb(self, msg: JointState):
        with self._lock:
            self.latest_joints = dict(zip(msg.name, msg.position))

    def _result_cb(self, msg: String):
        with self._lock:
            try:
                self._latest_result = _json.loads(msg.data)
            except Exception:
                pass

    def _spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.05)

    # ── 카메라 루프 (백그라운드) ──────────────────────────────────────────

    def _cam_loop(self):
        while self._cam_running:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=500)
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())
                det = self.detector.detect(img)

                with self._lock:
                    self._det_result = det
                    self._last_img = img.copy()

                # RViz에 이미지 발행
                if det.vis_img is not None:
                    vis = det.vis_img.copy()
                    if self._calib_status:
                        cv2.putText(vis, self._calib_status, (8, CAM_H - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
                    self._publish_image(vis)
            except Exception:
                continue

    def _publish_image(self, img: np.ndarray):
        if _bridge is not None:
            try:
                msg = _bridge.cv2_to_imgmsg(img, encoding='bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera_color_optical_frame'
                self.pub_image.publish(msg)
                return
            except Exception:
                pass
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.height = img.shape[0]
        msg.width  = img.shape[1]
        msg.encoding = 'bgr8'
        msg.step    = img.shape[1] * 3
        msg.data    = img.tobytes()
        self.pub_image.publish(msg)

    # ── 로봇 이동 ──────────────────────────────────────────────────────────

    def go_home(self, timeout: float = 20.0) -> bool:
        print('[이동] 홈 복귀...')
        ok = self._send_cmd_wait({'action': 'go_home'}, timeout=timeout)
        time.sleep(SETTLE_SEC)
        return ok

    def move_to_joints(self, joint_positions: List[float]) -> bool:
        """절대 관절각(rad) 자세로 이동."""
        joints_dict = {n: float(v) for n, v in zip(JOINT_NAMES, joint_positions)}
        ok = self._send_cmd_wait({'action': 'move_joints', 'joints': joints_dict})
        if ok:
            time.sleep(SETTLE_SEC)
        return ok

    def _send_cmd_wait(self, payload: dict, timeout: float = MOVE_TIMEOUT) -> bool:
        with self._lock:
            self._latest_result = None
        msg = String()
        msg.data = _json.dumps(payload)
        self.pub_cmd.publish(msg)
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._spin_once()
            with self._lock:
                res = self._latest_result
            if res is not None:
                return res.get('status') == 'success'
            time.sleep(0.05)
        print('  [경고] 응답 타임아웃')
        return False

    # ── 샘플 캡처 ──────────────────────────────────────────────────────────

    def capture_sample(self, idx: int, is_validation: bool = False
                        ) -> Optional[CalibSample]:
        """
        현재 자세에서 (tcp_pose, ChArUco target pose) 쌍을 캡처한다.

        실패 조건:
        - /feedback/tcp_pose 미수신
        - ChArUco 미검출 또는 quality gate 실패
        - solvePnP 실패
        """
        with self._lock:
            pose = self.latest_pose
            joints = self.latest_joints
            det = self._det_result

        if pose is None:
            print(f'  [스킵 #{idx}] /feedback/tcp_pose 미수신')
            return None
        if det is None or not det.pose_valid:
            corners_info = f"corners={det.n_corners}" if det else "no detection"
            print(f'  [스킵 #{idx}] ChArUco 품질 미달 ({corners_info})')
            return None

        p = pose.pose
        R_gb = quat_to_R(p.orientation.x, p.orientation.y,
                          p.orientation.z, p.orientation.w)
        t_gb = np.array([p.position.x, p.position.y, p.position.z])

        R_tc, _ = cv2.Rodrigues(det.rvec)
        t_tc    = det.tvec

        sample = CalibSample(
            idx=idx,
            timestamp=time.time(),
            joint_positions=joints or {},
            gripper_pos=(float(t_gb[0]), float(t_gb[1]), float(t_gb[2])),
            R_gripper_base=R_gb,
            t_gripper_base=t_gb,
            R_target_cam=R_tc,
            t_target_cam=t_tc,
            n_charuco_corners=det.n_corners,
            reproj_error=det.reproj_err,
            is_validation=is_validation,
        )
        print(f'  [캡처 #{idx}{"V" if is_validation else ""}] '
              f'tcp=({t_gb[0]:.3f},{t_gb[1]:.3f},{t_gb[2]:.3f})  '
              f'corners={det.n_corners}  reproj={det.reproj_err:.2f}px')
        return sample

    def try_capture_sample(self, idx: int, attempts: int = 3,
                            is_validation: bool = False) -> Optional[CalibSample]:
        """최대 attempts회 시도, 성공하면 반환."""
        for a in range(attempts):
            sample = self.capture_sample(idx, is_validation=is_validation)
            if sample is not None:
                return sample
            time.sleep(0.5)
        return None

    # ── RViz 마커 발행 ─────────────────────────────────────────────────────

    def publish_markers(self, samples: List[CalibSample], n_target: int):
        ma = MarkerArray()

        # 기존 마커 전체 삭제
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)

        for s in samples:
            R = s.R_gripper_base
            t = s.t_gripper_base
            qx, qy, qz, qw = R_to_quat(R)
            color = (0.0, 1.0, 0.2) if not s.is_validation else (1.0, 0.8, 0.0)

            for axis, ax_col in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
                m = Marker()
                m.header.frame_id = 'base_link'
                m.header.stamp    = self.get_clock().now().to_msg()
                m.ns   = 'calib_val_poses' if s.is_validation else 'calib_poses'
                m.id   = s.idx * 10 + axis
                m.type = Marker.ARROW
                m.action = Marker.ADD
                m.pose.position.x = t[0]
                m.pose.position.y = t[1]
                m.pose.position.z = t[2]
                ox, oy, oz, ow = qx, qy, qz, qw
                if axis == 1:
                    ox, oy, oz, ow = quat_mul((qx, qy, qz, qw), (0, 0, 0.707, 0.707))
                elif axis == 2:
                    ox, oy, oz, ow = quat_mul((qx, qy, qz, qw), (0, -0.707, 0, 0.707))
                m.pose.orientation.x = ox
                m.pose.orientation.y = oy
                m.pose.orientation.z = oz
                m.pose.orientation.w = ow
                m.scale.x = 0.04; m.scale.y = 0.005; m.scale.z = 0.005
                m.color = ColorRGBA(
                    r=float(ax_col[0]) * color[0],
                    g=float(ax_col[1]) * color[1],
                    b=float(ax_col[2]) * color[2], a=0.85)
                m.lifetime = RosDuration(sec=0)
                ma.markers.append(m)

            # 라벨
            txt = Marker()
            txt.header.frame_id = 'base_link'
            txt.header.stamp    = self.get_clock().now().to_msg()
            txt.ns    = 'calib_labels'
            txt.id    = s.idx + 10000
            txt.type  = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = t[0]
            txt.pose.position.y = t[1]
            txt.pose.position.z = t[2] + 0.06
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.04
            txt.color   = ColorRGBA(r=1.0, g=1.0, b=0.2, a=1.0)
            txt.text    = f'{"V" if s.is_validation else "C"}{s.idx}'
            txt.lifetime = RosDuration(sec=0)
            ma.markers.append(txt)

        # 상태 텍스트
        n_calib = sum(1 for s in samples if not s.is_validation)
        status_txt = Marker()
        status_txt.header.frame_id = 'base_link'
        status_txt.header.stamp    = self.get_clock().now().to_msg()
        status_txt.ns     = 'calib_status'
        status_txt.id     = 99999
        status_txt.type   = Marker.TEXT_VIEW_FACING
        status_txt.action = Marker.ADD
        status_txt.pose.position.z = 1.1
        status_txt.pose.orientation.w = 1.0
        status_txt.scale.z = 0.07
        ok_c = (0.2, 1.0, 0.2) if n_calib >= MIN_CALIB_SAMPLES else (1.0, 0.7, 0.0)
        status_txt.color = ColorRGBA(r=ok_c[0], g=ok_c[1], b=ok_c[2], a=1.0)
        status_txt.text  = f'Calib:{n_calib}/{n_target}  (green=calib, yellow=val)'
        status_txt.lifetime = RosDuration(sec=0)
        ma.markers.append(status_txt)

        self.pub_markers.publish(ma)

    def set_status_text(self, text: str):
        with self._lock:
            self._calib_status = text

    def cleanup(self):
        self._cam_running = False
        self._cam_thread.join(timeout=2.0)
        try:
            self.pipe.stop()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# 12. ChArUco board PNG 생성
# ══════════════════════════════════════════════════════════════════════════════

def generate_board_image(board_cfg: BoardConfig,
                          dpi: float = 300.0,
                          output_path: str = '/tmp/charuco_board.png') -> str:
    """
    ChArUco board를 PNG로 저장한다.
    실제 인쇄 크기 = 물리적 크기가 되도록 DPI를 기반으로 픽셀 크기를 계산.

    최종 인쇄 후 실측해서 board_cfg의 square_length/marker_length와 일치하는지
    반드시 확인하라. 프린터 scaling으로 실제 크기가 달라질 수 있음.
    """
    w_mm, h_mm = board_cfg.board_physical_size_mm()
    # margin 10mm
    margin_mm = 10.0
    px_per_mm = dpi / 25.4
    w_px = int((w_mm + 2 * margin_mm) * px_per_mm)
    h_px = int((h_mm + 2 * margin_mm) * px_per_mm)
    margin_px = int(margin_mm * px_per_mm)

    board = board_cfg.create_board()
    img = board.draw((w_px - 2 * margin_px, h_px - 2 * margin_px))

    # margin 추가
    out = np.ones((h_px, w_px), dtype=np.uint8) * 255
    out[margin_px:margin_px + img.shape[0], margin_px:margin_px + img.shape[1]] = img

    cv2.imwrite(output_path, out)

    print(f'\n[Board 생성] {output_path}')
    print(f'  물리 크기: {w_mm:.0f} x {h_mm:.0f} mm')
    print(f'  칸 크기: {board_cfg.square_length * 1000:.1f} mm')
    print(f'  Marker 크기: {board_cfg.marker_length * 1000:.1f} mm')
    print(f'  총 칸: {board_cfg.squares_x} x {board_cfg.squares_y}')
    print(f'  내부 코너 수: {board_cfg.total_inner_corners()}')
    print(f'  이미지 해상도: {w_px} x {h_px} px @ {dpi:.0f} DPI')
    print(f'\n  ⚠️  인쇄 후 실측으로 칸 크기가 {board_cfg.square_length*1000:.1f}mm인지 확인!')
    print(f'     크기가 다르면 board_cfg의 square_length를 실측값으로 수정하세요.')
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# 13. 결과 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_calibration_result(T_cam_gripper: np.ndarray, solver: str, board_cfg: BoardConfig):
    xyz = T_cam_gripper[:3, 3]
    roll, pitch, yaw = R_to_rpy(T_cam_gripper[:3, :3])

    print('\n' + '='*70)
    print(f'  Hand-Eye Calibration 결과  (solver: {solver})')
    print('='*70)
    print(f'\n  [T_cam_gripper] camera_color_optical_frame → gripper_flange')
    print(f'  Translation (m): x={xyz[0]:.5f}  y={xyz[1]:.5f}  z={xyz[2]:.5f}')
    print(f'  Rotation (rad):  r={roll:.5f}  p={pitch:.5f}  y={yaw:.5f}')
    print(f'  Rotation (deg):  r={math.degrees(roll):.2f}°  p={math.degrees(pitch):.2f}°  y={math.degrees(yaw):.2f}°')

    # URDF d435_camera_joint 업데이트 값
    try:
        xyz_d435, rpy_d435 = compute_d435_joint_origin(T_cam_gripper)
        print(f'\n  [d435_camera_joint 업데이트 값] (nero_with_camera.urdf)')
        print(f'  <origin xyz="{xyz_d435[0]:.8f} {xyz_d435[1]:.8f} {xyz_d435[2]:.8f}"')
        print(f'           rpy="{rpy_d435[0]:.8f} {rpy_d435[1]:.8f} {rpy_d435[2]:.8f}" />')
    except Exception as e:
        print(f'\n  [경고] d435_camera_joint 계산 실패: {e}')

    # Sanity check
    dist = np.linalg.norm(xyz)
    if dist > 1.0:
        print(f'\n  ⚠️  Translation 크기 {dist*1000:.0f}mm — 물리적으로 비정상적으로 큼')
        print('     frame convention 또는 board 크기를 확인하세요')
    elif dist < 0.001:
        print(f'\n  ⚠️  Translation 크기 {dist*1000:.1f}mm — 너무 작음')
    else:
        print(f'\n  ✅ Translation 크기: {dist*1000:.1f}mm (정상 범위)')
    print('='*70)


# ══════════════════════════════════════════════════════════════════════════════
# 14. 모드별 실행 함수
# ══════════════════════════════════════════════════════════════════════════════

def run_generate_board(args):
    """ChArUco board 이미지 생성."""
    cfg = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary=args.dictionary,
    )
    out = args.board_output or '/tmp/charuco_board.png'
    generate_board_image(cfg, dpi=args.dpi, output_path=out)


def run_dry_run(args):
    """실제 robot 이동 없이 캘리브레이션 계획 자세를 출력한다."""
    print('\n=== DRY-RUN: 캘리브레이션 계획 자세 ===')
    print(f'\n캘리브레이션 자세 ({len(CALIB_POSES)}개):')
    print(f'  {"#":>3}  {"j1":>8}  {"j2":>8}  {"j3":>8}  {"j4":>8}  {"j5":>8}  {"j6":>8}  {"j7":>8}')
    print('  ' + '-'*75)
    for i, p in enumerate(CALIB_POSES, 1):
        vals = '  '.join(f'{math.degrees(v):+7.1f}°' for v in p)
        print(f'  {i:>3}  {vals}')

    print(f'\n검증 자세 ({len(VALIDATION_POSES)}개):')
    print(f'  {"#":>3}  {"j1":>8}  {"j2":>8}  {"j3":>8}  {"j4":>8}  {"j5":>8}  {"j6":>8}  {"j7":>8}')
    print('  ' + '-'*75)
    for i, p in enumerate(VALIDATION_POSES, 1):
        vals = '  '.join(f'{math.degrees(v):+7.1f}°' for v in p)
        print(f'  V{i:>2}  {vals}')

    print(f'\n총 이동: {len(CALIB_POSES) + len(VALIDATION_POSES)}회 + 홈 복귀 2회')
    print(f'예상 소요 시간: {(len(CALIB_POSES) + len(VALIDATION_POSES)) * (MOVE_TIMEOUT/2 + SETTLE_SEC) / 60:.0f}~{(len(CALIB_POSES) + len(VALIDATION_POSES)) * (MOVE_TIMEOUT + SETTLE_SEC) / 60:.0f}분 (이동 시간 불확실)')
    print('\n⚠️  실제 실행 전 확인사항:')
    print('  1. ChArUco board가 바닥에 수평으로 고정되어 있는지')
    print('  2. perception_node가 중지되어 있는지 (카메라 점유 충돌)')
    print('  3. robot 작업공간에 장애물이 없는지')
    print('  4. 모든 자세에서 board가 camera FOV에 들어오는 위치인지')


def run_calibrate(args):
    """자동 캘리브레이션 실행."""
    board_cfg = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary=args.dictionary,
    )
    quality = QualityGate(min_charuco_corners=args.min_corners)

    print('\n=== ChArUco Hand-Eye Calibration ===')
    print(f'Board: {board_cfg.squares_x}x{board_cfg.squares_y}  '
          f'칸={board_cfg.square_length*1000:.1f}mm  '
          f'marker={board_cfg.marker_length*1000:.1f}mm  '
          f'dict={board_cfg.dictionary}')
    w, h = board_cfg.board_physical_size_mm()
    print(f'Board 물리 크기: {w:.0f}x{h:.0f}mm  |  내부 코너: {board_cfg.total_inner_corners()}')
    print(f'Quality gate: 최소 {quality.min_charuco_corners} corners, '
          f'최대 {quality.max_reproj_error}px reproj error')
    print(f'자세: {len(CALIB_POSES)} calibration + {len(VALIDATION_POSES)} validation')
    print()

    if args.dry_run:
        run_dry_run(args)
        return

    rclpy.init()
    node = ChArUcoCalibNode(board_cfg, quality)
    calib_samples: List[CalibSample] = []
    val_samples:   List[CalibSample] = []

    try:
        # 1. 홈 복귀
        node.go_home()
        time.sleep(1.0)

        # 2. 캘리브레이션 자세 순회
        print(f'\n──── 캘리브레이션 샘플 수집 ({len(CALIB_POSES)}개 자세) ────')
        for i, pose in enumerate(CALIB_POSES):
            deg_str = ', '.join(f'j{j+1}={math.degrees(v):.0f}°'
                                for j, v in enumerate(pose) if abs(v) > 0.01)
            print(f'\n[자세 {i+1}/{len(CALIB_POSES)}] {deg_str or "(홈)"}')

            node.set_status_text(
                f'자세 {i+1}/{len(CALIB_POSES)} | 캡처:{len(calib_samples)}')

            if not node.move_to_joints(pose):
                print('  [스킵] 이동 실패')
                continue

            sample = node.try_capture_sample(i + 1, attempts=3)
            if sample is None:
                print(f'  [스킵] 검출 실패 — 자세 {i+1} 건너뜀')
                continue

            calib_samples.append(sample)
            node.publish_markers(calib_samples + val_samples, len(CALIB_POSES))

        print(f'\n캘리브레이션 샘플: {len(calib_samples)}/{len(CALIB_POSES)}')

        if len(calib_samples) < MIN_CALIB_SAMPLES:
            print(f'\n[오류] 샘플 부족: {len(calib_samples)}개 < {MIN_CALIB_SAMPLES}개. '
                  '종료합니다.')
            return

        # 3. Hand-eye solver
        print(f'\n──── Hand-Eye Solver ({args.solver}) ────')
        T_cam_gripper, solver_used = solve_hand_eye(calib_samples, method=args.solver)
        print_calibration_result(T_cam_gripper, solver_used, board_cfg)

        # (선택) 여러 method 비교
        if args.compare_methods:
            print('\n──── 다중 방법 비교 ────')
            all_T = compare_methods(calib_samples)
            for name, T in all_T.items():
                xyz = T[:3, 3]
                r, p, y = R_to_rpy(T[:3, :3])
                print(f'  [{name:12s}] '
                      f't=({xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f})  '
                      f'rpy=({math.degrees(r):.1f},{math.degrees(p):.1f},{math.degrees(y):.1f})°')

        # 4. 내부 일관성 검증
        int_val = internal_validation(calib_samples, T_cam_gripper)
        print_internal_validation(int_val)

        # 5. 검증 자세 (독립 검증 샘플 수집)
        print(f'\n──── 독립 검증 샘플 수집 ({len(VALIDATION_POSES)}개 자세) ────')
        for i, pose in enumerate(VALIDATION_POSES):
            deg_str = ', '.join(f'j{j+1}={math.degrees(v):.0f}°'
                                for j, v in enumerate(pose) if abs(v) > 0.01)
            print(f'\n[검증 {i+1}/{len(VALIDATION_POSES)}] {deg_str}')

            node.set_status_text(f'검증 {i+1}/{len(VALIDATION_POSES)}')

            if not node.move_to_joints(pose):
                print('  [스킵] 이동 실패')
                continue

            sample = node.try_capture_sample(len(CALIB_POSES) + i + 1,
                                             attempts=3, is_validation=True)
            if sample is None:
                print(f'  [스킵] 검출 실패')
                continue

            val_samples.append(sample)
            node.publish_markers(calib_samples + val_samples, len(CALIB_POSES))

        # 6. 독립 검증
        val_result = None
        if len(val_samples) >= 3:
            print(f'\n──── 독립 검증 ({len(val_samples)}개 샘플) ────')
            val_result = independent_validation(
                val_samples, T_cam_gripper,
                T_board_base_gt=None,       # 외부 GT 없음 → calib_mean 사용
                calib_samples=calib_samples,
            )
            print_independent_validation(val_result)
        else:
            print(f'\n[경고] 검증 샘플 {len(val_samples)}개 — 독립 검증 스킵')

        # 7. 홈 복귀
        node.go_home()

        # 8. 결과 저장
        output_path = args.output or OUTPUT_JSON
        all_samples = calib_samples + val_samples
        saved_path = save_calibration(
            T_cam_gripper, board_cfg, solver_used,
            all_samples, int_val, val_result, output_path=output_path
        )
        print(f'\n[저장] {saved_path}')

        # 9. URDF 업데이트 (요청 시)
        if args.update_urdf:
            urdf_path = args.urdf_path or URDF_PATH
            try:
                xyz_d, rpy_d = compute_d435_joint_origin(T_cam_gripper)
                update_urdf_d435_joint(urdf_path, xyz_d, rpy_d)
            except Exception as e:
                print(f'\n[경고] URDF 업데이트 실패: {e}')
        else:
            try:
                xyz_d, rpy_d = compute_d435_joint_origin(T_cam_gripper)
                print(f'\n[URDF] --update-urdf 플래그 없음. 수동으로 적용하려면:')
                print(f'       d435_camera_joint origin:')
                print(f'       xyz="{xyz_d[0]:.8f} {xyz_d[1]:.8f} {xyz_d[2]:.8f}"')
                print(f'       rpy="{rpy_d[0]:.8f} {rpy_d[1]:.8f} {rpy_d[2]:.8f}"')
            except Exception:
                pass

    except KeyboardInterrupt:
        print('\n[중단] 사용자가 종료했습니다.')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


def run_validate(args):
    """저장된 calibration을 사용한 독립 검증 전용 모드."""
    T_cam_gripper = load_calibration(args.calib_file or OUTPUT_JSON)
    if T_cam_gripper is None:
        print(f'[오류] Calibration 파일을 찾을 수 없습니다: {args.calib_file or OUTPUT_JSON}')
        print('       먼저 --mode calibrate를 실행하세요.')
        sys.exit(1)

    board_cfg = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary=args.dictionary,
    )
    quality = QualityGate(min_charuco_corners=args.min_corners)

    print('\n=== 독립 검증 모드 (Validation Mode) ===')
    print(f'Calibration 로드: {args.calib_file or OUTPUT_JSON}')
    print(f'검증 자세: {len(VALIDATION_POSES)}개')

    if args.dry_run:
        run_dry_run(args)
        return

    rclpy.init()
    node = ChArUcoCalibNode(board_cfg, quality)
    val_samples: List[CalibSample] = []

    try:
        node.go_home()
        time.sleep(1.0)

        for i, pose in enumerate(VALIDATION_POSES):
            deg_str = ', '.join(f'j{j+1}={math.degrees(v):.0f}°'
                                for j, v in enumerate(pose) if abs(v) > 0.01)
            print(f'\n[검증 {i+1}/{len(VALIDATION_POSES)}] {deg_str}')
            node.set_status_text(f'검증 {i+1}/{len(VALIDATION_POSES)}')

            if not node.move_to_joints(pose):
                print('  [스킵] 이동 실패')
                continue

            sample = node.try_capture_sample(i + 1, attempts=3, is_validation=True)
            if sample is not None:
                val_samples.append(sample)
                node.publish_markers(val_samples, len(VALIDATION_POSES))

        node.go_home()

        if len(val_samples) < 3:
            print(f'\n[오류] 검증 샘플 {len(val_samples)}개 — 최소 3개 필요')
            return

        print(f'\n검증 샘플: {len(val_samples)}/{len(VALIDATION_POSES)}')

        # GT를 외부에서 제공하는 경우
        T_gt = None
        if args.gt_pos:
            # --gt-pos "x y z" (미터)
            try:
                xyz_gt = [float(v) for v in args.gt_pos.split()]
                T_gt = np.eye(4)
                T_gt[:3, 3] = xyz_gt
                print(f'[GT] 외부 GT 사용: {xyz_gt}m')
            except Exception as e:
                print(f'[경고] GT 파싱 실패: {e}')

        # GT 없을 때: calibration JSON의 internal_validation.mean_m을 fallback GT로 사용.
        # 단, 이 값은 calibration 샘플의 평균 추정치이므로 실제 GT보다 낙관적임.
        if T_gt is None:
            try:
                calib_path = args.calib_file or OUTPUT_JSON
                with open(calib_path) as _f:
                    _calib_data = json.load(_f)
                _mean_m = _calib_data.get('internal_validation', {}).get('mean_m')
                if _mean_m and len(_mean_m) == 3:
                    T_gt = np.eye(4)
                    T_gt[:3, 3] = np.array(_mean_m)
                    print(f'[GT] calibration JSON mean 사용 (근사): {[round(v, 4) for v in _mean_m]}m')
                    print('     ⚠️  실제 독립 GT보다 낙관적일 수 있음. --gt-pos로 실측값을 권장.')
            except Exception as _e:
                print(f'[경고] calibration JSON mean 로드 실패: {_e}')

        if T_gt is None:
            print('\n[오류] Ground truth를 결정할 수 없습니다.')
            print('       --gt-pos "x y z" (m) 로 board의 base_link 기준 위치를 지정하거나')
            print('       먼저 --mode calibrate를 실행하여 calibration 파일을 생성하세요.')
            return

        val_result = independent_validation(
            val_samples, T_cam_gripper,
            T_board_base_gt=T_gt,
            calib_samples=None,
        )
        print_independent_validation(val_result)

        # 결과 저장
        out = args.output or os.path.join(CALIB_DIR, 'validation_result.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'validation': val_result,
                       'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
        print(f'\n[저장] {out}')

    except KeyboardInterrupt:
        print('\n[중단]')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


def run_visualize(args):
    """Live ChArUco 검출 미리보기 (robot 이동 없음)."""
    board_cfg = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary=args.dictionary,
    )
    quality = QualityGate(min_charuco_corners=args.min_corners)

    print('\n=== ChArUco 시각화 모드 ===')
    print('RViz에 /calib/image (Image) 디스플레이를 추가하세요.')
    print('Ctrl+C 로 종료.')

    rclpy.init()
    node = ChArUcoCalibNode(board_cfg, quality)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            with node._lock:
                det = node._det_result
            if det is not None:
                status = (f'[{"VALID" if det.pose_valid else "INVALID"}] '
                          f'corners={det.n_corners}  '
                          f'reproj={det.reproj_err:.2f}px')
                print(f'\r{status}', end='', flush=True)
    except KeyboardInterrupt:
        print('\n[종료]')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# 15. CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='NERO ChArUco Eye-in-Hand Hand-Eye Calibration System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument('--mode', choices=['calibrate', 'validate', 'visualize', 'generate-board'],
                   default='calibrate', help='실행 모드')
    p.add_argument('--dry-run', action='store_true',
                   help='robot을 이동하지 않고 계획만 출력')

    # Board 설정
    board = p.add_argument_group('ChArUco Board 파라미터')
    board.add_argument('--squares-x', type=int, default=10, help='가로 칸 수 (기본: 10)')
    board.add_argument('--squares-y', type=int, default=7,  help='세로 칸 수 (기본: 7)')
    board.add_argument('--square-length', type=float, default=0.040,
                       help='칸 크기 m (기본: 0.040=40mm, 실측값으로 반드시 수정)')
    board.add_argument('--marker-length', type=float, default=0.028,
                       help='ArUco marker 크기 m (기본: 0.028=28mm)')
    board.add_argument('--dictionary', default='DICT_4X4_50',
                       choices=[k for k in dir(cv2.aruco) if k.startswith('DICT_')],
                       help='ArUco dictionary (기본: DICT_4X4_50)')

    # Solver
    solver_g = p.add_argument_group('Hand-Eye Solver')
    solver_g.add_argument('--solver', choices=list(SOLVER_METHODS.keys()),
                           default='TSAI', help='Hand-eye solver (기본: TSAI)')
    solver_g.add_argument('--compare-methods', action='store_true',
                           help='모든 solver 방법 결과 비교')

    # Quality gate
    qg = p.add_argument_group('Quality Gate')
    qg.add_argument('--min-corners', type=int, default=12,
                    help='최소 ChArUco corner 수 (기본: 12)')

    # 입출력
    io_g = p.add_argument_group('입출력')
    io_g.add_argument('--output', type=str, default=None,
                      help=f'출력 JSON 경로 (기본: {OUTPUT_JSON})')
    io_g.add_argument('--calib-file', type=str, default=None,
                      help='검증 모드에서 사용할 calibration JSON')
    io_g.add_argument('--board-output', type=str, default=None,
                      help='board 이미지 출력 경로 (기본: /tmp/charuco_board.png)')
    io_g.add_argument('--dpi', type=float, default=300.0,
                      help='board PNG DPI (기본: 300)')

    # URDF
    urdf_g = p.add_argument_group('URDF 업데이트')
    urdf_g.add_argument('--update-urdf', action='store_true',
                         help='캘리브레이션 후 nero_with_camera.urdf 자동 업데이트')
    urdf_g.add_argument('--urdf-path', type=str, default=None,
                         help=f'URDF 경로 (기본: {URDF_PATH})')

    # 검증
    val_g = p.add_argument_group('검증')
    val_g.add_argument('--gt-pos', type=str, default=None,
                        help='Board GT 위치 (m) 예: "0.5 0.0 0.0" (base_link 기준)')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'generate-board':
        run_generate_board(args)
    elif args.mode == 'visualize':
        run_visualize(args)
    elif args.mode == 'validate':
        run_validate(args)
    elif args.mode == 'calibrate':
        if args.dry_run:
            run_dry_run(args)
        else:
            run_calibrate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
