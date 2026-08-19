#!/usr/bin/env python3
"""
auto_hand_eye_calib.py
Hand-eye 캘리브레이션 자동화 스크립트.

[전제]
- perception_node가 중지된 상태여야 함 (카메라 점유 충돌 방지)
- 체스보드가 테이블에 고정되어 있어야 함 (캘리브레이션 도중 절대 이동 금지)
- 체스보드는 홈 자세 + joint7=90° 에서 카메라에 보이는 위치에 있어야 함

[사용법]
  python3 -m sj_pickplace.scripts.auto_hand_eye_calib

[동작]
1. 홈 복귀
2. 미리 정의된 포즈 시퀀스로 순서대로 이동
3. 각 포즈에서 체스보드 코너 검출 시도
4. 성공하면 (tcp_pose, 체스보드 코너) 샘플 캡처
5. 8개 이상 모이면 calibrateHandEye 실행 → URDF 값 출력 + JSON 저장
"""

import os
import sys
import json
import time
import math
import threading
from typing import List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import json as _json

try:
    import pyrealsense2 as rs
except ImportError:
    print('[오류] pyrealsense2 미설치')
    sys.exit(1)

# ── 체스보드 설정 ─────────────────────────────────────────────────────────────
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_M   = 0.025

# ── 카메라 설정 ───────────────────────────────────────────────────────────────
CAM_W, CAM_H, CAM_FPS = 640, 480, 30

# ── 저장 경로 ─────────────────────────────────────────────────────────────────
CALIB_DIR   = os.path.expanduser(os.environ.get("NERO_CALIB_DIR", "~/.nero_calib"))
OUTPUT_JSON = os.path.join(CALIB_DIR, "flange_to_camera.json")

# ── 관절 이동 허용 오차(rad) 및 안정화 대기 ──────────────────────────────────
JOINT_TOL      = 0.03   # 목표 관절각과 현재값 차이가 이 이하면 도착으로 판단
SETTLE_SEC     = 1.5    # 도착 판정 후 진동 안정화 대기
MOVE_TIMEOUT   = 15.0   # 포즈 이동 최대 대기 시간(초)
MIN_SAMPLES    = 8

# ── 캘리브레이션 포즈 시퀀스 ─────────────────────────────────────────────────
# [j1, j2, j3, j4, j5, j6, j7] 절대 라디안
# 기준: 홈(모두 0)에서 joint7을 다양하게 회전 + 팔 자세 소폭 변화
# joint7 회전이 hand-eye 에서 가장 중요한 회전 다양성 제공원.
_D = math.radians  # 도 → 라디안 단축
CALIB_POSES = [
    [0.0,  0.0,     0.0,    0.0,   0.0,   0.0,   _D(60)],   # 기본 -30°
    [0.0,  0.0,     0.0,    0.0,   0.0,   0.0,   _D(90)],   # 기본 0°
    [0.0,  0.0,     0.0,    0.0,   0.0,   0.0,   _D(120)],  # 기본 +30°
    [0.0,  _D(-12), 0.0,    0.0,   0.0,   0.0,   _D(70)],   # j2 내림
    [0.0,  _D(-12), 0.0,    0.0,   0.0,   0.0,   _D(90)],
    [0.0,  _D(-12), 0.0,    0.0,   0.0,   0.0,   _D(110)],
    [0.0,  _D(-8),  _D(8),  0.0,   0.0,   0.0,   _D(75)],   # j2+j3 조합
    [0.0,  _D(-8),  _D(8),  0.0,   0.0,   0.0,   _D(95)],
    [0.0,  _D(-8),  _D(8),  0.0,   0.0,   0.0,   _D(115)],
    [0.0,  _D(-5),  0.0,   _D(8),  0.0,   0.0,   _D(80)],   # j4 소폭 변화
    [0.0,  _D(-5),  0.0,   _D(8),  0.0,   0.0,   _D(100)],
]

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']


def quat_to_R(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def rot_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
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


class AutoHandEyeNode(Node):
    def __init__(self):
        super().__init__('auto_hand_eye_calib')
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.latest_pose: Optional[PoseStamped] = None
        self.latest_joints: Optional[dict] = None
        self._latest_result: Optional[dict] = None
        self._lock = threading.Lock()

        self.create_subscription(PoseStamped, '/feedback/tcp_pose',
                                  self._pose_cb, qos_be)
        self.create_subscription(JointState, '/feedback/joint_states',
                                  self._joint_cb, qos_be)
        self.create_subscription(String, '/pick_result',
                                  self._result_cb, qos_rel)

        self.pub_cmd = self.create_publisher(String, '/arm_command', qos_rel)

        # RealSense 초기화
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
        print(f'[카메라] fx={intr.fx:.1f} fy={intr.fy:.1f} '
              f'cx={intr.ppx:.1f} cy={intr.ppy:.1f}')

        # 체스보드 오브젝트 포인트
        objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                                0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp = objp * SQUARE_SIZE_M

        self.R_gripper2base: List[np.ndarray] = []
        self.t_gripper2base: List[np.ndarray] = []
        self.R_target2cam:   List[np.ndarray] = []
        self.t_target2cam:   List[np.ndarray] = []

    def _pose_cb(self, msg):
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

    def _send_cmd_wait(self, payload: dict, timeout: float = MOVE_TIMEOUT) -> bool:
        """arm_command 발행 후 pick_result 성공 응답까지 대기."""
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

    def go_home(self):
        print('[이동] 홈 복귀 중...')
        ok = self._send_cmd_wait({'action': 'go_home'}, timeout=20.0)
        time.sleep(SETTLE_SEC)
        print('[이동] 홈 복귀 완료' if ok else '[경고] 홈 복귀 응답 없음')

    def move_to_pose(self, joint_positions: list) -> bool:
        """절대 관절각(rad)으로 이동. planning_node move_joints 사용."""
        joints_dict = {name: float(joint_positions[i])
                       for i, name in enumerate(JOINT_NAMES)}
        ok = self._send_cmd_wait({'action': 'move_joints', 'joints': joints_dict})
        if ok:
            time.sleep(SETTLE_SEC)
        return ok

    def get_color_frame(self) -> Optional[np.ndarray]:
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            return np.asanyarray(color.get_data()) if color else None
        except Exception:
            return None

    def detect_chessboard(self) -> Optional[np.ndarray]:
        """체스보드 코너를 서브픽셀 정확도로 검출. 실패 시 None."""
        img = self.get_color_frame()
        if img is None:
            return None
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not found:
            return None
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return corners

    def capture_sample(self) -> bool:
        with self._lock:
            pose = self.latest_pose
        if pose is None:
            print('  [스킵] tcp_pose 수신 대기 중')
            return False

        corners = self.detect_chessboard()
        if corners is None:
            print('  [스킵] 체스보드 미검출')
            return False

        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist)
        if not ok:
            print('  [스킵] solvePnP 실패')
            return False

        R_t2c, _ = cv2.Rodrigues(rvec)
        t_t2c    = tvec.reshape(3)

        p = pose.pose
        R_g2b = quat_to_R(p.orientation.x, p.orientation.y,
                           p.orientation.z, p.orientation.w)
        t_g2b = np.array([p.position.x, p.position.y, p.position.z])

        self.R_gripper2base.append(R_g2b)
        self.t_gripper2base.append(t_g2b)
        self.R_target2cam.append(R_t2c)
        self.t_target2cam.append(t_t2c)

        n = len(self.R_gripper2base)
        print(f'  [캡처 #{n}] tcp=({t_g2b[0]:.3f},{t_g2b[1]:.3f},{t_g2b[2]:.3f})')
        return True

    def solve_and_save(self):
        n = len(self.R_gripper2base)
        if n < MIN_SAMPLES:
            print(f'\n[오류] 샘플 {n}개 — 최소 {MIN_SAMPLES}개 필요. 종료합니다.')
            return False

        print(f'\n[캘리브레이션] {n}개 샘플로 계산 중...')
        R_c2g, t_c2g = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam,   self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
        T = np.eye(4)
        T[:3, :3] = R_c2g
        T[:3, 3]  = t_c2g.reshape(3)

        # 품질 검증
        errors = []
        for Rg, tg, Rt, tt in zip(self.R_gripper2base, self.t_gripper2base,
                                    self.R_target2cam,   self.t_target2cam):
            T_g2b = np.eye(4); T_g2b[:3, :3] = Rg; T_g2b[:3, 3] = tg
            T_t2c = np.eye(4); T_t2c[:3, :3] = Rt; T_t2c[:3, 3] = tt
            errors.append((T_g2b @ T @ T_t2c)[:3, 3])
        spread = np.array(errors).std(axis=0)
        print(f'[검증] base→target 표준편차: '
              f'x={spread[0]*1000:.1f}mm y={spread[1]*1000:.1f}mm z={spread[2]*1000:.1f}mm')
        if spread.max() > 0.01:
            print('  ⚠️  10mm 초과 — 자세 다양성 부족하거나 체스보드 이동 가능성')

        # 저장
        os.makedirs(CALIB_DIR, exist_ok=True)
        with open(OUTPUT_JSON, 'w') as f:
            json.dump({'note': 'tcp_link→camera_color_optical_frame (eye-in-hand)',
                       'T_flange_camera': T.tolist()}, f, indent=2)
        print(f'\n[저장] {OUTPUT_JSON}')

        xyz        = T[:3, 3]
        roll, pitch, yaw = rot_to_rpy(T[:3, :3])
        print('\n=== URDF camera_joint origin (이 값을 nero_with_camera.urdf에 반영) ===')
        print(f'<origin xyz="{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}" '
              f'rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}" />')
        print('=' * 70)
        return True

    def cleanup(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def main():
    rclpy.init()
    node = AutoHandEyeNode()

    try:
        print('\n=== 자동 Hand-Eye 캘리브레이션 ===')
        print(f'체스보드 설정: {CHESSBOARD_SIZE}, 사각 {SQUARE_SIZE_M*1000:.0f}mm')
        print(f'목표 샘플: {len(CALIB_POSES)}포즈, 최소 {MIN_SAMPLES}개 캡처\n')

        # 1. 홈 복귀
        node.go_home()
        time.sleep(1.0)

        # 2. 각 포즈 순회
        for i, pose in enumerate(CALIB_POSES):
            deg_str = ', '.join(f'j{j+1}={math.degrees(v):.0f}°'
                                for j, v in enumerate(pose) if abs(v) > 0.01)
            print(f'\n[포즈 {i+1}/{len(CALIB_POSES)}] {deg_str}')

            arrived = node.move_to_pose(pose)
            if not arrived:
                print('  [스킵] 이동 실패')
                continue

            # 체스보드 검출 3회 시도
            captured = False
            for attempt in range(3):
                if node.capture_sample():
                    captured = True
                    break
                time.sleep(0.5)
            if not captured:
                print('  [스킵] 체스보드 검출 실패')

        # 3. 홈 복귀
        print('\n[이동] 홈 복귀...')
        node.go_home()

        # 4. 계산 및 저장
        node.solve_and_save()

    except KeyboardInterrupt:
        print('\n[중단] 사용자가 종료했습니다.')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
