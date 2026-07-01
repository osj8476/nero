#!/usr/bin/env python3
"""
hand_eye_calibration.py
Eye-in-hand 캘리브레이션: tcp_link → camera_color_optical_frame 고정 변환을 구한다.

[사용 전제]
- 로봇이 실행 중이고 /feedback/tcp_pose (geometry_msgs/PoseStamped,
  base_link 기준)가 publish 되고 있음
- RealSense가 USB로 연결돼 있고 pyrealsense2 사용 가능
- 체스보드(코너 개수 CHESSBOARD_SIZE, 사각 한 변 SQUARE_SIZE_M)를
  작업 테이블 등 로봇 시야 안 고정 위치에 둠 (캘리브레이션 도중 절대
  움직이면 안 됨 — 보드가 base_link 기준 고정이라는 게 이 방법의 전제)

[사용 절차]
1. CHESSBOARD_SIZE, SQUARE_SIZE_M 을 실제 체스보드 실측값으로 수정
2. 로봇을 teach mode 등 수동 조작 가능한 상태로 두고 이 스크립트 실행
3. 카메라가 체스보드를 잘 보는 자세로 팔을 이동 (teach pendant/teleop 등)
4. 팔을 완전히 멈춘 뒤 콘솔에서 Enter → 현재 자세에서
   (tcp_pose, chessboard 코너) 샘플 캡처
5. 팔 각도/거리/방향을 다양하게 바꿔가며 최소 8~10곳 이상 반복
   (★ 회전이 다양해야 함. 같은 각도로 평행이동만 하면 수학적으로 안 풀림)
6. 'c' 입력 → cv2.calibrateHandEye 실행, 결과를 저장하고 URDF에 넣을
   xyz/rpy 값을 출력
7. 출력된 값을 agx_arm.urdf.xacro 의 camera_joint origin 에 반영
   (camera_link 는 flange_link 가 아니라 **tcp_link** 의 자식으로 붙일 것 —
   /feedback/tcp_pose 가 바로 tcp_link 의 pose이기 때문에 이 스크립트가
   구하는 extrinsic도 tcp_link 기준이다)

[의존성]
    pip install opencv-python --break-system-packages
    (pyrealsense2, rclpy는 이미 설치돼 있다고 가정)
"""

import os
import json
from typing import List, Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


# ── 체스보드 설정 (반드시 실측값으로 수정) ──────────────────────────────
CHESSBOARD_SIZE = (9, 6)       # 내부 코너 개수 (가로, 세로)
SQUARE_SIZE_M = 0.025          # 사각 한 변 길이 (미터)

CALIB_DIR = os.path.expanduser(os.environ.get("NERO_CALIB_DIR", "~/.nero_calib"))
OUTPUT_PATH = os.path.join(CALIB_DIR, "flange_to_camera.json")

CAM_W, CAM_H, CAM_FPS = 640, 480, 30


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rot_to_rpy(R: np.ndarray):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


class HandEyeCalibNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calib_node')
        self.latest_pose: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, '/feedback/tcp_pose',
                                  self._pose_cb, 10)

        if rs is None:
            self.get_logger().error('pyrealsense2 미설치')
            raise RuntimeError('pyrealsense2 not available')

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        profile = self.pipe.start(cfg)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                            [0, intr.fy, intr.ppy],
                            [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(intr.coeffs[:5], dtype=np.float64)
        self.get_logger().info(
            f'RealSense color intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} '
            f'cx={intr.ppx:.2f} cy={intr.ppy:.2f}')

        # 체스보드 3D 코너 (보드 자체 좌표계, z=0 평면)
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp = objp * SQUARE_SIZE_M

        self.R_gripper2base: List[np.ndarray] = []
        self.t_gripper2base: List[np.ndarray] = []
        self.R_target2cam: List[np.ndarray] = []
        self.t_target2cam: List[np.ndarray] = []

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose = msg

    def get_color_frame(self):
        frames = self.pipe.wait_for_frames(timeout_ms=1000)
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def capture_sample(self) -> bool:
        """현재 자세 + 체스보드 코너를 한 세트 캡처. 성공하면 True."""
        if self.latest_pose is None:
            print('  ⚠️ 아직 /feedback/tcp_pose 수신 안 됨')
            return False

        img = self.get_color_frame()
        if img is None:
            print('  ⚠️ 카메라 프레임 없음')
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not found:
            print('  ⚠️ 체스보드 코너 검출 실패 — 각도/거리 조정 후 재시도')
            return False

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist)
        if not ok:
            print('  ⚠️ solvePnP 실패')
            return False

        R_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = tvec.reshape(3)

        p = self.latest_pose.pose
        R_gripper2base = quat_to_R(p.orientation.x, p.orientation.y,
                                    p.orientation.z, p.orientation.w)
        t_gripper2base = np.array([p.position.x, p.position.y, p.position.z])

        self.R_gripper2base.append(R_gripper2base)
        self.t_gripper2base.append(t_gripper2base)
        self.R_target2cam.append(R_target2cam)
        self.t_target2cam.append(t_target2cam)

        print(f'  ✅ 샘플 #{len(self.R_gripper2base)} 캡처 완료 '
              f'(tcp=[{t_gripper2base[0]:.3f}, {t_gripper2base[1]:.3f}, '
              f'{t_gripper2base[2]:.3f}])')
        return True

    def solve(self) -> Optional[np.ndarray]:
        n = len(self.R_gripper2base)
        if n < 6:
            print(f'⚠️ 샘플 {n}개뿐 — 최소 6~8개, 가능하면 10개 이상 권장. '
                  f'샘플을 더 모은 뒤 다시 c 를 입력하세요.')
            return None

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        T = np.eye(4)
        T[:3, :3] = R_cam2gripper
        T[:3, 3] = t_cam2gripper.reshape(3)

        # 재투영 일관성으로 대략적인 품질 체크:
        # 모든 샘플에서 추정한 base→target 위치가 서로 얼마나 흩어져 있는지
        errors = []
        for Rg, tg, Rt, tt in zip(self.R_gripper2base, self.t_gripper2base,
                                    self.R_target2cam, self.t_target2cam):
            T_g2b = np.eye(4); T_g2b[:3, :3] = Rg; T_g2b[:3, 3] = tg
            T_t2c = np.eye(4); T_t2c[:3, :3] = Rt; T_t2c[:3, 3] = tt
            T_b2t_est = T_g2b @ T @ T_t2c
            errors.append(T_b2t_est[:3, 3])
        spread = np.array(errors).std(axis=0)
        print(f'\n[검증] 추정된 base→target 위치의 샘플간 표준편차 '
              f'(작을수록 좋음): x={spread[0]*1000:.1f}mm '
              f'y={spread[1]*1000:.1f}mm z={spread[2]*1000:.1f}mm')
        if spread.max() > 0.01:
            print('  ⚠️ 10mm 이상 — 샘플 자세 다양성 부족했거나 체스보드가 '
                  '캘리브레이션 도중 움직였을 가능성. 재측정 권장.')

        return T

    def save(self, T: np.ndarray):
        os.makedirs(CALIB_DIR, exist_ok=True)
        with open(OUTPUT_PATH, 'w') as f:
            json.dump({
                'note': 'tcp_link -> camera_color_optical_frame 4x4 변환 (eye-in-hand)',
                'T_flange_camera': T.tolist(),
            }, f, indent=2)
        print(f'\n✅ 저장 완료: {OUTPUT_PATH}')

        xyz = T[:3, 3]
        roll, pitch, yaw = rot_to_rpy(T[:3, :3])

        print('\n--- URDF camera_joint origin (tcp_link 기준, 아래 값 그대로 복사) ---')
        print(f'<origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" '
              f'rpy="{roll:.4f} {pitch:.4f} {yaw:.4f}" />')


def main():
    rclpy.init()
    node = HandEyeCalibNode()

    print('=== Hand-Eye Calibration (eye-in-hand, tcp_link 기준) ===')
    print('체스보드가 잘 보이는 자세로 팔을 이동한 뒤 완전히 멈추고:')
    print('  Enter : 현재 자세 샘플 캡처')
    print('  c     : 캘리브레이션 계산 + 저장')
    print('  q     : 종료 (저장 안 함)\n')

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            cmd = input('> ').strip().lower()
            if cmd == 'q':
                break
            elif cmd == 'c':
                T = node.solve()
                if T is not None:
                    node.save(T)
                    break
            else:
                node.capture_sample()
    except KeyboardInterrupt:
        pass
    finally:
        node.pipe.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
