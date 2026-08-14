#!/usr/bin/env python3
"""
calib_visual.py
Hand-eye 캘리브레이션 — RViz 시각화 + 수동 조작 버전.

[사용법]
  python3 /home/bpdl/sj_real/nero/sj_pickplace/scripts/calib_visual.py

[RViz 에 추가할 디스플레이]
  - Image        topic: /calib/image          (카메라+체스보드 오버레이)
  - MarkerArray  topic: /calib/markers         (캡처한 tcp 포즈 좌표축)

[키 입력 — 스크립트 실행 중인 터미널에서]
  Enter  : 현재 자세 샘플 캡처 (체스보드 검출 성공 시만)
  c      : 캘리브레이션 계산 + 저장
  q      : 종료

[진행 순서]
  1. 스크립트 실행
  2. RViz에서 로봇을 원하는 자세로 이동 (체스보드가 카메라에 보이도록)
  3. 터미널에 "[FOUND]" 뜨면 Enter 로 캡처
  4. 자세를 바꿔가며 8~10회 반복 (wrist 회전 다양성이 핵심)
  5. c 입력 → 결과 확인 후 URDF에 반영
"""

import os
import sys
import json
import time
import math
import threading
import select
import termios
import tty
from typing import List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

try:
    import pyrealsense2 as rs
except ImportError:
    print('[오류] pyrealsense2 미설치')
    sys.exit(1)

try:
    from cv_bridge import CvBridge
    _bridge = CvBridge()
except ImportError:
    _bridge = None

# ── 체스보드 설정 ─────────────────────────────────────────────────────────────
CHESSBOARD_SIZE = (9, 7)
SQUARE_SIZE_M   = 0.025

# ── 저장 경로 ─────────────────────────────────────────────────────────────────
CALIB_DIR   = os.path.expanduser(os.environ.get("NERO_CALIB_DIR", "~/.nero_calib"))
OUTPUT_JSON = os.path.join(CALIB_DIR, "flange_to_camera.json")
URDF_PATH   = '/home/bpdl/sj_real/nero/sj_pickplace/nero_with_camera.urdf'

CAM_W, CAM_H, CAM_FPS = 640, 480, 30
MIN_SAMPLES = 8


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


def R_to_quat(R: np.ndarray):
    """회전행렬 → (x,y,z,w) 쿼터니언."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return x, y, z, w


class CalibVisualNode(Node):
    def __init__(self):
        super().__init__('calib_visual')
        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)

        self.latest_pose: Optional[PoseStamped] = None
        self._lock = threading.Lock()
        self._detected_corners = None   # 마지막 프레임 체스보드 코너
        self._last_img: Optional[np.ndarray] = None

        self.create_subscription(PoseStamped, '/feedback/tcp_pose',
                                  self._pose_cb, qos_be)
        self.pub_image   = self.create_publisher(Image,       '/calib/image',   qos_rel)
        self.pub_markers = self.create_publisher(MarkerArray, '/calib/markers', qos_rel)

        # RealSense
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

        objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                                0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp = objp * SQUARE_SIZE_M

        # 샘플 버퍼
        self.R_gripper2base: List[np.ndarray] = []
        self.t_gripper2base: List[np.ndarray] = []
        self.R_target2cam:   List[np.ndarray] = []
        self.t_target2cam:   List[np.ndarray] = []

        # 백그라운드 카메라 루프
        self._running = True
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

    def _pose_cb(self, msg):
        with self._lock:
            self.latest_pose = msg

    # ── 카메라 루프 (백그라운드) ──────────────────────────────────────────────
    def _cam_loop(self):
        while self._running:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=500)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                img = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
                vis = img.copy()
                n = len(self.R_gripper2base)

                if found:
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, found)
                    status_text = f'[FOUND] Enter=캡처  샘플:{n}/{MIN_SAMPLES}'
                    cv2.rectangle(vis, (0, 0), (CAM_W, 36), (0, 180, 0), -1)
                    cv2.putText(vis, status_text, (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    corners = None
                    status_text = f'[체스보드 미검출]  샘플:{n}/{MIN_SAMPLES}'
                    cv2.rectangle(vis, (0, 0), (CAM_W, 36), (0, 0, 180), -1)
                    cv2.putText(vis, status_text, (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 조작 안내
                guide = 'Enter=캡처  c=계산+저장  q=종료'
                cv2.putText(vis, guide, (8, CAM_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                with self._lock:
                    self._detected_corners = corners
                    self._last_img = vis

                # RViz에 이미지 퍼블리시
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
            except Exception:
                pass
        else:
            # cv_bridge 없을 때 수동 변환
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_color_optical_frame'
            msg.height = img.shape[0]
            msg.width  = img.shape[1]
            msg.encoding = 'bgr8'
            msg.step    = img.shape[1] * 3
            msg.data    = img.tobytes()
            self.pub_image.publish(msg)

    # ── 샘플 캡처 ─────────────────────────────────────────────────────────────
    def capture_sample(self) -> bool:
        with self._lock:
            pose    = self.latest_pose
            corners = self._detected_corners

        if pose is None:
            print('  [스킵] tcp_pose 미수신')
            return False
        if corners is None:
            print('  [스킵] 체스보드 미검출 — 카메라에 체스보드가 보이게 조정하세요')
            return False

        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist)
        if not ok:
            print('  [스킵] solvePnP 실패')
            return False

        R_t2c, _ = cv2.Rodrigues(rvec)
        t_t2c    = tvec.reshape(3)

        p     = pose.pose
        R_g2b = quat_to_R(p.orientation.x, p.orientation.y,
                           p.orientation.z, p.orientation.w)
        t_g2b = np.array([p.position.x, p.position.y, p.position.z])

        self.R_gripper2base.append(R_g2b)
        self.t_gripper2base.append(t_g2b)
        self.R_target2cam.append(R_t2c)
        self.t_target2cam.append(t_t2c)

        n = len(self.R_gripper2base)
        print(f'  ✅ 샘플 #{n} 캡처  tcp=({t_g2b[0]:.3f}, {t_g2b[1]:.3f}, {t_g2b[2]:.3f})')
        self._publish_markers()
        return True

    # ── RViz 마커 ─────────────────────────────────────────────────────────────
    def _publish_markers(self):
        ma = MarkerArray()
        n  = len(self.R_gripper2base)

        # 기존 마커 삭제
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)

        for idx, (R, t) in enumerate(zip(self.R_gripper2base, self.t_gripper2base)):
            qx, qy, qz, qw = R_to_quat(R)
            color_ratio = idx / max(n - 1, 1)

            # 좌표축 화살표 3개 (X=빨강, Y=초록, Z=파랑)
            for axis, color in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
                m = Marker()
                m.header.frame_id = 'base_link'
                m.header.stamp    = self.get_clock().now().to_msg()
                m.ns              = 'calib_poses'
                m.id              = idx * 3 + axis + 1
                m.type            = Marker.ARROW
                m.action          = Marker.ADD
                m.pose.position.x = t[0]
                m.pose.position.y = t[1]
                m.pose.position.z = t[2]
                m.pose.orientation.x = qx
                m.pose.orientation.y = qy
                m.pose.orientation.z = qz
                m.pose.orientation.w = qw
                m.scale.x = 0.04
                m.scale.y = 0.005
                m.scale.z = 0.005
                # 축 방향 회전 적용
                if axis == 1:   # Y축
                    m.pose.orientation.x, m.pose.orientation.y, \
                    m.pose.orientation.z, m.pose.orientation.w = \
                        self._quat_mul(qx, qy, qz, qw, 0, 0, 0.707, 0.707)
                elif axis == 2:  # Z축
                    m.pose.orientation.x, m.pose.orientation.y, \
                    m.pose.orientation.z, m.pose.orientation.w = \
                        self._quat_mul(qx, qy, qz, qw, 0, -0.707, 0, 0.707)
                m.color = ColorRGBA(r=float(color[0]), g=float(color[1]),
                                    b=float(color[2]), a=0.85)
                m.lifetime = Duration(sec=0)
                ma.markers.append(m)

            # 샘플 번호 텍스트
            txt = Marker()
            txt.header.frame_id = 'base_link'
            txt.header.stamp    = self.get_clock().now().to_msg()
            txt.ns              = 'calib_labels'
            txt.id              = idx + 1000
            txt.type            = Marker.TEXT_VIEW_FACING
            txt.action          = Marker.ADD
            txt.pose.position.x = t[0]
            txt.pose.position.y = t[1]
            txt.pose.position.z = t[2] + 0.06
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.04
            txt.color   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            txt.text    = f'#{idx+1}'
            txt.lifetime = Duration(sec=0)
            ma.markers.append(txt)

        # 상태 텍스트 (화면 고정 위치)
        status_txt = Marker()
        status_txt.header.frame_id = 'base_link'
        status_txt.header.stamp    = self.get_clock().now().to_msg()
        status_txt.ns     = 'calib_status'
        status_txt.id     = 9999
        status_txt.type   = Marker.TEXT_VIEW_FACING
        status_txt.action = Marker.ADD
        status_txt.pose.position.x = 0.0
        status_txt.pose.position.y = 0.0
        status_txt.pose.position.z = 0.9
        status_txt.pose.orientation.w = 1.0
        status_txt.scale.z = 0.06
        ok_color = (0.2, 1.0, 0.2) if n >= MIN_SAMPLES else (1.0, 0.8, 0.0)
        status_txt.color = ColorRGBA(r=ok_color[0], g=ok_color[1], b=ok_color[2], a=1.0)
        status_txt.text  = f'캘리브레이션 샘플: {n}/{MIN_SAMPLES}'
        if n >= MIN_SAMPLES:
            status_txt.text += '  ✓ c 눌러 계산 가능'
        status_txt.lifetime = Duration(sec=0)
        ma.markers.append(status_txt)

        self.pub_markers.publish(ma)

    @staticmethod
    def _quat_mul(ax, ay, az, aw, bx, by, bz, bw):
        return (
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz,
        )

    # ── 캘리브레이션 계산 ─────────────────────────────────────────────────────
    def solve_and_save(self) -> bool:
        n = len(self.R_gripper2base)
        if n < MIN_SAMPLES:
            print(f'\n[오류] 샘플 {n}개 — 최소 {MIN_SAMPLES}개 필요')
            return False

        print(f'\n[계산] {n}개 샘플로 calibrateHandEye 실행 중...')
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
        spread_max_mm = spread.max() * 1000
        print(f'\n[검증] base→target 표준편차: '
              f'x={spread[0]*1000:.1f}mm  y={spread[1]*1000:.1f}mm  z={spread[2]*1000:.1f}mm')
        if spread_max_mm > 10:
            print(f'  ⚠️  최대 {spread_max_mm:.1f}mm — 10mm 초과. '
                  'wrist 회전 다양성이 부족하거나 체스보드가 움직였을 가능성.')
        else:
            print(f'  ✅ 최대 {spread_max_mm:.1f}mm — 양호')

        # JSON 저장
        os.makedirs(CALIB_DIR, exist_ok=True)
        with open(OUTPUT_JSON, 'w') as f:
            json.dump({'note': 'tcp_link→camera_color_optical_frame (eye-in-hand)',
                       'T_flange_camera': T.tolist()}, f, indent=2)
        print(f'\n[저장] {OUTPUT_JSON}')

        xyz = T[:3, 3]
        roll, pitch, yaw = rot_to_rpy(T[:3, :3])
        origin_str = (f'<origin xyz="{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}" '
                      f'rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}" />')

        print('\n' + '='*70)
        print('URDF camera_joint origin (nero_with_camera.urdf 에 반영)')
        print(origin_str)
        print('='*70)

        # URDF 자동 반영 여부 확인
        self._apply_to_urdf(origin_str, xyz, roll, pitch, yaw)
        return True

    def _apply_to_urdf(self, origin_str, xyz, roll, pitch, yaw):
        """nero_with_camera.urdf 의 camera_joint origin 을 자동으로 교체."""
        if not os.path.exists(URDF_PATH):
            print(f'[경고] URDF 파일 없음: {URDF_PATH}')
            return
        with open(URDF_PATH, 'r') as f:
            content = f.read()

        # camera_joint 블록에서 origin 라인을 찾아 교체
        import re
        # camera_joint 이후 첫 번째 <origin ...> 을 찾아 교체
        pattern = (r'(joint name="camera_joint"[^>]*>.*?<origin\s+)'
                   r'(xyz="[^"]*"\s+rpy="[^"]*")')
        new_origin_attrs = (f'xyz="{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}" '
                            f'rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}"')
        new_content, count = re.subn(pattern, r'\g<1>' + new_origin_attrs,
                                     content, count=1, flags=re.DOTALL)
        if count == 0:
            print('[경고] URDF에서 camera_joint origin 패턴을 찾지 못했습니다.')
            print(f'       수동으로 아래 줄을 반영하세요:\n  {origin_str}')
            return

        # 백업 저장
        backup = URDF_PATH + '.bak'
        with open(backup, 'w') as f:
            f.write(content)
        with open(URDF_PATH, 'w') as f:
            f.write(new_content)
        print(f'[URDF] 자동 반영 완료 (백업: {backup})')
        print('  → planning_node / robot_state_publisher 재시작 후 적용됩니다.')

    def cleanup(self):
        self._running = False
        self._cam_thread.join(timeout=2.0)
        try:
            self.pipe.stop()
        except Exception:
            pass


# ── 터미널 키 입력 (non-blocking) ─────────────────────────────────────────────
def _key_available() -> bool:
    return select.select([sys.stdin], [], [], 0.0)[0] != []


def _read_key() -> str:
    return sys.stdin.read(1)


def main():
    rclpy.init()
    node = CalibVisualNode()

    print('\n=== Hand-Eye 캘리브레이션 (RViz 시각화 모드) ===')
    print(f'체스보드: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}, '
          f'사각 {SQUARE_SIZE_M*1000:.0f}mm\n')
    print('RViz 에 아래 디스플레이를 추가하세요:')
    print('  Image       →  /calib/image')
    print('  MarkerArray →  /calib/markers\n')
    print('조작 (이 터미널에서):')
    print('  Enter : 샘플 캡처 (체스보드 검출 시)')
    print('  c     : 캘리브레이션 계산 + 저장')
    print('  q     : 종료\n')

    # raw 모드로 전환해 Enter 없이 즉시 키 인식
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            if _key_available():
                ch = _read_key()

                if ch in ('\r', '\n', ' '):   # Enter or Space → 캡처
                    node.capture_sample()

                elif ch.lower() == 'c':
                    node.solve_and_save()
                    break

                elif ch.lower() == 'q':
                    print('\n[종료]')
                    break

    except KeyboardInterrupt:
        print('\n[중단]')
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
