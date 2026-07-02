#!/usr/bin/env python3
"""
perception_node.py  (박스 전용 단순화 버전 v2)
=================================================================================
[변경 이력 vs 기존 nero_ai perception_node]
- 멀티서버 클러스터(CLUSTER_N, 라운드로빈) 제거 → 박스 서버 하나에만 디스패치
- RealSense 노출값 수동 설정 추가 (어두운 환경 대응)
- 이미지 원본 크기(640x480)로 서버에 전송 (다운스케일 시 탐지 누락 방지)
- /detected_objects 퍼블리시 포맷 기존과 동일 유지 (planning_node 변경 불필요)

[2026-07 eye-in-hand 캘리브레이션 통합]
- pixel_to_robot_xyz(호모그래피 기반 평면 가정) 제거
- pixel_to_camera_xyz(표준 핀홀 역투영) + tf2(camera_color_optical_frame
  -> base_link)로 교체. 카메라가 tcp_link에 붙어 팔과 함께 움직이는
  eye-in-hand 구조이므로, 매 detection마다 그 순간의 실제 팔 자세를
  반영한 tf 변환이 필요함 (고정 오프셋 계산으로는 안 됨).

[환경변수]
  BOX_SERVER_URL  : 박스 서버 주소 (기본: http://127.0.0.1:8002/detect)
  BOX_HEALTH_URL  : 헬스체크 주소 (기본: http://127.0.0.1:8002/health)
  CAM_EXPOSURE    : RealSense 노출값 (기본: 500, 0이면 자동노출)
  DISPATCH_RATE_HZ: 추론 요청 주기 (기본: 10Hz)
  TARGET_LABEL    : 탐지 대상 클래스 (기본: box)
  BASE_FRAME      : tf 변환 목표 프레임 (기본: base_link)
  CAMERA_OPTICAL_FRAME : 카메라 광학 프레임 (기본: camera_color_optical_frame)
"""

import os
import json
import base64
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (PointStamped 변환 등록용)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from sj_pickplace.camera_calibration import (
    CameraIntrinsics, set_intrinsics, pixel_to_camera_xyz,
)


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
TARGET_LABEL     = os.environ.get("TARGET_LABEL", "box")
BOX_SERVER_URL   = os.environ.get("BOX_SERVER_URL", "http://127.0.0.1:8002/detect")
BOX_HEALTH_URL   = os.environ.get("BOX_HEALTH_URL", "http://127.0.0.1:8002/health")
REQUEST_TIMEOUT  = float(os.environ.get("REQUEST_TIMEOUT", "3.0"))
DISPATCH_RATE_HZ = float(os.environ.get("DISPATCH_RATE_HZ", "10.0"))

# 카메라 해상도
CAM_W   = int(os.environ.get("CAM_W", "640"))
CAM_H   = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))

# 노출값: 0이면 자동노출, 그 외 수동값 (500 권장)
CAM_EXPOSURE = int(os.environ.get("CAM_EXPOSURE", "500"))

# 오탐 필터
MIN_BBOX_SIZE = 0.02
DEDUP_THRESH  = 0.08

# tf 변환 대상 프레임 (URDF에서 tcp_link 하위에 붙인 광학 프레임과 일치해야 함)
BASE_FRAME = os.environ.get("BASE_FRAME", "base_link")
CAMERA_OPTICAL_FRAME = os.environ.get("CAMERA_OPTICAL_FRAME", "camera_color_optical_frame")
TF_TIMEOUT_SEC = float(os.environ.get("TF_TIMEOUT_SEC", "0.2"))


def filter_detections(dets):
    """크기 필터 + 중복 제거."""
    filtered = []
    for d in dets:
        w = d["x_max"] - d["x_min"]
        h = d["y_max"] - d["y_min"]
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            continue
        cx = d["x_min"] + w / 2
        cy = d["y_min"] + h / 2
        too_close = any(
            abs(cx - (f["x_min"] + (f["x_max"] - f["x_min"]) / 2)) < DEDUP_THRESH
            and abs(cy - (f["y_min"] + (f["y_max"] - f["y_min"]) / 2)) < DEDUP_THRESH
            and f["label"] == d["label"]
            for f in filtered
        )
        if not too_close:
            filtered.append(d)
    return filtered


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        if rs is None:
            self.get_logger().error('pyrealsense2 미설치')
            raise RuntimeError("pyrealsense2 not installed")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(String, "/detected_objects", qos)
        self.pub_image = self.create_publisher(Image, "/camera/color/image_raw", qos)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", qos)

        # ── RealSense 초기화 (이 안에서 intrinsics도 등록됨) ──
        self._init_camera()

        # ── tf2: eye-in-hand라 매 순간 camera->base_link 변환이 바뀜 ──
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.frame_lock = threading.Lock()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        if not self._wait_for_box_server():
            self.get_logger().error(
                f'박스 서버 응답 없음 ({BOX_SERVER_URL}). vlm_boxyolo.py 켜져있는지 확인.')
            raise RuntimeError("box detection server not available")

        threading.Thread(target=self._capture_loop, daemon=True).start()
        self.timer = self.create_timer(1.0 / DISPATCH_RATE_HZ, self._dispatch_inference)

        self.get_logger().info(
            f'PerceptionNode 시작 | target={TARGET_LABEL} | '
            f'server={BOX_SERVER_URL} | exposure={CAM_EXPOSURE} | '
            f'tf: {CAMERA_OPTICAL_FRAME} -> {BASE_FRAME}')

    def _init_camera(self):
        self.rs_pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, CAM_FPS)
        profile = self.rs_pipe.start(cfg)

        self.rs_align = rs.align(rs.stream.color)

        # depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 노출값 설정 (0이면 자동노출 유지)
        if CAM_EXPOSURE > 0:
            try:
                color_sensor = profile.get_device().query_sensors()[1]
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, CAM_EXPOSURE)
                self.get_logger().info(f'RealSense 노출값 수동 설정: {CAM_EXPOSURE}')
            except Exception as e:
                self.get_logger().warn(f'노출값 설정 실패 (자동노출 유지): {e}')
        else:
            self.get_logger().info('RealSense 자동노출 사용')

        self.get_logger().info(
            f'RealSense: {CAM_W}x{CAM_H}@{CAM_FPS}fps | depth_scale={self.depth_scale}')

        # ── intrinsics 등록: 반드시 color 스트림 기준 (depth/IR 아님) ──
        # align(rs.stream.color) 를 썼으므로 depth도 color 픽셀 좌표계에 맞춰져
        # 있음. 따라서 역투영도 color intrinsics로 해야 함 — 다른 렌즈(좌측 IR)
        # intrinsics를 쓰면 렌즈 baseline만큼(15~25mm) 어긋난다.
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        set_intrinsics(CameraIntrinsics.from_realsense_profile(color_profile))
        intr = color_profile.get_intrinsics()
        self._cam_info = CameraInfo()
        self._cam_info.width = intr.width
        self._cam_info.height = intr.height
        self._cam_info.k = [intr.fx, 0.0, intr.ppx, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 1.0]
        self._cam_info.distortion_model = "plumb_bob"

    def _capture_loop(self):
        while rclpy.ok():
            try:
                frames = self.rs_pipe.wait_for_frames(timeout_ms=1000)
                aligned = self.rs_align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue
                color_np = np.asanyarray(color.get_data())
                depth_np = np.asanyarray(depth.get_data()).astype(np.float32)
                depth_np *= self.depth_scale
                with self.frame_lock:
                    img_msg = Image()
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    img_msg.header.frame_id = "camera_color_optical_frame"
                    img_msg.height = color_np.shape[0]
                    img_msg.width = color_np.shape[1]
                    img_msg.encoding = "bgr8"
                    img_msg.step = color_np.shape[1] * 3
                    img_msg.data = color_np.tobytes()
                    self.pub_image.publish(img_msg)
                    self._cam_info.header = img_msg.header
                    self.pub_info.publish(self._cam_info)
                    self.latest_color = color_np
                    self.latest_depth = depth_np
            except Exception as e:
                self.get_logger().warn(f'프레임 수신 실패: {e}')
                time.sleep(0.05)

    def _wait_for_box_server(self, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(BOX_HEALTH_URL, timeout=1.0)
                if r.status_code == 200:
                    self.get_logger().info(f'박스 서버 ready: {r.json()}')
                    return True
            except Exception:
                pass
            self.get_logger().info('박스 서버 대기 중...')
            time.sleep(2.0)
        return False

    def _dispatch_inference(self):
        with self.frame_lock:
            if self.latest_color is None or self.latest_depth is None:
                return
            color = self.latest_color.copy()
            depth = self.latest_depth.copy()

        threading.Thread(
            target=self._send_and_publish, args=(color, depth), daemon=True
        ).start()

    def _send_and_publish(self, color: np.ndarray, depth: np.ndarray):
        try:
            # 원본 크기(640x480)로 전송 — 다운스케일 시 탐지 누락 방지
            ok, buf = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return
            img_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            payload = {"image_b64": img_b64, "labels": [TARGET_LABEL]}
            r = requests.post(BOX_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return
            data = r.json()
        except requests.exceptions.RequestException:
            return
        except Exception as e:
            self.get_logger().warn(f'요청 처리 오류: {e}')
            return

        raw_dets = data.get("detections", [])
        filtered = filter_detections(raw_dets)

        ch, cw = color.shape[:2]
        now_stamp = self.get_clock().now().to_msg()
        objs = []
        for d in filtered:
            cx_norm = (d["x_min"] + d["x_max"]) / 2
            cy_norm = (d["y_min"] + d["y_max"]) / 2
            cx_px = int(cx_norm * cw)
            cy_px = int(cy_norm * ch)

            depth_m = self._sample_depth(depth, cx_px, cy_px)

            # ── 픽셀+depth → 카메라 광학 좌표계 3D 점 (핀홀 역투영) ──
            pt_cam = pixel_to_camera_xyz(cx_px, cy_px, depth_m)
            if pt_cam is None:
                continue  # depth 무효하거나 intrinsics 미설정 → 이 물체는 스킵

            # ── 카메라 좌표계 점 → base_link (eye-in-hand이므로 매번 tf 조회) ──
            pt_stamped = PointStamped()
            pt_stamped.header.frame_id = CAMERA_OPTICAL_FRAME
            pt_stamped.header.stamp = rclpy.time.Time().to_msg()
            pt_stamped.point.x = pt_cam["x"]
            pt_stamped.point.y = pt_cam["y"]
            pt_stamped.point.z = pt_cam["z"]

            try:
                pt_base = self.tf_buffer.transform(
                    pt_stamped, BASE_FRAME, timeout=Duration(seconds=TF_TIMEOUT_SEC))
                xyz = {
                    "x": round(pt_base.point.x, 3),
                    "y": round(pt_base.point.y, 3),
                    "z": round(pt_base.point.z, 3),
                }
            except Exception as e:
                self.get_logger().warn(
                    f'tf 변환 실패 ({CAMERA_OPTICAL_FRAME} -> {BASE_FRAME}): {e}',
                    throttle_duration_sec=5.0)
                continue

            objs.append({
                "label": d["label"],
                "bbox": [d["x_min"], d["y_min"], d["x_max"], d["y_max"]],
                "center_2d": {"x": cx_norm, "y": cy_norm},
                "center_3d": xyz,
                "depth_m": round(float(depth_m), 3) if depth_m else None,
                "confidence": round(float(d.get("confidence", 0.0)), 3),
            })

        msg = String()
        msg.data = json.dumps({"objects": objs}, ensure_ascii=False)
        self.pub.publish(msg)

    @staticmethod
    def _sample_depth(depth: np.ndarray, cx: int, cy: int) -> Optional[float]:
        h, w = depth.shape
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > 0.05) & (patch < 2.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def destroy_node(self):
        try:
            self.rs_pipe.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
