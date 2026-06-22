#!/usr/bin/env python3
"""
perception_node.py  (박스 전용 단순화 버전)
RealSense RGB-D + 박스 전용 커스텀 YOLO 서버(vlm_boxyolo.py) 클라이언트.

[기존 버전과의 차이]
- 기존: YOLO-World 멀티서버 클러스터에 라운드로빈 디스패치 (cup/bottle/box/book 등 다중 라벨)
- 지금: 박스 단일 클래스만 탐지하는 단일 서버(vlm_boxyolo.py)에 디스패치
  → CLUSTER_N, 라운드로빈, 헬스체크 과반수 로직 등 멀티서버 복잡도 제거
  → /detected_objects 퍼블리시 포맷은 기존과 동일하게 유지 (planning_node 등 하위 노드 변경 불필요)

[추후 다른 객체(cup, bottle 등) 다시 추가할 때]
  BOX_SERVER 외에 일반 객체용 서버(vlm_yoloworld.py)를 별도로 띄우고,
  _send_and_publish에서 두 서버 응답을 합쳐서 publish하도록 확장하면 됨.
  (지금은 범위를 박스로만 좁혀서 일단 단순하게 둠)
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
from std_msgs.msg import String

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from nero_ai.camera_calibration import pixel_to_robot_xyz


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
TARGET_LABEL = os.environ.get("TARGET_LABEL", "box")

BOX_SERVER_URL = os.environ.get("BOX_SERVER_URL", "http://127.0.0.1:8002/detect")
BOX_HEALTH_URL = os.environ.get("BOX_HEALTH_URL", "http://127.0.0.1:8002/health")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "2.0"))
DISPATCH_RATE_HZ = float(os.environ.get("DISPATCH_RATE_HZ", "20.0"))

# 추론 입력 해상도 (네트워크 + GPU 부하 절감)
INFER_W = int(os.environ.get("INFER_W", "320"))
INFER_H = int(os.environ.get("INFER_H", "180"))

# 카메라 해상도
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))

# 오탐 필터
MIN_BBOX_SIZE = 0.02
DEDUP_THRESH = 0.08


def filter_detections(dets):
    """크기 필터 + 중복 제거. dets: [{label,x_min,y_min,x_max,y_max}]."""
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
            self.get_logger().error(
                'pyrealsense2 미설치. pip3 install pyrealsense2 --break-system-packages')
            raise RuntimeError("pyrealsense2 not installed")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(String, '/detected_objects', qos)

        self._init_camera()

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
            f'PerceptionNode(박스 전용) 시작 | target: {TARGET_LABEL} | server: {BOX_SERVER_URL}')

    def _init_camera(self):
        self.rs_pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, CAM_FPS)
        profile = self.rs_pipe.start(cfg)
        self.rs_align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(
            f'RealSense: {CAM_W}x{CAM_H}@{CAM_FPS}fps | depth_scale={self.depth_scale}')

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
            small = cv2.resize(color, (INFER_W, INFER_H))
            ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
        objs = []
        for d in filtered:
            cx_norm = (d["x_min"] + d["x_max"]) / 2
            cy_norm = (d["y_min"] + d["y_max"]) / 2
            cx_px = int(cx_norm * cw)
            cy_px = int(cy_norm * ch)

            depth_m = self._sample_depth(depth, cx_px, cy_px)
            xyz = pixel_to_robot_xyz(cx_px, cy_px, cw, ch, depth_m=depth_m)

            objs.append({
                "label": d["label"],
                "bbox": [d["x_min"], d["y_min"], d["x_max"], d["y_max"]],
                "center_2d": {"x": cx_norm, "y": cy_norm},
                "center_3d": xyz,
                "depth_m": round(float(depth_m), 3) if depth_m else None,
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
