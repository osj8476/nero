#!/usr/bin/env python3
"""
perception_node.py
RealSense RGB-D + MoonDream2 멀티서버 클라이언트 버전.

[변경점]
1. USB 웹캠 → Intel RealSense (pyrealsense2)
   - RGB + depth 동시 수신, 정렬(align)
2. 단일 모델 인스턴스 → N개 서버에 round-robin 분산
   - moondream_server 가 :8000 ~ :8000+N-1 에서 돈다고 가정
   - perception_node 는 얇은 클라이언트만 됨 (GPU 메모리 점유 X)
3. depth 값으로 실제 z 측정 → camera_calibration.pixel_to_robot_xyz 에 전달

다른 RGB-D 카메라(Orbbec 등) 쓰면 _init_camera / _capture_loop 만 갈아끼우면 됨.
"""

import os
import json
import base64
import threading
import time
from collections import deque
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
TARGET_OBJECTS = os.environ.get(
    "TARGET_OBJECTS", "cup,bottle,scissors,phone,box,book"
).split(",")
TARGET_OBJECTS = [t.strip() for t in TARGET_OBJECTS if t.strip()]

# 추론 클러스터
CLUSTER_HOST = os.environ.get("CLUSTER_HOST", "127.0.0.1")
CLUSTER_N = int(os.environ.get("CLUSTER_N", "10"))
BASE_PORT = int(os.environ.get("BASE_PORT", "8000"))
SERVER_URLS = [
    f"http://{CLUSTER_HOST}:{BASE_PORT + i}/detect"
    for i in range(CLUSTER_N)
]
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

        # /detected_objects 는 최신만 중요
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(String, '/detected_objects', qos)

        # ── RealSense 초기화 ──
        self._init_camera()

        # ── 공유 상태 ──
        self.frame_lock = threading.Lock()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None  # depth in meters
        self.frame_idx = 0

        # ── 클러스터 헬스체크 ──
        ready = self._wait_for_cluster()
        if ready == 0:
            self.get_logger().error(
                f'클러스터 응답 없음. scripts/run_cluster.sh start {CLUSTER_N} 실행했는지 확인.')
            raise RuntimeError("no moondream servers available")
        self.get_logger().info(f'클러스터 {ready}/{CLUSTER_N} 서버 ready')

        # ── 스레드 시작 ──
        threading.Thread(target=self._capture_loop, daemon=True).start()

        # 디스패치 타이머 (한 프레임당 1요청 round-robin)
        self.timer = self.create_timer(
            1.0 / DISPATCH_RATE_HZ, self._dispatch_inference)

        self.get_logger().info(
            f'PerceptionNode 시작 | 대상: {TARGET_OBJECTS} | '
            f'서버: {CLUSTER_N}개 @ {CLUSTER_HOST}:{BASE_PORT}+')

    def _init_camera(self):
        self.rs_pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, CAM_FPS)
        profile = self.rs_pipe.start(cfg)

        # depth → color 정렬 (같은 픽셀 좌표에서 depth 조회 가능하게)
        self.rs_align = rs.align(rs.stream.color)

        # depth scale: raw uint16 → meters
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
                depth_np *= self.depth_scale  # → meters
                with self.frame_lock:
                    self.latest_color = color_np
                    self.latest_depth = depth_np
            except Exception as e:
                self.get_logger().warn(f'프레임 수신 실패: {e}')
                time.sleep(0.05)

    def _wait_for_cluster(self, timeout: float = 90.0) -> int:
        """클러스터가 ready 될 때까지 대기. 응답한 서버 개수 반환."""
        deadline = time.time() + timeout
        ready_count = 0
        while time.time() < deadline:
            ready_count = 0
            for i in range(CLUSTER_N):
                url = f"http://{CLUSTER_HOST}:{BASE_PORT + i}/health"
                try:
                    r = requests.get(url, timeout=0.5)
                    if r.status_code == 200:
                        ready_count += 1
                except Exception:
                    pass
            if ready_count >= max(1, CLUSTER_N // 2):  # 과반수면 시작
                return ready_count
            self.get_logger().info(
                f'클러스터 대기 중... ({ready_count}/{CLUSTER_N})')
            time.sleep(5.0)
        return ready_count

    def _dispatch_inference(self):
        """주기적으로 한 프레임을 한 서버에 던짐. 응답은 별도 스레드에서 처리."""
        with self.frame_lock:
            if self.latest_color is None or self.latest_depth is None:
                return
            color = self.latest_color.copy()
            depth = self.latest_depth.copy()

        # 라운드 로빈
        url = SERVER_URLS[self.frame_idx % len(SERVER_URLS)]
        self.frame_idx += 1

        # 비동기 던지기
        threading.Thread(
            target=self._send_and_publish,
            args=(url, color, depth),
            daemon=True,
        ).start()

    def _send_and_publish(self, url: str,
                           color: np.ndarray, depth: np.ndarray):
        try:
            # 추론용 다운스케일
            small = cv2.resize(color, (INFER_W, INFER_H))
            ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return
            img_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            payload = {
                "image_b64": img_b64,
                "labels": TARGET_OBJECTS,
            }
            r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return
            data = r.json()
        except requests.exceptions.RequestException:
            return  # 한 서버 응답 못 받아도 다른 라운드에서 회복
        except Exception as e:
            self.get_logger().warn(f'요청 처리 오류: {e}')
            return

        # 결과 → /detected_objects 메시지로 변환
        raw_dets = data.get("detections", [])
        filtered = filter_detections(raw_dets)

        ch, cw = color.shape[:2]
        objs = []
        for d in filtered:
            cx_norm = (d["x_min"] + d["x_max"]) / 2
            cy_norm = (d["y_min"] + d["y_max"]) / 2
            cx_px = int(cx_norm * cw)
            cy_px = int(cy_norm * ch)

            # ── depth 조회 (해당 픽셀 주변 5x5 평균) ──
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
        """픽셀 주변 5x5 평균 depth (0 제외) — RealSense 노이즈 완화."""
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