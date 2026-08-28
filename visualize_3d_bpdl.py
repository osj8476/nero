#!/usr/bin/env python3
"""
visualize_3d_bpdl.py  (v4)
/camera/color/image_raw + /camera/camera_info + /detected_objects 구독.
- bbox 색상: 라벨별 고유 색
- bbox 위: 라벨명 + confidence
- bbox 중간 위: cam XYZ (초록)
- bbox 중간 아래: base_link XYZ (노란)

실행:
  source /opt/ros/jazzy/setup.bash
  source /home/bpdl/ros2_ws/install/setup.bash
  python3 /home/bpdl/sj_real/nero/visualize_3d_bpdl.py
종료: q
"""

import json
import signal
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo


def _label_color(label: str) -> tuple:
    """라벨 해시로 결정론적 BGR 색상 생성 (항상 동일 라벨 = 동일 색)."""
    h = (hash(label) * 2654435761) & 0xFFFFFF
    b = max(80, (h >> 16) & 0xFF)
    g = max(80, (h >>  8) & 0xFF)
    r = max(80,  h        & 0xFF)
    return (b, g, r)


class Visualizer3D(Node):
    def __init__(self):
        super().__init__('visualizer_3d')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_img  = self.create_subscription(Image,      '/camera/color/image_raw', self._img_cb,  qos)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera_info',      self._info_cb, qos)
        self.sub_obj  = self.create_subscription(String,     '/detected_objects',         self._obj_cb,  qos)

        self.lock = threading.Lock()
        self.latest_img  = None
        self.latest_objs = []
        self.fx = self.fy = self.cx_i = self.cy_i = None

    def _img_cb(self, msg):
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((msg.height, msg.width, 3))
        # [2026-08-26 수정] ROS 이미지가 rgb8인데 변환 없이 그대로 cv2에
        # 넘겨서(OpenCV는 BGR 기대) R/B 채널이 뒤바뀌어 보이던 버그.
        # perception_node_sim.py의 동일 변환 로직과 맞춤.
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with self.lock:
            self.latest_img = img.copy()

    def _info_cb(self, msg):
        with self.lock:
            k = msg.k
            self.fx   = k[0]
            self.fy   = k[4]
            self.cx_i = k[2]
            self.cy_i = k[5]

    def _obj_cb(self, msg):
        try:
            with self.lock:
                self.latest_objs = json.loads(msg.data).get("objects", [])
        except Exception:
            pass

    def get_state(self):
        with self.lock:
            return (
                self.latest_img.copy() if self.latest_img is not None else None,
                list(self.latest_objs),
                self.fx, self.fy, self.cx_i, self.cy_i,
            )


_running = True

def _stop(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGINT,  _stop)
signal.signal(signal.SIGTERM, _stop)


def main():
    rclpy.init()
    node = Visualizer3D()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("[info] 시작. q 누르면 종료.")

    try:
        while _running:
            img, objs, fx, fy, cx_i, cy_i = node.get_state()

            if img is None:
                cv2.waitKey(30)
                continue

            H, W = img.shape[:2]
            disp = img.copy()

            for obj in objs:
                bbox    = obj.get("bbox", [])
                c2d     = obj.get("center_2d", {})
                c3d     = obj.get("center_3d", {})
                depth_m = obj.get("depth_m")
                conf    = obj.get("confidence", 0.0)
                label   = obj.get("label", "")

                if len(bbox) < 4:
                    continue

                x1 = int(bbox[0] * W); y1 = int(bbox[1] * H)
                x2 = int(bbox[2] * W); y2 = int(bbox[3] * H)
                cx_px = int(c2d.get("x", 0) * W)
                cy_px = int(c2d.get("y", 0) * H)

                color = _label_color(label)

                # ── bbox + 중심점 ──────────────────────────────────────────
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                cv2.circle(disp, (cx_px, cy_px), 6, (0, 0, 255), -1)

                # ── 라벨 + confidence (bbox 위) ────────────────────────────
                tag = f"{label} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_y = max(y1 - 6, th + 4)
                cv2.rectangle(disp, (x1, tag_y - th - 4),
                              (x1 + tw + 4, tag_y + baseline), color, -1)
                cv2.putText(disp, tag, (x1 + 2, tag_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # ── 카메라 XYZ (초록, bbox 중앙 위) ───────────────────────
                if fx and depth_m:
                    cam_x = (cx_px - cx_i) * depth_m / fx
                    cam_y = (cy_px - cy_i) * depth_m / fy
                    cam_z = depth_m
                    coord = f"cam({cam_x:.2f},{cam_y:.2f},{cam_z:.2f})"
                    (tw, _), _ = cv2.getTextSize(
                        coord, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                    tx = x1 + (x2 - x1 - tw) // 2
                    cv2.putText(disp, coord, (tx, (y1 + y2) // 2 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)

                # ── base_link XYZ (노란, bbox 중앙 아래) ──────────────────
                bx = c3d.get("x"); by = c3d.get("y"); bz = c3d.get("z")
                if bx is not None:
                    b_coord = f"base({bx:.2f},{by:.2f},{bz:.2f})"
                    (tw, _), _ = cv2.getTextSize(
                        b_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                    tx = x1 + (x2 - x1 - tw) // 2
                    cv2.putText(disp, b_coord, (tx, (y1 + y2) // 2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)

            cv2.imshow("3D Detection (base_link)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
