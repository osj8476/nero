#!/usr/bin/env python3
"""
visualize_3d_ros.py  (v3)
/camera/color/image_raw + /camera/camera_info + /detected_objects 구독.
- 초록색: 카메라 좌표계 XYZ
- 노란색: base_link 기준 XYZ (실물 로봇 joint state 반영)

실행:
  source /opt/ros/jazzy/setup.bash
  source ~/sj/ros2_ws/install/setup.bash
  /usr/bin/python3 ~/sj/visualize_3d_ros.py
종료: q
"""

import json
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo


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


def main():
    rclpy.init()
    node = Visualizer3D()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("[info] 시작. q 누르면 종료.")

    try:
        while True:
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

                # 카메라 좌표계 XYZ 계산 (핀홀 역투영)
                if fx and depth_m:
                    cam_x = (cx_px - cx_i) * depth_m / fx
                    cam_y = (cy_px - cy_i) * depth_m / fy
                    cam_z = depth_m
                else:
                    cam_x = cam_y = cam_z = None

                # base_link XYZ
                bx = c3d.get("x"); by = c3d.get("y"); bz = c3d.get("z")

                # bbox + 중심점
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(disp, (cx_px, cy_px), 6, (0, 0, 255), -1)

                # bbox 중앙 위 — 카메라 XYZ (초록)
                if cam_x is not None:
                    coord = f"cam({cam_x:.2f},{cam_y:.2f},{cam_z:.2f})"
                    (tw, th), _ = cv2.getTextSize(coord, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
                    tx = x1 + (x2-x1-tw)//2
                    cv2.putText(disp, coord, (tx, (y1+y2)//2 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

                # bbox 중앙 아래 — base_link XYZ (노란)
                if bx is not None:
                    b_coord = f"base({bx:.2f},{by:.2f},{bz:.2f})"
                    (tw, th), _ = cv2.getTextSize(b_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
                    tx = x1 + (x2-x1-tw)//2
                    cv2.putText(disp, b_coord, (tx, (y1+y2)//2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)

            cv2.imshow("3D Detection (base_link)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
