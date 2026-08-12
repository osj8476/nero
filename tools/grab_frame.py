#!/usr/bin/env python3
"""현재 카메라 프레임 1장을 잡아 지정 경로에 저장한다.

probe_raw_detect.py에서 YOLO 서버 호출 부분을 뺀 축소판 --
로봇 자세를 바꿔가며 반복 호출해서 학습 데이터셋용 원본 이미지를
모을 때 쓴다 (라벨링은 이 스크립트의 역할이 아님, sam_labeler.py로
별도 수행).

사용:
  python3 grab_frame.py --out /home/bpdl/dataset/v6_stack_raw/images/f0001.jpg
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/camera/color/image_raw"


class Grabber(Node):
    def __init__(self):
        super().__init__('grab_frame_once')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)
        self.img = None
        self.create_subscription(Image, IMAGE_TOPIC, self.cb, qos)

    def cb(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding not in ('rgb8', 'bgr8'):
            print(f'미지원 encoding: {msg.encoding}', file=sys.stderr)
            return
        img = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.img = img.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="저장할 이미지 경로 (.jpg)")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    rclpy.init()
    node = Grabber()
    t0 = time.time()
    while node.img is None and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)

    if node.img is None:
        print("프레임을 못 받았습니다 (타임아웃)", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), node.img)
    print(f'저장: {out_path} (shape={node.img.shape})')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
