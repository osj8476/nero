#!/usr/bin/env python3
"""
visualize_detection.py
RealSense 카메라 피드에 박스 탐지 결과를 실시간으로 그려주는 시각화 스크립트.

실행:
  python3 visualize_detection.py
  python3 visualize_detection.py --server http://127.0.0.1:8002 --conf 0.4

종료: q 키
"""

import argparse
import base64
import time

import cv2
import numpy as np
import requests

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pip3 install pyrealsense2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8002/detect")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--label", default="box")
    args = parser.parse_args()

    # RealSense 초기화
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    print("[info] RealSense 시작. 'q' 누르면 종료.")

    last_dets = []
    last_infer_ms = 0.0
    last_sent = 0.0
    INFER_INTERVAL = 0.1  # 10Hz로 서버 요청

    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            h, w = img.shape[:2]

            now = time.time()
            if now - last_sent >= INFER_INTERVAL:
                last_sent = now
                small = cv2.resize(img, (320, 240))
                _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(buf.tobytes()).decode()
                try:
                    r = requests.post(args.server,
                                      json={"image_b64": b64, "labels": [args.label]},
                                      timeout=1.0)
                    if r.status_code == 200:
                        data = r.json()
                        last_dets = data.get("detections", [])
                        last_infer_ms = data.get("inference_ms", 0.0)
                except Exception:
                    pass

            # bbox 그리기
            disp = img.copy()
            for d in last_dets:
                if d.get("confidence", 1.0) < args.conf:
                    continue
                x1 = int(d["x_min"] * w)
                y1 = int(d["y_min"] * h)
                x2 = int(d["x_max"] * w)
                y2 = int(d["y_max"] * h)
                conf = d.get("confidence", 0.0)

                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_txt = f"{d['label']} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(disp, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(disp, label_txt, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 상태 표시
            status = f"det: {len(last_dets)}  infer: {last_infer_ms:.1f}ms  q:quit"
            cv2.putText(disp, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Box Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
