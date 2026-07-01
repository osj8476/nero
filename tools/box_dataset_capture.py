#!/usr/bin/env python3
"""
box_dataset_capture.py
RealSense로 직육면체 박스 YOLO 학습용 이미지를 수집하는 스크립트.

[목적]
커스텀 YOLO 학습에 쓸 박스 원본 이미지를 다양한 각도/거리/배경/조명에서 모은다.
여기서 모은 이미지는 이후 라벨링 단계(Roboflow / labelImg / 가라벨+보정)에서
bbox 라벨을 붙이게 된다.

[조작]
  s : 현재 프레임 1장 저장
  a : 자동 캡처 모드 ON/OFF (--interval 간격마다 자동 저장 → 박스를 천천히 움직이며 촬영)
  q : 종료

[실행 예]
  pip install pyrealsense2 opencv-python
  python box_dataset_capture.py --out dataset/box/images --interval 0.8

[촬영 가이드]
  - 최소 200~300장 권장 (많을수록 좋음, 다양성이 양보다 중요)
  - 각도: 정면 / 측면 / 위에서 / 비스듬히 — 골고루
  - 거리: 가까이~멀리 (실제 pick 거리 범위 포함)
  - 배경: 책상 위, 다른 물체와 같이, 빈 배경 등 섞어서
  - 조명: 형광등 켠/끈 상태, 그림자 있는 경우도 일부 포함
  - 박스가 잘려서 화면 밖으로 나가는 경우도 일부 섞으면 실전에서 더 안정적
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pyrealsense2 미설치: pip install pyrealsense2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset/box/images", help="이미지 저장 폴더")
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--interval", type=float, default=1.0, help="자동 캡처 간격(초)")
    parser.add_argument("--prefix", default="box")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 이미 저장된 파일 수 확인 (재실행 시 이어서 번호 매기기)
    existing = [f for f in os.listdir(args.out) if f.startswith(args.prefix)]
    idx = len(existing)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.cam_w, args.cam_h, rs.format.bgr8, args.cam_fps)
    pipe.start(cfg)

    auto_mode = False
    last_capture = 0.0

    print(f"[info] 저장 경로: {args.out}")
    print(f"[info] 시작 인덱스: {idx}")
    print("[info] s=저장 / a=자동캡처 토글 / q=종료")

    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            disp = img.copy()
            mode_txt = f"AUTO({args.interval}s)" if auto_mode else "MANUAL"
            cv2.putText(disp, f"saved: {idx}  mode: {mode_txt}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp, "s:save  a:auto  q:quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("box capture", disp)

            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            save = False
            if key == ord('s'):
                save = True
            elif key == ord('a'):
                auto_mode = not auto_mode
                last_capture = now
            elif key == ord('q'):
                break

            if auto_mode and (now - last_capture) >= args.interval:
                save = True
                last_capture = now

            if save:
                ts = datetime.now().strftime("%H%M%S%f")
                fname = f"{args.prefix}_{idx:04d}_{ts}.jpg"
                path = os.path.join(args.out, fname)
                cv2.imwrite(path, img)
                idx += 1
                print(f"[save] {fname}  (total {idx})")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print(f"[done] 총 {idx}장 저장됨 → {args.out}")


if __name__ == "__main__":
    main()
