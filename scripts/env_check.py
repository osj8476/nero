#!/usr/bin/env python3
"""
env_check.py
박스 YOLO 학습/추론 전에 환경이 준비됐는지 점검하는 스크립트.

확인 항목:
  1. torch / CUDA / GPU
  2. ultralytics import
  3. pyrealsense2 + RealSense 카메라 1프레임 캡처
  4. (옵션) rclpy / ROS2 환경
  5. (옵션) 기존 YOLO-World 서버 health 응답

실행:
  python env_check.py
  python env_check.py --check-server http://127.0.0.1:8000
"""

import argparse
import sys


def check_torch():
    print("\n[1] torch / CUDA")
    try:
        import torch
        print(f"  torch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_ok}")
        if cuda_ok:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM: {vram:.1f} GB")
        else:
            print("  [WARN] CUDA 미사용 — 학습/추론이 CPU로 돌아 매우 느려짐")
        return True
    except ImportError:
        print("  [FAIL] torch 미설치 → pip install torch")
        return False


def check_ultralytics():
    print("\n[2] ultralytics")
    try:
        import ultralytics
        print(f"  ultralytics version: {ultralytics.__version__}")
        return True
    except ImportError:
        print("  [FAIL] 미설치 → pip install ultralytics")
        return False


def check_realsense():
    print("\n[3] RealSense 카메라")
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  [FAIL] pyrealsense2 미설치 → pip install pyrealsense2")
        return False

    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        frames = pipe.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        ok = color is not None
        pipe.stop()
        print(f"  카메라 연결: {'OK' if ok else 'FAIL (프레임 없음)'}")
        return ok
    except Exception as e:
        print(f"  [FAIL] 카메라 오픈 실패: {e}")
        return False


def check_ros2():
    print("\n[4] ROS2 (rclpy)")
    try:
        import rclpy
        print("  rclpy import OK")
        return True
    except ImportError:
        print("  [WARN] rclpy 미설치/미소싱 — ROS2 환경 source 했는지 확인 (source /opt/ros/<distro>/setup.bash)")
        return False


def check_server(url: str):
    print(f"\n[5] 서버 health 체크: {url}")
    try:
        import requests
        r = requests.get(f"{url}/health", timeout=2.0)
        print(f"  status {r.status_code}: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"  [WARN] 응답 없음 ({e}) — 서버가 안 켜져있으면 정상")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-server", default=None, help="예: http://127.0.0.1:8000")
    args = parser.parse_args()

    results = {
        "torch/cuda": check_torch(),
        "ultralytics": check_ultralytics(),
        "realsense": check_realsense(),
        "ros2": check_ros2(),
    }
    if args.check_server:
        results["server"] = check_server(args.check_server)

    print("\n" + "=" * 40)
    print("결과 요약")
    for k, v in results.items():
        print(f"  {k:12s}: {'OK' if v else 'FAIL/WARN'}")
    print("=" * 40)


if __name__ == "__main__":
    main()
