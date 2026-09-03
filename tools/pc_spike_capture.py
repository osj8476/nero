#!/usr/bin/env python3
"""
pc_spike_capture.py  — grasp generation 착수 전 point cloud sanity spike (1/2)

RealSense에서 depth(미터) + color + color-stream intrinsics 를 한 프레임씩
저장한다. 기존 box_dataset_capture.py / grab_frame.py 는 컬러만 저장해서
point cloud 품질 평가에 못 쓴다 — 이 스크립트는 depth 를 같이 남긴다.

핵심:
  - rs.align(rs.stream.color) 후 저장 (camera_calibration.py 모듈 주석의
    "depth intrinsics 쓰지 말고 color stream profile 에서 뽑아라" 원칙).
  - depth 는 device depth_scale 로 곱해 **미터**로 저장 (NERO 전체 관례,
    point_cloud.py DEFAULT_DEPTH_* 와 동일 단위).

실행 (카메라 물려 있는 머신에서):
    pip install pyrealsense2 opencv-python numpy
    python3 pc_spike_capture.py --out ~/pc_spike --prefix scene

조작:  s = 현재 프레임 저장 / q = 종료 / a = 자동(--interval 간격) 토글

촬영 가이드 (spike 목적: "실물 depth 가 grasp net 에 쓸 만한가"):
  - 실제 pick 대상 물체를 실제 observation 자세·거리에서 (무지 골판지 box,
    cup, bottle, 얇은 물체 각 1~2씬)
  - 물체 하나만 있는 씬 / 2~3개 섞인 씬 / 물체가 테이블 가장자리에 걸친 씬
  - 최소 8~10 프레임. 같은 씬을 각도 조금씩 바꿔가며 몇 장.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pyrealsense2 미설치: pip install pyrealsense2")


def depth_to_viz(depth_m: np.ndarray, dmax: float = 2.0) -> np.ndarray:
    """depth(m) → 컬러맵 미리보기. 무효(0/NaN)는 검정."""
    d = np.nan_to_num(depth_m, nan=0.0)
    d = np.clip(d / dmax, 0.0, 1.0)
    viz = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    viz[depth_m <= 0.0] = 0
    return viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/pc_spike", help="저장 폴더")
    ap.add_argument("--prefix", default="scene")
    ap.add_argument("--cam-w", type=int, default=1280)
    ap.add_argument("--cam-h", type=int, default=720)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--interval", type=float, default=1.0, help="자동 캡처 간격(초)")
    ap.add_argument("--no-preview", action="store_true", help="창 없이 자동 캡처만")
    ap.add_argument("--auto-n", type=int, default=0, help=">0 이면 그만큼 자동 저장 후 종료")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.cam_w, args.cam_h, rs.format.z16, args.cam_fps)
    cfg.enable_stream(rs.stream.color, args.cam_w, args.cam_h, rs.format.bgr8, args.cam_fps)
    profile = pipe.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # z16 unit → meters
    align = rs.align(rs.stream.color)

    dev = profile.get_device()
    print(f"[capture] {dev.get_info(rs.camera_info.name)} "
          f"sn={dev.get_info(rs.camera_info.serial_number)} depth_scale={depth_scale:.6f}")

    saved = 0
    auto = args.auto_n > 0
    last_auto = 0.0
    try:
        # 노출 안정화
        for _ in range(15):
            pipe.wait_for_frames()

        while True:
            frames = align.process(pipe.wait_for_frames())
            df = frames.get_depth_frame()
            cf = frames.get_color_frame()
            if not df or not cf:
                continue

            color = np.asanyarray(cf.get_data())               # (H,W,3) bgr8
            depth_raw = np.asanyarray(df.get_data())            # (H,W) uint16
            depth_m = depth_raw.astype(np.float32) * depth_scale

            # aligned color stream 의 intrinsics (이걸 써야 함)
            intr = cf.profile.as_video_stream_profile().get_intrinsics()
            K = np.array([intr.fx, intr.fy, intr.ppx, intr.ppy], dtype=np.float64)

            do_save = False
            if auto and (time.time() - last_auto) >= args.interval:
                do_save = True
                last_auto = time.time()

            if not args.no_preview:
                viz = np.hstack([color, depth_to_viz(depth_m)])
                hole = float((depth_m <= 0.0).mean())
                cv2.putText(viz, f"saved={saved}  hole={hole*100:.1f}%  auto={auto}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("pc_spike_capture (s=save q=quit a=auto)", viz)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord('s'):
                    do_save = True
                if k == ord('a'):
                    auto = not auto

            if do_save:
                idx = saved
                stem = out_dir / f"{args.prefix}_{idx:03d}"
                np.savez_compressed(
                    str(stem) + ".npz",
                    depth_m=depth_m, color=color, K=K,
                    depth_scale=depth_scale,
                    cam_wh=np.array([args.cam_w, args.cam_h]),
                    timestamp=time.time(),
                    note=datetime.now().isoformat(),
                )
                cv2.imwrite(str(stem) + "_color.png", color)
                cv2.imwrite(str(stem) + "_depthviz.png", depth_to_viz(depth_m))
                saved += 1
                print(f"[capture] saved {stem}.npz  hole={(depth_m<=0).mean()*100:.1f}%")
                if args.auto_n and saved >= args.auto_n:
                    break

    finally:
        pipe.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()
    print(f"[capture] done. {saved} frames → {out_dir}")


if __name__ == "__main__":
    main()
