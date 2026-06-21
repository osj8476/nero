"""
realsense_demo.py  -  Windows 환경 RGB-D 클러스터 기반 객체 탐지 데모
========================================================================
Intel RealSense (D435/D455 등) + MoonDream2 클러스터 서버 연동.

[서버 개수]
  --servers 1  -> 단일 서버 baseline (성능 기준)
  --servers 2  -> 2배 throughput
  --servers 3  -> 3배 throughput (RTX 4090 24GB에서 최적)

화면 표시:
- 객체 중심점 + 라벨 표시 + 레이블
- 레이블 아래에 "1.23 m" 형태로 depth 표시
- 상단 통계: 서버 수, latency, throughput, display fps
- --show-depth 옵션으로 depth 컬러맵도 표시

실행 (PowerShell):
    python realsense_demo.py --servers 1
    python realsense_demo.py --servers 3 --show-depth
    python realsense_demo.py --servers 2 --targets "cup,bottle,phone"

설치:
    pip install pyrealsense2 opencv-python numpy requests
"""

import argparse
import base64
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import requests

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError(
        "pyrealsense2 미설치\n"
        "Windows 설치:  pip install pyrealsense2"
    )


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--base-port", type=int, default=8000)
parser.add_argument("--servers", type=int, default=1,
                    help="서버 개수 (1, 2, 3)")
parser.add_argument("--targets", default="cup,bottle,scissors,phone,box,book",
                    help="탐지 레이블, 쉼표구분")
parser.add_argument("--fps", type=float, default=30.0,
                    help="디스패치 fps (서버의 처리속도에 맞게)")
parser.add_argument("--infer-w", type=int, default=320)
parser.add_argument("--infer-h", type=int, default=180)
parser.add_argument("--cam-w", type=int, default=640)
parser.add_argument("--cam-h", type=int, default=480)
parser.add_argument("--cam-fps", type=int, default=30)
parser.add_argument("--show-depth", action="store_true",
                    help="depth 컬러맵도 별도 표시")
parser.add_argument("--timeout", type=float, default=3.0)
parser.add_argument("--conf", type=float, default=0.0,
                    help="최소 confidence 임계값 (0.0~1.0, 기본값: 0.0 = 필터 없음)")
args = parser.parse_args()


TARGETS = [t.strip() for t in args.targets.split(",") if t.strip()]
SERVER_URLS = [
    f"http://{args.host}:{args.base_port + i}/detect"
    for i in range(args.servers)
]

MIN_BBOX_SIZE = 0.02
DEDUP_THRESH = 0.08

COLORS = {
    "cup":      (0, 255, 127),
    "bottle":   (0, 191, 255),
    "scissors": (255, 165, 0),
    "phone":    (255, 0, 128),
    "box":      (180, 105, 255),
    "book":     (0, 255, 200),
    "person":   (0, 255, 127),
    "wallet":   (0, 191, 255),
    "keyboard": (255, 200, 0),
    "mouse":    (200, 100, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def filter_detections(dets):
    filtered = []
    for d in dets:
        if d.get("confidence", 1.0) < args.conf:
            continue
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


class State:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.color: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None

        self.det_lock = threading.Lock()
        self.detections = []

        self.frame_idx = 0
        self.frames_dispatched = 0
        self.frames_returned = 0
        self.latency_ms_recent = deque(maxlen=30)
        self.inference_ms_recent = deque(maxlen=30)
        self.throughput_recent = deque(maxlen=30)
        self.last_return_time = time.time()
        self.running = True


state = State()


def init_realsense():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.cam_w, args.cam_h,
                      rs.format.bgr8, args.cam_fps)
    cfg.enable_stream(rs.stream.depth, args.cam_w, args.cam_h,
                      rs.format.z16, args.cam_fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"[realsense] {args.cam_w}x{args.cam_h}@{args.cam_fps}fps "
          f"depth_scale={depth_scale}")
    return pipe, align, depth_scale


def capture_loop(pipe, align, depth_scale):
    while state.running:
        try:
            frames = pipe.wait_for_frames(timeout_ms=1000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = cv2.flip(np.asanyarray(color_frame.get_data()), 1)
            depth = cv2.flip(np.asanyarray(depth_frame.get_data()).astype(np.float32), 1)
            depth *= depth_scale
            with state.frame_lock:
                state.color = color
                state.depth = depth
        except Exception as e:
            print(f"[capture] error: {e}")
            time.sleep(0.05)


def wait_for_cluster(timeout=180.0):
    deadline = time.time() + timeout
    ready_count = 0
    last_ready = -1
    while time.time() < deadline:
        ready_count = 0
        for i in range(args.servers):
            try:
                r = requests.get(
                    f"http://{args.host}:{args.base_port + i}/health",
                    timeout=0.5)
                if r.status_code == 200:
                    ready_count += 1
            except Exception:
                pass
        if ready_count == args.servers:
            return ready_count
        if ready_count != last_ready:
            print(f"[client] 클러스터 대기 중... {ready_count}/{args.servers}")
            last_ready = ready_count
        time.sleep(2.0)
    return ready_count


def dispatch_one_frame():
    with state.frame_lock:
        if state.color is None:
            return
        color = state.color.copy()
        depth = state.depth.copy() if state.depth is not None else None

    url = SERVER_URLS[state.frame_idx % len(SERVER_URLS)]
    state.frame_idx += 1
    state.frames_dispatched += 1

    threading.Thread(
        target=_send_request,
        args=(url, color, depth, time.time()),
        daemon=True,
    ).start()


def _send_request(url, color, depth, dispatched_at):
    try:
        small = cv2.resize(color, (args.infer_w, args.infer_h))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        r = requests.post(url, json={
            "image_b64": img_b64,
            "labels": TARGETS,
        }, timeout=args.timeout)
        if r.status_code != 200:
            return
        data = r.json()
    except requests.exceptions.Timeout:
        return
    except requests.exceptions.ConnectionError:
        return
    except Exception as e:
        print(f"[client] request error: {e}")
        return

    raw = data.get("detections", [])
    filtered = filter_detections(raw)
    inference_ms = float(data.get("inference_ms", 0))
    latency_ms = (time.time() - dispatched_at) * 1000.0

    ch, cw = color.shape[:2]
    new_dets = []
    now = time.time()
    for d in filtered:
        cx_norm = (d["x_min"] + d["x_max"]) / 2
        cy_norm = (d["y_min"] + d["y_max"]) / 2
        cx_px = int(cx_norm * cw)
        cy_px = int(cy_norm * ch)

        depth_m = None
        if depth is not None:
            depth_m = _sample_depth(depth, cx_px, cy_px)

        new_dets.append({
            **d,
            "depth_m": depth_m,
            "stamp": now,
        })

    with state.det_lock:
        state.detections = new_dets
        state.frames_returned += 1
        state.latency_ms_recent.append(latency_ms)
        state.inference_ms_recent.append(inference_ms)
        dt = now - state.last_return_time
        if 0 < dt < 5.0:
            state.throughput_recent.append(1.0 / dt)
        state.last_return_time = now


def _sample_depth(depth, cx, cy):
    h, w = depth.shape
    x0, x1 = max(0, cx - 2), min(w, cx + 3)
    y0, y1 = max(0, cy - 2), min(h, cy + 3)
    patch = depth[y0:y1, x0:x1]
    valid = patch[(patch > 0.05) & (patch < 3.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def main():
    print(f"[client] targets:  {TARGETS}")
    print(f"[client] servers:  {args.servers} @ "
          f"{args.host}:{args.base_port}~{args.base_port + args.servers - 1}")
    print(f"[client] dispatch: {args.fps}fps")
    print()

    print("[client] 클러스터 ready 대기 (모델 로딩에 30~120초 걸릴 수 있음)...")
    ready = wait_for_cluster()
    if ready == 0:
        print("[client] 서버 응답 없음.")
        print(f"    python vlm_moondream.py --port 8000")
        return
    print(f"[client] 클러스터 ready ({ready}/{args.servers})")

    try:
        pipe, align, depth_scale = init_realsense()
    except Exception as e:
        print(f"[client] RealSense 초기화 실패: {e}")
        return

    threading.Thread(
        target=capture_loop, args=(pipe, align, depth_scale), daemon=True
    ).start()

    print("[client] 첫 프레임 대기...")
    while state.color is None and state.running:
        time.sleep(0.05)
    print("[client] 카메라 준비 완료")

    def dispatcher():
        interval = 1.0 / args.fps
        next_t = time.time()
        while state.running:
            now = time.time()
            if now >= next_t:
                dispatch_one_frame()
                next_t += interval
                if next_t < now:
                    next_t = now + interval
            time.sleep(0.001)
    threading.Thread(target=dispatcher, daemon=True).start()

    cv2.namedWindow("MoonDream2 Cluster", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MoonDream2 Cluster", args.cam_w, args.cam_h)
    if args.show_depth:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    display_fps_recent = deque(maxlen=30)
    last_t = time.time()

    print(f"\n[client] 시작! 'q' 종료\n")
    while state.running:
        with state.frame_lock:
            if state.color is None:
                time.sleep(0.01)
                continue
            display = state.color.copy()
            depth_for_show = state.depth.copy() if state.depth is not None else None

        with state.det_lock:
            dets = list(state.detections)
            latencies = list(state.latency_ms_recent)
            inferences = list(state.inference_ms_recent)
            throughputs = list(state.throughput_recent)
            returned = state.frames_returned
            dispatched = state.frames_dispatched

        for d in dets:
            x1 = int(d["x_min"] * args.cam_w)
            y1 = int(d["y_min"] * args.cam_h)
            x2 = int(d["x_max"] * args.cam_w)
            y2 = int(d["y_max"] * args.cam_h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            c = COLORS.get(d["label"], DEFAULT_COLOR)
            lb = d["label"]
            depth_m = d.get("depth_m")

            # 원형 마커
            cv2.circle(display, (cx, cy), 3, c, -1)
            cv2.circle(display, (cx, cy), 4, (0, 0, 0), 2)
            # 십자선
            cv2.line(display, (cx - 7, cy), (cx + 7, cy), c, 2)
            cv2.line(display, (cx, cy - 7), (cx, cy + 7), c, 2)

            if depth_m is not None:
                text = f"{lb}  {depth_m:.2f} m"
            else:
                text = f"{lb}  (no depth)"

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
            cv2.rectangle(display,
                          (cx - tw // 2 - 6, cy + 18),
                          (cx + tw // 2 + 6, cy + th + 26),
                          c, -1)
            cv2.putText(display, text, (cx - tw // 2, cy + th + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

        now = time.time()
        display_fps_recent.append(now - last_t)
        last_t = now
        disp_fps = (len(display_fps_recent) / sum(display_fps_recent)
                    if sum(display_fps_recent) > 0 else 0)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_inference = sum(inferences) / len(inferences) if inferences else 0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        in_flight = dispatched - returned

        targets_short = ','.join(TARGETS[:3])
        if len(TARGETS) > 3:
            targets_short += f"...(+{len(TARGETS) - 3})"

        lines = [
            f"Servers: {args.servers}   Targets: {targets_short}",
            f"Throughput: {avg_throughput:.1f} det/s   in-flight: {in_flight}",
            f"Latency: {avg_latency:.0f}ms   Inference: {avg_inference:.0f}ms",
            f"Display: {disp_fps:.1f}fps   Dispatched: {dispatched}  Done: {returned}",
        ]
        box_h = 65
        by = args.cam_h - box_h - 5
        cv2.rectangle(display, (5, by), (245, args.cam_h - 5), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(display, line, (10, by + 14 + i * 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 220, 0), 1)

        cv2.imshow("MoonDream2 Cluster", display)

        if args.show_depth and depth_for_show is not None:
            depth_vis = np.clip(depth_for_show, 0.1, 3.0)
            depth_vis = ((depth_vis - 0.1) / 2.9 * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    state.running = False
    try:
        pipe.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()

    print("\n=== 시연 통계 ===")
    print(f"서버 개수:           {args.servers}")
    print(f"디스패치 총:         {state.frames_dispatched}")
    print(f"응답 완료 총:        {state.frames_returned}")
    if latencies:
        print(f"평균 latency:        {sum(latencies)/len(latencies):.0f} ms")
    if throughputs:
        print(f"평균 throughput:     {sum(throughputs)/len(throughputs):.1f} det/s")
    print()


if __name__ == "__main__":
    main()