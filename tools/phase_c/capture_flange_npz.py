#!/usr/bin/env python3
"""
capture_flange_npz.py  —  PC. flange 카메라 한 프레임을 CGN 입력 npz 로 저장.

ROS 토픽만 씀 (Isaac 스크립트 안 건드림):
  /camera/depth/image_raw   (32FC1, distance_to_image_plane, m)
  /camera/camera_info       (K)
  /detected_objects         (perception_node_sim → YOLO@Thor 의 bbox, 정규화)
  TF  base_link -> camera_color_optical_frame

출력 npz (two_cgn.py 규약):
  depth_m (H,W) f32 | K [fx,fy,cx,cy] | W_T_cam (4,4)  [= base_link_T_optical @ diag(1,-1,-1,1),
  two_cgn 내부의 flip 이 되돌려 base_link_T_optical 로 씀]
  seg (H,W) i32  (bbox 안 = 1)  | box_id [1] | bottle_id [-1] | box_world [x,y,z]

사용:
  ros2 run 없이:  python3 capture_flange_npz.py --label box --out ~/grasp/boxpick.npz
  (source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash 필요)
"""
import argparse, base64, json, os, sys, time, urllib.request
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# Thor YOLO+SAM 서버. 환경변수 BOX_SERVER_URL 은 낡은 주소(윈도우 랩탑)일 수 있어 안 씀 —
# SAM_SERVER_URL 로만 오버라이드.
SAM_URL = os.environ.get("SAM_SERVER_URL", "http://163.239.19.132:8002/detect")


def sam_mask(rgb, label, H, W):
    """RGB + label -> YOLO@Thor(/detect, want_mask) -> (mask HxW bool, bbox_norm) 또는 (None,None)."""
    try:
        import cv2
        rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
        ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        print(f"[capture] SAM 요청: jpg {len(buf)//1024}KB -> {SAM_URL}", file=sys.stderr)
        body = json.dumps({"image_b64": b64, "labels": [label], "want_mask": True}).encode()
        req = urllib.request.Request(SAM_URL, data=body, headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=90).read())
        dets = [d for d in r["detections"] if d["label"] == label and d.get("mask_b64")]
        if not dets:
            return None, None
        d = max(dets, key=lambda x: x["confidence"])
        m = cv2.imdecode(np.frombuffer(base64.b64decode(d["mask_b64"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m > 127, [d["x_min"], d["y_min"], d["x_max"], d["y_max"]]
    except Exception as e:
        print(f"[capture] SAM 요청 실패: {e}", file=sys.stderr)
        return None, None


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])


BOTTLE_LABELS = {"bottle", "cup", "vase", "cylinder", "can", "wine glass", "banana"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="box",
                    help="/detected_objects 에서 이 라벨의 bbox 를 씀. "
                         "bottle/cup/vase/... 는 CGN 에서 원통(side) 취급.")
    ap.add_argument("--out", default="/tmp/cgn_capture.npz")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--pad", type=float, default=0.02, help="bbox 정규화 패딩")
    ap.add_argument("--floor-z", type=float, default=0.012, help="bbox 모드: world z 가 이 값 이하 = 바닥, seg 제외")
    ap.add_argument("--world-aabb", type=float, nargs=6, metavar=("X", "Y", "Z", "HX", "HY", "HZ"),
                    help="YOLO 대신: 이 world AABB(중심+half-extent) 안의 depth 점만 seg. "
                         "라벨은 --label 로 지정 (예: --label bottle).")
    ap.add_argument("--sam", action="store_true",
                    help="YOLO->SAM 경로: RGB 를 Thor 서버로 보내 SAM 인스턴스 마스크로 seg (bbox 직사각형 대신).")
    a = ap.parse_args()
    a.is_bottle = a.label in BOTTLE_LABELS

    rclpy.init()
    n = Node("capture_flange_npz")
    buf = Buffer(); TransformListener(buf, n)

    got = {}

    def on_info(m):
        got["K"] = np.array(m.k).reshape(3, 3)
        got["wh"] = (m.width, m.height)

    def on_depth(m):
        if m.encoding not in ("32FC1", "16UC1"):
            n.get_logger().warn(f"depth encoding {m.encoding}?")
        d = np.frombuffer(m.data, dtype=np.float32 if m.encoding == "32FC1" else np.uint16)
        d = d.reshape(m.height, m.width).astype(np.float32)
        if m.encoding == "16UC1":
            d = d / 1000.0
        got["depth"] = d

    def on_det(m):
        try:
            obj = json.loads(m.data).get("objects", [])
        except Exception:
            return
        cand = [o for o in obj if o.get("label") == a.label]
        if cand:
            cand.sort(key=lambda o: -o.get("bbox_size", 0) if "bbox_size" in o else 0)
            got["bbox"] = cand[0]["bbox"]
            got["obj"] = cand[0]

    def on_rgb(m):
        arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
        got["rgb"] = arr[..., ::-1].copy() if m.encoding.startswith("bgr") else arr.copy()

    be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST, depth=10)
    n.create_subscription(CameraInfo, "/camera/camera_info", on_info, qos_profile_sensor_data)
    n.create_subscription(Image, "/camera/depth/image_raw", on_depth, qos_profile_sensor_data)
    n.create_subscription(String, "/detected_objects", on_det, be)
    if a.sam:
        n.create_subscription(Image, "/camera/color/image_raw", on_rgb, qos_profile_sensor_data)

    if a.sam:
        pre = ("K", "depth", "rgb")
    elif a.world_aabb:
        pre = ("K", "depth")
    else:
        pre = ("K", "depth", "bbox")
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.2)
        if all(k in got for k in pre):
            try:
                got["tf"] = buf.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time())
                break
            except Exception:
                pass
    for k in pre + ("tf",):
        if k not in got:
            print(f"[capture] 못 받음: {k}", file=sys.stderr); rclpy.shutdown(); sys.exit(1)

    K3 = got["K"]; fx, fy, cx, cy = K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]
    depth = got["depth"]; H, W = depth.shape

    tr = got["tf"].transform
    R = quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
    base_T_opt = np.eye(4)
    base_T_opt[:3, :3] = R
    base_T_opt[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    W_T_cam = base_T_opt @ flip                     # two_cgn 이 @flip 으로 되돌림

    seg = np.zeros((H, W), dtype=np.int32)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    zf = depth.copy(); zf[~np.isfinite(zf)] = 0.0
    Xc = (uu - cx) / fx * zf
    Yc = (vv - cy) / fy * zf
    pts_opt = np.stack([Xc, Yc, zf, np.ones_like(zf)], -1)
    pts_base = pts_opt @ base_T_opt.T                # H,W,4  in base_link

    if a.sam:
        smask, sbbox = sam_mask(got["rgb"], a.label, H, W)
        if smask is None:
            print("[capture] SAM 마스크 못 받음", file=sys.stderr); rclpy.shutdown(); sys.exit(1)
        m = smask & (zf > 0.1) & (zf < 2.0) & (pts_base[..., 2] > a.floor_z)
        seg[m] = 1
        info_bbox = [round(x, 3) for x in sbbox] if sbbox else None
    elif a.world_aabb:
        cxw, cyw, czw, hx, hy, hz = a.world_aabb
        m = ((zf > 0.1) &
             (np.abs(pts_base[..., 0] - cxw) < hx) &
             (np.abs(pts_base[..., 1] - cyw) < hy) &
             (pts_base[..., 2] > czw - hz) & (pts_base[..., 2] < czw + hz))
        seg[m] = 1
        info_bbox = None
    else:
        x0, y0, x1, y1 = got["bbox"]
        x0 = max(0.0, x0 - a.pad); y0 = max(0.0, y0 - a.pad)
        x1 = min(1.0, x1 + a.pad); y1 = min(1.0, y1 + a.pad)
        px0, px1, py0, py1 = int(x0 * W), int(x1 * W), int(y0 * H), int(y1 * H)
        m = np.zeros((H, W), bool)
        m[py0:py1, px0:px1] = True
        m &= (zf > 0.1) & (zf < 2.0) & (pts_base[..., 2] > a.floor_z)
        # bbox 사각형 → 물체 뒤/위 배경이 섞임. 2단계 정제:
        #  (1) bbox 안 유효점의 depth 하위 30% (= 가장 가까운 = 물체) 만 남김
        #  (2) 그 점들의 world-z 중앙값 기준 ±10cm 밴드만
        if m.sum() > 50:
            zc = zf[m]
            dcut = np.percentile(zc, 35)
            m &= (zf <= dcut + 0.03)
            zmed = float(np.median(pts_base[..., 2][m]))
            m &= (pts_base[..., 2] < zmed + 0.10)
        seg[m] = 1
        info_bbox = [px0, py0, px1, py1]

    npx = int((seg == 1).sum())
    if npx < 50:
        print(f"[capture] seg 픽셀 {npx} — 물체 안 보임/AABB 빗나감", file=sys.stderr)
        rclpy.shutdown(); sys.exit(1)

    obj_world = np.median(pts_base[seg == 1][:, :3], axis=0)

    ids = dict(bottle_id=np.array([1]), box_id=np.array([-1])) if a.is_bottle \
        else dict(box_id=np.array([1]), bottle_id=np.array([-1]))
    wk = "cyl_world" if a.is_bottle else "box_world"
    np.savez(a.out, depth_m=zf.astype(np.float32), seg=seg,
             K=np.array([fx, fy, cx, cy], dtype=np.float64),
             W_T_cam=W_T_cam.astype(np.float64),
             **{wk: obj_world.astype(np.float64)}, **ids)
    print(json.dumps({"out": a.out, "label": a.label,
                      "target": "bottle(side)" if a.is_bottle else "box(top)",
                      "mode": "yolo_sam" if a.sam else ("world_aabb" if a.world_aabb else "yolo_bbox"),
                      "seg_px": npx, "bbox_px": info_bbox,
                      "obj_world": [round(float(v), 3) for v in obj_world],
                      "K": [round(float(v), 1) for v in (fx, fy, cx, cy)]}, indent=1))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
