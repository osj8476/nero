#!/usr/bin/env python3
"""
seg_bench.py  —  Phase 1b/1c: segmentation 벤치마크 + Vision 프로토타입

pc_spike_capture.py 의 .npz(RGB+depth+K) 에서:
  bbox → segmentation(여러 방식) → masked point cloud → 마스크/PC 품질 비교

Phase 1b (벤치): 여러 seg 방식을 마스크 품질·속도로 비교해 1개 선택
  (2) YOLO bbox → SAM 계열 (mobile_sam, sam2.1_t, sam2.1_b, HQ-SAM는 별도)
  (3) YOLO-seg / FastSAM (마스크 직접)
Phase 1c (프로토타입): 방식 1개로 masked PC 를 .ply 로 뽑아 육안 확인

--------------------------------------------------------------------------
1. bbox 라벨링 (한 번):
     python3 seg_bench.py ~/pc_spike/ --label
   각 프레임에서 대상 물체를 드래그로 박스 → Enter/Space, c=취소, q=중단.
   <stem>.bbox.json 사이드카 저장. (--yolo <weights> 로 자동 검출도 가능)

2. 벤치:
     python3 seg_bench.py ~/pc_spike/ --models mobile_sam,sam2.1_t,fastsam,yolo-seg \
         --ply-out ~/pc_spike/seg
--------------------------------------------------------------------------

의존: pip install ultralytics opencv-python numpy   (모델은 자동 다운로드)

지표 (GT 없이):
  seg_ms            : 세그 1회 시간
  tight            : mask area / bbox area  (SAM<1 = bbox보다 타이트, 낮을수록 좋음)
  smooth           : 등주비 4πA/P²  (1=원, 낮으면 경계 들쭉날쭉)
  hole%            : mask 안 픽셀 중 무효 depth 비율
  extent(PCA) cm   : masked PC 를 PCA 3주축으로 편 크기 = 물체 실제 크기여야
  bg_leak%         : masked PC 중 씬 지배평면(테이블)에서 8mm 이내 점 비율 (낮을수록 좋음)
  bimod            : masked PC depth 이봉성 (마스크 좋으면 단봉 <0.555 근처)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python 미설치: pip install opencv-python")


# ── point cloud helpers (point_cloud.py 와 동일 핀홀, ROS import 회피) ──
def deproject_mask(depth_m, mask, K, dmin=0.05, dmax=3.0):
    fx, fy, cx, cy = K
    v = np.isfinite(depth_m) & (depth_m > dmin) & (depth_m < dmax) & mask.astype(bool)
    vs, us = np.nonzero(v)
    if len(us) == 0:
        return np.zeros((0, 3))
    z = depth_m[vs, us].astype(np.float64)
    return np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=1)


def fit_plane_ransac(pts, iters=150, thresh=0.008, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    if n < 50:
        return None
    best_in, best = 0, None
    for _ in range(iters):
        s = pts[rng.choice(n, 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        na = np.linalg.norm(nrm)
        if na < 1e-9:
            continue
        nrm /= na
        d = -nrm @ s[0]
        ninl = int((np.abs(pts @ nrm + d) < thresh).sum())
        if ninl > best_in:
            best_in, best = ninl, (nrm, d)
    return best


def pca_extent(pts):
    """PCA 3주축을 따라 5~95% 폭 (cm). top-down 가정 없음 = 얕은 각도에서도 유효."""
    if len(pts) < 30:
        return None
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    lo, hi = np.percentile(proj, [5, 95], axis=0)
    return (hi - lo) * 100.0


def bimodality_coeff(x):
    x = x[np.isfinite(x)]
    if len(x) < 50 or x.std() < 1e-4:   # 거의 상수(평면 마스크 등)면 무의미
        return float("nan")
    n = len(x); m = x.mean(); s = x.std()
    g = ((x - m) ** 3).mean() / s ** 3
    k = ((x - m) ** 4).mean() / s ** 4 - 3.0
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def mask_smoothness(mask):
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return float("nan")
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c); p = cv2.arcLength(c, True)
    return 4 * np.pi * a / (p * p + 1e-9)


def write_ply(path, xyz, rgb=None):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, pt in enumerate(xyz):
            if rgb is not None:
                c = rgb[i]
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(c[2])} {int(c[1])} {int(c[0])}\n")
            else:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")


# ── segmentation backends (ultralytics 직접) ──
_MODEL_CACHE = {}

def _load(name):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    from ultralytics import SAM, FastSAM, YOLO
    table = {
        "mobile_sam": ("mobile_sam.pt", SAM),
        "sam2.1_t":   ("sam2.1_t.pt", SAM),
        "sam2.1_b":   ("sam2.1_b.pt", SAM),
        "sam_b":      ("sam_b.pt", SAM),
        "fastsam":    ("FastSAM-s.pt", FastSAM),
        "yolo-seg":   ("yolov8s-seg.pt", YOLO),
    }
    if name not in table:
        raise ValueError(f"모르는 모델: {name} (가능: {list(table)})")
    weight, cls = table[name]
    m = cls(weight)
    _MODEL_CACHE[name] = m
    return m


def segment(name, image_bgr, bbox):
    """→ (mask HxW bool, seg_ms). 실패 시 (None, ms)."""
    m = _load(name)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    t0 = time.perf_counter()
    if name == "fastsam":
        r = m.predict(image_bgr, bboxes=[[x1, y1, x2, y2]], verbose=False)
    elif name == "yolo-seg":
        r = m.predict(image_bgr, verbose=False)
    else:  # SAM 계열
        r = m.predict(image_bgr, bboxes=[[x1, y1, x2, y2]], verbose=False)
    ms = (time.perf_counter() - t0) * 1000

    H, W = image_bgr.shape[:2]
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None, ms
    masks = r[0].masks.data.cpu().numpy().astype(bool)
    if name == "yolo-seg":
        # bbox IoU 최대 인스턴스 선택
        best, bi = 0.0, 0
        for i, mk in enumerate(masks):
            ys, xs = np.nonzero(mk)
            if len(xs) == 0:
                continue
            ix1, iy1, ix2, iy2 = xs.min(), ys.min(), xs.max(), ys.max()
            iw = max(0, min(x2, ix2) - max(x1, ix1)); ih = max(0, min(y2, iy2) - max(y1, iy1))
            inter = iw * ih
            iou = inter / ((x2-x1)*(y2-y1) + (ix2-ix1)*(iy2-iy1) - inter + 1e-9)
            if iou > best:
                best, bi = iou, i
        mask = masks[bi]
    else:
        mask = masks[0]
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, ms


# ── bbox 라벨링 ──
def label_mode(files):
    for f in files:
        side = f.with_suffix(".bbox.json")
        d = np.load(f, allow_pickle=True)
        img = d["color"]
        print(f"  {f.name} — 물체 드래그, Enter/Space=확정, c=스킵, q=중단")
        r = cv2.selectROI(f"label {f.name}", img, showCrosshair=True)
        cv2.destroyAllWindows()
        if r == (0, 0, 0, 0):
            print("    스킵"); continue
        x, y, w, h = r
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        side.write_text(json.dumps({"bbox": bbox}))
        print(f"    저장 {side.name}  bbox={bbox}")


def get_bbox(f, yolo_weights):
    side = f.with_suffix(".bbox.json")
    if side.exists():
        return json.loads(side.read_text())["bbox"]
    if yolo_weights:
        from ultralytics import YOLO
        d = np.load(f, allow_pickle=True)
        r = YOLO(yolo_weights).predict(d["color"], verbose=False)
        if r and r[0].boxes is not None and len(r[0].boxes) > 0:
            b = r[0].boxes.xyxy.cpu().numpy()
            areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            return b[areas.argmax()].tolist()
    return None


def bench_one(f, models, ply_dir, roi_dmax, yolo_weights=None):
    d = np.load(f, allow_pickle=True)
    depth = d["depth_m"].astype(np.float32)
    color = d["color"]
    K = list(d["K"])
    bbox = get_bbox(f, yolo_weights)
    if bbox is None:
        print(f"\n=== {f.name} — bbox 없음 (--label 먼저 or --yolo)"); return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bbox_area = (x2 - x1) * (y2 - y1)

    # 씬 지배평면 (bg_leak 기준) — bbox 주변 확장 영역에서
    scene = deproject_mask(depth, np.ones_like(depth, bool), K, dmax=roi_dmax)
    plane = fit_plane_ransac(scene[np.random.default_rng(0).choice(
        len(scene), min(8000, len(scene)), replace=False)]) if len(scene) > 200 else None

    print(f"\n=== {f.name}   bbox=[{x1},{y1},{x2},{y2}]")
    hdr = f"  {'method':<11} {'seg_ms':>7} {'tight':>6} {'smooth':>7} {'hole%':>6} " \
          f"{'extent(cm)':>18} {'bg_leak%':>9} {'bimod':>6}"
    print(hdr)
    rows = []
    for name in models:
        try:
            mask, ms = segment(name, color, bbox)
        except Exception as e:
            print(f"  {name:<11} 실패: {type(e).__name__}: {e}"); continue
        if mask is None:
            print(f"  {name:<11} {ms:7.1f}  마스크 없음"); continue

        area = int(mask.sum())
        tight = area / (bbox_area + 1e-9)
        smooth = mask_smoothness(mask)
        mpts = deproject_mask(depth, mask, K, dmax=roi_dmax)
        mask_px = area
        hole = 1.0 - (len(mpts) / (mask_px + 1e-9)) if mask_px else float("nan")
        ext = pca_extent(mpts)
        if plane is not None and len(mpts):
            nrm, pd = plane
            leak = float((np.abs(mpts @ nrm + pd) < 0.008).mean())
        else:
            leak = float("nan")
        bim = bimodality_coeff(mpts[:, 2]) if len(mpts) else float("nan")
        ext_s = f"{ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f}" if ext is not None else "  -"
        print(f"  {name:<11} {ms:7.1f} {tight:6.2f} {smooth:7.2f} {hole*100:6.1f} "
              f"{ext_s:>18} {leak*100:9.1f} {bim:6.2f}")
        rows.append(dict(method=name, seg_ms=ms, tight=tight, smooth=smooth,
                         hole=hole, extent=ext, bg_leak=leak, bimod=bim))

        if ply_dir is not None:
            ply_dir.mkdir(parents=True, exist_ok=True)
            rgb = color[[np.nonzero(np.isfinite(depth) & (depth > 0.05) &
                        (depth < roi_dmax) & mask.astype(bool))][0]] if len(mpts) else None
            vs, us = np.nonzero(np.isfinite(depth) & (depth > 0.05) & (depth < roi_dmax) & mask.astype(bool))
            rgb = color[vs, us] if len(us) else None
            write_ply(ply_dir / f"{f.stem}__{name}.ply", mpts, rgb)
            ov = color.copy()
            ov[mask.astype(bool)] = (0.5 * ov[mask.astype(bool)] + np.array([0, 128, 255]) * 0.5).astype(np.uint8)
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(ply_dir / f"{f.stem}__{name}_overlay.png"), ov)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".npz 파일 또는 폴더")
    ap.add_argument("--label", action="store_true", help="bbox 라벨링 모드")
    ap.add_argument("--yolo", default=None, help="bbox 자동검출용 YOLO weights (사이드카 없을 때)")
    ap.add_argument("--models", default="mobile_sam,sam2.1_t,fastsam,yolo-seg",
                    help="비교할 seg 방식 (쉼표). mobile_sam,sam2.1_t,sam2.1_b,sam_b,fastsam,yolo-seg")
    ap.add_argument("--ply-out", default=None, help="masked .ply + overlay 저장 폴더")
    ap.add_argument("--dmax", type=float, default=1.5, help="depth 상한(m)")
    args = ap.parse_args()

    p = Path(args.path).expanduser()
    files = sorted(p.glob("*.npz")) if p.is_dir() else [p]
    if not files:
        print("npz 없음", file=sys.stderr); sys.exit(1)

    if args.label:
        label_mode(files)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ply_dir = Path(args.ply_out).expanduser() if args.ply_out else None
    allrows = []
    for f in files:
        r = bench_one(f, models, ply_dir, args.dmax, yolo_weights=args.yolo)
        if r:
            allrows.extend(r)

    if allrows:
        print("\n[방식별 평균]")
        print(f"  {'method':<11} {'seg_ms':>7} {'tight':>6} {'smooth':>7} {'hole%':>6} {'bg_leak%':>9} {'bimod':>6}")
        for name in models:
            rs = [x for x in allrows if x["method"] == name]
            if not rs:
                continue
            def avg(k): return np.nanmean([x[k] for x in rs])
            print(f"  {name:<11} {avg('seg_ms'):7.1f} {avg('tight'):6.2f} {avg('smooth'):7.2f} "
                  f"{avg('hole')*100:6.1f} {avg('bg_leak')*100:9.1f} {avg('bimod'):6.2f}")

    print("\n[판단 기준]")
    print("  좋은 seg: bg_leak% 낮음(<~3), smooth 높음(>~0.4), extent 실제 물체 크기와 일치,")
    print("            bimod 낮음(<0.555, 물체만이라 단봉), seg_ms 는 Thor 배포 예산 내")
    print("  (2)SAM 계열 vs (3)fastsam/yolo-seg 를 bg_leak·smooth·속도로 트레이드오프 비교")
    print("  선택한 방식으로 segmentation_backend.SamSegmentationBackend 의 SAM_MODEL 설정")


if __name__ == "__main__":
    main()
