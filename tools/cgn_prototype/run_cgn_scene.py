#!/usr/bin/env python3
"""
run_cgn_scene.py — Phase 2c: full-scene vs local-region CGN 비교 (재설계 최대 미지수 판정)

2c 예비 결과: masked-only .ply 입력 → pretrained CGN score ~0.19~0.22 (marginal).
원인 후보: (a) scene context 없음  (b) sim→real 도메인 갭.
이 스크립트가 (a)를 직접 테스트: 같은 씬을 세 방식으로 돌려 score 분포 비교.

입력: pc_spike 캡처 npz (키: depth_m, color, K=[fx,fy,cx,cy], depth_scale)
      + 옆의 <name>.bbox.json (키: bboxes = [[x0,y0,x1,y1], ...])  ← 타겟 지정

방식:
  full     — depth+K 전체 씬 PC → predict_scene_grasps (segment 없음)
  region   — bbox → SAM mask → segmap → local_regions=True, filter_grasps=True
  segonly  — SAM mask 안의 점만 (masked-only, 2c 예비와 동일 조건) → predict_grasps

실행:
  source ~/grasp/cgn_venv/bin/activate
  export CGN_REPO=~/grasp/contact_graspnet_pytorch
  python run_cgn_scene.py ~/grasp/pc_spike_thor/A_box_000.npz --sam ~/grasp/mobile_sam.pt
  python run_cgn_scene.py ~/grasp/pc_spike_thor/A_box_000.npz --modes full region --out /tmp/box0.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 공유 헬퍼는 cgn_prototype 에서 (같은 폴더). 중복 정의 안 함.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgn_prototype import load_capture, load_bboxes, sam_masks  # noqa: E402

CGN_REPO = os.environ.get("CGN_REPO", str(Path.home() / "grasp" / "contact_graspnet_pytorch"))


def write_scene_ply(path, pc_scene, grasps4x4, scores, top_k=40):
    """씬 점(회색) + grasp 와이어(초록). 실험용 4x4-행렬 버전 —
    GraspOut 버전은 cgn_prototype.write_grasp_ply(scene_pc=...)."""
    V, C, E = [], [], []
    step = max(1, len(pc_scene) // 60000)
    for p in pc_scene[::step]:
        V.append(p); C.append((110, 110, 110))
    order = np.argsort(-np.asarray(scores))[:top_k]
    smax = float(np.asarray(scores).max()) if len(scores) else 1.0
    for i in order:
        g = grasps4x4[i]; t = g[:3, 3]; a = g[:3, 2]; b = g[:3, 0]
        w = 0.05
        tip = t - 0.1358 * a
        f1, f2 = tip + w * b, tip - w * b
        pts = [t, tip, f1, f2, f1 + 0.03 * a, f2 + 0.03 * a]
        edg = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
        off = len(V)
        val = int(80 + 175 * (float(scores[i]) / (smax + 1e-9)))
        for p in pts:
            V.append(p); C.append((20, val, 20))
        E.extend([(off + u, off + v) for u, v in edg] if False else [(off + u, off + v) for u, v in edg])
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(E)}\nproperty int vertex1\nproperty int vertex2\nend_header\n")
        for p, c in zip(V, C):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for u, v in E:
            f.write(f"{u} {v}\n")


def summ(name, scores, contacts, openings, target_pc=None):
    if scores is None or len(scores) == 0:
        print(f"  {name:11s}  grasp 0개")
        return
    s = np.asarray(scores).reshape(-1)
    o = np.asarray(openings).reshape(-1) if openings is not None else np.array([np.nan])
    line = (f"  {name:11s}  N={len(s):4d}  max {s.max():.3f}  p90 {np.percentile(s,90):.3f}  "
            f"med {np.median(s):.3f}  open {np.nanmedian(o)*100:.1f}cm")
    if target_pc is not None and len(contacts):
        c = np.asarray(contacts).reshape(-1, 3)
        try:
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(target_pc).query(c)
        except Exception:
            dist = np.linalg.norm(c[:, None, :] - target_pc[None, :, :], axis=2).min(1)
        on = dist < 0.03
        line += f"  on-target(<3cm) {100*on.mean():.0f}%"
        if on.any():
            line += f"  [on-tgt max {s[on].max():.3f} med {np.median(s[on]):.3f}]"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--modes", nargs="+", default=["full", "region", "segonly"],
                    choices=["full", "region", "segonly"])
    ap.add_argument("--sam", default=str(Path.home() / "grasp" / "mobile_sam.pt"))
    ap.add_argument("--z-range", type=float, nargs=2, default=[0.2, 1.5])
    ap.add_argument("--forward-passes", type=int, default=1)
    ap.add_argument("--target", type=int, default=0, help="bbox 인덱스 (near-target 지표용)")
    ap.add_argument("--filter", action="store_true", help="region 모드에서 filter_grasps (contact가 segment 안)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ply-out", dest="ply_out", default=None, help="씬+region grasp 와이어 .ply")
    args = ap.parse_args()

    sys.path.insert(0, CGN_REPO)
    sys.path.insert(0, os.path.join(CGN_REPO, "Pointnet_Pointnet2_pytorch"))
    os.chdir(CGN_REPO)

    import torch
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO

    ckpt_dir = os.path.join(CGN_REPO, "checkpoints", "contact_graspnet")
    cfg = config_utils.load_config(ckpt_dir, batch_size=1)
    est = GraspEstimator(cfg)
    cio = CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, "checkpoints"), model=est.model)
    cio.load("model.pt")
    est.model.eval()

    depth, K, color = load_capture(args.npz)
    bboxes = load_bboxes(args.npz)
    print(f"[scene] {Path(args.npz).name}  depth {depth.shape}  "
          f"valid {100*(depth>0).mean():.0f}%  bbox {len(bboxes)}개  gripper_w {cfg['DATA']['gripper_width']}")

    # segmap 만들기 (bbox → SAM)
    segmap = np.zeros(depth.shape, dtype=np.int32)
    tgt_pc = None
    if bboxes and color is not None:
        masks, how = sam_masks(color, bboxes, args.sam)
        for i, mk in enumerate(masks, start=1):
            segmap[mk] = i
        print(f"[scene] segmap: {how}, {len(masks)} seg, 픽셀 {[int((segmap==i).sum()) for i in range(1,len(masks)+1)]}")

    pc_full, pc_segments, _ = est.extract_point_clouds(
        depth, K, segmap=segmap if segmap.any() else None, z_range=args.z_range)
    print(f"[scene] pc_full {pc_full.shape}  segments {[f'{k}:{v.shape[0]}' for k,v in pc_segments.items()]}")
    tid = args.target + 1
    if tid in pc_segments:
        tgt_pc = np.asarray(pc_segments[tid])
        print(f"[scene] 타겟#{args.target} seg pc {tgt_pc.shape}  "
              f"bbox {np.round(tgt_pc.max(0)-tgt_pc.min(0),3)}  centroid {np.round(tgt_pc.mean(0),3)}")

    results = {}
    with torch.no_grad():
        if "full" in args.modes:
            g, s, c, o = est.predict_scene_grasps(
                pc_full, pc_segments={}, local_regions=False, filter_grasps=False,
                forward_passes=args.forward_passes)
            _gl = [np.asarray(g[k]).reshape(-1, 4, 4) for k in g if np.asarray(g[k]).size]
            _sl = [np.asarray(s[k]).reshape(-1) for k in s if np.asarray(s[k]).size]
            _cl = [np.asarray(c[k]).reshape(-1, 3) for k in c if np.asarray(c[k]).size]
            _ol = [np.asarray(o[k]).reshape(-1) for k in o if np.asarray(o[k]).size]
            gg = np.concatenate(_gl) if _gl else np.zeros((0, 4, 4))
            ss = np.concatenate(_sl) if _sl else np.zeros(0)
            cc = np.concatenate(_cl) if _cl else np.zeros((0, 3))
            oo = np.concatenate(_ol) if _ol else np.zeros(0)
            summ("full", ss, cc, oo, tgt_pc)
            results["full"] = (gg, ss, cc, oo)

        if "region" in args.modes and pc_segments:
            g, s, c, o = est.predict_scene_grasps(
                pc_full, pc_segments=pc_segments, local_regions=True, filter_grasps=args.filter,
                forward_passes=args.forward_passes)
            for k in sorted(g):
                summ(f"region#{k}", np.asarray(s[k]).reshape(-1), np.asarray(c[k]).reshape(-1, 3),
                     np.asarray(o[k]).reshape(-1), tgt_pc)
            tid = args.target + 1
            if tid in g:
                results["region"] = (np.asarray(g[tid]).reshape(-1, 4, 4), np.asarray(s[tid]).reshape(-1),
                                     np.asarray(c[tid]).reshape(-1, 3), np.asarray(o[tid]).reshape(-1))

        if "segonly" in args.modes and pc_segments:
            tid = args.target + 1
            seg_pc = pc_segments[tid]
            gg, ss, cc, oo = est.predict_grasps(
                torch.from_numpy(seg_pc.astype(np.float32)), forward_passes=args.forward_passes)
            gg = gg.cpu().numpy() if hasattr(gg, "cpu") else np.asarray(gg)
            ss = ss.cpu().numpy() if hasattr(ss, "cpu") else np.asarray(ss)
            cc = cc.cpu().numpy() if hasattr(cc, "cpu") else np.asarray(cc)
            oo = oo.cpu().numpy() if hasattr(oo, "cpu") else np.asarray(oo)
            summ("segonly", ss.reshape(-1), cc.reshape(-1, 3), oo.reshape(-1), tgt_pc)
            results["segonly"] = (gg, ss, cc, oo)

    if args.ply_out and "region" in results:
        gg, ss, _, _ = results["region"]
        write_scene_ply(args.ply_out, pc_full, gg, ss)
        print(f"[scene] ply → {args.ply_out}  ({len(gg)} region grasps, top40 그려짐)")

    if args.out:
        np.savez(args.out, pc=pc_full,
                 **{f"{m}_grasps": results[m][0] for m in results},
                 **{f"{m}_scores": results[m][1] for m in results},
                 **{f"{m}_contacts": results[m][2] for m in results})
        print(f"[scene] → {args.out}")


if __name__ == "__main__":
    main()
