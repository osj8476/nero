#!/usr/bin/env python3
"""
run_cgn.py  —  Contact-GraspNet (pytorch port) 최소 러너 (viz 없음, headless)

⚠️ DEPRECATED (2026-09-07) — masked .ply / bare-xyz npz 입력 = "segonly" 경로.
   Phase 2c 에서 이게 틀렸다고 판명: CGN 에 물체 점만 주면 국소 씬 컨텍스트가
   없어 score ~0.19, edge-pinch 만 나옴. 올바른 경로는 전체 씬 PC + segment →
   local_regions. → run_cgn_scene.py (실험/비교) 또는 cgn_prototype.py --model (프로토타입).
   이 파일은 depth+K+segmap npz 를 load_available_input_data 로 처리하는 경우엔
   아직 유효 (segmap 있으면 --local-regions 동작). masked .ply 만 폐기.

Phase 2a: contact_graspnet_pytorch 를 감싸서 point cloud → grasp pose 를 낸다.
inference.py 는 open3d/mayavi viz 를 top-level import 해서 headless 에서 무거움 —
이건 GraspEstimator 만 직접 호출.

입력: .ply (seg_bench 출력) 또는 .npy/.npz (키: 'xyz' Nx3, 옵션 'xyz_color';
      또는 'depth'+'K'+옵션 'segmap')  — 미터 단위, 카메라 좌표계
출력: <out>.npz  키: grasps_cam (M,4,4), scores (M,), contact_pts (M,3),
      openings (M,), pc (N,3)

실행:
  export CGN_REPO=~/grasp/contact_graspnet_pytorch          # 포트 위치
  source ~/grasp/cgn_venv/bin/activate
  python run_cgn.py <in.ply|in.npz> --out /tmp/cgn_pred.npz [--forward-passes 3] [--z-range 0.2 1.2]
  # segmap 있으면:  --local-regions --filter-grasps
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

CGN_REPO = os.environ.get("CGN_REPO", str(Path.home() / "grasp" / "contact_graspnet_pytorch"))


def _load_ply_xyz(path):
    with open(path) as f:
        props = []
        while True:
            ln = f.readline().strip()
            if ln.startswith("property"):
                props.append(ln.split()[-1])
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            if ln == "end_header":
                break
        xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
        rows = np.array([[float(v) for v in f.readline().split()] for _ in range(n)])
    return rows[:, [xi, yi, zi]].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help=".ply 또는 .npy/.npz")
    ap.add_argument("--out", default="/tmp/cgn_pred.npz")
    ap.add_argument("--ckpt", default=None, help="checkpoint dir (기본: 포트의 contact_graspnet)")
    ap.add_argument("--forward-passes", type=int, default=1)
    ap.add_argument("--z-range", type=float, nargs=2, default=[0.2, 1.8])
    ap.add_argument("--local-regions", action="store_true")
    ap.add_argument("--filter-grasps", action="store_true")
    ap.add_argument("--arg-configs", nargs="*", default=[],
                    help="예: TEST.second_thres:0.19 TEST.first_thres:0.23 (후보 더/덜)")
    args = ap.parse_args()

    sys.path.insert(0, CGN_REPO)
    sys.path.insert(0, os.path.join(CGN_REPO, "Pointnet_Pointnet2_pytorch"))
    os.chdir(CGN_REPO)   # 포트가 상대경로 checkpoint 를 씀

    import torch
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.data import load_available_input_data

    ckpt_dir = args.ckpt or os.path.join(CGN_REPO, "checkpoints", "contact_graspnet")
    cfg = config_utils.load_config(ckpt_dir, batch_size=1, arg_configs=args.arg_configs)

    est = GraspEstimator(cfg)
    cio = CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, "checkpoints"), model=est.model)
    cio.load("model.pt")
    est.model.eval()
    print(f"[run_cgn] model loaded ({ckpt_dir})  device={est.device}  gripper_width={cfg['DATA']['gripper_width']}")

    # ── 입력 로드 ──
    inp = Path(args.inp).expanduser()
    pc_segments = {}
    pc_colors = None
    segmap = None
    if inp.suffix == ".ply":
        pc_full = _load_ply_xyz(inp)
    elif inp.suffix in (".npz", ".npy") and "xyz" in getattr(np.load(str(inp), allow_pickle=True), "files", []):
        # 포트의 load_available_input_data 는 bare xyz 에서 cam_K UnboundLocalError → 직접 처리
        _d = np.load(str(inp), allow_pickle=True)
        pc_full = np.asarray(_d["xyz"]).reshape(-1, 3).astype(np.float32)
        if "xyz_color" in _d.files:
            pc_colors = _d["xyz_color"]
    else:
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(str(inp))
        if pc_full is None:
            pc_full, pc_segments, pc_colors = est.extract_point_clouds(
                depth, cam_K, segmap=segmap, z_range=args.z_range)

    if (args.local_regions or args.filter_grasps) and not pc_segments and segmap is None:
        print("[run_cgn] segmap 없음 → --local-regions/--filter-grasps 무시 (전체 씬)")
        args.local_regions = args.filter_grasps = False

    print(f"[run_cgn] pc_full {pc_full.shape}  z {pc_full[:,2].min():.2f}~{pc_full[:,2].max():.2f}m")

    # ── 추론 ──
    with torch.no_grad():
        grasps, scores, contacts, openings = est.predict_scene_grasps(
            pc_full, pc_segments=pc_segments,
            local_regions=args.local_regions, filter_grasps=args.filter_grasps,
            forward_passes=args.forward_passes)

    # dict (key=segment id, -1=full) → 평탄화
    G, S, C, O = [], [], [], []
    for k in grasps:
        g = np.asarray(grasps[k])
        if g.size == 0:
            continue
        G.append(g.reshape(-1, 4, 4))
        S.append(np.asarray(scores[k]).reshape(-1))
        C.append(np.asarray(contacts[k]).reshape(-1, 3))
        ok = np.asarray(openings[k]).reshape(-1)
        O.append(ok if ok.size == g.reshape(-1, 4, 4).shape[0] else np.full(g.reshape(-1, 4, 4).shape[0], np.nan))
    if not G:
        print("[run_cgn] grasp 0개")
        np.savez(args.out, grasps_cam=np.zeros((0, 4, 4)), scores=np.zeros(0),
                 contact_pts=np.zeros((0, 3)), openings=np.zeros(0), pc=pc_full)
        return
    G, S, C, O = np.concatenate(G), np.concatenate(S), np.concatenate(C), np.concatenate(O)
    order = np.argsort(-S)
    G, S, C, O = G[order], S[order], C[order], O[order]

    np.savez(args.out, grasps_cam=G, scores=S, contact_pts=C, openings=O, pc=pc_full)
    print(f"[run_cgn] {len(G)} grasp  score {S.max():.2f}~{S.min():.2f}  opening {np.nanmean(O)*100:.1f}cm avg")
    print(f"[run_cgn] → {args.out}")
    for i in range(min(5, len(G))):
        t = G[i][:3, 3]
        ax = G[i][:3, 2]   # 접근축 = z열
        print(f"  #{i} t={np.round(t,3)}  approach(z)={np.round(ax,2)}  score={S[i]:.2f}  w={O[i]*100:.1f}cm")


if __name__ == "__main__":
    main()
