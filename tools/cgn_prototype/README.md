# Phase 2 — Grasp 층 프로토타입

재설계 5-Phase 중 Phase 2. **기존 `sj_pickplace/` 안 건드림** — 검증 완료 후 Phase 4 에서 이관.

## 파일

- **`cgn_prototype.py`** — Phase 2b/2c 백엔드. scene depth+K (+SAM segmap) → `deproject`
  → `ContactGraspNetBackend.predict(pc_full, pc_segment)` → CGN `predict_scene_grasps(
  local_regions=True, filter_grasps=True)` → TCP 재계산 (`R_g=[b,a×b,a]`, `t_g=c+(w/2)b+d·a`,
  Gram-Schmidt, d=0.1358) → `w≤w_max / s≥thr` 필터 → 180° twin → 그리퍼 와이어 `.ply` /
  RViz MarkerArray. 모델 로드 실패/`--fallback` 이면 geometric 후보.
- `run_cgn_scene.py` — Phase 2c 실험 기록: 같은 씬을 `full` / `region` / `segonly` 세 방식으로
  돌려 score 분포 비교. 공유 헬퍼(`load_capture`/`load_bboxes`/`sam_masks`)는 `cgn_prototype`
  에서 import.
- `run_cgn.py` — ⚠️ DEPRECATED. masked `.ply` 입력 = "segonly" (아래 참조). depth+K+segmap
  npz 는 아직 유효.

## ★ Phase 2c 판정 (2026-09-07) — segonly 는 틀렸다

masked 물체 점만 CGN 에 넣으면(`run_cgn.py <masked.ply>`, 옛 `cgn_prototype` predict) score
~0.19, edge-pinch 만. **CGN 이 필요로 하는 국소 씬 컨텍스트(물체 주변 테이블면)를 버리기 때문.**

**올바른 방식 = 전체 씬 PC + 물체 segment → `local_regions=True, filter_grasps=True`**
(NVIDIA 릴리스의 `--local_regions --filter_grasps`). Thor `~/pc_spike*` 실측:

| 물체 | segonly | **region (local_regions)** |
|---|---|---|
| cup / bottle / clutter | 0.18~0.22, 0 grasp 도 | **0.29, opening 물체 크기 일치** |
| box (무광 스테이지, 45~70°, 단독) | 0.17 | **0.29, opening 5cm (몸통 감싸기)** |

→ **pretrained CGN(약한 pytorch 포트) 로 NERO 전 물체 클래스 충분. 파인튜닝 불필요.**

성공 조건: `local_regions` + 관측각 45~70° + 거리 0.5~0.7m + **무광 스테이지 + 물체 단독**
(검은 테이블/광택 매트면 물체 옆면 depth dropout → rim-pinch 만).

## CONFIG (NERO AGX 그리퍼, 확정)

| | 값 | 출처 |
|---|---|---|
| `d` (flange→fingertip) | **0.1358 m** | `planning_node.TOP_TCP_OFFSET`, URDF `gripper_joint1` z, 2026-07 실측 확정 |
| `w_max` (최대 개방폭) | **0.10 m** | URDF `gripper_joint1/2` prismatic limit 0.05 × 2 |
| `s_threshold` | 0.15 | CGN pytorch 포트 score 천장 ~0.29, config first/second_thres 0.15 |
| 프레임 | `camera_color_optical_frame` (OpenCV) | 호출부에서 `_cam_to_base` 로 base_link (joint1≈0 에서만) |

CGN 4x4 의 translation 은 **버리고** contact `c` + NERO `d` 로 다시 세움 (CGN 은 Franka
baseline d≈0.10). 이게 CLAUDE.md "TOP/SIDE TCP offset 합치지 마라" 문제를 없앤다 —
grasp 마다 자기 approach 축 하나로 offset 통일.

## 실행

```bash
source ~/grasp/cgn_venv/bin/activate
export CGN_REPO=~/grasp/contact_graspnet_pytorch

# scene npz (+ 옆에 <name>.bbox.json: {"bboxes":[[x0,y0,x1,y1]]}) → CGN grasp
python3 cgn_prototype.py ~/grasp/pc_spike6/A_box_001.npz --model --ply-out /tmp/g/

# RViz (ROS 워크스페이스 sourced)
python3 cgn_prototype.py <scene.npz> --model --ros --frame camera_color_optical_frame
#   RViz: Fixed Frame = camera_color_optical_frame, Add → MarkerArray → /cgn_grasps

# 모델 없이 파이프/viz (geometric fallback, masked .ply 도)
python3 cgn_prototype.py <scene.npz|masked.ply> --fallback --ply-out /tmp/g/
```

`--ply-out` → `<name>__grasps.ply` (씬 회색 점 + 초록 그리퍼 와이어, twin 은 짙은 초록).
MeshLab / CloudCompare 로.

## Phase 4 매핑 (learned_grasp_backend.py — 여기서 안 건드림)

```
현재:    ContactGraspNetBackend.predict(pc_full, pc_segment)              -> List[GraspOut]
Phase 4: LearnedGraspBackend.predict(point_cloud, geometry_features, grasp_intent) -> List[LearnedGraspOutput]
  point_cloud = pc_full,  pc_segment 은 segmentation_backend 마스크에서 추가 인자로.
  GraspOut ↔ LearnedGraspOutput 필드 동형 (position/approach_vector/grasp_axis/score/width).
```

## 다음 (Phase 2d / 3)

- 2d: grasp(camera frame) → base_link TF (`_cam_to_base`, canonical observation 자세)
- 3a: 후보 + 180° twin 둘 다 pick_ik 루프 → reachability 필터
- Isaac Sim 그리퍼 물리(close+attach) — Phase 4 정량평가 전제
