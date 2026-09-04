# Phase 2 — Grasp 층 프로토타입

재설계 5-Phase 중 Phase 2. **box 기준으로 착수** (cup/thin/clutter + Phase 3 는 재촬영 후).
**기존 `sj_pickplace/` 안 건드림** — 검증 완료 후 Phase 4 에서 이관.

## 파일

- `cgn_prototype.py` — Phase 2b. Contact-GraspNet backend 스캐폴딩:
  masked PC(.ply) → 정규화 → `ContactGraspNetBackend.predict()` → 공식 변환
  (`R_g=[b,a×b,a]`, `t_g=c+(w/2)b+d·a`, Gram-Schmidt, 180° twin) → 그리퍼
  와이어프레임 `.ply` / RViz MarkerArray.
  **`_run_model()` 딱 이 함수만 Phase 2a 대기.** 그 전엔 `--fallback`.

## CONFIG (NERO AGX 그리퍼, 확정)

| | 값 | 출처 |
|---|---|---|
| `d` (flange→fingertip) | **0.1358 m** | `planning_node.TOP_TCP_OFFSET`, URDF `gripper_joint1` z, 2026-07 실측 확정 |
| `w_max` (최대 개방폭) | **0.10 m** | URDF `gripper_joint1/2` prismatic limit 0.05 × 2 |
| 프레임 | `camera_color_optical_frame` | 호출부에서 `_cam_to_base` 로 base_link (joint1≈0 에서만) |

`get_gripper_teaching_pendant_param()` 로 실물에서 `max_range_config`(0.07/0.10) 크로스체크는
로봇팔 생기면. 지금은 URDF 값.

## Phase 2a — Contact-GraspNet 런타임 (이 PC, Thor 접근 불가)

1. `git clone https://github.com/elchun/contact_graspnet_pytorch` (또는 NVIDIA 공식 TF2)
2. env: PyTorch + CUDA. Isaac Sim 과 VRAM 공유 → `nvidia-smi` 로 여유 확인
   (Isaac Sim 안 띄운 상태에서 CGN 로드 시 VRAM, 둘 다 띄웠을 때 여유).
   부족하면 Docker 로 격리하거나 CGN 을 별도 프로세스로.
3. pretrained checkpoint 다운로드
4. 단독 테스트: 저장된 PC → grasp 출력 (로봇/ROS 없이)
5. `cgn_prototype.py::_run_model()` 에 연결:
   - A) 같은 env 면 `import contact_graspnet_pytorch` 후 forward
   - B) 별도 프로세스면 PC 를 HTTP/소켓으로 POST
   릴리스 플래그: `--local_regions --filter_grasps --forward_passes N`

## Phase 2b — 지금 (모델 없이 파이프 검증)

```bash
# geometric fallback 으로 파이프 + 공식 + viz 검증
python3 cgn_prototype.py <box_masked.ply> --fallback --ply-out /tmp/g
```
box masked .ply 는 `seg_bench.py --ply-out` 출력 (HDD `phase1_thor_run/seg_overlays_mobile_sam/`).

## Phase 2c — box .ply 9개 → grasp → 시각화

**RViz** (ROS 워크스페이스 sourced):
```bash
python3 cgn_prototype.py <ply> --model --ros --frame camera_color_optical_frame
# RViz: Fixed Frame = camera_color_optical_frame, Add → MarkerArray → /cgn_grasps
# 물체 PC 도 같이 보려면 PointCloud2 로 발행하거나 .ply 를 RViz 에
```

**오프라인** (MeshLab / CloudCompare / Isaac Sim):
```bash
python3 cgn_prototype.py <ply> --model --ply-out /tmp/g
# /tmp/g/<name>__grasps.ply (초록 그리퍼 와이어프레임) + 원본 물체 .ply 를 같이 로드
```

**Isaac Sim**: `.ply` 를 USD 로 import 하거나, MarkerArray 를 Isaac Sim ROS2 bridge 로.

### 판정 (box 기준, pretrained 충분한지)

- 윗면에 top-down grasp, 측면에 side grasp 이 **말이 되게** (위치가 물체 표면, approach 가
  물체를 향함, 폭이 물체 크기와 맞음) → **pretrained OK, 파인튜닝 불필요, 진행**
- 체계적으로 나쁨 (방향 뒤집힘, 위치 cm 어긋남, 명백한 표면에 후보 0개) → 파인튜닝 계획
  (Isaac Sim + NERO 물체 메시 + RealSense 노이즈 모델, ~하루)

이게 재설계 최대 미지수(Contact-GraspNet 나쁜소식 B1: sim→real 도메인 갭)를 **재촬영 없이**
답한다.

## 병렬 (지금 가능)

- **Isaac Sim 그리퍼 물리**: sim 그리퍼가 물체를 실제로 close + attach 하나
  (Phase 4 정량평가 전제)
- `d`/`w_max`: URDF 값 확정 (위 표) — 실물 크로스체크는 추후
