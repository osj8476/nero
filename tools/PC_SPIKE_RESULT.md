# Point Cloud Sanity Spike — 결과 (2026-09-03)

`PC_SPIKE.md` 프로토콜대로 실물 RealSense D435i 로 2 라운드 촬영·분석.
질문: **실물 depth 가 grasp net 입력으로 쓸 만한가?**

## 결론 (요약)

- **raw depth 를 그대로 Contact-GraspNet / VGN 에 넣는 건 아직 아님.**
  게이트(중앙ROI hole `<10%` · near-cluster extent ≈ 물체크기 · bimodality `>0.555`)
  를 통과 못 함.
- 단, 실패 원인이 명확해짐:
  1. **물체/작업면 depth 분리가 안 됨** (ROI depth bimodality 단봉 프레임 다수)
     → **SAM mask crop 이 전제**여야 한다.
  2. 물체 뒤 **occlusion shadow**(스테레오 베이스라인 상 한쪽으로 생기는 구멍).
  3. **금속/반투명 물체**는 IR 반사·투과로 몸통 depth 전멸 — 조건 보정으로 해결 안 됨.
- **모델 결정: `SAM mask crop → Contact-GraspNet` 을 1순위**로 착수.
  마스크 내부에서는 물체가 형태를 유지한다(2라운드 육안 확인). 클러터·반사물엔
  **VGN 다중뷰 TSDF 융합**을 폴백으로.
- **다음 착수 과제: SAM segmentation 실기 검증**
  (`sj_pickplace/segmentation_backend.py` 의 `SamSegmentation` 은 배선만 됨, 미검증 —
  [[vlm_capability_tiers]]). 이게 grasp net 구현보다 먼저 와야 한다.
  검증되면 `sj_pickplace/learned_grasp_backend.py` 에 Contact-GraspNet 클래스를 붙인다.

## 촬영 조건

| | 1라운드 | 2라운드 |
|---|---|---|
| 해상도 | 1280×720 | 848×480 (D4xx 네이티브) |
| visual preset | 없음 | High Accuracy |
| 후처리 필터 | 없음 | disparity → spatial → temporal |
| 작업면 | 검은 무광 테이블 | 녹색 PVC 바닥 (일부 씬은 검은 테이블) |
| 관측거리 (ROI p50) | 0.35~0.6 m | 0.65~0.95 m |
| 씬 | A_box(솔리드 큐브)·B_cup·C_bottle(반투명 물통)·D_thin·E_clutter·F_edge | A_box(**실제 골판지 box**)·B_cup·C_bottle(불투명 텀블러)·D_thin·E_clutter·F_edge |
| 프레임 수 | 19 | 19 |

카메라: Intel RealSense D435i, SN 243722074750, FW 5.15.1.55. `pyrealsense2` 2.58.2
는 이 머신(aarch64 Tegra) 시스템 python3 에 이미 설치돼 있어 venv 불필요.

## 정량 결과

| 지표 | 1라운드 | 2라운드 |
|---|---|---|
| 중앙ROI hole_ratio | 9~36% (median ~20%) | 0~14% (median ~9%) — 검은 테이블 F_edge 제외 |
| ROI depth bimodality | 단봉 프레임 다수 | 단봉 프레임 여전히 다수 |
| near-cluster extent | 15~32 cm | 20~44 cm — **이 각도대에선 지표 신뢰 불가**(아래) |

### near-cluster extent 지표 주의

`pc_spike_report.py` 의 near-cluster extent 는 "최근접 depth 밴드(±0.12m) 점들의
bbox" 로, **top-down 시야(최근접 밴드 ≈ 물체)를 가정**하고 만들어졌다. 2라운드처럼
관측각이 얕으면 이 밴드가 멀어지는 작업면 평면을 통째로 슬라이스해서 dx 가
20~44cm 로 부풀려진다 — 물체 재구성이 나빠서가 아니다. 2라운드 판단은
`hole_ratio` + 육안(depthviz/ply) + `bimodality` 로 했다.
→ **TODO: 지표를 얕은 각도에 견디게 수정**(예: 밴드 폭 축소, ROI 중심 근방으로
lateral 제한, 또는 관측각 추정 후 경고).

## 육안 소견 (depthviz / ply)

2라운드 기준:

- **B_cup / E_clutter**: 컵·박스 footprint 또렷. 물체 뒤에 occlusion shadow(검은 halo).
- **C_bottle (불투명 텀블러)**: 몸통이 형태 유지하며 잡힘. → 1라운드의 반투명
  "WATER" 물통은 몸통 depth 전멸이었음(IR 투과). **불투명이면 OK, 투명/반투명은 불가.**
- **E_clutter 안의 금속 텀블러**: 반사로 몸통 구멍.
- **A_box (2R)**: 상자가 바닥에서 거의 안 떠 보임 — 카메라가 거의 수평(grazing).
  관측각이 너무 얕았던 촬영 실수.
- **F_edge**: 검은 무광 테이블 + grazing angle → 구멍 재발. **검은 무광 작업면이
  일관되게 최악.**

## 실기 반영 (촬영 조건 권고)

- 관측각 **45~70° 내려보기**. 거의 수평(grazing) 금지. 2R E_clutter 각도가 하한.
- 거리 0.6~0.8 m.
- 작업면이 검은 무광이면 별도 대응(매트 등) 필요 — depth 품질이 색/재질에 크게 좌우됨.
- 반투명/투명·고반사 금속 물체는 depth-grasp 경로 밖. 별도 트랙(알려진 geometry / VLM).

## 산출물

- 캡처: `tools/pc_spike_capture.py` (848×480 + High Accuracy preset + 후처리 필터로 갱신,
  `--no-filters`/`--preset none`/`--hole-fill` 옵션. npz 에 `preset`/`filters` 필드 기록)
- 리포트: `tools/pc_spike_report.py` (변경 없음 — near-cluster extent 지표 수정은 TODO)
- 원본 캡처 데이터(.npz/.ply)는 커밋 안 함 (`~/pc_spike/`, 재현 가능).

## 관련

- [[grasp_geometry_pipeline]] — "extents 0.27~0.44m" 배경누출 증상. 이번 스파이크가
  그 원인(마스크에 배경 포함 + 얕은 각도)을 실물 depth 쪽에서 재확인.
- [[vlm_capability_tiers]] — `SamSegmentation` / `DepthPlane` backend 배선 상태.
- `PC_SPIKE.md` — 실행 지시·촬영 프로토콜·판단 게이트.
