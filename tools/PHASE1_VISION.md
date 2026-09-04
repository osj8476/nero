# Phase 1 — Vision 층 확정 (Jetson Thor 실행 지시)

재설계 5-Phase 중 Phase 1. **Vision 층(YOLO bbox → SAM 마스크 → masked point cloud)이
grasp/placement/static맵/ground_object 모두의 공유 기반.** 여기를 닫기 전 Phase 2(grasp
net) 착수 금지 — masked PC 품질이 grasp 품질의 상한.

배경: point cloud spike 결과 raw depth 게이트 탈락, 원인은 물체/작업면 depth 분리 불가
(bimodality 단봉) → SAM crop 필수. 3편 논문(OK-Robot/Contact-GraspNet/6-DOF GraspNet)
모두 "SAM 마스크된 단일물체 PC" 전제.

## 환경 (Thor)

```bash
source ~/pc_spike_venv/bin/activate    # 없으면 python3 -m venv 로 생성
pip install ultralytics opencv-python numpy pyrealsense2
```
- seg 모델(mobile_sam.pt, sam2.1_t.pt, FastSAM-s.pt, yolov8s-seg.pt)은 ultralytics 가
  첫 실행 시 자동 다운로드 → 인터넷 필요. 오프라인이면 `yolo/weights/` 에 미리.
- `pyrealsense2` 가 Thor arm64 에 이미 있음 (D435i 스파이크에서 확인).

---

## 1a — 입력 품질 미니 스파이크 (Contact-GraspNet B1~B4 계량)

### ① 국소 depth 노이즈  (`depth_noise.py`)
```bash
python3 tools/depth_noise.py --distances 0.5 0.65 0.8 --frames 30
```
평평한 벽/테이블을 정면으로. 각 거리에서 Enter.
- **기준**: spatial std p90 `< 5mm` → Contact-GraspNet 5mm contact 반경에 OK.
  `> 8mm` → N프레임 평균/필터 강화 필요.
- raw 비교: `--no-filters --preset none`

### ② observation 자세 각도
- 스파이크 권고 = **45~70° 내려보기, grazing(거의 수평) 금지**. Contact-GraspNet
  학습 분포(top-down tabletop)와도 일치.
- 확인: 실제 `go_home`/observation 자세에서 카메라 광축과 테이블 평면이 이루는 각.
  grazing 이면 observation 자세 joint 값 조정 필요 (planning_node/mcp_robot_server
  의 observation 포즈).
- 측정 팁: `depth_noise.py` 의 RANSAC 평면 normal 과 카메라 z축(0,0,1) 사이 각.

### ③ AGX 그리퍼 스펙 + 제어 방식
`~/Downloads/pyAgxArm/docs/piper/piper_api.md` + `init_effector` 확인:
- **position-controllable 인가, full open/close 만인가?** (예측 width 를 명령으로 쓸 수 있나)
- **max opening width** (m) — Contact-GraspNet w_max config
- **d** = 그리퍼 baseline(손끝 접점) → 그리퍼 base(플랜지) 거리 — `t_g = c + (w/2)b + d·a`
- finger length, 손끝 두께
→ 3개 수치 + 제어방식을 RESULT 문서에 기록.

---

## 1b — segmentation 벤치마크  (`seg_bench.py`)

### 1) bbox 라벨링 (한 번)
```bash
python3 tools/seg_bench.py ~/pc_spike/ --label
```
각 프레임에서 대상 물체를 드래그 → Enter. `<stem>.bbox.json` 저장.
(또는 `--yolo yolo/best.pt` 로 자동 검출)

### 2) 벤치
```bash
python3 tools/seg_bench.py ~/pc_spike/ \
    --models mobile_sam,sam2.1_t,fastsam,yolo-seg \
    --ply-out ~/pc_spike/seg
```
지표(GT 없이):
| 지표 | 좋은 값 | 의미 |
|---|---|---|
| `seg_ms` | Thor 예산 내 | 세그 1회 시간 |
| `tight` | SAM<1 | mask area / bbox area (타이트할수록 배경 적음) |
| `smooth` | `>~0.4` | 등주비. 낮으면 경계 들쭉날쭉 |
| `hole%` | 낮음 | mask 안 무효 depth 비율 |
| `extent(PCA)` | 물체 실제 크기 | masked PC 를 PCA 3주축으로 편 크기 (top-down 가정 없음) |
| `bg_leak%` | `<~3` | masked PC 중 씬 지배평면(테이블) 8mm 이내 점 비율 |
| `bimod` | `<0.555` | masked PC depth 단봉 (물체만이면) |

`seg/*.ply` + `*_overlay.png` 육안 확인.

### 선택
(2) SAM 계열 vs (3) FastSAM/YOLO-seg 를 **bg_leak · smooth · seg_ms** 로 트레이드오프.
- SAM 계열이 경계·bg_leak 우수, 느림
- FastSAM/YOLO-seg 빠름, 경계 거침(단 `hole%`·`bg_leak%` 가 허용 범위면 OK)
- 선택한 걸로 `segmentation_backend.SamSegmentationBackend` 의 `SAM_MODEL` 설정
  (yolo-seg 를 고르면 별도 backend 클래스 필요 — Phase 4)

---

## 1c — Vision 프로토타입 (게이트)

`seg_bench.py` 를 선택한 방식 1개로 실행 + `--ply-out`:
```bash
python3 tools/seg_bench.py ~/pc_spike/ --models mobile_sam --ply-out ~/pc_spike/proto
```
**게이트 통과 기준**: 여러 물체(box/cup/bottle/클러터)·여러 각도에서
- masked `.ply` 안에서 물체가 형태를 유지
- `bg_leak% < 3`, `bimod < 0.555` 가 반복 재현
- `extent(PCA)` 가 물체 실제 크기와 일치

통과 → Phase 2 (Contact-GraspNet) 착수.
탈락 → 원인별: 경계 나쁨(다른 SAM 모델/HQ-SAM), hole 큼(관측각·거리·조명),
클러터에서 마스크 새어나감(point prompt 추가).

---

## 보고할 것 (RESULT 문서)

1. `depth_noise.py` 출력 (거리별 spatial/temporal std)
2. observation 자세 각도 (측정값)
3. AGX 그리퍼: 제어방식 + max width + d + finger 치수
4. `seg_bench.py` 방식별 평균 표 + `seg/` 육안 소견 (box/클러터/edge 씬)
5. 선택한 seg 방식 + 이유
6. Phase 1c 게이트 통과 여부
