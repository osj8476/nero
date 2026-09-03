# Point Cloud Sanity Spike — Jetson Thor 실행 지시

## 목적

grasp 6-DoF pose generation(Contact-GraspNet / VGN / AnyGrasp) 구현에 **착수하기 전에**
답해야 하는 질문 하나:

> **실물 RealSense depth 가 grasp net 입력으로 쓸 만한가?**

이게 #1 리스크다. `docs/wiki/grasp_geometry_pipeline.md` 가 이미 지적했듯이 —
15cm box 가 extents 0.27~0.44m 로 나오고, 무지 골판지 box 는 stereo depth 가 잘
뚫린다. 같은 나쁜 point cloud 가 grasp net 에 들어가면 grasp net 도 쓰레기 후보를 낸다.

Isaac Sim depth 는 항상 깨끗해서 이 리스크를 검증 못 한다 → **실물 카메라 필수.**
로봇 팔에 마운트 안 해도 되고, 같은 카메라 모델로 실제 물체를 실제 거리에서 찍기만
하면 된다.

## 환경 준비 (Thor)

```bash
python3 -m venv ~/pc_spike_venv && source ~/pc_spike_venv/bin/activate
pip install pyrealsense2 opencv-python numpy
```

- ROS 불필요, 로봇 불필요, Isaac Sim 불필요 — pyrealsense2 standalone.
- RealSense(D435/D455 등)를 Thor USB 에 직결.
- `pyrealsense2` 가 Thor(arm64)에서 pip 로 안 깔리면: librealsense 를 소스빌드하거나
  (`-DBUILD_PYTHON_BINDINGS=ON`), 데스크탑/노트북에 카메라 물려서 캡처만 하고 .npz 를
  Thor 로 옮겨도 된다 (report 는 numpy 만 있으면 어디서든 돈다).

## 1단계 — 캡처

```bash
python3 tools/pc_spike_capture.py --out ~/pc_spike --prefix scene
```

조작: `s` 저장 · `a` 자동캡처 토글 · `q` 종료
(창 없는 환경이면 `--no-preview --auto-n 10 --interval 1.5`)

### 촬영 프로토콜 (이대로)

실제 pick 대상을 **실제 observation 자세·거리**에서:

| 씬 | 물체 | 프레임 |
|---|---|---|
| A | 무지 골판지 box 1개 단독 | 2~3 (각도 조금씩) |
| B | cup 1개 단독 | 2 |
| C | bottle 1개 단독 | 2 |
| D | 얇은 물체(pen/가위 등) 1개 | 2 |
| E | box + cup + bottle 섞인 클러터 | 2 |
| F | box 가 테이블 **가장자리에 걸친** 상태 | 2 |

총 12~15 프레임. 대상 물체가 대략 화면 중앙에 오게.

## 2단계 — 리포트

```bash
python3 tools/pc_spike_report.py ~/pc_spike/ --ply-out ~/pc_spike/ply
```

씬마다 출력:
- `hole_ratio` (전체 / 중앙 ROI) — depth 구멍 비율
- `ROI depth p05/p50/p95` — 작업 거리
- `near-cluster extent` — 중앙 최근접 depth 대역 점들의 bbox 크기 (cm)
- `bimodality` — ROI depth 히스토그램이 물체/테이블 두 봉우리로 갈리나

`ply/*.ply` 는 MeshLab / CloudCompare / Isaac Sim 에서 육안 확인.

## 판단 게이트

| 리포트 결과 | 결론 |
|---|---|
| 중앙ROI hole `< ~10%` · near-cluster extent ≈ 물체 실제 크기(10~20cm) · bimodality `> 0.555` | ✅ depth 그대로 **Contact-GraspNet / VGN 입력 OK** → 모델 확정, 구현 착수 |
| hole 큼 (`> 20%`) · extent 과대 (`> 25cm` = 배경 섞임) · 단봉 (bimodality `< 0.555`) | ⚠️ 대응 필요: (1) 카메라 자세·거리 조정 재촬영 (2) **SAM mask 로 물체만 crop 후 입력** — 이 경우 SAM 실기 검증이 선행 과제 (3) **VGN 다중뷰 TSDF 융합** 으로 노이즈 완화 |

특히 **씬 A (무지 골판지 box)** 가 hole 심하면 → 모델 선택이 VGN 쪽으로 기울고,
SAM segmentation 실기 검증이 grasp 구현보다 먼저 와야 한다.

## 보고할 것

1. `pc_spike_report.py` 전체 출력 (텍스트 그대로)
2. `ply/` 중 씬 A, E, F 의 `.ply` 육안 소감 (물체가 형태를 유지하나 / 구멍 / 배경 덩어리)
3. 캡처 환경: 카메라 모델, 해상도, 대상까지 거리, 조명
4. `pyrealsense2` 가 Thor 에서 pip 로 깔렸는지 (안 깔렸으면 어떻게 우회했는지)

이 결과로 모델(Contact-GraspNet vs VGN vs GraspNet-baseline)을 확정하고
`learned_grasp_backend.py` 에 구현 클래스를 붙인다.
