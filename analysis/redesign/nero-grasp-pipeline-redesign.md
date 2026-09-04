---
name: nero-grasp-pipeline-redesign
description: "NERO pick&place 파이프라인 \"갈아엎기\" 진단 + 우선순위 논문 리스트 (2026-09-03 세션)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6ba50bd9-f824-411e-816c-7dcb14f08a0e
  modified: 2026-09-04T11:39:43.152Z
---

# NERO pick&place 재설계 — 진단 및 참고 논문

2026-09-03 세션. 사용자가 `/brutal`로 근본 원인 진단 요청. 상세 파트별 논문 분석은 [[okrobot]].

## 진단 (한 문장)
개별 문제 3개가 아니라 "grasp pose를 계산하려는" 접근법 하나가 3방향으로 터지는 것.
손으로 못 맞추는 걸 손으로 맞추고 있음(wiki 전체가 "미검증/신뢰불가").

## 갈아엎을 것
1. analytic grasp orientation 스택 전체 (`grasp_kinematics.py` atan2, CLAUDE.md 변환표,
   `resolve_grasp_dir`, `WAIST_Z`, approach-angle 스윕, Hough `angle_base_deg`,
   `geometry_3d.py` RANSAC+PCA) → **learned 6-DoF grasp 후보 생성 + batch reachability
   필터 + VLM re-rank**
2. grasp pose = 계산된 값 → grasp pose = 랭킹된 후보 집합
3. IK: KDL(단일 시드) → **pick_ik**(memetic global). 플래너: OMPL RRTConnect → **STOMP/CHOMP**
   (MoveIt2 내장 trajectory optimization). 마지막 구간: **Cartesian 직선 접근**(`p−k·a`,
   MoveIt2 `computeCartesianPath` 또는 이미 config에 있는 **Pilz LIN**). cuRobo/nvblox는
   나중에 필요하면(§"cuRobo 재검토 조건")
4. Claude가 inner loop 안 (CLAUDE.md에 "Claude가 매 호출마다 변환표를 손으로 적용"이라
   적힌 순간 설계 실패) → LLM 태스크플래너 → 결정론적 BT skill 실행기 → 실패 시에만 LLM
5. VLM이 좌표/각도 출력(자체 문서상 수 cm 오차) → VLM은 mark/point/시맨틱 라벨만,
   geometry는 detector + point cloud

## IK / 모션 결정 (2026-09-03)
- **sim 기준으로 먼저 간다.** 실물은 `pyAgxArm.move_p`(펌웨어 IK, CAN 너머 블랙박스,
  텔레메트리만) **끊는 전제** — 호스트에서 관절각 계산 → 실물은 `move_j`로 전송.
  펌웨어 IK는 sim에 못 가져옴(MCU 안, 라이브러리 구현 없음). 통일하려면 양쪽 다 안 씀.
- **IK 솔버 = pick_ik** (TRAC-IK와 성공률 동급, but 패키징 안전 + MoveIt2 유지 + 커스텀
  cost 지원). 실 변경: `nero_gripper_moveit_config/config/kinematics.yaml` 를 KDL →
  `pick_ik/PickIkPlugin` (mode: global) 로 교체함. `sudo apt install ros-$ROS_DISTRO-pick-ik`
  + colcon build 필요. 구 파일의 깨진 `arm喔:` 항목 제거(SRDF 그룹은 arm/gripper 뿐).
  이전 KDL 설정은 파일 상단 주석에 보존.
- **cuRobo 논문은 지금 안 읽음.** cuRobo는 IK뿐 아니라 collision-free 모션생성까지 하는
  시스템이라 pick_ik와 대체 범위가 다름. pick_ik + STOMP + Cartesian 접근으로 시작.
- KDL이 이미 `kinematics_solver_attempts: 50`, timeout 0.3 으로 쓰이고 있었음 =
  KDL 실패와 오래 싸워온 증거. pick_ik 전환 효과 클 것.

## Grasp 6-DoF pose generation (2026-09-03, 현재 집중 대상)

목표: **side / top-down 하드코딩 쿼터니언 문제 해결.** IK 솔버 딥다이브는 나중.

### 쿼터니언 문제가 사라지는 원리 (NERO 코드 기준)
- 지금: VLM "SIDE" → `resolve_grasp_dir` → `grasp_kinematics.py` roll/pitch/yaw 공식 +
  `_APPROACH_DIRECTION_DEG`(4방향 이산) + approach 스윕 → pose 1개 → IK 자주 실패
- 바뀜: point cloud → generator → 각 grasp이 완전 SE(3) pose(위치+쿼터니언+approach vector).
  top/side/기울어짐 전부 점수 붙은 pose로 출력. "mode" 없음. TCP offset은 각 grasp 자기
  approach 축 방향 하나
- **`grasp_types.py`에 `GraspCandidate` dataclass 이미 있음** (position, quaternion xyzw,
  approach_vector, score, source). `geometry_3d.py`/Hough 대신 generator가 채우게 하면 됨
- VLM 역할 축소: (a) 어느 물체(segmentation mask) (b) 선택적 시맨틱 re-rank. geometry 안 냄.
  `GraspIntent.orientation`, `grasp_type` → 삭제하거나 soft prior로만

### generator 방법론 결정 (2026-09-04, Contact-GraspNet vs 6-DOF GraspNet 정독 후)
**주 generator = Contact-GraspNet** ([[contact-graspnet]] 방법론 A~E 분석 완료).
6-DOF GraspNet([[6dof-graspnet]])은 생성형(CVAE 샘플→평가→refine)이라 무겁고 자유공간
샘플이 불완전 depth 에 불리 — CGN Part A 가 명시적으로 비판. **6DGN 에서 빌려올 것:
near-miss 후보를 버리지 말고 ≤1cm SE(3) 볼 안에서 perturb 해 pick_ik 재확인** (evaluator
망 없이 pick_ik 가 objective). 두 논문 공통 결론 = grasp gen 은 단일물체 pose 만,
reach/collision 은 downstream — NERO 3층 아키텍처 검증됨.

### 방법론
| 방법 | 입력 | 라이선스/구현 | 컴퓨트 | NERO |
|---|---|---|---|---|
| Contact-GraspNet (NVIDIA, ICRA21) | depth→PC + 타겟 mask | NVIDIA research(비상용), 공식 TF2 / 커뮤니티 PyTorch | 중 | **개념 1순위** |
| VGN (ETH, CoRL20) | TSDF 40³ | BSD, 공식 PyTorch | 소(~10ms) | **배포 1순위**, sim학습 → Isaac Sim 궁합 |
| GraspNet-1B baseline (SJTU, CVPR20) | RGB-D→PC | 오픈, PyTorch, 실데이터 학습 | 중 | 강한 오픈 모델 원하면 |
| AnyGrasp (SJTU, T-RO23) | RGB-D→PC | SDK 바이너리+라이선스(연구무료/상용유료) | 중(~4GB) | 성능 천장, OK-Robot이 씀 |
| HGGD (RA-L23) | RGB-D | 오픈 PyTorch | 소~중 real-time | VGN 대안 |
| M2T2 (NVIDIA, CoRL23) | PC | 오픈 | 중~대 | grasp+placement 통합 (Part C) |
| GPD (ten Pas, IJRR17) | PC | 오픈, ROS pkg | 소 | 고전 sampling+CNN, 폴백 |

### 읽을 것 (집중, 5)
1. **Contact-GraspNet** (ICRA 2021) — "contact point 예측 → 6-DoF", mask 필터.
   파트별 분석 [[contact-graspnet]] (Part A 완료 — 표현식·NERO 적용점 7개)
2. **"6-DOF GraspNet: Variational Grasp Generation for Object Manipulation"**
   (Mousavian, Eppner, Fox — NVIDIA, ICCV 2019, arXiv:1905.10520) — "grasp를 생성/평가한다"는
   패러다임 원전(CVAE). 왜 계산이 아니라 생성인지. 입력이 object PC(씬 아님) → seg 먼저 필요
3. **VGN** (CoRL 2020) — 실제 배포할 것. TSDF, 빠름, 오픈
4. **GraspNet-1Billion** (CVPR 2020) — grasp quality를 어떻게 점수매기는지(force-closure).
   후보 랭킹 설계
5. **AnyGrasp** (T-RO 2023) — 성능 천장
(옵션: HGGD / EdgeGraspNet — 효율형 최신)

### 용어 정리 — "6-DoF grasp" ≠ "7-DOF 팔" (2026-09-03 확인)
- **6-DoF grasp** = grasp **타겟 pose**가 6자유도(위치+방향) = SE(3). 로봇 기구학 무관,
  point cloud만 보고 "그리퍼 여기 이 방향" + width 출력
- **7-DOF 팔** = 관절 7개 = redundancy. 같은 grasp pose를 만드는 관절해가 여러 개
- 파이프라인: generator → 6-DoF grasp pose(로봇 무관) → pick_ik(7-DOF, redundancy 활용) → 관절각
- **7-DOF는 유리**: null-space로 관절한계/충돌/특이점 피한 IK 해 찾을 확률↑ (pick_ik global이 이득 큰 이유)
- **피할 것: "4-DoF grasp"** (top-down만: xyz+yaw) — 정확히 side/기울어진 grasp를 버림. "6-DoF" 명시된 논문만
- "...Net" = neural network. 학습 기반 = point cloud → grasp pose. 손계산 쿼터니언 대체 신호.
  "Net" 없으면(GPD, antipodal sampling) 해석적/샘플링

### 이름 비슷한 논문 혼동 주의
| 논문 | 그룹 | 내용 |
|---|---|---|
| 6-DOF GraspNet (ICCV19) | NVIDIA | 단일물체 PC → CVAE 생성/평가. 패러다임 원전 |
| 6-DOF Grasping ... in Clutter (Murali, Mousavian et al., ICRA20) | NVIDIA | 위의 clutter 확장 |
| Contact-GraspNet (ICRA21) | NVIDIA | scene PC → contact point. 더 빠르고 실용적 |
| GraspNet-1Billion (CVPR20) | **SJTU(다른 그룹)** | 데이터셋 + baseline. graspnetAPI |
| AnyGrasp (T-RO23) | SJTU | GraspNet-1B baseline 프로덕션 후속. SDK |

### 가중치 — 기존 오픈소스 pre-trained 그대로 시작 (재학습 계획 X)
| 모델 | pre-trained | 학습데이터 | 비고 |
|---|---|---|---|
| VGN | ✅ 공식(ETH) | 시뮬(pybullet) | sim→sim, Isaac Sim 도메인갭 작음 |
| GraspNet-1B baseline | ✅ 공식 checkpoint | **실데이터, RealSense/Kinect** | 너희 카메라 궁합 좋음 |
| Contact-GraspNet | ✅ 공식(NVIDIA TF2) + 커뮤니티 PyTorch | 시뮬(ACRONYM) | depth→PC만 |
| AnyGrasp | ✅ SDK 내장(폐가중치) | 비공개 | 라이선스 신청(연구무료) + HW 지문 라이선스 파일 |
| 6-DOF GraspNet | ✅ 공식 | 시뮬(ShapeNet) | object PC 입력 → seg 필요 |
| M2T2 | ✅ 공식(NVIDIA) | 시뮬 | |

**그대로 쓸 때 조정할 것(재학습 아님)**: (1) 그리퍼 파라미터 — 대부분 Franka Panda 기준
(폭 8cm), 너희 AGX 그리퍼 스펙으로 `gripper_width_max`/`finger_depth`/TCP offset 교체
(VGN/Contact-GraspNet config로 노출) (2) 입력 전처리 — depth 단위, workspace crop, voxel/TSDF
해상도, intrinsics (3) 좌표계 — 출력 grasp(카메라 프레임) → base_link TF(`_cam_to_base` 있음)

**재학습 필요해지는 경우(나중, 아마 불필요)**: sim-trained 모델로 실기 성공률 낮을 때 →
Isaac Sim 도메인 랜덤화 fine-tune / 특이물체(무지 골판지 box) 분포 크게 다를 때 / 그리퍼 물리적으로 많이 다를 때

**시작 전략**: VGN(가벼움) 또는 GraspNet-baseline(RealSense 실데이터) 가중치 그대로 →
Isaac Sim RViz 시각화 → 성공률 보고 재학습 판단. 처음부터 학습 계획 세우지 말 것.

### 구체적 경로 (spike 결과 반영: [타겟 mask]는 선택 아니라 필수)
1. grasp 생성 서비스: RealSense depth → **SAM mask crop(필수)** → masked PC → generator
   → `[(pose,score,width)]` → `GraspCandidate` 매핑
2. sim 검증: Isaac Sim 알려진 물체 → 후보 생성 → RViz grasp 마커 시각화
3. 필터/랭킹: reachability(IK) + collision(planning scene) + score → 최상위 reachable
4. planning_node 배선: `grasp_dir`+쿼터니언 계산 경로 → "generator 최상위 후보"로 교체,
   구 경로는 플래그 뒤 폴백
5. VLM `infer_grasp` → "타겟 mask 제공 + (선택) 시맨틱 re-rank"로 축소

### 계획 (2026-09-03): Contact-GraspNet + 6-DOF GraspNet 읽고 → 구현 착수
읽기와 **병렬로 point cloud spike 먼저** 진행 중.

### Point cloud spike — 완료 (2026-09-03, Thor D435i, commit `a0af4d8`)
목적: "실물 RealSense depth 가 grasp net 입력으로 쓸 만한가" = #1 리스크.
결과 문서: `tools/PC_SPIKE_RESULT.md`. 다른 Claude 세션이 Thor에서 실행.

**판정: raw depth 를 Contact-GraspNet/VGN 에 그대로 넣는 건 아직 아님. 게이트 탈락.**

- 2라운드: 1280×720/필터없음 → 848×480(D4xx 네이티브)+High Accuracy preset+
  disparity→spatial→temporal 필터. 중앙ROI hole_ratio median ~20% → ~9% 개선.
- **핵심 실패 원인: ROI depth bimodality 단봉 프레임 다수 = 물체/작업면이 depth 만으로
  분리 안 됨** → 이건 depth 품질 문제가 아니라 segmentation 문제. **SAM mask crop 이 전제.**
- occlusion shadow(스테레오 베이스라인 그림자), 금속/반투명 물체 몸통 depth 전멸(IR
  반사/투과) = 조건 보정으로 해결 안 됨. 불투명이면 OK, 투명/반투명/고반사는 depth 경로 밖.
- `pc_spike_report.py` near-cluster extent 지표: top-down 가정이라 얕은 관측각에서 작업면
  평면을 슬라이스해 20~44cm로 부풀려짐 → **수정 TODO** (SAM crop 되면 마스크 내부 점
  bbox 직접 측정으로 대체 가능).
- 촬영 조건 권고(실기 반영): 관측각 **45~70° 내려보기, grazing 금지**, 거리 0.6~0.8m,
  **검은 무광 작업면이 일관되게 최악**(매트 등 대응).

**모델 결정: `SAM mask crop → Contact-GraspNet` 1순위.** 마스크 내부에선 물체가 형태
유지(육안 확인). 클러터/반사물엔 VGN 다중뷰 TSDF 융합 폴백.

**→ 시퀀스 변경: segmentation 실기 검증이 grasp net 구현보다 선행.**
`sj_pickplace/segmentation_backend.py` `SamSegmentation` 은 배선만 됨/미검증
([[vlm_capability_tiers]]). 검증되면 `learned_grasp_backend.py` 에 Contact-GraspNet 클래스.

### YOLO / SAM / point cloud — 관계와 흐름 (2026-09-04)
**전부 붙여서 씀. 택1 아님.** YOLO·SAM은 2D RGB에서만 작동, depth는 별도 스트림,
마스크가 "어느 depth 픽셀이 물체인지" 잇는 다리.

| | 정체 | 입력 | 출력 | 아는 것 |
|---|---|---|---|---|
| YOLO | detector | RGB | bbox + 라벨 + conf | "어느 게 컵" (의미) |
| SAM | promptable segmenter | RGB + **prompt(bbox)** | 픽셀 마스크 | 경계만. 뭘 가리켰는지 모름 (class-agnostic) |
| 역투영 | — | 마스크 + **depth** + intrinsics | PC (N,3) | — |
| Contact-GraspNet | — | PC | 6-DoF grasp 후보 | — |

흐름: `RGB → YOLO → {"cup", bbox} → SAM(bbox=prompt) → 마스크 → point_cloud.py 역투영
→ masked PC → _cam_to_base TF → Contact-GraspNet`

| 단계 | 걸러지는 것 |
|---|---|
| YOLO 후 | 다른 물체, 화면 대부분 (bbox 사각형만) |
| **SAM 후** | **사각형 안 테이블도 제외 = spike bimodality 문제 해결점** |
| 역투영 후 | depth 구멍 픽셀 (occlusion/검은면/금속 — SAM이 못 고침) |

하나 빼면: YOLO 빼면 SAM이 뭘 자를지 모름 / SAM 빼면 bbox 사각형에 테이블 섞임 (spike
실패) / depth 빼면 2D 마스크뿐, 3D grasp 불가.

NERO 코드 매핑: YOLO=`vlm_boxyolo.py`(운영중) / SAM=`segmentation_backend.py`
`SamSegmentation`(배선됨, 벤치로 모델 확정) / 역투영=`point_cloud.py`
`mask_depth_to_pointcloud()`(구현됨) / intrinsics=`camera_calibration.py` / TF=`_cam_to_base`
(joint1≈0에서만) / CGN=`learned_grasp_backend.py` `LearnedGraspBackend` ABC(클래스 미구현).

### segmentation 접근 결정 (2026-09-04)
- YOLO 와 SAM 은 택1 아님. **YOLO = 의미, SAM = 경계.** SAM prompt = YOLO bbox.
- **벤치마킹할 2안**:
  - (2) **YOLO bbox → SAM(또는 MobileSAM/SAM2/HQ-SAM) 마스크** — 경계 정밀, 모델 2개,
    class-agnostic
  - (3) **YOLO-seg / FastSAM** — 마스크 직접 출력, 모델 1개·빠름, 마스크 품질 낮음
    (프로토타입 ~160px 업샘플), 학습 클래스만
  현재 `best.pt`/`yolov8n.pt` 는 detection(bbox) 모델이라 (3)은 seg 재학습 or FastSAM 필요.
  실물 프레임(box/cup/클러터)에서 마스크 품질 + 속도 비교 → 결정.

### SAM 논문 (읽기 순서)
1. **Segment Anything** (Kirillov et al., Meta, ICCV 2023) — 원전. promptable seg,
   bbox/point/mask prompt, ambiguity(3-mask), class-agnostic
2. **SAM 2** (Ravi et al., Meta, 2024) — ultralytics 지원, 이미지 경로도 SAM1보다 빠르고 정확
3. **HQ-SAM** ("Segment Anything in High Quality", Ke et al., NeurIPS 2023) — frozen SAM +
   작은 adapter로 **마스크 경계 품질↑.** 헐거운 마스크 = point cloud crop에 테이블 누출 → 직결
4. **MobileSAM** ("Faster Segment Anything", Zhang et al., 2023) — 경량 인코더 distillation, Thor 배포
엣지 변형: EfficientSAM(CVPR24), FastSAM(2023, YOLOv8-seg 계열), EdgeSAM(2023),
  **NanoSAM**(NVIDIA, 논문X — TensorRT distill SAM, Jetson 전용, Thor에 가장 직접적)
패턴: **Grounded SAM** ("Assembling Open-World Models", Ren et al., 2024) — detector+SAM 조립.
  너희 = Grounding DINO 자리에 YOLO. / LangSAM(도구, OK-Robot이 씀)
우선순위 밖: SAM 3D/SAM-6D(마스크 3D lift — 오버킬), TinySAM(압축+양자화),
  Grounding DINO(ECCV24 — Grounded-SAM detector 절반, YOLO로 대체하니 낮음)

툴 (`tools/`, commit `e9ecd8e`+`a0af4d8`): `pc_spike_capture.py`(848×480+preset+필터,
`--no-filters`/`--preset none`/`--hole-fill` 옵션), `pc_spike_report.py`(extent 지표 수정 TODO),
`PC_SPIKE.md`(지시서), `PC_SPIKE_RESULT.md`(결과). 원본 .npz/.ply 는 커밋 안 함.

### 착수 타이밍 분석 (2026-09-03)
**"논문 2개 읽은 직후"는 착수에 충분치 않음. 논문 + point cloud sanity spike 후가 맞음.**
spike는 읽기와 병렬로 지금 가능(독립적).

준비된 것(green): pick_ik 검증됨(IK 필터), `GraspCandidate` dataclass 존재, RealSense
eye-in-hand, Isaac Sim, `_cam_to_base` TF, MoveIt2 planning scene, `segmentation_backend.py` 뼈대.

리스크(위험도順):
1. **point cloud 품질 — #1 리스크.** `grasp_geometry_pipeline.md`가 이미 지적: NoOp seg
   배경 누출, 15cm box가 extents 0.27~0.44m, point_count 상한. 같은 나쁜 입력이 grasp net에도
   들어감. 무지 골판지 box = stereo depth 구멍 악명. **garbage PC → garbage grasp.**
2. **타겟 segmentation.** 6-DOF GraspNet은 object-only PC **필수**(하드 의존). Contact-GraspNet은
   mask 있으면 좋고 없으면 scene-wide 가능(소프트). SAM 실기 미검증(vlm_capability_tiers.md).
3. **모델 런타임 환경.** Contact-GraspNet 공식 = TF2(구버전, CUDA 핀). 커뮤니티 PyTorch 포트는
   충실도 편차. VGN = 깔끔한 PyTorch. 어디서 돌릴지(PC 3080Ti는 Isaac Sim과 공유 / Thor는
   Blackwell 문제). env 세팅에 며칠 날아갈 수 있음.
4. **Isaac Sim 정량 검증.** grasp 좋은지 알려면 sim 실행(그리퍼 close+attach 물리 튜닝돼 있나?)
   또는 RViz 육안. 시작은 육안, 반복 개선엔 sim GT 필요.
5. **planning_node 배선.** PC 쪽 미push 코드 있음(vlm_capability_tiers.md). 통합 = planning_node
   건드림 → 그 코드 상태 먼저 정리. 또는 planning_node 밖에서 프로토타입 먼저.
6. **그리퍼 파라미터.** AGX 그리퍼 max width / finger length / TCP offset 실측값 필요(사소하지만 블로킹).

지금 병렬로(읽기와 동시, 독립): point cloud spike(실물 씬 depth 캡처 → PC 시각화 → 구멍/노이즈/
배경 평가), 그리퍼 스펙 수집, 컴퓨트 위치 결정.

읽기 후 실제로 블로킹: 모델 최종 선택(입력 요구사항을 논문에서 이해해야), 통합 배선.

green-light 체크리스트: [ ] 실물 PC 품질 확인 [ ] 컴퓨트 위치 + torch/CUDA env 계획
[ ] SAM 실기 mask 되나 [ ] AGX 그리퍼 스펙 [ ] planning_node 미push 코드 정리
[ ] Isaac Sim 그리퍼 물리(close+attach) 동작 확인

**첫 구현 마일스톤(로봇/MoveIt/planning_node와 완전 분리)**: 저장된 point cloud 파일 →
grasp pose 리스트 → RViz 마커. 이게 되면 그다음 통합. end-to-end부터 하지 말 것.

모델 선택 가이드: 씬 PC 깨끗 → Contact-GraspNet / 노이즈·물체 격리 필요 → VGN(TSDF 융합이
노이즈 완화) 또는 GraspNet-baseline / 6-DOF GraspNet은 clean object-only PC 필요라 1순위 아님.

### pick_ik smoke test — 통과 (2026-09-03)
- ROS Humble, `ros-humble-pick-ik` 설치됨 (`/opt/ros/humble/lib/libpick_ik_plugin.so`)
- `nero_gripper_moveit_config` 리빌드 → `move_group` 정상 기동 ("You can start planning now!")
- `ros2 param get /move_group robot_description_kinematics.arm.kinematics_solver`
  → `pick_ik/PickIkPlugin`, `mode: global` 확인 (내 kinematics.yaml 그대로 로드됨)
- `/compute_ik` (arm, gripper_flange, FK로 만든 reachable pose) → `error_code.val=1` SUCCESS.
  해가 seed와 다른 config로 나옴 → 실제 최적화 동작 확인
- **arm 그룹이 7-DOF (joint1~7, base_link→gripper_flange chain, redundant)** — KDL 단일시드가
  특히 나쁜 케이스. joint7은 CLAUDE.md에서 카메라 프레이밍에 쓰는 실제 가동관절
- 남은 경고는 무해: "world" TF 없음(robot_state_publisher 미실행), "empty JointState"(seed 안 줌)

### 정량 평가는 나중에 한 번에
generator가 내는 실제 grasp 후보 N개로 IK 성공률 pick_ik vs KDL 측정. 손으로 고른 pose로
벤치하면 오해 소지. grasp 후보 나온 뒤 같이.

## 목표 아키텍처 (3층)
- LLM(Claude): 태스크 그래프, 실패 시 재계획. 태스크당 2~3회
- 결정론적 skill 실행기(behavior tree): perception→grasp→motion→execute, LLM 왕복 없음
- VLM: 씬당 1회 시맨틱 그라운딩, 캐시
- 모션: pick_ik(IK) + STOMP(경로) + Cartesian(최종 접근). grasp 후보는 pick_ik 루프로
  reachability 필터(20~50개면 GPU 배치 불필요)

### 관절 궤적은 누가 계산하나 (2026-09-04, 반복 확인용)
**MoveIt2 가 계산한다.** grasp generator 는 6-DoF pose(SE(3) 타겟)만 냄, 관절각 아님.
그다음:
- MoveIt2 내부에서 **pick_ik**(IK 플러그인) 가 goal pose → goal 관절 config
- **STOMP**(플래너 플러그인) 가 현재 관절상태 → goal config 의 collision-free 궤적
  (JointTrajectory = 관절위치+속도+time_from_start 웨이포인트 열)
- Cartesian 마지막 구간(`p−k·a`)도 MoveIt2 `computeCartesianPath` 또는 Pilz LIN
  (이미 config 에 있음)이 IK 해서 JointTrajectory 로
- 결과 `moveit_msgs/RobotTrajectory` 는 **하드웨어 무관** — 그냥 숫자
- **분기는 마지막 실행 단계 뿐**: 같은 JointTrajectory 를 `FollowJointTrajectory` 액션으로
  → sim: Isaac Sim `arm_controller` / 실물: `pyAgxArm` 을 감싼 `ros2_control` HW 인터페이스
  가 `move_j`(또는 joint MIT)로 CAN 전송
- 현재 divergence 원인: sim 은 MoveIt2 경유, 실물은 `move_p`(펌웨어 자체 IK+궤적). 고치려면
  pyAgxArm 을 ros2_control 로 래핑 → 실물도 MoveIt2 궤적 받아 실행만
- planning_node(또는 BT 실행기)는 궤적을 **직접 계산 안 함** — MoveIt2 plan API 호출만

## 차용 공식 (Contact-GraspNet + 6-DOF GraspNet + OK-Robot) — 2026-09-04

### A. Grasp pose 재구성 — Contact-GraspNet Eq (1)(2)
```
t_g = c + (w/2)·b + d·a
R_g = [ b │ a×b │ a ]        (열벡터, 회전행렬)
```
c=contact point(PC 실측 점), a=approach 단위벡터, b=baseline(손끝 닫히는 축), w=폭,
d=그리퍼 baseline→base 거리(하드웨어 상수, AGX≈0.19~0.20m). R_g→쿼터니언(xyzw)→
`GraspCandidate.quaternion`. **`d·a` 항이 TCP offset 통일** — TOP/SIDE offset "합치지 마라"
문제 소멸.

### B. a,b 직교정규화 — Contact-GraspNet Eq (6)
```
b̂ = z1/‖z1‖
â = (z2 − ⟨b̂,z2⟩·b̂) / ‖z2 − ⟨b̂,z2⟩·b̂‖
```
raw 벡터에서 유효한 a⊥b. 직접 grasp 만들/refine 시 이걸로 정규화 → R_g 항상 valid
rotation, roll/pitch/yaw axis-flip 버그 회피.

### C. Grasp 간 거리 / refinement — CGN Eq (7)(8) / 6DGN Eq (3)(6)
```
v_i(g) = v·R_g^T + t_g                  v = 그리퍼 위 미리정의 점 5개(control points)
L(g,ĝ) = (1/n) Σ_u min_u ‖v_u(g) − v_u(ĝ)‖      min_u = 그리퍼 180° 대칭 고려
```
용도: 후보 중복제거(L<ε), 180° twin 생성(둘 다 pick_ik → reachability 2배).
**near-miss refinement (6DGN Eq 6 차용)**: `Δg = η·(∂S/∂T)·(∂T/∂g)`, η로 병진 스텝 ≤1cm.
NERO 변형: evaluator 망 없음 → S="pick_ik reachable+collision-free?" (미분 불가) →
**국소 탐색**: near-miss 후보 주변 ≤1cm 병진 + 소각도 회전 볼에서 perturb → IK 재확인.

### D. 전처리 정규화 — 6-DOF GraspNet §3
```
origin = X̄ = mean(관찰 point cloud),  axes ∥ camera frame
```
CGN backend: masked PC mean 빼고 넣기, 출력 grasp에 mean 다시 더하기.

### E. 후보 랭킹 cost field — OK-Robot §II-A 패턴 (수식 자체는 폐기, 구조 차용)
```
s(x) = s1 + 8·s2 + 8·s3
  s1 = ‖x−x_o‖                            거리
  s2 = 40 − min(‖x−x_o‖, 40)              standoff 여유(너무 가까우면 페널티)
  s3 = 1/‖x−x_obs‖ (근접 시), else 0       장애물 역거리
```
NERO 대응:
- 관찰 자세 선택: `score = w1·중심정렬오차 + w2·bbox잘림 + w3·occlusion + w4·unreachable` 최소화
- Grasp 랭킹(Phase 3c): `score = graspness(ŝ) − w1·θ⁴ − w2·IK margin부족 − w3·collision clearance부족`
  θ=grasp normal과 바닥 normal 각도, **θ⁴ 페널티 = top-down 선호** (OK-Robot Part B `S − θ⁴/10`,
  calibration 오차에 강함). s2류 standoff 항, s3류 장애물 역거리 항 차용.

### F. Placement 기하 — OK-Robot Part C
```
정렬: X=로봇정면, Y=좌우, Z∥바닥normal / 정규화: 로봇(x,y)=(0,0), 바닥 z=0
(x_m, y_m) = segment된 컨테이너 클라우드 median (x,y)          ← 떨굴 위치
z_max = buffer + max{z │ 0≤x≤x_m, |y−y_m|<0.1}                ← 떨굴 높이 (테두리 + 버퍼)
"A on B" = "A near B": A_pt = argmin over (A top-10, B top-50) ‖A−B‖
```
`container_p20` depth 휴리스틱 대체. buffer = 그리퍼 길이 + 물체 늘어진 길이(OK-Robot은 0.2m).

## 학습 판단 — pretrained vs 파인튜닝 (2026-09-04)
**"Net"이 pretrained 배포하면 내가 학습 안 함** (YOLO/SAM 처럼). Contact-GraspNet은
ACRONYM(ShapeNet 8872메시, grasp 17.7M) + 렌더 tabletop 10k씬, **Franka 그리퍼**로 학습.

**Config ≠ 학습**: `d`, `w_max`, 그리퍼 collision 메시, intrinsics, workspace crop, depth 범위,
입력 점 수 = 그냥 설정 (`d` 0.1034→0.19 는 상수 하나 고치는 것, 학습 아님).
**학습/파인튜닝** = s/a/b/w 예측 head 가중치.

도메인 갭(pretrained 성능 깎을 수 있는 것): ① depth 센서 갭(렌더≠RealSense, 제일 큼)
② 그리퍼 갭(Franka 8cm/d=0.10 vs AGX 10cm/d=0.19) ③ 관측각(top-down vs 비스듬)
④ 물체 분포(무지 골판지 box).

**순서**: pretrained 돌림 → NERO 실물 masked PC(box .ply)에 먹여 RViz 시각화 → box에 grasp
말 되면 학습 불필요 진행 / 체계적으로 나쁘면(방향 틀림, 위치 cm 어긋남, 명백한 표면에 후보 0개)
파인튜닝.

**파인튜닝 = from scratch 재학습 아님**: pretrained 이어서, Isaac Sim에서 NERO 물체 메시 +
random pose + **RealSense 노이즈 모델**(depth_noise 측정값에 맞춤) + **AGX 그리퍼 기하로**
(a,b,w) 라벨 생성 → 몇 천 iter (~하루). s(노이즈 적응)/w(10cm 재보정)/a·b(NERO 그리퍼 자세) 이동.

**폴백 사다리**: pretrained CGN → 파인튜닝 CGN → (파인튜닝 고통스러우면) VGN from scratch
학습(작은 망, 시뮬 학습이라 Isaac Sim 데이터 생성 쉬움).

## 우선순위 논문 / 문서 (2026-09-03 갱신 — cuRobo 빠짐)
1. **OK-Robot** (Liu et al., 2024) — 목적지 지도. VLM+detector+learned grasp 통합, "통합에서
   뭐가 깨지나". [[okrobot]] (A/B/C/D 분석 완료)
2. **pick_ik docs** (MoveIt2) — 지금 쓰는 IK 솔버. 파라미터(mode/scale/threshold), 커스텀 cost
3. **MoveIt2 STOMP / CHOMP docs** — OMPL 대체 플래너
4. **STOMP 논문** (Kalakrishnan et al., ICRA 2011) — trajectory optimization 개념, 짧은 클래식
5. **Contact-GraspNet** (Sundermeyer et al., ICRA 2021) — analytic orientation 대체. point
   cloud → 6-DoF grasp 분포
6. **VGN** (Breyer et al., CoRL 2020) — Contact-GraspNet 경량 대안, TSDF, Jetson 현실적
7. **Integrated Task and Motion Planning** 서베이 (Garrett et al., 2021) — "grasp 정하고 IK
   체크" 실패가 TAMP 문제라는 프레이밍
8. **Text2Motion** (Lin et al., 2023) — LLM 플래닝 + 실행 전 feasibility 체크 = pre-flight 필터
9. **M2T2** (NVIDIA, CoRL 2023) — grasp + placement 한 모델 (Part C 놓을 곳 커버)
10. **Closing the Loop for Robotic Grasping** (Viereck et al., CoRL 2017) — 마지막 접근
    visual servoing. "OMPL 성공해도 그립 정확도 감소" 해법

참고(순위 밖, 필요할 때):
- cuRobo(NVIDIA, ICRA 2023) + Isaac ROS cuMotion — §"재검토 조건" 충족 시
- nvblox(Millane et al., ICRA 2024) — GPU ESDF, cuRobo world용
- AnyGrasp(T-RO 2023), GraspNet-1Billion(CVPR 2020), GIGA(RSS 2021)
- CabiNet(NVIDIA, ICRA 2023 — 어수선한 리셉터클 collision), TAX-Pose(CoRL 2022),
  Paxton "Stable Configurations"(CoRL 2021), AdaPoinTr(가려진 컨테이너 완성)
- REFLECT(CoRL 2023), LLM3(2024), PDDLStream(ICAPS 2020), ProgPrompt(ICRA 2023),
  Inner Monologue(2022), Code as Policies(ICRA 2023), ReKep(2024)
- GraspGPT(RA-L 2023), RoboPoint(2024), LERF-TOGO(CoRL 2023) — VLM re-rank
- Reuleaux(2018) — reachability map
- 도구: BehaviorTree.CPP(ROS2, Nav2/MoveIt Pro가 씀), TRAC-IK(pick_ik 대안), Pinocchio+HPP-FCL

## cuRobo 재검토 조건
1. STOMP 경로가 너무 느리거나 품질 부족
2. grasp 후보 100개+ 를 궤적 feasibility까지 배치로 걸러야 함
3. 모션 계산을 Thor로 이관 (그때 Thor/Blackwell/JetPack7 지원: torch 휠, cuRobo 커널 빌드,
   cuMotion Jetson 매트릭스 확인 필요 — jetson_thor_vlm_upgrade_task.txt와 같은 조사 패턴)
   ※ "sim 기준" 이면 cuRobo는 PC 3080Ti(Ampere)에서 완전 지원 — 설치 리스크는 Thor 이관 시에만

## 로드맵 — 5 Phase (2026-09-04, OK-Robot + CGN + 6DGN + spike 종합)

원칙: **Vision(Phase 1) 완전히 닫기 전 Phase 2 착수 금지** — masked PC 품질이 grasp
품질의 상한. 세 논문 모두 "SAM 마스크된 단일물체 PC" 전제. **본격 코드 반영 + 기존 코드
삭제는 Phase 4 (프로토타입 검증 후) 한 번에** (사용자 지시).

### Phase 0 — 완료 ✅
- pick_ik 전환 + smoke test (KDL→`pick_ik/PickIkPlugin`, `/compute_ik` SUCCESS)
- point cloud spike (raw depth 게이트 탈락 → SAM crop 필수)

### Phase 1 진행 상황 (2026-09-04, Thor 1b 실행됨 — HDD `nero_vision_bundle/phase1_thor_run/`)
- **1b seg 벤치 완료** (19프레임 A_box/B_cup/C_bottle/D_thin/E_clutter/F_edge, .npz는 HDD에
  없고 masked .ply + 로그만). 촬영 품질 미흡(일부 grazing, cup/bottle 원통 얕은 각도로
  z-extent 1.5~2.4cm=테두리만) → **재촬영 필요, 단 지금 어려워 추후로 미룸**.
- **잠정 seg 결정**:
  - **yolo-seg(COCO yolov8s-seg) 탈락** — tight 평균 1.96, 골판지 box(COCO 아님)에서 파탄,
    클러터에서 인스턴스 오배정. NERO 클래스 seg 재학습 필요(Phase 4).
  - **mobile_sam ≈ sam2.1_t 품질** 동일, mobile_sam이 빠름(warm ~65ms vs ~127ms) → mobile_sam.
  - **fastsam**: 3~4배 빠름(~16ms), **bg_leak 일관되게 더 낮음**(경계 보수적), smooth 약간 낮음.
    point-cloud-crop엔 bg_leak가 더 중요 → fastsam도 강력 후보. `SamSegmentationBackend`는
    `ultralytics.SAM`이라 FastSAM 추가 = 5줄 backend 변경(Phase 4).
  - **point prompt 미검증** (벤치는 bbox만). 클러터용으로 재촬영 시 확인.
- **box(불투명, 프로덕션 타겟) vision 파이프라인은 작동**: 9개 box masked .ply 전부 extent
  ~21×12×5~7cm 일관, 12~34k pts, bg_leak <5% → **Contact-GraspNet 입력 준비됨**.
  cup/bottle/thin/clutter는 재촬영 후.
- **1a-③ AGX 그리퍼 (pyAgxArm docs + URDF에서 확인)**:
  - **바이너리 아님.** `move_gripper_m(value_m, force_N)` — 폭 위치제어, m 단위, 정밀도 1e-6,
    force [0,3.0]N. → **예측 width `w`가 실제 제어 신호로 사용 가능** (이전 "0 or 1" 가정 폐기).
  - **w_max ≈ 0.10 m** (URDF: prismatic 손가락 각 0.05 × 2. API `max_range_config` 흔한값
    0.07/0.1 — `get_gripper_teaching_pendant_param()`로 확정).
  - **d ≈ 0.19~0.20 m** (fingertip contact → `gripper_flange`: URDF 0.0055 flange→base +
    0.1358 base→prismatic + 손가락 ~0.05~0.06). Franka(Contact-GraspNet 기본 d=0.1034)보다
    훨씬 길다 → **`d` 반드시 재설정.** w_max는 0.08→0.10이라 width head 대략 OK, 약간 rescale.
  - closed-loop close: `move_gripper_m(w, force=1.0)` 후 `get_gripper_status().force` 로 접촉 확인.
- **1a-① depth 노이즈, 1a-② observation 각도**: 아직 안 함 (물리 접근 필요).

### 재촬영 체크리스트 (추후)
1. observation 각도 45~70° 내려보기 (grazing 금지) — F_edge/일부 씬이 grazing이었음
2. 거리 0.6~0.8m (F_edge_000이 0.43m로 너무 가까움)
3. 원통 물체(cup/bottle): 각도·거리 개선해도 얕으면 테두리만 → 측면에서 더 내려보거나 다중뷰
4. D_thin: bg_leak 100%(테이블 평면과 구분 불가) — depth-grasp 경로 밖, 별도 트랙 확정
5. point prompt 추가 비교 (클러터에서 bbox 안 여러 물체)
6. `depth_noise.py` 실행, observation 자세에서 카메라-테이블 각도 측정

### Phase 1 — Vision 층 확정 (모두가 쓰는 기반) ← **착수, 툴 커밋 `4a86555`**
| # | 과제 | 게이트 | 툴 |
|---|---|---|---|
| 1a | ① 국소 depth 노이즈 실측 vs 5mm ② observation 자세 각도 확정(45~70° down) ③ **AGX 그리퍼 스펙+제어 방식**(position-controllable? max width? d?) | 수치 3개 | `tools/depth_noise.py` (① 자동), ②③ 수동 |
| 1b | seg 벤치: (2)YOLO bbox→SAM 계열 vs (3)FastSAM/YOLO-seg. 실물 box/cup/클러터 마스크 경계·bg_leak·Thor 속도 | 1개 방식 선택 | `tools/seg_bench.py` (지표: bg_leak%/smooth/extent PCA/hole%/bimod. `--label` bbox 드래그) |
| 1c | Vision 프로토타입: `seg_bench.py --models <선택> --ply-out`. 여러 물체·각도 masked .ply 육안 | **bg_leak%<3, bimod<0.555, extent=실제 크기 반복 재현** | seg_bench.py |

지시서 `tools/PHASE1_VISION.md`. `SamSegmentationBackend`(`segmentation_backend.py`)는 이미
잘 구현됨(ultralytics SAM, bbox+point prompt, lazy load, NoOp 폴백) — 벤치로 SAM_MODEL만 정하면 됨.
`pc_spike_report.py` extent 지표는 seg_bench PCA extent 로 대체(주석 추가).
커밋 `4a86555` (스파이크 3파일 + 이번 4파일만, 사용자 미커밋 작업 안 건드림). push 는 사용자.

### Phase 2 — Grasp 층 (프로토타입, 기존 코드 안 건드림)
| # | 과제 |
|---|---|
| 2a | Contact-GraspNet 런타임 — PyTorch 포트(`contact_graspnet_pytorch`) + Docker + 별도 서비스. 컴퓨트: PC 3080Ti(Isaac Sim VRAM 공존 확인) 우선, 안 되면 Thor |
| 2b | `learned_grasp_backend.py`에 `ContactGraspNetBackend`. 입력=1c masked PC(`--local_regions`), mean centering. 출력→`LearnedGraspOutput`. **width=예측값 대신 masked PC b방향 단면 폭 기하측정.** d=AGX 상수 |
| 2c | **첫 마일스톤: 저장 masked PC → grasp 후보 → RViz 마커.** s threshold, 그리퍼 180° twin. 로봇/MoveIt 분리 |
| 2d | grasp → base_link TF (`_cam_to_base`, canonical observation 자세에서만) |

### Phase 3 — Grasp→모션 통합 (프로토타입 브랜치/플래그)
| # | 과제 |
|---|---|
| 3a | pick_ik 루프 reachability 필터 (후보 + 180° twin 둘 다) |
| 3b | near-miss refinement (6DGN) — pick_ik 실패 후보를 ≤1cm SE(3) 볼에서 perturb 재확인 |
| 3c | 랭킹 `graspness − w1·θ⁴(top-down) − w2·IK margin − w3·collision clearance` + VLM 시맨틱 re-rank. **argmax(s) 금지** |
| 3d | STOMP 경로 + Cartesian 마지막 구간(`p−k·a`) + closed-loop 그리퍼. 플래그 뒤, 기존 경로 폴백 |

### Phase 4 — 검증 후 한 번에 반영 (사용자 지시)
- Isaac Sim end-to-end pick 성공률 (**그리퍼 물리 close+attach 동작 확인 선행**) + 실물 검증
- 통과 시 일괄: 삭제(`grasp_kinematics.py` atan2 / 변환표 / `resolve_grasp_dir` / `WAIST_Z` /
  approach 스윕 / Hough `angle_base_deg` / `geometry_3d.py` PCA), 교체(OMPL→STOMP, planning_node
  배선), 통일(pyAgxArm `ros2_control` 래핑 → `move_p` 중단 `move_j`), `infer_grasp` VLM 역할 축소

### Phase 5 — 상위 층 (별도 트랙, Phase 4 후)
- static 시맨틱 맵 (OK-Robot Part A) — 컨테이너/가구 좌표, `ground_object` sweep 대체
- 결정론적 BT skill 실행기 + Claude 태스크 층 후퇴
- placement (OK-Robot Part C rim 휴리스틱 or M2T2)
- Qwen3-VL 벤치마크

병렬 가능(Phase 1과 무관): planning_node 미커밋 코드 정리, 컴퓨트 위치 결정.

한계(spike로 확정): 투명/반투명/고반사 금속 물체는 depth-grasp 경로 밖 → 별도 트랙
(알려진 geometry / VLM). 검은 무광 작업면도 depth 품질 저하 요인. SAM crop은 필요조건이지
충분조건 아님(마스크 내부 구멍은 남음).

## 제안안 문서 (2026-09-04)
`/brutal` 진단 + AS-IS/TO-BE 파이프라인 + 핵심 변경 5가지 + 실측 근거 + 로드맵 11항 +
리스크 표를 텍스트 제안서로 작성(사용자가 아티팩트 거부, 터미널 텍스트로). 랩 발표/제안용.
핵심 변경 5: Grasp(analytic→learned 후보), IK(KDL→pick_ik ✅), 모션(OMPL→STOMP+Cartesian),
LLM(inner loop→task 층), 인지(VLM 좌표→SAM mask+static 맵).

## 관련 파일
- `ros2_ws/src/agx_arm_sim/Moveit2/nero_gripper_moveit_config/config/kinematics.yaml`
  (2026-09-03 KDL→pick_ik 교체함. 활성 config — `gazebo_moveit.launch.py`,
  `start_nero_isaac_all.sh` `MOVEIT_PKG`)
- `nero/yolo/vlm_grasp_server.py`, `vlm_grasp_server_networked.py`
- `ros2_ws/src/nero_sj_pickplace/sj_pickplace/grasp_types.py` (`resolve_grasp_dir` — 은퇴 대상)
- `ros2_ws/src/nero_sj_pickplace/sj_pickplace/planning_node.py` (OMPL→STOMP, Cartesian 접근)
- `docs/wiki/{grasp_geometry_pipeline,grasp_kinematics_ik,vlm_capability_tiers}.md`
- `tools/{depth_noise,seg_bench,pc_spike_capture,pc_spike_report}.py` + `PHASE1_VISION.md`/`PC_SPIKE*.md`
- `sj_pickplace/{point_cloud,segmentation_backend,learned_grasp_backend,grasp_pose_generator,camera_calibration}.py`
- `.claude/skills/grasp-kinematics-design/` — grasp_kinematics.py 수정 시 필독(폐기된 접근 이력). Phase 4에서 삭제 작업 전 읽을 것
- `~/Downloads/pyAgxArm` (실물 드라이버 — `move_p`=펌웨어 IK 끊고 `move_j` 사용)
- `sjfolder/jetson_thor_vlm_upgrade_task.txt` (Qwen3-VL 벤치 — 우선순위 낮춤)
- `sjfolder/sj_folder/curobo.pdf` (62p 확장판 — 지금 안 읽음)
