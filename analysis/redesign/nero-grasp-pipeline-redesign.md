---
name: nero-grasp-pipeline-redesign
description: "NERO pick&place 파이프라인 \"갈아엎기\" 진단 + 우선순위 논문 리스트 (2026-09-03 세션)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6ba50bd9-f824-411e-816c-7dcb14f08a0e
  modified: 2026-09-07T09:20:07.505Z
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

### 로봇 제어 방식 — 이전(룰기반) vs 재설계 (2026-09-07 정리)

**이전**: Claude ↔ MCP(`mcp_robot_server`) → `planning_node.py` 룰기반.
`pick_object(grasp_dir='side', side_approach_deg=…)` 처럼 **Claude가 grasp 모드/각도를 골라서** 넘김
→ planning_node가 `infer_grasp`(VLM) → `grasp_kinematics.py` 하드코딩 쿼터니언 + approach 스윕
→ `resolve_grasp_quat()` pose 1개 → OMPL plan → 실행. **Claude가 inner loop 안**(CLAUDE.md 변환표
매 호출 적용), LLM 지연이 임계경로. /brutal이 뜯어내라고 한 것.

**재설계 (3층, Thor에서 구동)**:
1. **태스크 층 (Claude)** — "컵을 바구니에" → 태스크 그래프 `[컵 위치, 바구니 위치, pick(컵), place(바구니)]`.
   태스크당 2~3회, 모션당 아님. 실패 시에만 재계획.
2. **결정론적 BT skill 실행기** (LLM 왕복 없음):
   - `perceive`: RealSense → YOLO(bbox) → SAM(mask) → local-region point cloud
   - `grasp_gen`: PC → **Contact-GraspNet** → 6-DoF 후보 + 180° twin  ← `ContactGraspNetBackend`(2c-tool 완료)
   - `filter/rank`: 후보별 pick_ik reachability + collision 체크 → `graspness − w1·θ⁴ − w2·IK margin − w3·clearance`
     로 랭킹 (+ 선택적 VLM 시맨틱 re-rank). **argmax 금지**
   - `plan`: MoveIt2 STOMP → JointTrajectory + Cartesian 마지막 구간(`p−k·a`)
   - `execute`: `FollowJointTrajectory` → sim `arm_controller` / 실물 `pyAgxArm`(ros2_control `move_j`).
     그리퍼: sim `gripper_controller`→`/isaac_joint_command`→AG (물리 파지 OK 확인) / 실물 `move_gripper_m(w, force)` 폐루프
   - `verify`: 파지 성공? placement 검증
3. **VLM** — 씬당 1회 시맨틱 그라운딩, 캐시

**MCP 역할 축소**: `pick_object(target='cup')` 수준 (grasp_dir/side_approach_deg/angle 파라미터 삭제 — Phase 4).
"grasp 자세를 어떻게" 책임이 planning_node 밖 **별도 grasp 서비스**(SAM→CGN→필터→랭킹)로. planning_node는
"이 후보 실행"하는 얇은 노드 → Phase 5에서 BT 실행기가 더 대체.

**현재 상태**: pick_ik ✅ / CGN backend 프로토타입 ✅(미배선) / STOMP ⬜(아직 OMPL) / BT 실행기 ⬜(Phase 5) /
Cartesian = Pilz LIN config 있음. **오늘 로봇 돌리면 아직 이전 룰기반 경로.** 새 경로는 조각별로 만들어
Phase 4에서 일괄 통합·이전 코드 삭제 (사용자 지시).

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

## 컴퓨트 분담 — Thor = 두뇌, PC = 시뮬레이터 (2026-09-06)

### 하드웨어 제약
- **메인 PC**: RTX 3080Ti(12GB VRAM, Ampere) + **시스템 RAM 16GB (← 병목)**. Isaac Sim
  하나로 빠듯. x86. 사용자 연구실 책상에 있음.
- **Jetson Thor**: Blackwell GPU, arm64, **128GB 통합 메모리(CPU·GPU 공유 LPDDR5X)**.
  헤드리스 서버(선반, 로봇셀 연결). 고성능 추론기. 통합 메모리 = VRAM 한계 없음 +
  host↔device 복사 없음(zero-copy).

### 최적화 기준 (프로젝트 설계 원칙)
**"Thor 128GB 통합 메모리를 최대한 채워서 성능을 산다"가 컴퓨트 설계의 기준.**
PC의 16GB RAM에 뭘 올릴지 고민하지 말고, PC는 Isaac Sim 전용으로 비우고
학습모델·큰 데이터는 전부 Thor. 모델 크기 선택 시 "메모리에 맞나"가 아니라
"품질 최선"으로 고르고 Thor 메모리로 감당.

### 역할 분담 (명확히)

| 구분 | 어디 | 무엇 |
|---|---|---|
| **워크스테이션** | 당신 PC | 코드 편집(VS Code), git push, RViz/rqt/plotjuggler(Thor ROS 그래프 구독), `ssh thor`. **여기 앉아서 작업** |
| **시뮬레이터** | 당신 PC | **Isaac Sim만** (렌더·물리, PC여야만 하는 유일한 것) + Isaac Sim ROS2 브리지 |
| **컴퓨트 서버** | Thor (헤드리스) | 인지(YOLO·SAM2) · grasp 생성(CGN + checkpoint 2개 + learned evaluator) · VLM(Qwen3-VL-8B) · 시맨틱 맵(VoxelMap+CLIP) · **MoveIt2/pick_ik/STOMP** · (되면)cuRobo · fine-tune 학습 |
| **실행 대상** | PC의 Isaac Sim / 실물 팔 | Thor가 계산한 JointTrajectory 를 받아 실행만. 실물이면 PC 안 거침(Thor→관절 명령) |

**Thor는 sim·실물 공통 두뇌** → Sim2Real 통일 자연스러움 (JointTrajectory 는 하드웨어 무관).

### 개발 워크플로우 (Thor 앞에 앉을 일 없음)
```
PC에서 개발 → git push → Thor pull
Thor: 스택을 서비스로 상시 구동 (start_nero_isaac_all.sh 를 Thor용으로 분리)
PC: Isaac Sim + RViz 띄우고 관찰·조작 (grasp 마커 = Thor 계산 결과)
Thor 노드 이터레이션·GPU 확인 = PC 터미널에서 ssh (여전히 책상에서)
랩 네트워크 ROS2 DDS — 이미 있음 (Thor 가 지금도 YOLO/VLM 을 PC 에 서빙, Tailscale)
```

### 128GB가 푸는 것 (우선순위)
1. 모델을 "맞추려고"가 아니라 "품질"로 크게 — VLM 3B→**Qwen3-VL-8B/Thinking**
   (`jetson_thor_vlm_upgrade_task.txt` 근거 생김), SAM MobileSAM→**SAM2**,
   **CGN 체크포인트 2개 상주**(sigma_001+0025, 노이즈별 선택), CGN + **learned evaluator**(6DGN식) 동시
2. **전부 warm, lazy load 폐지** — 시작 시 전부 로드, 첫 호출 지연 제거
3. **멀티뷰 스캔 배치** — joint1/joint7 스윕 N프레임 → 융합 PC 하나 → SAM+CGN 통째 (occlusion 완화)
4. 시맨틱 맵(OK-Robot VoxelMap, voxel별 CLIP) 상주 — GB급, 즉시 쿼리
5. zero-copy — depth→PC→마스크→CGN 텐서를 통합 메모리 한 공간에
6. **fine-tune도 Thor** — 데이터 생성(Isaac Sim)은 PC, 학습은 Thor(큰 배치)

### 주의
- **nvmap 누수**(Tegra 통합 메모리, 프로세스 多 → CUDA 컨텍스트 파편화) → **프로세스 수
  최소화, 모델 in-process 로드, 메모리 모니터**
- Thor CPU 바운드 확인 (MoveIt2+브리지+인지 동시)
- 개발툴(RViz 등)은 PC에서 Thor ROS 가리키게

### 지금은
월요일 npz 테스트는 PC 그대로 (Isaac Sim 렌더가 PC, CGN도 PC 셋업됨).
**Thor 전 스택 이관 = Phase 4 인프라 결정.** 그것도 "Thor에 노드 배포"지 "Thor에서 작업" 아님.

### Thor 접근 시 수정 우선순위 체크리스트 (2026-09-07)
**원칙**: 큰 결정을 여는 순서. RealSense·실물 팔은 Thor 쪽(스파이크가 Thor D435i) →
depth 실측·재촬영은 Thor 앞에서만. **기존 코드 삭제/교체는 여기 없음 — Phase 4 (검증 후 일괄).**

**SSH 확정 (2026-09-07)**: `ssh thor` (= `bpdl@163.239.19.132`, 랩 LAN, ed25519 키
`~/.ssh/id_ed25519`, `~/.ssh/config` 등록). Thor sshd 활성화함. Tailscale 미설치(랩 LAN이면 불필요).
Thor: JetPack7 / L4T R38.4.0 / Ubuntu 24.04 / CUDA 13.0 / kernel 6.8.12-tegra / 14코어 /
**122GB RAM** / NVMe 937G(774 여유) / ROS **jazzy** (PC MoveIt config는 humble — Phase 4 주의) /
py3.12 / venv `~/phase1/pc_spike_venv`(torch 2.12+cu130) + `~/vllm-venv`.
**도는 서비스 없음** (VLM/YOLO/box 서버 다 내려가 있음, 재구성 필요).

**Tier 0 — ✅ 완료 (2026-09-07)**
- [x] Thor `~/pc_spike/` 19 npz + ply + seg + bbox.json → `~/grasp/pc_spike_thor/` (215MB rsync).
  키 = `depth_m`+`K`+`color` (segmap 없음). Thor docs → `~/grasp/thor_phase1_docs/`
- [x] 시스템 스냅샷 (위 SSH 확정 참조)
- [x] 서비스 확인 — 없음
- [x] **2c 해결** (위 Phase 2 표 2c-★): segonly→region 방식 오류였음. cup/bottle/clutter OK
- [ ] 실물 팔(pyAgxArm CAN)이 Thor에 물려있는지 — 미확인

**Tier 1 — depth 품질 실측 (재촬영 여부 판단)**
- [ ] `python3 tools/depth_noise.py --distances 0.5 0.65 0.8 --frames 30` (평면 정면)
  → 국소 노이즈 p50/p90 vs **5mm**(CGN contact 반경). `PHASE1_RESULT.md`에 기록.
  p90<5mm OK / >8mm 필터·평균 강화
- [ ] observation 자세 관측각 측정 (`tools/check_view_angle.py` — RANSAC 평면 normal ↔ 카메라 z)
  → **45~70° 하향인지** (1c 탈락 원인이 grazing 31~41°였음)
- [ ] grazing이면 `saved_poses.json` observation `joint7`을 45~60° 하향으로 재저장

**Tier 2 — 재촬영 (Tier 1 나쁘거나 cup/thin/clutter 필요)**
- [ ] `pc_spike_capture.py` 재촬영 — **`.npz`에 depth+K+segmap 저장** (CGN full-scene 입력).
  `PHASE1_VISION.md` 프로토콜: 관측각 45~70°, 거리 0.6~0.8m, box 단독 2~3 / cup 2 /
  bottle(불투명) 2 / 얇은 2 / 클러터 2 / 가장자리 2. 원통은 측면에서 더 내려보거나 2뷰
- [ ] `seg_bench.py --label` → `--models mobile_sam,fastsam,sam2.1_t --ply-out` 재실행
  **+ point prompt 비교** (클러터 bbox 안 다물체). mobile_sam vs fastsam 최종 결정
- [ ] `run_cgn.py <재촬영.npz> --local-regions --filter-grasps` → masked-only 대비 score/분포
  개선되는지 → **scene-context 가설 판정** (Tier 0 원본 npz로 먼저 해봤으면 크로스체크)
- [ ] D_thin = depth-grasp 경로 밖 확정, 별도 트랙 문서화

**Tier 3 — 그리퍼 스펙 확정 (팔이 Thor에 연결돼 있으면)**
- [ ] `get_gripper_teaching_pendant_param()` CAN 쿼리 → `max_range_config` **0.07 vs 0.10** 확정
  (지금 URDF 값 0.10 추정)
- [ ] `move_gripper_m(0.05, 1.0)` 테스트 (팔 정지, 그리퍼만) — 위치제어·force 피드백 동작
- [ ] `d`(fingertip→flange) 물리 측정 or 그리퍼 STL → URDF `0.1358` 검증/보정

**Tier 4 — Thor 배포 조사 (Phase 4 인프라용, 나중)**
- [ ] Thor JetPack용 PyTorch 휠 → `contact_graspnet_pytorch` 돌아가나 (`env.sh`에 torch
  2.12.0+cu130 흔적 있음 — 재확인)
- [ ] `ultralytics` SAM2/MobileSAM Thor arm64
- [ ] cuRobo / Isaac ROS cuMotion Blackwell+JetPack 지원 매트릭스 (§"cuRobo 재검토 조건")
- [ ] Thor↔PC ROS2 DDS 통신 확인 (이미 되고 있어야 함 — Tailscale)
- [ ] `start_nero_isaac_all.sh` → Thor용(인지·grasp·MoveIt2) / PC용(Isaac Sim·RViz) 분리 계획

**막힘 없음 (Thor 없이 지금)**: Isaac Sim 렌더 씬 CGN 테스트(2c-next), `cgn_prototype.py`
정제, Isaac Sim 그리퍼 물리(close+attach) 확인

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

### Phase 2 — Grasp 층 (프로토타입, 기존 코드 안 건드림) — 2026-09-04 착수
**box 기준으로 진행** (cup/thin/clutter는 재촬영 후). Thor 접근 불가 → 이 PC에서.

| # | 과제 | 상태 |
|---|---|---|
| 2a | ✅ **완료 (2026-09-04, 이 PC RTX 3080Ti)**. `elchun/contact_graspnet_pytorch` clone `~/grasp/contact_graspnet_pytorch` (pretrained `model.pt` 포함, **pointnet2 순수 pytorch = CUDA 컴파일 불필요**). venv `~/grasp/cgn_venv` (uv `--system-site-packages`, torch 2.5.1+cu121 재사용). deps: trimesh/pyquaternion/addict/configargparse/pyrender/opencv-headless. 러너 `tools/cgn_prototype/run_cgn.py` (commit `1813f1b`) — viz import 우회, `.ply/.npz` → `predict_scene_grasps` → `.npz`. 번들 test scene 326 grasp OK. **VRAM: 로드+추론 시 ~1~3GB, 12GB 중 → Isaac Sim 공존 가능** |
| 2b | **`tools/cgn_prototype/cgn_prototype.py` — commit `83c2b3a`.** `ContactGraspNetBackend` 스캐폴딩: 정규화(mean centering) → predict() → 공식 변환(Gram-Schmidt Eq6, `R_g=[b,a×b,a]`, `t_g=c+(w/2)b+d·a`) → w≤w_max/s≥thr 필터 → 180° twin → 그리퍼 와이어프레임 `.ply`/RViz MarkerArray. `_run_model()`만 2a 대기, 그 전엔 `--fallback`. HDD box .ply로 스모크테스트 통과 | ✅ 나 |
| 2c | **예비 결과 (2026-09-04)**: pretrained CGN(pytorch 포트) → NERO box masked .ply. **marginal**: max score ~0.19~0.22, 번들 test scene 은 0.29(포트 자체가 TF 원본보다 약함 — README "results may vary"). uniq 위치 69개(코너 클러스터는 top-score 만), opening 2.5~2.8cm(21cm box 에 과소=edge grasp), **approach 는 대략 top-down**(광학계 y+ mean 0.82), bottle/thin → 0개. **원인 미확정** — 후보 (a) masked object 만 입력(scene context 없음) (b) 도메인 갭. **시도한 것, 효과 없음**: gripper_width 0.08→0.10(0.19→0.22), 임계값 낮춤, forward-passes 6, 합성 테이블면 붙이기(F_edge 소폭↑ / A_box 0 grasp — 합성 평면이 너무 인공적, 결론 안 남). **진짜 scene-context 테스트 = Thor 원본 depth+K+segmap npz 필요 (지금 못 가져옴).** | 예비 완료 |
| 2c-★ | **해결 (2026-09-07, SSH로 Thor `~/pc_spike/` 19 npz 회수 → `~/grasp/pc_spike_thor/`).** npz 키 = `depth_m`+`K`+`color` (segmap 없음, bbox.json 라벨 있음). 새 툴 `tools/cgn_prototype/run_cgn_scene.py`: bbox→MobileSAM→segmap → `extract_point_clouds` → 3방식 비교. **판정: 2c 예비의 "marginal"은 도메인 갭이 아니라 입력 방식 오류.** `run_cgn.py`가 masked `.ply`(=`segonly`)를 먹였는데, 이건 CGN이 필요로 하는 국소 씬 컨텍스트(물체 주변 테이블면)를 다 버림. **올바른 방식 = `local_regions=True`** (NVIDIA README `--local_regions --filter_grasps`): segment 주변 큐브를 크롭해 넣음. 결과: cup `segonly` 0.18~0.22 → `region` **0.25~0.29**, bottle/thin `segonly` **0 grasp** → `region` **0.29** (opening 4.9cm 정상), clutter 물체들 `region` 0.29 (med 0.28). 전부 포트 자체 번들 test scene(0.29) **동급**. → **pretrained CGN(약한 pytorch 포트조차) cup/bottle/clutter 는 파인튜닝 없이 충분. 진행.** | ✅ 해결 |
| 2c-box | **box(A_box)만 여전히 안 됨** — `region` 에서도 max ~0.17~0.19. 원인: 21×12cm 풋프린트 vs 그리퍼 8cm(포트가 그리퍼폭 config 무시, Franka 가중치에 박힘. `DATA.gripper_width` 0.10/0.12 → 변화 없음 확인). + grazing 각도라 윗면만 보임 → 12/21cm 스팬만. **해결책: 45~70° 재촬영으로 측면(5~7cm 높이) 노출** + (Phase 4) 필요시 AGX 폭으로 파인튜닝. box 는 프로덕션 타겟이지만 CGN 입장에선 최악 케이스. | box 재촬영 대기 |
| 2c-viz | `run_cgn_scene.py --ply-out` → 씬 점(회색)+grasp 와이어(초록, 밝기=score). `~/grasp/cgn_scene_viz/*.ply` + HDD `phase1_thor_run/`. MeshLab 육안 확인용 | 나 |
| 2c-tool | ✅ **완료 (2026-09-07, 서브에이전트)**. `cgn_prototype.py` `segonly`→`local_regions` 리팩터. **인터페이스: `predict(pc_full, pc_segment)`** — 두 `(N,3)` 배열(camera frame), SAM·역투영은 드라이버. `_run_model()` = `GraspEstimator` lazy-load 1회 → `predict_scene_grasps(local_regions=True, filter_grasps=True)`. **TCP 재계산 검증**: CGN Franka 깊이 버리고 `build_grasp(c,b,a,w,s,d=0.1358)` 재구성 → approach 축 성분 정확히 +13.6cm=d. `S_THRESHOLD` 0.30→0.15. 테스트: `pc_spike6/A_box_001` 56 grasp max 0.293 opening 5.7cm ✓. `run_cgn.py` deprecation 배너. `sj_pickplace/` 안 건드림, 미커밋. Phase-4 `LearnedGraspBackend.predict()` 매핑은 docstring/README에. | ✅ |
| 2d | grasp → base_link TF (`_cam_to_base`, canonical observation 자세에서만) | 2c 후 |

### Thor CGN 이관 (Phase A) — ✅ 완료 (2026-09-07)
- `~/grasp/contact_graspnet_pytorch` (rsync, .git 제외), `~/grasp/cgn_prototype/`, SAM 모델, pc_spike6 → Thor
- **fresh venv `~/grasp/cgn_venv`**: `python3 -m venv` (system-site-packages 안 씀 — pc_spike_venv는
  venv-numpy2 vs system-scipy1 ABI 충돌로 폐기). torch **2.12.0+cu130 aarch64**
  (`download.pytorch.org/whl/cu130`), numpy 2.5.3, ultralytics/trimesh/pyrender/tqdm/scipy/sklearn/cv2.
- **CGN 레포 패치 2곳** (torch 2.12 / numpy 2.x): `checkpoints.py:65` `torch.load(..., weights_only=False)`,
  `contact_grasp_estimator.py:341` `np.in1d`→`np.isin`
- 스모크: `A_box_001` region+filter → max **0.294** opening **5.7cm** 100% on-target = PC와 동일.
  **device cuda:0 (Thor GPU), ~1.2s/씬.**

### Isaac Sim → CGN 6-DoF 검사 (Phase B) — ✅ 검증됨 (2026-09-07)
**결론: 렌더 → point cloud → CGN → 6-DoF grasp pose 파이프라인 작동.**
- Isaac Sim: 팔 관측자세(`saved_poses.json`), 5cm TestBox를 카메라 시야(-0.5,0,0.03)에 배치.
  flange 카메라(`/World/agx_arm/link7/gripper_flange/Camera`)에서 replicator 애너테이터로
  RGB+depth+**GT semantic seg**(id 3='box', 2='floor')+K 렌더 → npz. **depth 100% valid (sim은 깨끗).**
  ⚠️ `rep.orchestrator.step()`은 재생 중 데드락 — `omni.kit.app.get_app().update()` 루프로 렌더.
  ⚠️ USD 카메라 vert aperture가 4:3용 → `fx=fy=f/ha·W` (square px) 로 K 보정 필요.
- Thor CGN (GT segmap): **190 grasp, score max 0.283, opening 5.7cm** (5cm 큐브 몸통 감쌈).
  world 프레임 변환(`world_T_optical = camT.T @ diag(1,-1,-1,1)`): 상위 grasp 위치가
  **박스 위 x,y ±1cm, approach ≈ (0,0,-1) 클린 top-down.** z는 flange 기준 약간 낮음(Phase C에서 조정).
- sim score(0.25~0.28) < 실물 재촬영(0.29): 작은 큐브·완벽 평면(ACRONYM 메시엔 디테일 있음) 추정.
  파이프라인 검증엔 무관.

**⚠️ SAM 마스크 bg_leak 확인됨**: `run_cgn_scene` seg pc bbox 가 cup 에서 z-extent 0.20m, bottle 0.41m
(grazing 각도 → 마스크가 테이블면 슬라이스 포함). `region` 모드는 큐브 크롭이라 영향 적지만
`filter_grasps`/on-target 지표는 부정확. 재촬영 시 45~70° + point prompt 로 개선 (Tier 2).

**CONFIG 확정 (URDF/코드)**: `d = 0.1358 m` (`planning_node.TOP_TCP_OFFSET`, flange→fingertip
2026-07 실측), `w_max = 0.10 m` (URDF `gripper_joint1/2` prismatic 0.05×2).
**AGX 그리퍼 = 위치제어** `move_gripper_m(w_m, force_N)` → 예측 w 직접 명령 가능.

### Isaac Sim 그리퍼 물리 — ✅ 작동 확인 (2026-09-07, MCP execute_script)
**결론: 순수 마찰로 파지 성립. fixed-joint attach 불필요** (5cm/0.2kg 박스 기준).
- 씬: `/World/agx_arm` articulation (9 DOF: joint1~7 + gripper_joint1/2 prismatic), `/World/TestBox*`
  5cm 큐브 0.2kg 마찰 **1.5**(`BoxHighFriction`, 이미 튜닝됨). 손끝 개폐축 = **world Y**, 최대 gap **10cm**.
- **막고 있던 것**: ActionGraph `/World/ActionGraph/articulation_controller`(`IsaacArticulationController`)
  가 매 틱 관절을 `/isaac_joint_command`(ROS) 값으로 강제 → ROS 스택 없으면 0으로 홀드.
  MCP `apply_action`이 안 먹던 이유. **테스트하려면 이 노드 `set_disabled(True)` 후 복원.**
- **physx solver iteration 1→16** (`art.set_solver_position_iteration_count(16)`, 파지 안정성).
  ⚠️ 런타임만 — USD `physicsScene`/articulation prim 에 영구 반영 필요.
- 테스트: 5cm 박스 손끝 사이 배치 → close → 중력 하 유지(3mm 슬립) → 팔 arc 이동
  (joint1→0.5, joint2→-0.6, joint4→0.8) → **박스 계속 유지**, 손끝중점-박스 거리 5.9mm.
- **남은 것**: solver_iter USD 영구화 / ROS 경로 검증(`gripper_controller`→`/isaac_joint_command`→AG)
  / 큰·무거운 물체 한계 / 실제 approach→close 시퀀스 (지금은 텔레포트로 시작).

### Phase 3 — Grasp→모션 통합 (프로토타입 브랜치/플래그)
| # | 과제 |
|---|---|
| 3a | pick_ik 루프 reachability 필터 (후보 + 180° twin 둘 다) |
| 3b | near-miss refinement (6DGN) — pick_ik 실패 후보를 ≤1cm SE(3) 볼에서 perturb 재확인 |
| 3c | 랭킹 `graspness − w1·θ⁴(top-down) − w2·IK margin − w3·collision clearance` + VLM 시맨틱 re-rank. **argmax(s) 금지** |
| 3d | STOMP 경로 + Cartesian 마지막 구간(`p−k·a`) + closed-loop 그리퍼. 플래그 뒤, 기존 경로 폴백 |

### Phase C — Thor MoveIt/pick_ik/STOMP + 실행 검증 (2026-09-07)

**MoveIt 스택은 PC(Humble) 아니라 Thor(Jazzy)에.** STOMP MoveIt 플러그인이 Jazzy부터 정식
(`ros-jazzy-moveit-planners-stomp`), Humble엔 apt 패키지 없음. PC↔Thor **ROS2 DDS 안 됨**
(demo talker 안 보임 — Humble/Jazzy + 캠퍼스 스위치). 실행 브리지는 Phase 4.

**C1 ✅ Thor Jazzy 빌드**: `agx_arm_sim` + `nero_sj_pickplace` → `~/ros2_ws/src/` rsync.
`sudo apt install ros-jazzy-pick-ik`. `colcon build` 클린 (포팅 이슈 없음).
config 3개 Jazzy 포맷으로 교체 (Humble→Jazzy): `stomp_planning.yaml`·`pilz_..._planning.yaml`
= `planning_plugin`(str) → `planning_plugins`(list), `request_adapters` folded-str → list.
jazzy 기본값(`/opt/ros/jazzy/share/moveit_configs_utils/default_configs/`) 복사가 정답.
`kinematics.yaml`은 이미 pick_ik (Phase 0). `move_group.launch.py`에 stomp 파이프라인 추가,
default=stomp. 헤드리스 런치 `headless_plan_test.launch.py` (rsp+jsp+static tf+move_group,
ros2_control 없이). → **OMPL/STOMP/Pilz 3개 다 로드, "You can start planning now!"**

**C2 ✅ pick_ik reachability**: Isaac Sim 렌더 box CGN grasp 5개 → `/compute_ik` (group=arm,
seed=관측자세). **1/5 reachable** (4개 NO_IK_SOLUTION — 5cm 박스가 베이스에 가깝고 grasp이
거의 바닥). 필터가 목적대로 동작.

**C3 ✅ STOMP plan**: reachable grasp → `/move_action` (plan_only, pipeline_id=stomp,
joint goal). **error_code=1, 25 waypoints, 2.30s 궤적** → `c3_trajectory.json` 덤프.

**C4 ⚠️ Isaac Sim 실행 (프레임 수정 후 재검증, 2026-09-07)**:
- **프레임 변환 확정·검증**: `world_T_optical = W_T_cam @ diag(1,-1,-1,1)`,
  `W_T_cam = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(cam_prim)).T`
  (USD Gf.Matrix4d = row-vector → `.T` = col-vector world_T_cam. USD 카메라 -Z fwd/+Y up →
  optical +Z fwd/+Y down 는 `diag(1,-1,-1)`). **검증**: 박스 world → 픽셀 투영이 GT seg 중심과
  1px 이내, depth 0.775m 일치. **손가락-박스 XY 오차 34mm → 4.6mm.**
- **CGN grasp 행렬 z-col = approach = 물체를 향함(닫는 방향).** flange `t_g = c − d·z_col`,
  fingertip = `flange + d·approach`. (내 초기 fingertip 계산 부호 반대여서 헷갈렸음 — pose 자체는 정상)
- **재실행**: 궤적 → Isaac Sim apply_action. 팔이 grasp 자세 도달, **손끝중점-박스 XY 4.6mm**.
  BUT close 시 **비대칭**(한 손가락 0.025, 다른 손가락 -0.047=거의 열림) → 박스 안 잡힘.
  원인: **grasp이 너무 낮음** (fingertip z≈0.014, 바닥 근처) → 한 손가락이 바닥/박스에 끼임.
- **reachability 1/8** (pick_ik attempts 25, timeout 0.5 로 올려도). tilted approach(수직서 15°) +
  기구학 경계. standoff 2cm 주면 8/8 전부 IK 실패. → **Phase 3b(near-miss refine)·3c(θ⁴ top-down
  랭킹)·180° twin 필요.** 5cm 큐브 top-down 뷰 = 어려운 케이스 (Phase 2c 실물 7cm box는 깨끗).

**결론: 루프 전 구간 동작 (CGN→프레임변환[✅검증]→pick_ik→STOMP→실행→그리퍼). 남은 격차:
grasp 품질/reachability = Phase 3 (twin·refine·top-down 랭킹·standoff). 프레임은 해결.**
Thor move_group는 tmux `planmg` 세션 구동 중. kinematics.yaml attempts 4→25, timeout 0.2→0.5.

### Phase 3a+3c 검증 — twin + θ⁴ 랭킹 (2026-09-07)
**IK 진단**: pick_ik는 박스 위치(-0.5,0,0.2)에서 **top-down 자세는 넓게 도달** (yaw 0/45/135,
z 0.10~0.30, x -0.4~-0.55 전부 IK=1). CGN grasp이 IK 실패한 건 **CGN 자세가 수직서 15~40°
기울어서** — 도달 자세 밖. → θ⁴ 페널티가 정확히 이걸 해결.

**`rank_grasps.py`** (Thor `~/grasp/`): CGN grasp 전체(top-N 아님) + **180° twin**
(`R @ diag(-1,-1,1)`) → 각각 `/compute_ik` → 통과분을 `cost = score − w·θ⁴`
(θ=approach의 수직 편차, w=0.6) 로 랭킹. Isaac Sim 7×7×14cm 박스:
- **118 grasp × 2 = 236 후보 → 132 reachable** (이전 0~1/8 에서 대폭). θ⁴ 랭킹 1위 =
  θ=14°, 손끝 박스 몸통 중상단(z=0.093, 바닥 아님), d_box_xy 6mm, STOMP code=1.

**C4 재실행 (랭킹된 grasp)**: 팔 자세 정확히 도달 (arm_now≈target). **그러나 그리퍼 close가
여전히 비대칭** (한 손가락만 ~1cm 닫히고 stall, 박스 안 잡힘). 박스를 손끝 중점에 스냅해도,
4cm로 줄여도 동일. **원인 = 프레임/센터링 아님** (스냅해도 실패). 추정:
- CGN grasp 자세가 contorted arm config (`joint7=1.54` 등) → 힘 전달 나쁨 / 특이점 근처
- θ=14° 기울기 → 중력 shear 성분을 마찰이 못 버팀 (task2의 home 자세 수직 grasp은 잘 됐음)
- endpoint jump 실행 (전체 궤적 재생 아님) — 접근 운동학 무시

### Step 4 검증 — Cartesian 마지막 접근 + 랭킹 v2 (2026-09-07)

**closing-axis 진단**: rank v1 최고 grasp의 `b`(closing axis) = X-Y 대각선 → 7×7 박스를
대각선(9.9cm)으로 물려 함 → 손가락이 모서리에 걸림. rank v1 grasp config도 contorted
(`joint7=1.54, joint3=1.27`).

**`rank2.py`**: v1 + **(a) pre-grasp(−5cm·a)도 reachable 체크 (b) box-axis 정렬 항**
`cost = score − 0.5·θ⁴ − 0.15·(1−align)` (align = closing axis의 수평면 주축 정렬,
1.0=면정렬 / 0.707=45° 대각). → **236 후보 중 118개가 grasp+pre-grasp 둘 다 reachable.**
최고: θ=12°, **align=1.00** (면 정렬), **non-contorted** (`joint7=1.0, joint3=−0.06`).

**`step4b.py`**: STOMP(seed→pre-grasp joint goal) `code=1, 25wp` + `/compute_cartesian_path`
(pre→grasp 직선, `avoid_collisions=False`) **fraction=1.00, 5wp** → 30-pt 궤적 `c4_trajectory.json`.
grasp config 깨끗, fingertip (−0.499, 0.013, 0.102) box (−0.5,0,0.07) — 몸통.

**C4 재실행 (Cartesian + smooth 60-pt 리샘플 재생)**: 여전히 **파지 실패**. 접근 중 열린
그리퍼(10cm)가 7cm 박스를 툭 침 → 박스가 joint2 손가락 쪽으로 밀림 → close 시 joint2
손가락이 즉시 stall(−0.05 유지), joint1만 조금 → 박스 이탈. lift → DROPPED.
씬 상태도 여러 조작으로 drift(박스 스케일/위치).

**근본 원인 = 파이프라인 로직 아님.** sim 실행 충실도: (1) MCP apply_action 재생이
`FollowJointTrajectory` 컨트롤러의 매끄러운 보간이 아님 (2) 7cm box / 10cm 그리퍼 tight
클리어런스 (3) 센터링 ~1cm 오차 + fingertip 접촉면 모델 부정확. **task2(중앙정렬 박스,
home 자세, 팔 이동 중 유지)에서 물리는 이미 검증됨** — 실행이 깨끗하면 됨.

**→ 제대로 된 C4 = Phase 4의 실제 ROS 실행 경로** (`FollowJointTrajectory` → 컨트롤러 →
보간 실행), MCP jump-replay 아님. Thor↔PC 브리지(Zenoh) 필요.

**Step 1·2·4 성과**: reachability 병목 해결 (0/8 → 118/236, twin + θ⁴ + align + pre-grasp),
CGN→프레임→twin/rank→pick_ik→STOMP→Cartesian 전 구간 계획 검증. 실행 충실도만 Phase 4.
스크립트: `~/grasp/{rank_grasps,rank2,ik_probe,step4b}.py` (+ HDD `tools/phase_c/`).

### 속도 최적화 + CoM 랭킹 (3c 확장) — 2026-09-07

**bottle pick 전체 파이프라인 실행 → sim 파지 DROPPED** (box와 동일 caging 실패 —
손끝 10cm/물체 5cm, 1cm 센터링 오차로 한 손가락 먼저 접촉→물체 밀어냄. **해법 = fixed-joint
attach, Phase 4**). 파이프라인 로직은 전 구간 OK.

**병목**: rank2가 400 후보 × 2 IK = 800 `/compute_ik` 호출 (각 timeout 0.5s) = **~10분**.

**`rank3.py`** (새 표준, rank_grasps/rank2 대체):
1. **기하 prefilter** (IK 0): θ≤35° from vertical, opening≤0.10, contact가 coarse workspace
   box 안. 200×2 → ~100개, **5ms**.
2. **CoM 항 추가** (사용자 요청): `cost = score − 0.5·θ⁴ − 1.2·d_com`
   (d_com = contact ↔ object centroid 거리). bottle에서 grasp을 꼭대기→중간(d_com 4.4cm)으로 당김.
3. **fast IK**: 상위 K=24만, `req.ik_request.timeout=120ms`, twin은 원본 실패 시만.
   `kinematics.yaml` **timeout 0.5→0.08, attempts 25→4** (필터용).
→ 12~23 IK 호출, **~2s**. STOMP+Cartesian **1.2s**.

**전체 (warm)**: SAM 65ms + CGN 1.2s + rank3 ~2s + STOMP+Cart 1.2s ≈ **~5s** (이전 ~10분, **100배**).
CGN 모델 로드 10s는 1회 — Phase 4에서 CGN 상주 노드로.
스크립트 `~/grasp/rank3.py` (+ HDD `tools/phase_c/`).

### 파이프라인 시각화 아티팩트 (2026-09-07)
`https://claude.ai/code/artifact/2d19b714-4bb2-4994-b5c2-b74cd2de122e` — RGB→SAM→PC→CGN 후보→
선택 grasp 4패널, **2물체 비교**. `tools/pipeline_figure/make_panels.py` (`<npz> <bbox.json> <label>`,
npz에 GT seg 있으면 사용). 그리퍼 마커 = Π자 (palm 위, 손가락 아래, approach 화살표).
- **box** (실물 `pc_spike6/A_box_001`): ~50 grasp, max 0.294, 선택 0.29 / 5.7cm / θ17°
- **bottle** (Isaac Sim, Ø5cm×20cm 원통 `/World/WaterBottle`, `bottle` semantic): **200 grasp
  전부 0.294 saturated** — CGN(ACRONYM=병·머그 다수)이 깨끗한 원형 단면에 포화. Phase 2c 실물 cup(~0.29)과 일치.
  선택 grasp = 병 **윗부분** top-down 핀치 (θ19°, opening 4.9cm). CGN은 원통을 아무 높이서나 잡음 —
  무게중심 쪽 선호 항(3c 확장) 여지.

### Phase 4 — planning_node.py / grasp_kinematics.py 에서 사라지는 것 (2026-09-06 조사)

**`grasp_kinematics.py` (633줄) — 거의 통삭제.**
- 상수: `SIDE_TCP_OFFSET`, `SIDE_MIN_DIST`, `SIDE_PITCH_DEG`, `TOP_ANGLE_PITCH_DEG`,
  하드코딩 쿼터니언 `QUAT_TOP_DOWN(_R/_L/_RR)`, `QUAT_SIDE_FRONT`
- 테이블: `GRASP_DIR_MAP`, `LABEL_GRASP_HINT`, `SIDE_TAG`, `PINCH_TAG`
- 함수: `top_down_angle_quat`, `sim_top_down_angle_quat`, `sim_box_aligned_quat`,
  `side_quat_for`(+sim/pinch 변형 4개), `side_reachability_check`, `SIDE_BLIND_SWEEP_DEG`,
  `side_face_candidates_deg`(+from_normal), `auto_grasp_quat`, **`resolve_grasp_quat`**,
  `candidate_quat_for`(approach→quat, R_g=[b,a×b,a]로 대체), `YawCandidateSelector`
- 생존: `SequenceRejected`(이동), `quat_angle_diff`(유틸)
- ⚠️ `.claude/skills/grasp-kinematics-design/` 스킬이 이 파일 수정 시 필독 — **이번이 그 "의도된 통째 교체"**

**`planning_node.py` (2945줄) 에서 삭제/대체:**
- grasp 자세 결정 전체: `resolve_grasp_quat()` 호출(L465), `_top_down_quat_for`,
  `_box_aligned_quat_for`, `_abs_angle_to_quat`, `_pick_best_yaw_candidate`
- `TOP_TCP_OFFSET` 상수는 `d=0.1358`로 살지만 **`approach_z = pz + APPROACH_Z + TOP_TCP_OFFSET`
  식의 top/side 분기 계산이 사라짐** → `t_g = c + (w/2)b + d·a` 하나
- `is_side` 분기 전반 (approach_z/descend_z/lift_z를 top이냐 side냐로), `SIDE_APPROACH_STANDOFF_M`,
  `SIDE_LIFT_APPROACH_Z`, `SIDE_MIN_DIST`
- 접근각 후보 스윕: `_face_offset_candidates`, `_side_offset_candidates`, `blind_fallback`/
  `face_aligned` 분기 → near-miss refinement(≤1cm)로
- `_box_angle_deg` / `angle_base_deg` (Hough 2D 각도) 수신·사용 → CGN이 point cloud 기하에서
- `BEARING_OFFSET_DEG=188.0` + pre-rotate bearing 계산 (일부; joint1 사전회전은 pick_ik+STOMP면 필요성↓)
- `grasp_dir` 파라미터 (top/side/pinch), CLAUDE.md 변환표
- `geometry_3d.py` (RANSAC+PCA, 이미 "신뢰 불가")

**유지 (뼈대):**
- `_do_pick`/`_pick_sequence` 실행 시퀀스(approach→descend→close→lift) — 단 quat/approach를 CGN 후보에서
- `_scan_box_sequence`(joint 스윕) — 오히려 멀티뷰 배치로 강화
- `_home_sequence`, observation 자세, safety 전체
- `_sync_scene_collision_objects`(collision world — STOMP/pick_ik가 씀)
- `_compute_ik_joints` — pick_ik 경유, **필터의 핵심으로 승격**
- `_try_place_at` + placement 폐루프 검증, `_move_try_pilz`(Pilz LIN — Cartesian 접근에 씀), `_gripper`

**축소:** `_move_try_ompl_safe`→STOMP · `DESCEND_TOL_ORI_FALLBACK` 3단계(Cartesian 직선이면 덜 필요) ·
`on_command` pick 분기 대폭 슬림(grasp_dir/side_approach_deg/angle 처리 다 빠짐)

**이미 있는 seam:** `cmd.get('grasp_candidate')` (planning_node L453) — CGN 후보를 받는 자리가
**이미 있음**. redesign = 이 경로를 기본으로, else(`resolve_grasp_quat`) 삭제. `grasp_pose_generator.py`가
그 후보를 채우는 프로토타입 → `learned_grasp_backend.ContactGraspNetBackend` 연결.

**아키텍처 레벨:** planning_node가 "grasp 자세를 어떻게" 정하던 책임이 **별도 grasp 서비스**
(SAM→CGN→필터→랭킹)로 나감. planning_node는 "이 후보를 실행"하는 얇은 노드로 →
Phase 5에서 결정론적 BT skill 실행기로 더 대체 가능.

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
