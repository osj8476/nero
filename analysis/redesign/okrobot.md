---
name: okrobot
description: "OK-Robot 논문(2024) 파트별 분석 — NERO pick&place 갈아엎기 참고. Part A 완료, B/C 대기 중"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6ba50bd9-f824-411e-816c-7dcb14f08a0e
  modified: 2026-09-03T06:35:48.177Z
---

# OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics (Liu et al., 2024)

NERO pick&place 파이프라인 재설계("갈아엎기") 참고용으로 사용자가 파트별로 읽으며 분석 중.
사용자가 파트별로 읽으며 분석 중. Part A/B/C/D 분석 완료 → 사용자가 나중에 한 번에 정리 요청 예정.
관련: [[nero-grasp-pipeline-redesign]]

## NERO 현재 상태 (비교 기준)
- 고정 팔(Piper), Isaac Sim + ROS2, Jetson Thor에서 YOLO/VLM 서빙
- 인지: `vlm_boxyolo.py`(YOLO 듀얼: 커스텀 box + COCO 25종), `vlm_grasp_server.py`(Qwen2.5-VL-3B)
- 파이프라인: mcp서버→Claude판단→tool→VLM판단→tool→IK검증, 매번 라이브
- 아픈 곳: (1) "reach 불가" 오류 폭증 — grasp를 하나로 확정 후 IK 검증, fallback 후보 0개;
  orientation 하드코딩(특히 side) (2) collision/IK/OMPL — 하드코딩 orientation+단순 rulebase
  → IK오류+가동범위감소+OMPL 성공해도 그립정확도 감소 (3) 파이프라인 딜레이 심함
- wiki 문서 전부 "미검증/신뢰불가/확정값 아님"으로 끝남 (grasp_geometry_pipeline "신뢰 불가",
  geometry_3d.py PCA major_axis_yaw 편향, WAIST_Z 경험적 범위, SIDE_TCP_OFFSET 미검증)

## Part A (Open-vocab object navigation) — 분석 완료

네비게이션 수식(A*, 로봇반경 팽창, s1/s2/s3, Voronoi)은 고정 팔이라 폐기. **구조가 핵심.**

### 가져올 것
1. **객체 중심 시맨틱 메모리 (VoxelMap)** — 매 프레임 재탐지 제거. OK-Robot: 1회 스캔 →
   5cm voxel마다 CLIP 임베딩(detector-conf 가중 평균) → 쿼리 = dot product, 모델 호출 없음.
   NERO 매핑: observation 자세에서 joint1/joint7 스윕으로 작업공간 멀티뷰 1회 스캔 →
   각 뷰 YOLO-World(넓은 vocab) + SAM 마스크 → depth back-project → voxel CLIP 임베딩.
   이후 "컵/바구니 어디" = 벡터 조회. `ground_object` 컨테이너 프레이밍 sweep(첫 호출 1분+),
   `analyze_scene`(~13초) 라이브 왕복이 사라짐 → 문제 3의 핵심 원인 제거.

2. **"대략 위치(접근용)" vs "정밀 pose(조작용)" 분리** — OK-Robot: voxel 쿼리는 팔 닿는
   거리까지만, 조작은 근접 재관찰로. 좌표 재사용 안 함. NERO는 VLM bbox(자체 문서상 수 cm
   오차)를 물체찾기+grasp 입력 양쪽에 씀 → 분리: 맵/YOLO/VLM = 대충 어디, grasp 지점
   point cloud + segmentation = 실제 pose.

3. **목적함수 최소화 패턴을 자세 선택에 적용** — OK-Robot은 바닥 점마다 s(x) 평가 → 최소점
   base 배치. NERO는 base 없지만 "후보마다 스칼라 필드 평가 → 최소 선택" 패턴 자체를 차용:
   - 관찰 자세 선택: `center_view=True` blind joint7 sweep(스텝당 13초) 대신 후보 카메라
     자세마다 score=(중심정렬)+(bbox 안잘림, bbox_truncated 문제)+(occlusion)+(reachable)
     → next-best-view 최적화
   - 그립 후보 선택: score = 시맨틱매치 + reachability + collision clearance + 접근가능성
     (reachability 항은 cuRobo가 배치로 채움)
   교훈: 자세 하드코딩 금지, cost field 정의해서 최적화.

4. **스캔 단계에 넓은 고정 vocabulary 공급** — OK-Robot: ScanNet200 라벨 ~200개를 OWL-ViT에
   미리 줌, LLM이 라벨 추측 안 함. NERO `vlm_boxyolo.py`는 ALLOWED_COCO 25개 하드코딩 →
   스캔 때 YOLO-World 넓게 1회, 쿼리 시 필터. `vlm_capability_tiers.md`가 라이브
   `/detect_open_vocab`로 풀려는 "미등록 라벨" 문제가 구조적으로 사라짐.

5. **static 맵 + live 인지 분리** — OK-Robot 한계: 맵 운영 중 갱신 불가. NERO는 pick&place가
   씬 재배치하니 완전 static은 더 나쁨 → 분리: 안 움직이는 것(테이블/바구니/선반/벽 =
   collision + placement target) 1회 스캔해서 static 맵, 움직이는 물체만 live. CLAUDE.md
   "놓을 곳은 pick 전에 좌표 확보" 규칙이 코드가 됨 — 바구니 좌표가 static 맵에 있으니
   태스크 중간 재탐색 자체를 안 함.

6. (부수) detector를 논문 주장 아닌 preliminary query 실측으로 선택(OWL-ViT vs Detic),
   bbox는 SAM 마스크화. NERO segmentation_backend.py detector→SAM 파이프라인이 표준 확인.
   Qwen3-VL 벤치마크도 이렇게 실측 결정.

### 안 가져올 것
- A* 2D grid, 로봇반경 팽창, s1/s2/s3 가중치(8,8), Voronoi — 고정 팔
- Record3D iPhone 스캔 — 팔에 달린 RealSense로 대체

### Part A 한 줄
배울 건 네비게이션이 아니라 파이프라인 **순서**: 스캔 1회 → 지속되는 시맨틱+기하 맵 →
값싼 쿼리 → 대략 접근 → 근접 재인지 → 조작. NERO엔 voxel 맵이라는 층이 통째로 비어 있음.

## Part B (Open-vocab grasping) — 분석 완료

핵심: **쿼터니언을 계산하지 마라. 쿼터니언 집합(grasp pose 후보)을 생성하고 고른다.**
side 쿼터니언 하드코딩 문제의 직접 청사진.

### OK-Robot Part B 파이프라인
1. semantic memory가 준 물체 3D 위치로 head 카메라 조준 → RGB-D 캡처 → pointcloud
2. **AnyGrasp**: RGB + pointcloud → parallel-jaw용 collision-free 6-DoF grasp 전부 생성.
   출력: grasp point, width/height/depth, graspness score(uncalibrated conf)
3. **LangSam**: 언어 쿼리로 대상 물체 mask
4. 모든 grasp point를 이미지에 투영 → mask 안에 드는 것만 남김
5. 휴리스틱 선택: `score = S − (θ⁴/10)`, θ = grasp normal과 floor normal 각도.
   θ작을수록(top-down) 우대 — top-down이 hand-eye calibration 오차에 강함
   (**"horizontal grasp" = top-down = 그리퍼 몸체가 눕는 것; "vertical" = side.**
   수식이 anchor: θ⁴ 페널티 → θ→0 → grasp normal 수직 → 위에서 하강 = top-down)
6. 실행: pre-grasp 직선 접근 `⟨p−0.2a, p−0.08a, p−0.04a, p⟩` (a = approach vector),
   물체 가까이 스텝 작게 → 가벼운 물체 안 쓰러뜨림. 도착 후 그리퍼 closed-loop 닫기
7. lift → retract → wrist tuck (네비게이션 중 물체 유지용)

### NERO 현재 ↔ Part B 대응
| NERO 현재 | Part B |
|---|---|
| `/infer_grasp` VLM이 TOP/SIDE/PINCH 판정 | 없음 — grasp 종류를 VLM이 안 정함 |
| `resolve_grasp_dir` + `grasp_kinematics.py` atan2 → 쿼터니언 계산 | AnyGrasp가 6-DoF pose 직접 생성 |
| `_APPROACH_DIRECTION_DEG` 4방향 이산 + approach 스윕 ±15~90° | AnyGrasp approach vector `a` 직접 제공 |
| `ground_object` bbox → crop → crop에 infer_grasp | LangSam mask → grasp point 투영 필터 (bbox보다 정밀, VLM 왕복 없음) |
| planning_node blind/Hough 후보 탐색 | AnyGrasp가 collision-free 후보 전부 생성 |
| `TOP_TCP_OFFSET` vs `SIDE_TCP_OFFSET` ("절대 합치지 마라") | `p − k·a` 하나 — grasp 종류 무관 |
| `SIDE_TCP_OFFSET` 미검증, top/side `is_side` 분기 | grasp normal–floor 각도 θ 하나로 연속 처리, 분기 없음 |

### side 쿼터니언 문제가 왜 사라지나
1. AnyGrasp가 모든 grasp의 orientation을 줌 — "side 모드" 상수 없음. reachable + 시맨틱
   매치되는 후보 선택. `resolve_grasp_dir` 변환표/`WAIST_Z` 강등/approach 스윕 은퇴.
2. pre-grasp `p − k·a` 는 실제 접근축 `-a` 방향 — grasp 종류별 고정 오프셋 아님.
   `TOP_TCP_OFFSET`/`SIDE_TCP_OFFSET` 합치면 깨지는 이유 = 하나는 world Z, 하나는
   gripper frame 기준으로 오프셋을 재서 우연히 값이 같았던 것. `p − k·a`는 공통 공식 하나.

### θ 휴리스틱 — NERO에 제일 중요
NERO도 calibration 안 좋음(`perception_calibration.md`, `_cam_to_base` y 0.33→0.53
joint1 회전 시). → 여러 reachable grasp 중 top-down 가까운 걸 골라라. 계산 신뢰 못 하는
side를 억지로 하지 마라.
NERO 휴리스틱: `score = graspness − w1·θ⁴ − w2·(IK margin 부족) − w3·(collision clearance 부족)`.
top vs side를 하드 분기 말고 reachability를 점수에 넣기 (Part A #3 cost field). reachability
항은 cuRobo가 배치로.
⚠️ `WAIST_Z`와 부분 충돌: 허리 높이(z 0.3~0.5m)는 top-down IK 실패로 side 강등 + calibration
최악. AnyGrasp+reachability가 나쁜 calibration의 side를 마법으로 못 고침. 이 구간은
(a) hand-eye calibration 개선 (b) side에 closed-loop visual servoing (eye-in-hand니까 가능)
(c) 성공률 낮음 수용 — 셋 중 하나.

### approach 벡터 공식 대응 위치
`⟨p−0.2a, ..., p⟩` → `planning_node.py` 실행 단계 **마지막 구간**. OMPL/cuRobo는
`p−0.2a`(standoff)까지만 계획, 거기서 `p`까지는 `-a` 방향 Cartesian 직선 웨이포인트 3개.
짧은 직선이라 collision/IK 거의 안 터지고 정밀. = "마지막 몇 cm Cartesian" + 논문 8번
(Viereck 2017). OK-Robot 버전은 open-loop Cartesian + closed-loop 그리퍼 — visual servoing
보다 단순, NERO엔 이걸로 충분할 수도.

### 적용 가능성 / 우선순위
| 항목 | 대응 위치 | 난이도 | 우선 |
|---|---|---|---|
| **`p − k·a` Cartesian pre-grasp** | `planning_node.py` 실행 단계 | 낮음, 격리됨 | **먼저.** 현재 단일-pose 파이프라인 그대로, 마지막 구간만 교체 → "그립 정확도 감소" 줄어드는지 A/B |
| **AnyGrasp / Contact-GraspNet / VGN** | `resolve_grasp_dir`+`grasp_kinematics.py` 쿼터니언 수학 대체 | 중간 | 2순위. 입력 = RealSense RGB-D (있음). AnyGrasp 연구무료/상용유료, Contact-GraspNet BSD, VGN 최경량 |
| **LangSam/Grounded-SAM mask → grasp 필터** | "bbox→crop→infer_grasp" 대체 | 중간 | `segmentation_backend.py` 뼈대 있음. GroundingDINO conf 문제는 seg 시드용이라 덜 치명적. YOLO-World box→SAM도 가능 |
| **θ 기반 grasp 랭킹** | 신규, 작음 | 낮음 (후보 있어야) | AnyGrasp 이후 |
| **closed-loop 그리퍼 닫기** | 실행 단계 | Piper 그리퍼 force/current 피드백 있으면 쉬움 (확인 필요) | 부수 |
| "wrist tuck over body" | — | — | 불필요 (고정 팔). "pick↔place 사이 알려진 안전 자세 경유"만 차용 |
| "head camera point at object" | eye-in-hand = 관찰 자세로 팔 이동 | 이미 함 | eye-in-hand는 접근 중 카메라가 움직여서 연속 재인지(visual servoing) 가능 — OK-Robot 고정 head는 못 함. 기회 |

### Part B 한 줄
"grasp_type을 VLM이 정하고 → 쿼터니언을 공식으로 계산" 을 "grasp 모델이 pose 후보 다
생성 → mask로 시맨틱 필터 → θ+reachability로 랭킹 → `p−k·a`로 접근" 으로 바꾸는 레시피.
**`p − k·a` Cartesian 접근은 지금 당장 격리해서 넣을 수 있는 저위험 변경 — 이거 먼저.**

## Part C (Drop 휴리스틱) — 분석 완료

### OK-Robot이 하는 것
1. LangSam으로 drop 쿼리 → 컨테이너(sink/bin/box/bag) point cloud segment
2. 정렬: X=로봇 정면, Y=좌우, Z=floor normal. 로봇 (x,y)=(0,0), 바닥 z=0 정규화 → Pa
3. drop point = segment된 클라우드의 median (xm, ym)
4. drop height `zmax = 0.2 + max{z | 0≤x≤xm, |y−ym|<0.1}` — median-y 근처 좁은 띠에서
   로봇 쪽 절반의 **가장 높은 점(테두리) + 20cm 버퍼**
5. 그리퍼 그 위로 → 열어서 떨굼. "clutter 안 따짐, 평균적으로 잘 됨"

### NERO 대응 (현황: analyze_scene→placement_regions, find_placement, ground_object
center_view sweep 해킹 1분+, container_p20 depth, bbox_truncated, 2026-09-02 "회색
바구니 이미지 가장자리→앞슬라이스만→중심 x 80mm 당겨짐, depth 0.77~0.91m 출렁")

| Part C 아이디어 | NERO 적용 |
|---|---|
| **테두리 높이 + 버퍼** (내부 depth 아님) | drop이면 바구니 깊이 불필요. `container_p20` → `rim_max_z + buffer`. depth 출렁임 소멸 |
| **segment 클라우드 median (x,y)** (bbox 중심 3D화 아님) | Part B SAM 도입과 세트. mask back-project median > bbox 중심. 가장자리 truncation에 덜 민감 |
| **median-y 띠 + 로봇쪽 절반 샘플링** | 로봇이 넘어갈 쪽 테두리만 콕. "바구니 가장자리" 문제에서 보이는 부분만 |
| **crude해도 평균적으로 됨 수용** | drop-into-container면 analyze_scene+placement_regions+find_placement+container_p20 전체를 rim 휴리스틱 하나로 대체 가능 |

캐비엣:
- OK-Robot은 **떨군다**. 평면에 조심스럽게 place하려면 표면 높이 정밀 + collision 여전히 필요. rim 휴리스틱은 drop 전용
- eye-in-hand라 컨테이너 클라우드 잘 segment하려면 좋은 시점 필요 (center_view가 풀던 문제).
  **→ 진짜 해법: Part A(static 맵) + Part C(rim 휴리스틱). 바구니/싱크/선반은 가구 →
  static 스캔 맵에 1회 멀티뷰로 3D extent + rim 높이 저장 → drop 시 맵 조회, sweep 없음**
- 20cm 버퍼는 그들 튜닝값. NERO는 자체값(그리퍼 길이 + 물체 늘어진 길이)
- `0≤x≤xm`은 컨테이너가 로봇 정면 +x 가정. base_link 축 부호 확인

적용: rim 높이 + 버퍼는 **높음, 저위험** — container_p20 자리 바로 교체 테스트. segment
좌표는 Part B SAM과 세트.

### Part C 추가로 읽을 논문 (OK-Robot 휴리스틱만으론 구현 부족)
1. **M2T2** (NVIDIA, CoRL 2023) — pick+place 통합 트랜스포머, placement pose 직접 예측.
   단일 최강 답 (redesign 리스트 #7과 동일)
2. **CabiNet** (NVIDIA, ICRA 2023) — 어수선한 리셉터클(캐비닛/선반)에 놓을 때 neural
   collision detection. OK-Robot이 인정한 "clutter 안 따짐" 갭 메움. cuRobo와 세트
3. **Predicting Stable Configurations for Semantic Placement of Novel Objects**
   (Paxton et al., CoRL 2021) — 안정적 placement pose 예측
4. **TAX-Pose** (Pan et al., CoRL 2022) — "A를 B에/위에" 태스크용 두 물체 상대 pose 예측.
   관계적 배치
5. **AdaPoinTr** (또는 point cloud completion 일반) — 바구니 뒤쪽 가려진 부분 완성
   ("앞슬라이스만 보임" 문제)
6. (옵션) O2O-Afford (Mo et al., CoRL 2021), StructFormer (Liu et al., ICRA 2022)

## Part D (배포 / state machine) — 분석 완료

### OK-Robot 사실관계
- 스캔 <1분, VoxelMap 처리 <5분, 첫 pick-drop까지 총 <10분
- **error detection/correction 구현 안 함** — state machine = 단순 선형 체인
  (nav→grasp→nav→drop)
- LLM(GPT-4V) 역할 = 쿼리 문자열 생성만. 제어 루프에 0회
- 실행 전 nav 실패 물체(semantic memory가 위치 못 잡은 것) 미리 필터링
- trial 간 reset 없이 순차 실행

### NERO 대응 — Problem 3 답의 가장 센 형태
1. **기본 경로 = 결정론적 선형 체인, LLM 없음.** perception→grasp 선택→plan→execute→place.
   왕복 없이 처음부터 끝까지. MCP tool은 이미 스텝으로 존재 — Claude를 사이에서 빼고
   체이닝하는 executor만 필요
2. **LLM은 예외 시에만.** 스텝 실패(grasp IK 불가/물체 없음/place 막힘) → 그때 실패
   컨텍스트와 함께 Claude 에스컬레이션 → retry/재정렬/skip/사용자 문의. = "Inner Monologue"
3. **Pre-flight 필터.** pick 시퀀스 시작 전 검증: (a) 물체가 맵/스캔에 있나 (b) reachable
   grasp 후보 ≥1개 (c) drop 위치 확보 + reachable. 하나라도 실패면 **움직이기 전에** 보고.
   "reach 불가"를 실행 도중 발견하지 마라 → **"reach 불가 오류 폭증"의 직접 대응**

적용: happy-path 선형 BT/state machine = **높음, 큰 latency 이득**. pre-flight reachability
체크 = **높음, "reach 불가" 직접 완화**. 완전 "error recovery 없음" = **NERO엔 비권장**
(CLAUDE.md 규칙들이 전부 실패 흉터). LLM을 예외 핸들러로 유지.

### Part D 추가로 읽을 논문
1. **Integrated Task and Motion Planning** 서베이 (Garrett et al., Annual Review 2021) —
   "grasp 정하고 IK 체크" 실패가 TAMP 문제라는 프레이밍. 필수 개념
2. **Text2Motion** (Lin et al., 2023) — LLM 태스크 플래닝 + **실행 전 feasibility 체크**
   (pre-flight 필터 그것)
3. **REFLECT** (Liu et al., CoRL 2023) — 멀티모달 로그로 실패 설명 + 수정. "LLM은 예외 시에만"
4. **LLM3** (2024) — LLM TAMP + motion failure reasoning. "reach 불가 → replan" 루프
5. **PDDLStream** (Garrett et al., ICAPS 2020) — 어떤 grasp가 나머지를 feasible하게 하는지
   샘플링 TAMP
6. **ProgPrompt** (Singh et al., ICRA 2023) — LLM이 assertion 박힌 플랜 프로그램 생성
7. **Behavior Trees in Robotics and AI** (Colledanchise & Ögren, 2018) + **BehaviorTree.CPP**
   (ROS2 라이브러리, 도구 — Nav2/MoveIt Pro가 씀) — 결정론적 skill 층 구현체
8. (이미 redesign 리스트) Code as Policies #10, Inner Monologue, SayCan

## 최종 정리 — 대기 중 (사용자가 한 번에 정리 요청 예정)
