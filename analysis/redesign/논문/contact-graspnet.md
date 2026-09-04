---
name: contact-graspnet
description: "Contact-GraspNet (Sundermeyer et al., ICRA 2021) 파트별 분석 — 방법론 A~E 완료 + 종합(좋은/나쁜 소식). NERO grasp generation 구현 참고"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 6ba50bd9-f824-411e-816c-7dcb14f08a0e
  modified: 2026-09-04T09:28:19.649Z
---

# Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes
Sundermeyer, Mousavian, Triebel, Fox — NVIDIA, ICRA 2021

NERO grasp 6-DoF pose generation 구현의 개념/구현 참고. [[nero-grasp-pipeline-redesign]].
사용자가 파트별로 읽으며 분석 중.

## Part A — Grasp Representation (분석 완료)

### 핵심 아이디어: contact grasp representation
- 관찰: 실현 가능한 2지 그립 대부분은 **두 contact 중 최소 하나가 grasp 전에 보인다.**
  안 보이는 contact 뿐인 grasp은 애매하거나 물체 자세를 흐트러뜨림.
- 그래서 GT 6-DoF grasp 분포 g∈G → 대응 **contact point c∈R³** 로 매핑.
  visible contact 는 depth 센서로 관찰 가능한 표면 위 → 기록된 point cloud 의
  근접 점으로 3D 위치 표현.
- **6-DoF grasp 학습 문제를 3-DoF 회전 R_g + 그립 폭 w 추정으로 축소.**
  contact point c 자체는 point cloud 에서 옴 (예측 아님).

### 수식 (parallel-jaw)
contact point c (그리퍼 baseline 이 메시와 교차하는 점) 에서:
```
t_g = c + (w/2)·b + d·a        (1)  그리퍼 base 위치
R_g = [ b | a×b | a ]           (2)  회전 (열벡터)
```
- a: 단위 approach vector (다가가는 방향)
- b: 단위 grasp baseline vector (손끝이 닫히는 축)
- d: 상수 — 그리퍼 baseline → 그리퍼 base 거리 (그리퍼 하드웨어 상수)
- w: 그립 폭

즉 네트워크가 관찰된 점마다 예측: (여기가 grasp contact 인가 = confidence) + a + b + w.
c 는 point cloud 에서 직접.

### 논문이 주장하는 장점
1. **차원 축소** → unconstrained SE(3) 예측보다 학습 쉬움
2. **정확도↑** — grasp 가 관찰된 씬 기하에 묶임 (c 가 실측 점, 떠다니는 예측 아님)
3. axis-angle 대비 **ambiguity/discontinuity 없음**
4. **후보 샘플링 = contact point 샘플링** — 관찰 가능 표면 전체를 덮어 6-DoF grasp
   분포의 모드를 잘 표현
5. 3D 뷰가 좋지만 **박스 정면 뷰만으로도 radial mapping 덕에 합리적 grasp**

### 아키텍처
PointNet++ 로 point cloud 처리, 국소 3D 이웃에서 계층적 feature 집계.
**예측이 입력 point cloud 의 3D 점에 직접 연결됨** — 이 표현이 그 능력을 활용.
= per-point 예측 헤드.

## NERO 적용점

### 1. analytic orientation 스택 직접 대체
`grasp_kinematics.py` 가 R_g 를 각도 공식(atan2)으로 계산 → Contact-GraspNet 이
contact 마다 (a, b, w) 예측 → `R_g=[b, a×b, a]`, `t_g=c+(w/2)b+da`.
이게 곧 `GraspCandidate` (position, quaternion, approach_vector, width).

**`learned_grasp_backend.py` 의 `LearnedGraspOutput` 이 정확히 이 표현으로 설계됨:**
`position` / `approach_vector`(=a) / `grasp_axis`(=b, baseline) / `score` / `width`(=w).
docstring 에도 "GraspNet은 approach+binormal+width" 라고 이미 적혀 있음 (binormal=b).
→ 변환 코드는 R_g→quaternion(xyzw) 하나. `grasp_pose_generator.py` 가 backend별 담당.

### 2. d 상수 = TCP offset 통일 → TOP/SIDE offset 문제 소멸
`t_g = c + (w/2)b + d·a` 의 `d·a` 항이 approach 축 방향 TCP offset.
**grasp 종류 무관하게 항상 `d` 를 `a` 방향으로** — CLAUDE.md "TOP_TCP_OFFSET과
SIDE_TCP_OFFSET 절대 합치지 마라" 의 원인(두 프레임 규약: world Z vs gripper frame)이
여기선 없음. offset 은 예측된 approach vector 방향 하나. d = AGX 그리퍼 하드웨어 상수
(그리퍼 스펙 수집 과제와 연결).

### 3. contact 샘플링 = 랭킹된 후보 집합
SAM-masked point cloud 의 물체 표면 점들을 샘플 → 점마다 grasp 후보 →
pick_ik reachability 필터 → θ(top-down)/collision/VLM re-rank (roadmap #4~5).
Contact-GraspNet 이 per-point 예측이라 이 흐름과 자연 정합.

### 4. SAM crop 궁합
PointNet++ 가 scene point cloud 를 받아 per-point 예측 → **SAM-masked PC(물체만)**
넣으면 모든 예측이 물체 위. spike 가 "SAM crop 전제" 로 판정한 것과 정확히 맞음.

### 5. "최소 하나의 contact 가 보이면 된다" = 부분 단일 뷰 OK
NERO eye-in-hand 관측 자세(부분 뷰, 정면~비스듬)에서도 동작. 전체 물체 재구성 불필요.
"박스 정면 뷰만으로도 OK" = NERO 관측각(top-down 아님)에 유리.

### 6. 정확도 = 관찰 기하에 묶임 → calibration 오차 부분 완화
grasp position 이 실측 점 c. point cloud 가 base_link 에 잘 정합되면(`_cam_to_base`)
grasp position 정확도 = depth + TF 정확도. 공식 누적 오차 없음.

### 7. axis-angle discontinuity 없음 → 기존 버그 클래스 회피
`grasp_kinematics_ik.md` 의 pinch `roll=90,pitch=180 axis-flip` 버그 = axis-angle
표현 문제. contact representation 은 이 클래스 사이드스텝.

### 한계 (NERO 관점)
- **얇은 물체**(pen 등, spike D_thin): 두 contact 다 얇은 모서리 + depth 나쁨 → 약할 것
- 반사/투명 물체: contact point 자체가 point cloud 에 없음 (spike 확인) → depth 경로 밖
- 학습 데이터가 시뮬(ACRONYM) → 실기 성공률은 검증 필요

## Part B — Data Generation (분석 완료)

### 학습 데이터
**ACRONYM** [Eppner et al.]: ShapeNet 메시 8872개 + 마찰 변화 하에 시뮬 grasp 17.7M.
씬 point cloud 를 렌더링해서 학습 (offline + online, Fig 2).

### 라벨링 방식 (per-point)
씬 point cloud P={p1..pn} 의 각 점 pi 에:
```
s_i = 1  if  min_j ||p_i − c_j||² < r        (r = 5mm)
s_i = 0  otherwise
```
- c_j = **비충돌(non-colliding)** GT grasp g_j 의 메시 contact point (카메라 좌표계)
- P⁻ = {s_i=0} (5mm 내 feasible contact 없음), P⁺ = {s_i=1} (contact 적합 점)

각 p⁺_i 에 가장 가까운 GT grasp 할당 (식 4~5):
```
(w_i, R_i, t_i) = ( w_j, R_j, p⁺_i + (w_j/2)·b_j + d·a_j )
   j = argmin_k ||p⁺_i − c_k||²
```
**핵심: t_i 는 GT contact c_j 가 아니라 관찰된 점 p⁺_i 로 재앵커.** 회전·폭은 최근접
GT grasp 에서 복사, 위치는 실측 점 기준. = Part A 의 "grasp 가 관찰 기하에 묶임" 의 메커니즘.

결과: 충분한 coverage 면 GT 6-DoF grasp 분포를 point cloud 위에 조밀하게 투영.

### 모델이 실제로 학습하는 것
- per-point 이진 분류: 이 점이 graspable contact 인가 (s_i)
- per-positive-point 회귀: R_g, w, 그리고 t_g 를 주는 offset
- 추론은 **완전 feed-forward** — object pose estimation·CAD 매칭 없음

## NERO 적용점 (Part B)

### 1. 학습 도메인 = 시뮬 (ACRONYM/ShapeNet) → 도메인 갭 확인
pretrained 가중치는 ShapeNet 물체 + 렌더 depth 로 학습. NERO 실물(골판지 box/cup/bottle)은
ShapeNet 에 카테고리 있어 커버는 되지만 **RealSense depth 노이즈 ≠ 렌더 depth.**
실기 성공률 낮으면 → Isaac Sim 에서 NERO 실제 물체 메시로 ACRONYM 식 데이터 생성 +
도메인 랜덤화 fine-tune (roadmap "재학습 필요 시"). 큰 작업 — pretrained 실패 시에만.

### 2. r = 5mm → depth 정확도가 mm 스케일로 중요
contact 할당 반경 5mm. NERO depth 노이즈: place 검증에서 z 최대 26mm 관측(CLAUDE.md)
됐지만 그건 다른 상황 — 0.6~0.8m 밝은 불투명 표면 국소 depth 는 수 mm 예상. **경계선,
측정 필요.** 노이즈 크면 P⁺ 후보가 흔들려 grasp 품질 저하.

### 3. SAM crop "필요조건이지 충분조건 아님" 정량화
masked PC 가 성글면(occlusion shadow, 검은 면) P⁺ 후보가 적음 → grasp 후보 적음.
depth 구멍 = 그 영역엔 contact 예측 자체가 안 나옴.

### 4. "non-colliding GT" = 암묵적 collision 인식, 단 보장 아님
학습 분포가 비충돌 grasp 뿐이라 모델이 대충 충돌 회피 grasp 를 냄 — 하지만 **NERO
실제 planning scene 기준이 아니라 학습 씬 분포 기준.** 명시적 collision 필터(planning
scene/pick_ik) 를 위에 얹어야 함 (roadmap #5 collision clearance re-rank 와 정합).

### 5. 카메라 프레임 출력 → base_link TF
학습·추론 전부 카메라 좌표계. NERO 는 `_cam_to_base` 로 변환. **joint1 회전 시 `_cam_to_base`
y 틀어짐 이슈** → grasp 인지는 canonical observation 자세(joint1≈0)에서만. 회전된 자세에서
재관찰하면 grasp pose 변환 정확도 저하.

### 6. CAD 모델 불필요 (추론 시)
데이터 생성엔 메시 필요하지만 추론은 point cloud → grasp feed-forward. NERO 가 물체 CAD
없는 것과 궁합. p⁺_i 재앵커 덕에 부분 뷰·불완전 모델에 강건.

## Part C — Network (분석 완료)

### 아키텍처
PointNet++ set abstraction + feature propagation → **비대칭 U-net.**
- 입력: n=20000 랜덤 점 (R^{20000×3})
- 출력: 입력의 **FPS(farthest point sampling) 2048 점**에 대해서만 grasp 예측
  (GPU 메모리 + 씬 coverage 균형)
- **4개 head** (각 1D-Conv 2층), per-point 출력:
  - `s ∈ R` — grasp confidence (contact-ness 점수) = 랭킹 신호
  - `z1 ∈ R³` → baseline 방향 b
  - `z2 ∈ R³` → approach 방향 a
  - `o ∈ R^10` — 그립 폭 bin

### 그립 폭 = 10-bin 분류 (회귀 아님)
ŵ ∈ [0, w_max] 를 10개 등간격 bin 으로. data imbalance 대응. 최종 폭 = 최고 confidence
bin 의 중심값.

### 회전: in-network Gram-Schmidt 직교정규화
a, b 는 정의상 직교정규 → 학습에 주입:
```
b̂ = z1 / ||z1||
â = (z2 − ⟨b̂, z2⟩·b̂) / ||z2 − ⟨b̂, z2⟩·b̂||
```
z2 를 b̂ 직교 평면에 투영 후 정규화. **a 는 b 에 직교인 성분만 예측.**
차원 더 축소 + 3D 회전 회귀 용이 (Zhou et al. "continuity of rotation representations" [36]).

## NERO 적용점 (Part C)

### 1. 후보 수 = 추론당 최대 2048 (FPS 점), s 로 threshold
pick_ik 루프에 필요한 ~50개보다 훨씬 많음. 파이프라인:
predict → `s > threshold` 필터 → 정렬 → top-K → pick_ik reachability → collision →
VLM re-rank → best. `GraspCandidate.score` = s.

### 2. local-region / object-crop 추론 모드 = NERO SAM-crop 계획 그대로
Contact-GraspNet 공식 지원: "물체 segment → point cloud crop → crop 에 predict"
(타겟 grasping 방식). 릴리스 코드에 `--local_regions` / `--filter_grasps` 플래그.
**off-distribution 아님, 설계된 모드.** SAM crop → Contact-GraspNet 이 정확히 이것.

### 3. 그리퍼 폭 config
w_max 파라미터를 AGX 그리퍼로 설정 (10-bin 분류). Franka 8cm 와 가까우면 pretrained
head OK, 많이 다르면 폭 예측 clamp/rescale 필요 or 신뢰도 낮음 → 그리퍼 스펙 수집 과제.

### 4. 회전 출력이 구조적으로 well-conditioned
Gram-Schmidt + continuous 6D 표현 → `R_g=[b, a×b, a]` 항상 유효 회전 →
`GraspCandidate.quaternion` 변환 깔끔. `grasp_kinematics.py` 의 roll/pitch/yaw
axis-flip 버그 클래스(`grasp_kinematics_ik.md`) 회피.

### 5. 컴퓨트 = 가벼움 → PC/Thor 어디든
n=20000 입력, m=2048 출력, PointNet++ U-net. VLM 보다 훨씬 작음(~2~4GB).
PC 3080Ti (Isaac Sim 옆) 또는 Thor 둘 다 가능 → 컴퓨트 위치 결정에 제약 아님.

### 6. d 상수는 네트워크 밖 후처리
네트워크는 a, b, w 만. `t_g = c + (w/2)·b + d·a` 재구성은 코드에서, `d` = AGX 그리퍼
baseline→base 거리 (하드웨어 상수). Part A #2 (TCP offset 통일) 와 연결.

## Part D — Target Losses (분석 완료)

### 세 loss
1. **contact success `ŝ`** — 모든 출력점에서 BCE. 단 **에러 큰 top-k=512 점만 backprop**
   (data imbalance 대응, hard mining).
2. **geometry (a,b,w)** — **positive contact 점 p⁺_i 에서만** 평가.
3. **결합 6-DoF grasp loss `l_add-s`** (ADD-S = symmetric avg distance) — 핵심 혁신:
   - head 를 따로 감독하지 않고, **학습 중에 예측을 완전 6-DoF pose 로 합침**
   - 그리퍼 pose 를 나타내는 5개 3D 점 v 정의 → GT/예측 pose 로 변환
   - `l_add-s = (1/n+) Σ ŝ_i · min_u ||v_pred_i − v_gt_u||²`
   - 그리퍼 대칭성 고려(min_u), **예측 contact confidence ŝ_i 로 가중**

### `l_add-s` 의 장점 (NERO 직결)
1. **GT grasp 분포의 여러 모드 학습** — 다른 approach 방향 â 도 작은 에러 = **multimodal**
2. **ŝ_i 가중 = contact 분류와 pose 예측을 커플링.** 좋은 6-DoF pose 를 예측해야만
   contact confidence 가 올라감
3. **GT 에서 먼 곳의 잘못된 grasp (occlusion 인공 edge 등) = 높은 loss → 회피.**
   ← spike 가 걱정한 occlusion shadow 문제를 **모델이 명시적으로 학습해서 안 함**

width: weighted multi-label BCE, bin 크기 anti-proportional 가중 (작은 폭 과다).
총 loss: `l = 1·l_bce,k + 10·l_add-s + 1·l_width` (pose loss 10배 지배).

## Part E — Implementation Details

- Adam lr 0.001 → 0.0001 step decay
- set abstraction: 3 병렬 branch, query ball 반경 [0.02,0.04,0.08]/[0.04,0.08,0.16]/
  [0.08,0.16,0.32] = 2cm~32cm 멀티스케일 국소 feature
- **추론: point cloud 를 카메라 좌표계 mean 에서 centering** (정규화)
- **학습 씬: 10000 tabletop, ShapeNet 8~12개 random stable pose, rejection sampling 충돌회피**
- batch 3, 144k iter, V100 1장 ~40시간. 이전 방법(최대 1주) 대비 훨씬 빠름 = 표현이 효과적

## NERO 적용점 (Part D/E)

### 1. occlusion edge 방어가 학습에 내장됨 (Part D 장점 3)
spike 의 "occlusion shadow" 걱정 완화 — 모델이 GT 없는 인공 edge 에 grasp 안 내도록
학습됨. 단 "안 냄"이지 "메움"이 아님 — 가려진 영역엔 grasp 자체가 없음.

### 2. multimodal (Part D 장점 1) → top/side/angled 후보 자연 발생
= 하드코딩 단일 쿼터니언 죽이는 핵심. **단 argmax(ŝ) 만 취하면 안 됨** — 태스크에
맞는 approach 는 다를 수 있음. funnel: ŝ threshold → pick_ik reachability → VLM/task re-rank.

### 3. 그리퍼 대칭성 (l_add-s min_u)
parallel-jaw 는 approach 축 180° 회전에 불변. **각 후보마다 grasp 와 그 180° twin 이 동등**
→ 둘 다 pick_ik 에 넣으면 하나가 reachable 일 수 있음 (공짜 reachability 2배).

### 4. 학습 도메인 = top-down tabletop (Part E)
10k 씬이 위에서 내려보는 tabletop. NERO 관측각 얕으면 off-distribution.
→ observation 자세 45~70° 내려보기(spike 권고)가 모델 학습분포와도 맞음.

### 5. 추론 정규화 = mean centering
NERO 도 masked PC 를 mean centering 후 넣어야 (전처리 일치).

### 6. fine-tune 비용 = ~40h V100 1장
필요 시 현실적. Isaac Sim 에서 NERO 물체 + RealSense 노이즈 모델로 10k 씬 생성 가능.

## Part F — 종합 분석 (좋은 소식 / 나쁜 소식) — 방법론 A~E 종합

방법: point cloud → PointNet++ U-net → per-point (contact conf s, approach a, baseline b,
width w) → `t_g = c + (w/2)b + da`, `R_g = [b, a×b, a]`. 결합 ADD-S loss(contact conf 가중)
로 학습. 시뮬 10k ShapeNet tabletop 씬 pretrained.

### 좋은 소식

| # | 내용 |
|---|---|
| G1 | **`learned_grasp_backend.py` `LearnedGraspOutput` 인터페이스가 이 표현과 정확히 일치** (position/approach_vector/grasp_axis=b/score=s/width=w). 설계자가 이 논문 알고 만듦. 변환 = R_g→quaternion 하나 |
| G2 | **`--local_regions` = SAM crop 계획이 공식 지원 모드.** off-distribution 아님 |
| G3 | **grasp position 이 항상 실측 점에 앵커** (p⁺_i 재앵커). 떠다니는 SE(3) 예측·공식 누적오차 없음 → NERO calibration 불안 부분 완화 |
| G4 | **회전 표현이 Gram-Schmidt 로 구조적으로 valid·continuous** → roll/pitch/yaw axis-flip 버그 클래스(`grasp_kinematics_ik.md`) 회피 |
| G5 | **occlusion edge 허위 grasp 를 loss 로 명시적 억제** (Part D 장점 3) = spike 걱정 완화 |
| G6 | **multimodal** — contact 마다 여러 valid approach → top/side/angled 후보 자연 발생 = 하드코딩 단일 쿼터니언 죽이는 핵심 |
| G7 | **`d·a` offset = TCP offset 통일** → `TOP_TCP_OFFSET`/`SIDE_TCP_OFFSET` "절대 합치지 마라" 문제 소멸 |
| G8 | **컴퓨트 가벼움** (~2~4GB, PointNet++, VLM 아님) → PC/Thor 어디든. 컴퓨트 위치 결정에 제약 아님 |
| G9 | **추론 시 CAD 불필요.** feed-forward. NERO 물체 모델 없는 것과 궁합 |
| G10 | **confidence s = 깔끔한 랭킹 신호**, 추론당 ~2048 후보 = funnel 에 충분 |
| G11 | **fine-tune 비용 현실적** (~40h V100 1장, 이전 방법 1주 대비) |
| G12 | **그리퍼 대칭성** → 후보마다 180° twin 도 동등, 둘 다 pick_ik 에 = reachability 2배 |

### 나쁜 소식 + 개선 방향

| # | 나쁜 소식 | 개선 방향 |
|---|---|---|
| B1 | **시뮬 전용 학습(ShapeNet 렌더), depth 도메인 갭.** 실기 성공률 10~30% 하락 흔함 | (a) RealSense 후처리(spike R2: High Accuracy preset + spatial/temporal 필터)로 real→clean 근접 (b) 낮으면 Isaac Sim + RealSense 노이즈 모델로 NERO 물체 10k 씬 fine-tune (~40h) (c) `--forward_passes` test-time aug |
| B2 | **학습 씬 = top-down tabletop.** 얕은 관측각은 off-distribution (spike 에서 grazing 최악) | (a) observation 자세 45~70° 내려보기로 고정 (spike 권고 = 학습분포와도 일치) (b) `--local_regions` crop + mean centering 이 완화 (c) 얕을 수밖에 없으면 multi-view 융합(VGN) or 관측 전 팔 이동 |
| B3 | **5mm contact 반경 → mm 스케일 depth 정확도 요구.** NERO 국소 노이즈 >5mm 면 grasp 흔들림 | 평면 타겟 0.6~0.8m 에서 depth patch std 실측. 크면 spike R2 필터 + N프레임 평균 |
| B4 | **width head 가 Franka(w_max=8cm, d=0.1034) 학습.** AGX 그리퍼는 w_max≈10cm(대략 OK, 약간 rescale), **d≈0.19~0.20m 로 훨씬 길다 → `d` 반드시 재설정** (2026-09-04 URDF/docs 확인). `t_g=c+(w/2)b+d·a` 에서 d 오차 = approach축 위치 cm 오차 | `d`·w_max = URDF/`get_gripper_teaching_pendant_param()` 확정. **AGX 그리퍼는 바이너리 아님** — `move_gripper_m(w_m, force_N)` 폭 위치제어 → 예측 w 를 그리퍼 명령으로 직접 사용 가능. 예측 w 부정확하면 masked PC b방향 단면폭으로 대체 |
| B5 | **occlusion/금속/투명 = 모델이 못 보는 기하를 못 지어냄.** loss 가 허위 억제 → 가려진 영역엔 grasp 아예 없음 | (a) 투명/금속/고차폐 물체는 별도 트랙(알려진 geometry/VLM/사람) — 모델 문제 아님 (b) multi-view 융합이 occlusion shadow 특정하게 완화 (c) 문제 물체는 조명/무광 스프레이 |
| B6 | **카메라 프레임 출력 + `_cam_to_base` joint1 회전 시 drift** | grasp 인지는 canonical observation 자세(joint1≈0)에서만. 회전 자세 재관찰 금지 (NERO 기존 known-issue + 완화책) |
| B7 | **공식 코드 TF2, CUDA 핀.** PC(Isaac Sim CUDA 공유)/Thor(Blackwell) 세팅에 며칠 | (a) 커뮤니티 PyTorch 포트(`contact_graspnet_pytorch` 등) — deps 깔끔, 충실도 확인 (b) Docker 격리 (c) 별도 서비스로(이미 `learned_grasp_backend` 패턴) — env 오염 방지 |
| B8 | **NERO 실제 planning scene 기준 collision 인식 없음** (학습분포 기준일 뿐) | 명시적 collision 필터 위에 얹기(MoveIt planning scene/pick_ik) = roadmap #5. `--filter_grasps` 도 일부 |
| B9 | **multimodal → argmax(s) 가 원하는 grasp 아닐 수 있음** (태스크 무관 top-down 일 수) | funnel: s threshold → pick_ik reachability → **VLM/task re-rank** (roadmap #5). argmax 금지 |

### 결론
표현·인터페이스·모드가 NERO 계획과 거의 완벽히 맞음(G1~G7). 위험은 전부 **입력 품질
(depth/도메인/관측각)** 과 **그리퍼 config** 쪽이지 방법론 자체가 아님. → roadmap #3(seg
검증)·#4(통합) 전에 **B1~B4 를 스파이크 성격으로 먼저 계량**: 국소 depth 노이즈 실측,
observation 자세 각도, AGX 그리퍼 spec/제어 방식. B7(런타임)은 PyTorch 포트 + Docker 로 우회.
