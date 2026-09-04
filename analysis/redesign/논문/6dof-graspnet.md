---
name: 6dof-graspnet
description: "6-DOF GraspNet (Mousavian et al., ICCV 2019) 분석 + Contact-GraspNet 비교 + NERO 방법론 결정"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 6ba50bd9-f824-411e-816c-7dcb14f08a0e
  modified: 2026-09-04T06:54:10.821Z
---

# 6-DOF GraspNet: Variational Grasp Generation for Object Manipulation
Mousavian, Eppner, Fox — NVIDIA, ICCV 2019, arXiv:1905.10520

[[contact-graspnet]] 와 비교 대상. NERO grasp generator 방법론 결정용. [[nero-grasp-pipeline-redesign]].

## Section 3 — Method (분석)

### 개요
grasp pose generation = "그리퍼 닫으면 안정 grasp 되는 pose 집합 생성 + 물체를 잡는
모든 방법을 커버하는 다양한 집합". **단일 물체만** — reach 제약·다른 물체는 scope 밖,
"trajectory optimization 이 처리" (← NERO 아키텍처 검증: grasp gen 은 단일물체,
reach/collision 은 MoveIt/pick_ik/STOMP. 두 논문 다 이 분리에 동의).

파이프라인: **VAE 샘플러 → 반복 evaluation + refinement.** 입력 = 물체 point cloud.
P(G*|X) 학습, grasp g=(R,T)∈SE(3), **물체 좌표계**(원점=관찰 PC 무게중심 X̄, 축은 카메라
프레임 평행). G* 는 복잡·불연속·**multimodal** (머그: rim/handle/bottom 모드).

### 3.1 Variational Grasp Sampler (CVAE)
- encoder Q(z|X,g): (PC, grasp) → 잠재 subspace. decoder: z → 재구성 ĝ
- P(z)=N(0,I). `L_vae = Σ L(ĝ,g) − α·D_KL[Q, N(0,I)]`
- **재구성 loss `L(g,ĝ) = (1/n)Σ||T(g;p) − T(ĝ;p)||₁`** — 그리퍼 위 미리정의 점 p 를
  g/ĝ 로 변환한 것의 L1 거리 (= Contact-GraspNet 의 5점 ADD 와 같은 control-point 아이디어)
- 학습: 랜덤 뷰포인트 PC + GT grasp stratified 샘플
- 추론: encoder 제거, z~N(0,I) 샘플 → decode. **grasp 수 = z 샘플 수**
- encoder/decoder 둘 다 PointNet++

### 3.2 Grasp Pose Evaluation
- 샘플러가 positive 만 봄 → 모드 사이 transitional failure grasp 생김 → evaluator 로 pruning
- P(S|g,X) — 관찰 X 기준 + **미관찰 부분 extrapolate** 해야 함. 단일뷰 불완전 PC 로 분류
  (이전 방법은 고품질 센서/멀티뷰 필요 → 배포 제약)
- **핵심 표현 트릭**: 6D pose 를 점 feature 에 concat 하면 정확도 나쁨. 대신 **그리퍼를
  pose g 로 point cloud X_g 렌더 → 물체 X + 그리퍼 X_g 를 하나로 합치고 binary feature
  (물체 vs 그리퍼)**. PointNet 이 grasp-물체 상대 기하를 자연스럽게 씀
- cross-entropy. **hard negative mining**: positive 와 비슷한 pose 지만 충돌하거나 너무 먼
  grasp (`G− = {g− | ∃g∈G*: L(g,g−)<δ}`), positive 를 perturb 해서 생성

### 3.3 Iterative Grasp Pose Refinement
- 거부된 grasp 상당수가 성공에 가까움 → ∆g∈SE(3) 로 P(s=1) 올림
- evaluator 가 미분가능 → `∂S/∂g` 가 refinement 방향
- 점별 gradient = non-rigid → Euler각+translation 으로 파라미터화해 rigidity 강제
- `∆g = η × ∂S/∂T(g;p) × ∂T(g;p)/∂g`, η 로 스텝 제한 (**최대 translation 업데이트 ≤ 1cm**)
- 반복

## Contact-GraspNet(CGN) vs 6-DOF GraspNet(6DGN) 비교

| 측면 | 6DGN (ICCV19) | CGN (ICRA21) |
|---|---|---|
| 패러다임 | **생성형** (CVAE 샘플→평가→refine) | **판별형** (per-point 회귀, 1 forward) |
| grasp 앵커 | 자유 SE(3), 물체 무게중심 프레임 | **관찰된 실측 점** c + 3-DoF 회전 + width |
| 씬 범위 | **단일 물체만** (clutter scope 밖) | clutter 씬 (+ `--local_regions` 단일물체 모드) |
| 샘플링 | z~N(0,I) 개수만큼 | FPS 2048점, 1패스 ~2048 후보 |
| 평가 | **별도 evaluator 네트워크** (그리퍼-점 렌더 + binary feat) | confidence head `s` 가 같은 망에 내장 |
| refinement | **있음** — evaluator `∂S/∂g` 로 반복 개선 (≤1cm 스텝) | 없음 (1패스) |
| 추론 비용 | 높음 (z 다수 × evaluator × refine N회) | 낮음 (1 forward) |
| 불완전 depth | evaluator 가 extrapolate 하도록 설계 | contact 보이는 곳만 grasp ("최소 1개 visible" 가정) |
| 프레임 | 물체 프레임 (centroid) | 카메라 프레임 |
| 코드 | `pytorch_6dof-graspnet` (저자) | 공식 TF2 + 커뮤니티 PyTorch |

### 개념적 핵심 차이
1. **생성 vs 판별.** 6DGN 은 자유공간에서 샘플 후 평가·refine 로 "좁은 성공 subspace" 로
   밀어넣음. CGN 은 관찰 점마다 예측이라 애초에 자유공간 샘플 안 함 → CGN Part A 가
   "unconstrained SE(3) 방법"(=6DGN)을 명시적으로 비판. CGN 이 pose 정확도·속도 우위 주장.
2. **둘 다 NERO 엔 SAM crop 단일물체 입력 필요** (6DGN 필수, CGN local_regions). SAM 의존 동일.
3. **refinement 는 6DGN 만의 진짜 장점** — near-miss grasp 를 구조.

## NERO 방법론 결정

### 주 generator = Contact-GraspNet
근거:
- 1패스 2048 후보 → 빠름, pick_ik funnel 에 적합
- 관찰 기하 앵커 → 불완전 depth 에 강함 (자유공간 샘플 안 함)
- `--local_regions` = SAM crop 이 설계된 모드
- 배포 단순 (1망 vs 샘플러+evaluator+refiner)
- `LearnedGraspOutput` 인터페이스가 이미 CGN 표현
- multimodality (ADD-S loss) 로 top/side/angled 후보 자연 발생

### 6DGN 에서 빌려올 것
1. **반복 refinement 아이디어를 "reachability" 에 적용** — CGN 후보가 pick_ik 에서
   실패하면 버리지 말고, 작은 SE(3) 볼(≤1cm translation, 6DGN 의 η) 안에서 perturb 해
   IK 재확인. "reach 불가" 상당수가 near-miss. **evaluator 망 불필요 — pick_ik 가
   objective.** (단 CGN 이 이미 조밀 후보라 대부분 불필요, 시맨틱 선호 grasp 이 간신히
   unreachable 인 경우에만)
2. **evaluator 표현 트릭(그리퍼-점 렌더 + binary feature)** — 나중에 "이 물체+태스크에
   대한 grasp 적합도" 학습 re-ranker 원하면 이 방식. **지금은 아님.**

### 안 할 것
- 6DGN 풀 파이프라인(샘플러+evaluator+refiner)을 주력으로 — 무겁고 느리고 자유공간
  샘플(불완전 depth 에 불리), 단일물체 가정이 CGN local_regions 대비 이점 없음
- evaluator 네트워크 지금 구축

### 두 논문 공통 = NERO 아키텍처 검증
**grasp generator = 단일 물체 pose 만. reach·collision 은 downstream(MoveIt/pick_ik/STOMP).**
관심사 분리가 정석. (6DGN 명시, CGN Part 3 서두 명시)

### 얇은 물체
6DGN refinement 이 도움될 수 있으나, 본질은 depth 문제(spike D_thin) — 어느 방법도
나쁜 depth 는 못 고침.
