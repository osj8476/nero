# VLA/VLM 도입 설계 논의 — 그립 형상 일반화 & 자연어 MCP 제어

> **이 문서는 설계 논의 단계 기록이다. 코드에 반영된 내용 없음.**
> `grasp_kinematics.py`/`planning_node.py`/`vlm_boxyolo.py` 어디에도
> 아래 개념(grip_family, azimuth_deg 등)은 아직 존재하지 않는다.
> 실제 구현을 시작하기 전에 반드시 `.claude/skills/grasp-kinematics-design/SKILL.md`를
> 먼저 읽을 것 (CLAUDE.md 지시사항, 이 문서도 그 규칙 아래에 있음).

## 요약

지금 시스템은 박스 전용(top-down 90°만)이라, "서랍문 열기(pinch)",
"문 손잡이 잡기(side)" 같은 일반화된 물체 조작이 불가능하다. VLA/VLM을
도입해 (1) 자연어로 MCP 제어, (2) 낯선 물체에 대한 그립 형태 판단을
일반화하려는 목표로 아키텍처를 논의했다. 핵심 결론: **VLA를 저수준
연속 제어기로 쓰지 않고, 기존의 정밀 검증된 코드 파이프라인 앞단
(perception/판단 레이어)에만 꽂는다.**

## 현재 상태 / 결론

### 역할 분담 (합의된 아키텍처)

| 레이어 | 담당 | 비고 |
|---|---|---|
| 자연어 작업 판단 (지금 Claude MCP 세션이 하는 일) | 어떤 tool을, 어떤 인자로 호출할지 | 안 바뀜 — 이미 구조화된 데이터 위에서 판단 |
| VLM (perception 확장) | "이게 뭔지"(열린 어휘 라벨) + "대략 어디"(영역/마스크) + 그립 계열 판단(top_down/side/pinch) | `vlm_boxyolo.py`가 지금 박스만 검출하는 자리를 확장하는 개념 |
| 기존 코드 (`grasp_kinematics.py`) | 정밀 quaternion/좌표 계산, IK 후보 검증 | 100% 유지 — 이번 논의로 바뀌는 것 없음 |

VLA를 "이미지+언어→관절 액션"까지 통째로 쓰는 방식(진짜 VLA의 정의)은
명시적으로 배제했다 — 이유는 아래 "폐기된 접근" 참고.

### grip_family 확장 방향

지금 top-down만 있는 걸 `side`(180° 횡), `pinch`(180° 종)로 확장하되,
**pitch를 90°~180° 사이 연속값(10도 단위 등)으로 보간하지 않는다.**
top-down이 지금처럼 안정적인 이유는 `pitch=90°` 고정 + `roll` 스윕을
실물 컨트롤러로 전수 검증했기 때문이며([[grasp_kinematics_ik]] 참고),
그 실측에서 `roll=180°,pitch=0°`처럼 "직관적으로 맞을 것 같은" 조합이
NO_SOLUTION이었던 전례가 있다. side/pinch도 각 family마다 **독립적인
실물 orientation sweep**이 먼저 있어야 하고, family 사이를 부드러운
함수로 잇지 않는다.

또한 `SIDE_TCP_OFFSET`이 아직 실측 미검증(CLAUDE.md, [[grasp_kinematics_ik]])
이라, side/pinch 확장 자체가 이 값의 실측 검증에 의존한다.

### side/pinch에서 flange 위치가 유일하게 안 정해지는 문제

top-down은 접근 방향이 항상 월드 `-z`로 고정돼 있어
`flange_pos = object_pos + TCP_offset × (0,0,-1)` 한 줄로 끝난다.
side/pinch는 접근 방향(azimuth, 물체를 어느 쪽에서 잡을지)이 물체
위치만으로는 정해지지 않는 **새로운 자유도**다.

제안한 파라미터화:
```
flange_pos = object_pos + TCP_offset_magnitude * approach_unit_vector(family, azimuth_deg)
```
`azimuth_deg`는 기존 `angle_rel`과 같은 컨벤션(=`position_yaw` 기준
상대각)으로 받는다. 출처는 VLA/VLM의 제안값이지만, **그 값을 곧이곧대로
실행하지 않고** `YawCandidateSelector`와 동일한 패턴(FK 기반 후보
스냅 + IK 검증)으로 가까운 유효 후보에 스냅해야 한다 — 이것도 사전에
azimuth 스윕으로 유효 후보 목록을 만들어둬야 가능하다(아직 안 함).

### 박스가 아닌 물체의 grasp 좌표 추출

지금은 bbox 중심 = grasp 지점이지만, 이건 "박스+top-down" 조합의
우연한 단순화다. 막대(side)는 장축 방향이, 손잡이(pinch)는 특정
돌출부 지점이 필요해서 bbox 중심으로는 안 된다. 검토한 방법:

1. **마스크 + PCA**로 주축·중심 추출 (Grounded-SAM류로 마스크 획득 후)
2. **포인팅 특화 VLM**(예: Molmo)에게 직접 키포인트 2개 찍게 하기
3. **depth/포인트클라우드 프리미티브 피팅** (원기둥/작은 박스 등) — 가장 정밀하지만 구현 부담 큼

### 1차 검증 실험 계획 (착수 예정, 미시작)

아이작심에서 YOLO 대신 VLM을 붙여 "긴 막대→side, 서랍 손잡이→pinch"
그립계열 판단이 되는지 검증. **의도적으로 좌표추출 정확도 문제와
분리**한다 — 이번 실험은 VLM의 grip_family 판단(라벨→그립종류)만
검증하고, 좌표/축은 아이작심 ground truth pose를 그대로 사용한다.
두 문제(판단 정확도 vs 좌표추출 정확도)를 한 실험에 섞으면 실패 시
원인을 특정할 수 없기 때문.

파인튜닝은 처음부터 필요 없다고 판단 — grip_family/라벨 판단은
프롬프트+few-shot으로 우선 시도, 정밀 숫자값(좌표/azimuth)만 필요시
LoRA 등 경량 파인튜닝 고려.

## 이력

- 2026-08-18: VLA/VLM 도입 아키텍처 논의 (채팅 세션). 하이브리드 구조
  합의, grip_family(side/pinch) 확장 시 pitch 연속보간 대신 family별
  개별 실측 검증 원칙 확정, azimuth_deg 파라미터 개념 도출, 비박스
  물체 좌표추출 3가지 방법 검토, 1차 Isaac Sim 검증 실험 설계(변수
  분리 원칙). **코드 구현은 아직 없음.**

## 폐기된 접근 / 하지 말 것

- **VLA를 저수준 연속 제어기(관절 액션 직접 출력)로 쓰는 것** — 이
  프로젝트가 이번 세션 전체(sim2real IK 버그)에 걸쳐 쌓은 quaternion
  단위 정밀 검증/디버깅 가능성을 포기하는 방향이라 배제. VLA/VLM은
  perception·판단 레이어에서만 쓰고, 실행은 항상 기존 코드가 담당.
- **top-down(90°)~side(180°) 사이 pitch를 10도 단위 등으로 부드럽게
  보간 가능하다고 가정하는 것** — top-down의 안전구간(`pitch=90,
  roll∈[-90,0]`)조차 실측 전엔 "직관적으로 맞을 것 같던" 조합이
  NO_SOLUTION이었던 전례가 있어([[grasp_kinematics_ik]]), 중간 pitch
  값들의 IK 도달가능성·TCP offset 변화는 검증 전까지 전혀 알 수 없다.
  family별 이산 실측 스윕으로 대체.
- **VLM/VLA가 제안한 azimuth나 bbox 중심을 검증 없이 그대로 실행** —
  top-down의 `angle_deg%180` 후보 스냅 검증 없이 각도를 바로 썼다가
  x>0에서만 팔이 꼬였던 sim2real 사고와 같은 유형의 재발 위험. 반드시
  IK-reachable 후보로 스냅 후 실행.

## 관련 문서
- `.claude/skills/grasp-kinematics-design/SKILL.md` — grasp_kinematics.py
  수정 전 필독, 폐기된 접근 이력
- [[grasp_kinematics_ik]] — top-down 공식·TCP offset 상수 현재 상태,
  side 실측 미검증 근거
- [[mcp_pickplace_architecture]] — 지금 MCP tool 경계(pick_object 등),
  VLA/VLM 출력이 꽂힐 자리
- [[perception_dev_tools]] — `vlm_boxyolo.py` 현재 구조, VLM으로
  교체/확장 시 건드릴 지점
