# 그립 각도 선택(grasp_kinematics.py) & IK 도달범위 스캔 도구

대상 파일: `sj_pickplace/grasp_kinematics.py`,
`nero/tools/ik_top_scan.py`, `nero/tools/ik_full_scan.py`,
`nero/tools/ik_boundary_scan_dense.py`,
`nero/tools/ik_side_reachable_map.txt`

> 설계 이력/폐기된 접근은 이미
> `.claude/skills/grasp-kinematics-design/SKILL.md`에 정리돼 있으므로
> 여기서는 중복 서술하지 않고 요지만 링크한다. `grasp_kinematics.py`를
> 고칠 때는 **이 문서가 아니라 그 SKILL.md를 먼저 읽을 것** (CLAUDE.md
> 지시사항).

## 요약
`grasp_kinematics.py`는 "어떤 자세로 잡을까"만 순수 함수로 결정하는
모듈(ROS 의존 없음) — IK/FK 호출과 모션 실행은 `planning_node.py`에
남아있다. `nero/tools/ik_*.py` 세 스크립트는 이 결정 로직과 무관하게
독립적으로 로봇의 side-grip 도달 범위를 오프라인으로 스캔한 일회성
측정 도구다.

## 구조
- `euler_to_quat`, `quat_angle_diff`(쿼터니언 부호 등가성 처리),
  `top_down_angle_quat`/`sim_top_down_angle_quat`(실물/시뮬 백엔드
  컨벤션 차이), `side_quat_for`, `side_reachability_check`,
  `auto_grasp_quat`/`resolve_grasp_quat`(grasp_dir 문자열 또는
  라벨/좌표 휴리스틱을 quat으로 해석 — 예전엔 pick/place/move 여러
  분기에 중복돼 있던 로직을 여기로 통합), `YawCandidateSelector`
  (mod-90 대칭 yaw 후보 중 IK 도달 가능하면서 **FK 기준 기하학적 회전량이
  더 작은** 쪽을 선택 — IK 조인트공간 이동량 기준이 아님에 주의).

## TCP offset 상수 위치 정정
CLAUDE.md는 `TOP_TCP_OFFSET`/`SIDE_TCP_OFFSET` 둘 다
`grasp_kinematics.py`에 있다고 적혀 있었으나, 실제로는:
- `SIDE_TCP_OFFSET` (0.1358m) → `grasp_kinematics.py:78`
- `TOP_TCP_OFFSET` (0.1358m, `GRIPPER_FLANGE_TO_FINGERTIP`과 동일) →
  `planning_node.py:118`

둘 다 같은 파일에 있는 게 아니라 **파일은 다르고 값만 같다**. 두 상수를
하나로 합치면 안 된다는 CLAUDE.md의 핵심 규칙 자체는 여전히 유효 —
파일 위치 서술만 부정확했던 것이라 CLAUDE.md 쪽도 같이 정정해둔다.

`SIDE_TCP_OFFSET`은 `grasp_kinematics.py:78` 주석에 "top과 동일
실측값 적용 (side 별도 실측 전까지, 2026-08 기준 아직 검증 대기)"라고
명시돼 있음 — CLAUDE.md의 "side 실측 미검증" 규칙과 일치하는 코드
근거.

`SIDE_MIN_DIST = 0.32`(`grasp_kinematics.py:76`)는 이 거리 미만이면
IK 시도 없이 즉시 거부하는 컷오프 — 아래 IK 스캔 결과를 사람이 보고
수동으로 뽑아낸 요약값으로 추정되며(코드상 파일을 직접 읽어들이는
연결고리는 없음), 스캔을 다시 돌리게 되면 이 값도 재검토 대상.

## IK 도달범위 스캔 도구 (`nero/tools/`)
모두 독립 실행 스크립트(rclpy/pymoveit2)이며 **런타임 파이프라인에
연결돼 있지 않다** — import되는 곳이 코드베이스 어디에도 없음.
- `ik_top_scan.py`: top-down quat 고정, z/일부 (x,y) 조합만 수동 스윕,
  stdout에 OK/FAIL만 출력. 파일 저장 없음.
- `ik_full_scan.py`: side-grip(`roll=0,pitch=90,yaw=atan2(y,x)`) 630점
  거친 격자 스캔, `/tmp/ik_scan_result.csv`로 출력.
- `ik_boundary_scan_dense.py`: side-grip 0.02m 해상도 조밀 스캔
  (x∈[-0.45,0.45], y∈[-0.40,0.40], z 3레벨), 베이스 근접 특이점
  영역은 스킵. `/home/bpdl/sj/ik_side_reachable_map.txt`에 행 단위로
  즉시 flush하며 기록 — 지금 `nero/tools/ik_side_reachable_map.txt`에
  있는 파일의 출처.

## `ik_side_reachable_map.txt`
`x,y,z,ok` 헤더 + 5598행(각 좌표별 True/False) + 요약 블록(OK
3323/59.4%, FAIL 2275, 소요 2125초). **이 파일을 읽어들이는 코드가
현재 없다** — `grasp_kinematics.py`, `planning_node.py` 어디에도
파일명 참조가 없음. 순수 오프라인 분석 산출물이며, 도달범위 로직을
바꾸려면 이 데이터를 사람이 다시 해석해서 상수(`SIDE_MIN_DIST` 등)로
수동 반영해야 한다는 뜻.

## 관련 문서
- `.claude/skills/grasp-kinematics-design/SKILL.md` — 설계 이력,
  폐기된 접근(joint5 중간 yaw 트위스트, 런타임 ACM 조작 등), 채택된
  `(roll=-angle, pitch=90, yaw=0)` top-down 컨벤션의 근거
- [[mcp_pickplace_architecture]] — 실제 IK/모션 실행은 planning_node.py
- [[repo_layout]] — 이 도구들이 `nero/`에만 있고 `ros2_ws`엔 없는 이유
