# MCP 로봇 서버 & Pick/Place 오케스트레이션

대상 파일: `sj_pickplace/mcp_robot_server.py`, `sj_pickplace/planning_node.py`,
`sj_pickplace/placement_verification.py`, `sj_pickplace/task_planner.py`
([[repo_layout]] 참고 — 전부 `ros2_ws/src/nero_sj_pickplace`에만 존재)

## 요약
LLM 에이전트가 호출하는 MCP 툴(`pick_object`, `place_object`,
`scan_for_boxes` 등)은 `mcp_robot_server.py`가 노출하고, 실제 모션
플래닝/실행은 `planning_node.py`가 ROS2 노드로 담당한다. 두 프로세스는
직접 함수 호출이 아니라 **ROS2 토픽 기반 요청/응답 패턴**으로 통신한다.

## 아키텍처
- `mcp_robot_server.py`가 `/arm_command` 토픽에 JSON(`std_msgs/String`)
  요청을 publish하고, `planning_node.py`가 처리 후 `/pick_result`에
  결과를 publish한다. 매칭은 `threading.Event` + 타임아웃(`wait_for_result`)
  방식 — gRPC/서비스콜이 아니라 pub/sub로 요청-응답을 흉내낸 구조라는
  점이 특이하다.
- 인지 데이터(`/detected_objects`, best-effort QoS)는 단방향으로
  `mcp_robot_server`(list_detected_objects용 캐시)와 `planning_node`
  양쪽에 흘러들어간다.
- `placement_verification.py`는 ROS 의존성이 전혀 없는 순수 함수
  모듈(`verify_placement`) — `planning_node._try_place_at`가 같은
  프로세스/스레드에서 직접 호출한다.

## MCP 툴 목록 (mcp_robot_server.py)
| 툴 | 비고 |
|---|---|
| `list_detected_objects()` | `/detected_objects` 캐시 즉시 반환, ROS 왕복 없음. center_3d는 바닥이 아니라 기하학적 중심 |
| `get_joint_positions()` | `/joint_states` 캐시 |
| `save_pose(name)` / `list_saved_poses()` / `move_to_saved_pose(name)` | JSON 파일 기반 포즈 저장 |
| `scan_for_boxes(target_label)` | joint1 스윕, 서버측에 결과 캐시. timeout 60s |
| `get_scanned_boxes()` | 마지막 스캔 캐시 즉시 반환 |
| `pick_object(target_label, grasp_dir, from_scan, box_index)` | timeout 75s. `remaining_scanned_boxes` 반환 |
| `place_object(x,y,z, grasp_dir)` | timeout 60s. `place_pos`/`requested_place_pos`/`placement_verified`/`verification_reason` 반환 |
| `stack_boxes(box_indices, base_x/y/z, ..., allow_reorder=False)` | 서버측에서 pick→place 루프를 한 번의 MCP 호출로 수행. 기본(`allow_reorder=False`)은 실패/verified=False 시 조기 중단("partial"). `allow_reorder=True`면 pick 실패 시 다른 스캔된 박스로 대체를 시도(최대 2회, `task_planner.py`) — 자세한 내용은 아래 "stack_boxes 백트래킹" 참고 |
| `move_to_position` / `move_joints` / `move_joints_relative` / `go_home` / `get_system_status` | 기본 모션/상태 툴 |

`box_index`, `place_pos` 자동보정, `placement_verified`/`go_home` 직후
정착시간 등 **운영 규칙은 CLAUDE.md에 이미 확정돼 있으므로 여기서
반복하지 않는다** — 아래는 그 규칙들이 코드 레벨에서 왜/어떻게
구현됐는지에 대한 보충 설명만 다룬다.

## 주요 상수
- `mcp_robot_server.py`: `TIMEOUT_PICK=75s`, `TIMEOUT_PLACE=60s`,
  `TIMEOUT_MOVE=11s`, `TIMEOUT_HOME=16s`, `TIMEOUT_SCAN=60s`,
  `TIMEOUT_STACK_PER_BOX=90s` — 2026-07에 실제 타임아웃으로 인한
  거짓-실패/재시도 루프가 있어서 상향 조정됨. `DEFAULT_BOX_HEIGHT_M=0.05`
  (CLAUDE.md 확정값과 동일 소스).
- `planning_node.py`: `APPROACH_Z=0.10`, `LIFT_Z=0.10`,
  `PLACE_DROP_Z=0.01`, `DESCEND_MIN_FINGERTIP_Z=0.03`,
  `MIN_REACH_R_M=0.20`(도달범위 안쪽 데드존), `BOUNDARY_R_M=0.30`,
  `JOINT_JUMP_MAX_DEG_PER_SEC=150`(비상정지 트리거),
  `BEARING_OFFSET_DEG=188.0`(카메라-베이스 베어링 실측 오프셋 — 2026-08-17
  이전엔 접근 전 joint1 사전회전에도 썼으나 그 용도는 제거됨, 지금은
  placement 검증 재확인 시 "스캔 자세 재사용" 로직에서만 사용),
  `SCAN_DEDUP_DIST_M=0.05`.
- `placement_verification.py`: `DEFAULT_XY_TOL_M=0.035`,
  `DEFAULT_Z_TOL_M=0.04`(z 노이즈 최대 26mm 관측 때문에 느슨하게 설정
  — CLAUDE.md의 "z 오차 관대하게 판정" 규칙의 근거), `MAX_STALE_SEC=2.0`.
- `task_planner.py`: `DEFAULT_MAX_BACKTRACK_ATTEMPTS=2` — `stack_boxes`
  호출 전체에서 공유되는 예산(tier별이 아님). `TIMEOUT_STACK_PER_BOX`가
  박스당 이미 빠듯해서 예산을 낮게 잡음; `mcp_robot_server.py`가 이
  상수를 직접 import해서 `allow_reorder=True`일 때 타임아웃 계산에
  반영한다 (두 곳에 값을 따로 하드코딩하면 나중에 어긋날 위험이 있어
  단일 소스로 유지).

## 코드 레벨 비하인드 (재발 방지 이력)
- **pipeline_id/planner_id 리셋 누락 버그**: 직전 이동이 Pilz(PTP/LIN)
  플래너를 썼으면 그 설정이 남아서, 이후 joint-space
  `move_to_configuration` 호출이 별도 리셋(`pipeline_id=''`) 없이는
  조용히 ABORT됨. scan_box/home/move_joints/pre-rotate 등 여러 지점에
  개별적으로 픽스가 들어감 — 이 패턴의 새 호출부를 추가할 때 리셋을
  빠뜨리기 쉬우니 주의.
- **좌표 자동 시프트**: place 실패 시 원좌표 → y+0.05 → y-0.05 → (z가
  0.20m 넘으면) z를 0.05/0.10/0.15씩 낮춤 순서로 재시도. 2026-07-30~
  08-03 실측에서 한쪽 방향 시프트만으로는 불충분해서 이렇게 확장됨.
- **홈 복귀 후 정착 지연(1.5s)**: CLAUDE.md의 "go_home 직후 정착시간"
  규칙이 실제로는 이 하드코딩된 1.5s sleep으로 구현돼 있음. 상태값은
  정상으로 보이는데도 즉시 pick하면 타임아웃나는 현상 관찰 후 추가.
- **궤적 타당성 검사**(`_is_trajectory_reasonable`): OMPL이 직선거리
  대비 1.8배 넘게 돌아가거나 orientation 궤적이 3배 넘게 우회하면
  재계획 — descend 중 그리퍼가 뒤집히는 현상이 position-only 체크는
  통과해서 추가됨.
- **grasp yaw 방식 변경**: 접근 중간에 joint5를 돌려 yaw만 트는 방식은
  엘보 플립(~20도 조인트 점프)을 유발해 폐기, 현재는 pick 시퀀스
  시작 전에 `(roll=-angle, pitch=90, yaw=0)`을 한 번에 고정.
- **place 시 각도 미정렬**: 2026-08부터 place quat는 박스 각도에 안
  맞추고 angle=0 고정 — "회전+하강 동시 실패(넘어짐)"를 피하기 위한
  트레이드오프로 의도적으로 채택됨.
- **joint1 사전회전(pre-rotate) 완전 제거 (2026-08-17)**: `_do_pick`
  진입 시 approach 전에 joint1만 먼저 물체 방향(대략, `BEARING_OFFSET_DEG`
  보정 적용)으로 돌려놓던 로직(2026-07 도입, "물체 방향까지 큰 차이를
  approach와 동시에 풀면 IK가 어려워진다"는 가설)을 코드에서 완전히
  삭제. [[grasp_kinematics_ik]]의 그립각도 근본 수정(sim의
  `position_yaw` 반영/180도 axis-flip 보정/`YawCandidateSelector` 4후보
  확장/`sim_box_aligned_quat` 분리) 이후 재테스트한 결과, pre-rotate가
  더 이상 필요 없고 오히려 없는 쪽이 더 안정적으로 확인됨(사용자 실측
  확인, 2026-08-17). 즉 pre-rotate는 애초에 "부정확한 그립 각도
  계산이 유발하는 IK 어려움"을 완화하는 우회책이었고, 근본 원인(그립
  각도 계산 버그)이 고쳐지자 우회책 자체가 불필요해진 사례 — 향후
  비슷하게 "증상 완화용 사전 조치"가 남아있는 다른 지점을 발견하면
  근본 원인이 이미 고쳐졌는지부터 의심해볼 것. `BEARING_OFFSET_DEG`
  상수 자체는 삭제하지 않음(placement 검증 재확인 로직이 별도로 사용
  중 — 위 "주요 상수" 참고).
- **`move_joints`가 미지정 관절을 0으로 리셋하는 버그 (2026-08-11
  수정)**: `_move_joints_sequence`(`planning_node.py`)는 지정 안 한
  관절의 "현재값 유지"를 위해 `self.latest_joint_state`를 읽었는데,
  이 속성은 **실물 전용 토픽(`/feedback/joint_states`)**에서만
  채워짐(콜백 `_on_joint_state_safety_check`가 안전 조인트 급회전
  감지용으로만 구독). 시뮬레이션(`use_moveit2:=true`)에서는 이게
  항상 `None`이라 `except Exception: positions = [0.0] * 7` 폴백으로
  떨어져서, **관절 1개만 지정해서 `move_joints`를 호출해도 나머지
  전부가 0으로 리셋**되는 버그가 실측 확인됨(문서화된 "미지정 관절은
  현재 각도 유지"와 정반대 동작). 코드베이스 다른 모든 곳(`_do_pick`,
  IK 체크 등)은 이미 `self.latest_joint_state_sim`(`/joint_states`
  구독 — sim 전용이 아니라, 실물에서도 `start_nero.sh`의
  `ros2 run topic_tools relay /feedback/joint_states /joint_states`로
  값이 채워지는 실물+sim 공용 속성)을 쓰고 있어서
  `_move_joints_sequence` 한 곳만 잘못된 속성을 참조하고 있었음.
  수정: `self.latest_joint_state` → `self.latest_joint_state_sim`으로
  교체. `colcon build` 후 재현 테스트(관절 1개만 지정 → 나머지 유지
  확인)로 검증.

## stack_boxes 백트래킹 (task_planner.py, 2026-08 추가)
기존 `_stack_sequence`는 어느 tier든 pick/place가 실패/거부되거나
`placement_verified is False`면 무조건 전체 스택을 중단했다(비효율적
이라는 문제의식). `task_planner.run_stack_plan`이 이 로직을 대체하되,
**pick 실패와 place 실패를 비대칭으로 취급**한다:

- **pick 실패는 후보(어떤 박스)에 좌우되는 문제** — 그래서
  `allow_reorder=True`일 때만, 요청한 박스의 pick이 실패/거부되면
  같은 라벨의 다른 스캔된 박스(`fallback_pool`)로 대체를 시도한다.
  후보 랭킹은 (1) top-down 우선(side보다), (2) 도달범위 밴드
  (`BOUNDARY_R_M` 밖 편안 > 경계 링 > `MIN_REACH_R_M` 안 데드존),
  (3) confidence 순. side 그립 후보는 `side_reachability_check`로
  먼저 걸러서(무료, 예산 소모 없음) 확정 실패가 뻔한 시도를 안 함.
  대체는 전체 스택 호출에 공유되는 `DEFAULT_MAX_BACKTRACK_ATTEMPTS=2`
  예산 안에서만 허용 — 무한 재시도로 `TIMEOUT_STACK_PER_BOX` 예산을
  잡아먹지 않기 위함.
- **place 실패는 tier의 목표 좌표(`base_pos + tier*box_height_m`)
  문제라 어떤 박스를 들고 있어도 해결 안 됨** — 그래서 place 단계에는
  후보 교체 로직이 아예 없고, `SequenceRejected`/`Exception`(place
  자체가 물리적으로 실패)은 `allow_reorder` 값과 무관하게 지금까지와
  동일하게 무조건 중단한다. **이 비대칭이 설계의 핵심이며, place
  실패에도 후보 교체를 "완성"하려는 시도는 하지 말 것** —
  `task_planner.py` 코드 주석에도 명시돼 있음.
- **[2026-08-17 변경] `placement_verified is False`는 더 이상 중단
  사유가 아니다.** 원래는 "카메라로 재확인했더니 목표에 없음/비스듬함"
  이면 불안정한 위에 계속 쌓지 않도록 여기서도 무조건 중단했는데,
  사용자 판단으로 이 안전장치를 제거함 — 이제 `tiers[]`에 기록만 하고
  다음 tier로 계속 진행한다(`planning_node.py`가 경고 로그만 남김).
  **불안정한 스택 위에 계속 쌓일 위험이 실제로 있으므로, 호출부는
  `tiers[].placement_verified`를 반드시 확인해야 한다** — CLAUDE.md의
  placement_verified 규칙(개별 place_object 호출에 대한 에이전트
  행동 지침)은 이 변경과 별개로 여전히 유효.
- 요청한 박스가 대체됐으면 응답 `tiers[].box_used`/`substituted`,
  `skipped_candidates`로 확인 가능 — CLAUDE.md의 box_index 규칙에
  이 캐비엇이 추가돼 있음.
- 기본값은 `allow_reorder=False`로, 기존 호출자와 완전히 동일한 동작을
  보장한다(하위호환) — `fallback_pool`이 있어도 완전히 무시됨.

## placement_verification 상세
`verify_placement`는 place 전 스냅샷(`before_objects`)에서 기존 물체와
2cm 이내로 겹치는 감지는 필터링(쌓기 작업 시 아래 박스를 "방금 놓은
물체"로 오인하는 것 방지)한 뒤, 새로 나타난 후보 중 가장 가까운 것과
비교해 xy≤3.5cm, z≤4cm면 True. **알려진 미해결 한계: 박스가 넘어져도
xy/z만 맞으면 True로 오판할 수 있음(orientation 미검사, 의도적으로
범위 밖으로 둠)** — 중요 작업에서는 이 점 감안.

- **급회전 안전정지 임시 비활성화 (2026-08-14, 미검증)**: 스캔 중 joint1이
  빠르게 스윕할 때 `JOINT_JUMP_MAX_DEG_PER_SEC` 임계값을 초과해 emergency
  stop이 계속 트리거됐다. 임시 조치로 `planning_node.py`의 안전정지 로직을
  주석처리(`#`)했고, 대신 최대 속도를 `0.3 → 0.15`로 절반 줄임.
  **안전 기능이 꺼진 상태이므로 실물 운영 시 주의 필요.**
  장기적으로는 스캔 경로를 느리게 하거나 임계값을 구분해서 재활성화해야 함.

- **`from_scan` 시 align 각도 재조회 (2026-08-14)**: 기존에는 `from_scan=True`
  일 때 `skip_reacquire=True`로 스캔 시 측정한 각도를 그대로 썼다. 스캔
  거리에서 측정한 각도가 부정확하다는 실측 확인 후 `skip_reacquire=False`로
  변경 — align 시점(박스 바로 위)에서 각도를 재조회하고, perception이 실패할
  경우에만 스캔 각도로 fallback.

- **`_abs_angle_to_quat` 추가 (2026-08-14)**: `perception`의 `angle_base_deg`
  는 base_link 절대각인데, `_top_down_quat_for`는 position_yaw 기준 상대각을
  입력으로 기대했다. place/move 시퀀스에서 절대각을 직접 넘기던 호출부를
  `_abs_angle_to_quat`로 교체 — 내부에서 `atan2(y,x)` 변환 후 `top_down_angle_quat` 호출.

## 미완성/취약 지점
- 박스 높이 전역 고정값(0.05m) — 실측 깊이 기반 측정은 아직 TODO.
- z 노이즈(최대 26mm)는 완화가 아니라 관용치로 우회 중 — 미디언 필터링
  계획은 미구현.
- 실물 로봇 경로(`AgxArmStatus` import 실패 시)는 도착 확인 없이
  성공을 반환하는 폴백이 있음 — 실물 로봇 디버깅 시 이 폴백이 활성화돼
  있는지 먼저 확인할 것.
- 최근 코드리뷰 수정 다수가 `use_moveit2` 분기에만 적용돼 있어 실물
  하드웨어 경로가 상대적으로 덜 검증됨.

## 관련 문서
- [[repo_layout]]
- [[perception_calibration]] — `/detected_objects` 생산 측
- CLAUDE.md의 box_index, place_pos 자동보정, placement_verified,
  go_home 정착시간 규칙 (이 문서와 세트로 볼 것)
