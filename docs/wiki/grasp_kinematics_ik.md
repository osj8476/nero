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
  컨벤션 차이), `sim_box_aligned_quat`(sim 전용, `sim_top_down_angle_quat`
  과 달리 `position_yaw` 미포함 — "자유 twist 자세 최적화"용과 "실제
  박스각도 정렬"용을 헷갈리면 안 됨, 아래 이력 4차 수정 참고),
  `side_quat_for`, `side_reachability_check`,
  `auto_grasp_quat`/`resolve_grasp_quat`(grasp_dir 문자열 또는
  라벨/좌표 휴리스틱을 quat으로 해석 — 예전엔 pick/place/move 여러
  분기에 중복돼 있던 로직을 여기로 통합), `YawCandidateSelector`
  (box mod-90 × 그리퍼 mod-180 대칭을 조합한 4후보 중 IK 도달 가능하면서
  **FK 기준 기하학적 회전량이 더 작은** 쪽을 선택 — IK 조인트공간
  이동량 기준이 아님에 주의). `planning_node.py`에는 이 두 quat 함수를
  각각 감싸는 `_top_down_quat_for`(자유 twist)/`_box_aligned_quat_for`
  (박스 정렬) 디스패처가 있다.

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

## 이력
- 2026-08-15: `sim_top_down_angle_quat`(당시 `angle_deg`만 받아 yaw로 씀,
  물체 위치와 무관)이 x>0 근접 영역에서 IK가 팔을 안쪽으로 꼬아넣는
  branch를 고르는 원인으로 의심되어 두 단계로 수정.
  1. **1차 수정 (위치추종 yaw 도입)**: 실물 `top_down_angle_quat`이
     `yaw=position_yaw(=atan2(y,x))`로 위치에 고정되는 것과 같은
     메커니즘을 sim에도 적용 — `sim_top_down_angle_quat(x,y,angle_deg)`로
     시그니처 확장, `yaw=position_yaw+angle_deg`. `pitch=180`은 sim IK에서
     이미 검증된 값이라 유지(실물의 `pitch=90` 구조를 그대로 옮기지
     않음 — sim IK 백엔드에서 미검증이라 별도 판단 필요, 상세 근거는
     [[grasp_kinematics_ik]] 상단 SKILL.md 링크 참고).
     **주의**: 이 1차 수정을 배포하고 나서도 재현 테스트에서 문제가
     그대로 남아있었는데, 원인은 이 수정 자체가 아니라 `colcon build`를
     안 돌려서 `install/`이 `src/` 변경을 반영 못 한 것이었다(구버전
     바이너리가 계속 실행됨 — 로그의 approach quat이 서로 다른 물체
     위치에서 전부 동일하게 찍히는 것으로 확진). **`grasp_kinematics.py`/
     `planning_node.py`를 고친 뒤에는 `colcon build --packages-select
     sj_pickplace`로 재빌드하고 `planning_node` 프로세스를 재시작해야
     반영된다** — 안 그러면 "고쳤는데도 재현된다"는 오판을 하게 된다.
  2. **2차 수정 (+180° 보정, 근본 원인)**: 재빌드 후에도 x>0에서 꼬임이
     실측 재현됨. 원인 규명: `euler_to_quat(roll=0, pitch=180°, yaw)`는
     `Ry(180°)`를 yaw 회전보다 먼저 적용하는 합성이라(`R=Rz(yaw)·Ry(180°)·
     Rx(0)`), 로컬 기준축이 `Ry(180°)`에서 이미 반전된다(`(x,y,z)→
     (-x,y,-z)`). 그 결과 그리퍼가 실제로 향하는 수평 방향은 요청한
     `yaw`가 아니라 `yaw+180°`가 되어, `position_yaw`를 그대로 넣으면
     오히려 물체 반대방향(로봇 쪽으로 접히는 방향)을 향하도록 명령한
     셈이었다. 최종 수정: `yaw = position_yaw + angle_deg + π`.
  **검증 방법**: `/arm_command` 토픽으로 `override_pos`/`override_angle_deg`를
  직접 넣어 특정 (x,y)를 반복 재현(예: `x=0.3,y=0.2,angle=30`), 로그의
  `[top:sim(pitch180)] approach는 yaw 자유 -> quat=...` 값을 위 공식으로
  손으로 검산해 일치 확인 + 실제 pick 실행 후 육안으로 팔 자세 확인.
  x>0 영역 여러 지점(0.229,0.246 등)에서 재현 테스트 통과, 육안으로도
  자연스러운 자세 확인됨(2026-08-15, sim 한정 검증).
  **미검증/한계**: 실물 백엔드(`top_down_angle_quat`)는 이번 수정
  대상이 아니었고 원래부터 문제 없었음(별도 pitch=90 컨벤션이라
  이 axis-flip 자체가 발생하지 않음 — 위 SKILL.md 링크의 실측 근거
  참고). 또한 이번 수정은 "그리퍼가 향하는 수평 방향"만 바로잡은
  것이고, 7-DOF redundancy로 인한 IK branch 선택 자체(팔꿈치가 어느
  쪽으로 접히는지)를 제어하는 것은 아니다 — `_compute_ik_joints()`가
  이미 `robot_state.joint_state`로 현재 관절상태를 시드로 넣고 있지만
  (`planning_node.py:1449-1451`), 이건 `YawCandidateSelector`의
  "도달가능한가" 체크에만 쓰이고 실제 branch 품질 비교에는 아직
  안 쓰인다 — 향후 유사 증상 재발 시 이 지점부터 볼 것.
  3. **3차 수정 (`YawCandidateSelector` 2후보→4후보 확장)**: 위 수정
     배포 후 approach는 x>0에서 안정적이었으나, align에서 여전히 큰
     폭(약 90~170도)으로 그리퍼가 도는 사고가 별도로 재현됨. 로그 확진:
     박스(0.039,-0.476), 인식각도=81.1도 케이스에서 mod-90 두 후보
     (81.1도, 171.1도) 중 "가까운"(cost 작은) 81.1도가 IK NO_SOLUTION으로
     거부되어, 유일하게 성공한 171.1도(cost=171.1도, 사실상 180도
     반대)로 어쩔 수 없이 선택됨. 원인: 박스 mod-90 대칭(짧은변/긴변)과
     그리퍼 자체의 mod-180 대칭(좌우 대칭 그리퍼라 손끝 반대로 잡아도
     물리적으로 같은 그립)은 서로 다른 대칭인데 후보군이 mod-90만
     반영하고 있었음 — `quat_angle_diff`가 이미 [0,180도] wrap 처리를
     하므로, 171.1도의 mod-180 짝인 351.1도(=171.1+180)는 cost가
     8.9도(=360-351.1)로 훨씬 작았는데 애초에 후보에 없어 시도조차
     못 됨. `candidates_deg`를 `[base, base+90, base+180, base+270]`
     4개로 확장(기존 2개는 부분집합이라 순수 추가)하고, 선택된 cost가
     `LARGE_REORIENT_WARN_DEG=90`도를 넘으면 경고 로그를 남기도록 추가.
     동일 케이스 재현 테스트: 351.1도(cost 8.9도)가 정상 선택되어
     approach 대비 큰 회전 없이 해결 확인(2026-08-15, sim).
  4. **4차 수정 (align/descend/place의 `position_yaw` 오염 제거, 가장
     근본적인 버그)**: 3차 수정 이후 사용자가 "그리퍼가 박스 면이 아니라
     45도 돌아간 모서리를 잡는다"는 별개 증상을 보고. 원인 규명: 1차
     수정에서 만든 `sim_top_down_angle_quat(x,y,angle_deg)`의
     `yaw=position_yaw+angle_deg+180`은 approach(`angle_deg=0` 고정,
     "yaw 자유" — 아무 twist나 자세만 좋으면 됨)에는 맞는 공식이지만,
     align은 `angle_deg` 자리에 **perception이 tf로 base_link 기준
     절대각으로 이미 변환해 보낸 실제 박스 edge 각도**가 들어온다. 여기에
     `position_yaw`(물체 방향 bearing — 박스의 실제 회전과는 무관한 값)를
     또 더하면, 최종 그리퍼 방향이 정확히 `position_yaw`만큼 박스 edge와
     어긋난다. 실측 확진: 박스(0.233,0.250), 인식각도=12.9도 케이스에서
     `position_yaw=atan2(0.250,0.233)=47.0도`를 역산하니 사용자가 육안
     보고한 "약 45도 어긋남"과 정확히 일치. 실물 백엔드(`top_down_angle_quat`)
     는 애초에 이 문제가 없음 — `roll=-angle_deg`(박스정렬)와
     `yaw=position_yaw`(자세)가 서로 다른 Euler 축이라 안 섞임. sim만
     yaw 하나에 두 역할(자유 twist 자세 최적화 vs 박스 정렬)을 얹어서
     생긴 sim 전용 버그였다.
     같은 원인이 align(`_pick_best_yaw_candidate`)뿐 아니라, align 완료 후
     px/py가 갱신되면서 descend용 quat을 재계산하는 지점
     (`planning_node.py:_do_pick`, `_current_top_angle_deg` 사용)과,
     place/move가 `self._box_angle_deg`(직전 pick 때의 실제 박스각)로
     기본 자세를 잡는 지점(`planning_node.py:397,423`) 세 곳 모두에
     있었다 — align만 고치면 descend에서 다시 틀어진다. 이 세 곳은
     이번 세션 내내 관찰된 stack 검증 실패("z가 XXmm 어긋남, 비스듬히
     놓였거나 다른 물체 위에 걸쳤을 가능성")의 근본 원인이었을 가능성이
     높다(박스가 삐딱하게 놓이면 z 오차가 크게 나는 것과 정합).
     수정: `grasp_kinematics.py`에 `sim_box_aligned_quat(angle_deg)`
     신설(`position_yaw` 미포함, 순수 `euler(0,pi,angle_deg+pi)`),
     `planning_node.py`에 `_box_aligned_quat_for(pos,angle_deg)` 디스패처
     신설(sim은 `sim_box_aligned_quat`, real은 기존 `top_down_angle_quat`
     그대로 — real은 원래 안전). align 후보 콜백, descend 재계산, place/move
     기본자세 세 지점을 `_top_down_quat_for`(자유 twist 전용, `position_yaw`
     필요)에서 `_box_aligned_quat_for`로 교체. approach(`angle_deg=0`)와
     place entry_quat(`0.0`)은 자유 twist 자세 최적화가 목적이라
     `_top_down_quat_for` 그대로 유지 — 이 둘을 혼동해서 바꾸면 1차
     수정 이전의 원래 버그(팔 꼬임)가 재발한다.
     재현 테스트: 동일 박스(0.233→0.227,0.249) 재스캔+pick, 선택각도가
     인식각 그대로(12.9도→다음 스캔에서 62.8도) 나오는 것과 cost가
     `|position_yaw-candidate|` 공식과 일치하는 것을 로그로 확인, 육안으로
     "면을 정확히 잡음" 확인(2026-08-15, sim).
  **남은 미검증 항목**: 이 4차 수정이 이번 세션에서 반복 관찰된 stack
  tier 검증 실패를 실제로 줄이는지는 아직 별도로 재현 테스트 안 함 —
  다음에 stack_boxes를 다시 쓸 때 placement_verified 결과를 눈여겨볼 것.
  5. **후속: joint1 사전회전(pre-rotate) 제거 (2026-08-17)**: 위 1~4차
     수정으로 그립 각도 계산 자체가 근본적으로 고쳐지고 나니, 2026-07에
     "IK 부담 완화용"으로 도입했던 `_do_pick`의 pre-rotate 우회책이
     더 이상 필요 없고 오히려 없는 쪽이 더 안정적으로 확인됨(실측).
     상세(제거 범위, 남겨둔 `BEARING_OFFSET_DEG` 등)는
     [[mcp_pickplace_architecture]]의 "코드 레벨 비하인드" 참고 — 이
     문서(그립 각도 계산)가 아니라 그쪽(pick 시퀀스 오케스트레이션)
     책임이라 본문은 거기 있음.

## 관련 문서
- `.claude/skills/grasp-kinematics-design/SKILL.md` — 설계 이력,
  폐기된 접근(joint5 중간 yaw 트위스트, 런타임 ACM 조작 등), 채택된
  `(roll=-angle, pitch=90, yaw=0)` top-down 컨벤션의 근거
- [[mcp_pickplace_architecture]] — 실제 IK/모션 실행은 planning_node.py
- [[repo_layout]] — 이 도구들이 `nero/`에만 있고 `ros2_ws`엔 없는 이유
