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

## top_down_angle_quat 공식 & 절대각 변환 (2026-08-12)

### 공식
```python
# grasp_kinematics.py
def top_down_angle_quat(x, y, angle_deg):
    position_yaw = math.atan2(y, x)
    return euler_to_quat(math.radians(-angle_deg), math.radians(90), position_yaw)
```

`position_yaw = atan2(y, x)` 는 "로봇 베이스→물체 방향"을 yaw로 반영한다.
이 항이 없으면(yaw=0) 특정 위치(예: (-0.4, 0.1))에서 AGX IK solver가
NO_SOLUTION을 반환함 — IK 안정성을 위해 의도적으로 유지한다.

**왜 docstring은 `yaw=0`이 정답이라고 했나?** (2026-08 재검토)
docstring §채택된접근의 "pitch=90 후 roll 스윕 전 구간 IK 성공"은
근거리에서 검증한 결과로, 원거리(0.4m+)까지 성립하지 않는다.
실제로 `yaw=0` 변경 시 동일하게 NO_SOLUTION 발생 확인 — 공식 원복.

### 절대 좌표계 기준 각도 입력 (planning_node.py)

사용자(MCP 또는 pick 명령)는 절대 좌표계 각도(월드 프레임 기준)를 넘긴다.
`top_down_angle_quat`의 내부 기준은 radial(베이스→물체 방향)이므로,
호출 전에 변환이 필요하다:

```python
# planning_node.py — approach 후보 계산
_position_yaw_deg = math.degrees(math.atan2(pos['y'], pos['x']))
_angle_rel = (angle_deg_abs - _position_yaw_deg) % 180.0
_approach_angle_cands = [_angle_rel, (_angle_rel + 90.0) % 180.0]
```

**수학적 근거**: `euler(-angle_rel, 90°, position_yaw)` 의 jaw 방향(월드 프레임) =
`position_yaw + angle_rel` = `position_yaw + (angle_abs - position_yaw)` = `angle_abs` ✓

그리퍼가 절대 좌표 기준 정확히 요청한 각도를 향하면서,
position_yaw는 IK 안정성을 위해 보존된다.

### 실물 approach 다중 후보 + 폴백 (planning_node.py)

1. `angle_rel` 시도 → 성공이면 이 quat을 align/descend/lift 전 구간 유지
2. `angle_rel + 90°` 시도 (그리퍼 대칭)
3. 두 후보 모두 NO_SOLUTION → `QUAT_TOP_DOWN` 상수 폴백
   (시각적으로 top-down이 아닌 방향이지만 IK는 전 구간 성공)

approach에서 성공한 quat을 `_align_quat_override`로 align 단계에 그대로
전달해 방향 일관성을 유지한다 (re-compute하면 QUAT_TOP_DOWN 폴백이
덮어씌워지는 버그 있었음 — 2026-08 수정).

### 시뮬 vs 실물 frame 불일치

| | 공식 | pitch | 탑다운 축 |
|--|------|-------|----------|
| 시뮬(MoveIt2) | `euler(0, 180°, angle_deg)` | 180° | Z축 아래 |
| 실물(AGX) | `euler(-angle_rel, 90°, position_yaw)` | 90° | X축 아래 |

같은 Nero 7DOF 로봇이지만 gripper_flange 좌표계 정의가 90° 다르다.
시뮬 URDF(`nero_with_camera.urdf`)의 `gripper_flange_joint`가
`rpy="-1.5708 0 -1.5708"`로 정의돼 있어 발생한 차이.
실물에서 pitch=180° = QUAT_TOP_DOWN ≈ `[0,1,0,0]` 으로 그리퍼가
"옆으로 누운" 방향이 되는 이유.

### QUAT_TOP_DOWN 상수
`[0.008, 0.999, 0.023, 0.037]` — 실물에서 IK가 전 구간 성공하도록
실측 캘리브레이션된 고정 쿼터니언. 시뮬 공식 `euler(0, 180°, 0) ≈ [0,1,0,0]`
과 거의 동일. **시각적으로 top-down이 아닌 것처럼 보이지만**
실제로는 pick에 지장 없음 (arm이 -Z 방향으로 내려가므로).

## 관련 문서
- `.claude/skills/grasp-kinematics-design/SKILL.md` — 설계 이력,
  폐기된 접근(joint5 중간 yaw 트위스트, 런타임 ACM 조작 등)
- [[mcp_pickplace_architecture]] — 실제 IK/모션 실행은 planning_node.py
- [[repo_layout]] — 이 도구들이 `nero/`에만 있고 `ros2_ws`엔 없는 이유
