#!/usr/bin/env python3
"""
grasp_kinematics.py

[신설 — planning_node.py에서 그립 각도/자세 선택 로직만 분리]

기존 planning_node.py는 로봇 물리 실행(ROS2 액션, MoveIt2, 타이밍 재시도)과
"어떤 자세로 집을지" 결정 로직이 한 파일에 섞여 있었다. 이 모듈은 후자만
떼어낸 것으로, ROS2 의존성이 전혀 없는 순수 함수/상수로만 구성된다.
(pytest로 단위테스트 가능 — 실측 로그를 입력값으로 재생해서 검증할 수 있음)

책임 범위:
  - 물체 위치/라벨/사용자 지정값 -> 목표 쿼터니언 변환
  - top-down vs side 그립 중 어느 쪽이 적합한지 판단
  - side 그립의 도달가능성 사전 검사
  - 두 IK 후보 중 실제 회전량이 더 적은 쪽 선택 (yaw candidate 비교)

책임 밖 (planning_node.py에 남아있음):
  - IK/FK 서비스 호출 자체 (ROS2 서비스 클라이언트 필요)
  - 실제 이동 실행/재시도/타임아웃
  - approach/descend/lift 시퀀스 오케스트레이션

═══════════════════════════════════════════════════════════════════════════
## 폐기된 접근 — 재도입 금지 (planning_node.py 원본 주석에서 이관)
═══════════════════════════════════════════════════════════════════════════

1. [2026-07, 1차 시도, 폐기] approach 이후 joint5 값을 읽어 joint7 자리에
   넣어 "그리퍼 yaw만 살짝 돌리기"를 시도한 방식.
   → 직렬 링크 로봇은 조인트 하나만 움직여도 그 이후 링크 전체가 같이
   움직이므로 이 방식은 애초에 물리적으로 성립 불가능했다. 게다가 approach
   중간에 자세를 바꾸다보니 MoveIt2/펌웨어가 매번 새로 IK를 풀면서
   조인트1~4가 20도 이상 튀는 위험한 동작(엘보 플립)까지 관찰되어 폐기.

2. [2026-07, 복원 시도 후 재롤백] _seeded_ik / _best_yaw_solution /
   _move_joint_target(관절한계 게이트 + joint-goal 직접이동) 및
   _register_floor_collision(PlanningScene ACM 직접 조작) 조합.
   → project knowledge의 예전 아키텍처를 참고해 재도입했으나, approach
   자체가 전혀 안 움직이는 회귀가 발생함 (게이트가 후보를 전부 거부,
   ACM 직접 조작이 SRDF 기반 자기충돌 허용목록을 깨뜨림). 실측 검증된
   이전 상태로 전량 복원함. 바닥/자기충돌은 SRDF(disable_collisions)와
   joint_limits.yaml 레벨에서 처리하며, 런타임에 PlanningScene ACM을
   직접 재작성하지 않는다 — 이 원칙을 어기면 같은 회귀가 재발한다.

═══════════════════════════════════════════════════════════════════════════
## 채택된 접근 (현재 구현)
═══════════════════════════════════════════════════════════════════════════

- top-down 자세는 유지한 채 접근축 기준 yaw만 돌리려면, 요청 쿼터니언을
  (roll=-angle_deg, pitch=90°, yaw=0)로 구성해야 한다 (2026-07, 실물
  컨트롤러 orientation sweep 실측으로 검증됨: roll=180/pitch=0 조합은
  광범위하게 NO_SOLUTION, pitch=90 고정 후 roll 스윕은 전 구간 IK 성공,
  실측 yaw = 요청 roll - 90° 관계로 선형 대응).
  이 자세는 approach 진입 전 딱 한 번만 계산해서 pick 시퀀스
  (approach→descend→lift) 내내 동일하게 유지한다. 시퀀스 중간에 자세를
  바꾸는 로직이 없으므로 1번 문제(중간 전환 -> IK 분기 튐)가 구조적으로
  재발하지 않는다.

- 그립 각도 후보 비교는 IK 이동량이 아니라 FK 기반 순수 기하학적 회전각
  (_quat_angle_diff)으로 판단한다. 7DOF redundancy 때문에 IK 해가 어느
  branch로 나오는지가 우연에 좌우되어, 관절거리 cost는 실제 회전량과
  무관하게 나올 수 있기 때문이다 (2026-08 재설계).

═══════════════════════════════════════════════════════════════════════════
## 알려진 한계
═══════════════════════════════════════════════════════════════════════════

이 알고리즘은 두 후보 사이의 회전량만 비교하는 최소한의 안전장치다. 두
후보 자체가 perception이 인식한 각도에서 파생되므로, 각도 인식 자체가
틀리면 "둘 중 그나마 나은 선택"일 뿐 정답이 되지 않는다. 근본적인 인식
정확도는 이 모듈이 아니라 perception_node의 책임이다.
"""
import math


class SequenceRejected(RuntimeError):
    """pick/place가 물리적으로 도달불가로 사전 거부됨 (side reachability 등).
    일반 RuntimeError(실행 중 실패)와 구분해서 'failed' 대신 'rejected'로
    보고하기 위한 마커 클래스.
    [2026-08 이관] 원래 planning_node.py에 정의돼 있었으나, task_planner.py
    (ROS 비의존 순수 모듈)가 이 예외를 잡아야 해서 이곳으로 옮김
    (planning_node.py를 import하면 순환참조 발생). 동작/의미 변경 없음."""


# ── side 그립 전용 상수 (대규모 IK 그리드 전수조사 결과) ──────────────────
SIDE_MIN_DIST = 0.32     # 이 거리 미만이면 IK 시도 없이 즉시 거부
SIDE_PITCH_DEG = 90
SIDE_TCP_OFFSET = 0.1358  # 미터. top과 동일 실측값 적용 (side 별도 실측 전까지,
                          # 2026-08 기준 아직 검증 대기 — 변경 시 주의)

# ── top 그립 각도보정 전용 상수 ──────────────────────────────────────────
TOP_ANGLE_PITCH_DEG = 90.0  # 고정값 (실측상 항상 top-down으로 귀결됨)

# ── 사전 정의 쿼터니언 (xyzw, base_link 기준) ────────────────────────────
QUAT_TOP_DOWN    = [0.008,  0.999,  0.023,  0.037]
QUAT_TOP_DOWN_R  = [0.999, -0.008, -0.037,  0.023]
QUAT_TOP_DOWN_L  = [0.708,  0.697, -0.010,  0.043]
QUAT_TOP_DOWN_RR = [-0.643, 0.653,  0.041,  0.011]
QUAT_HOME        = [0.0, 0.0, 0.0, 1.0]
QUAT_SIDE_FRONT  = [0.481, -0.527,  0.427,  0.556]

# side 계열 식별용 마커 (쿼터니언 비교 대신 이 태그로 판별)
SIDE_TAG = 'DYNAMIC'

GRASP_DIR_MAP = {
    'top':         QUAT_TOP_DOWN,
    'top_down':    QUAT_TOP_DOWN,
    'top_down_r':  QUAT_TOP_DOWN_R,
    'top_down_l':  QUAT_TOP_DOWN_L,
    'top_down_rr': QUAT_TOP_DOWN_RR,
    'side':        SIDE_TAG,   # 물체 위치에 따라 _side_quat_for()로 동적 계산
    'side_front':  SIDE_TAG,
    'side_left':   SIDE_TAG,
    'side_right':  SIDE_TAG,
}

LABEL_GRASP_HINT = {
    'bottle':   'side_front',
    'cup':      'top_down',
    'book':     'side_front',
    'box':      'top_down',
    'ball':     'top_down',
    'scissors': 'side_front',
    'remote':   'top_down',
}


# ═══════════════════════════════════════════════════════════════════════
# 순수 기하 함수
# ═══════════════════════════════════════════════════════════════════════

def euler_to_quat(roll: float, pitch: float, yaw: float) -> list:
    """ZYX 오일러각(roll, pitch, yaw, 라디안) -> 쿼터니언(xyzw)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def quat_angle_diff(q1, q2) -> float:
    """두 쿼터니언(xyzw) 사이 각도 차이(라디안, 0~π).
    q와 -q가 같은 회전이므로 dot에 abs()를 적용한다."""
    dot = sum(a * b for a, b in zip(q1, q2))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2 * math.acos(dot)


def top_down_angle_quat(x: float, y: float, angle_deg: float) -> list:
    """top-down 유지(pitch=90) + 위치추종 yaw 반영 쿼터니언 (실물 백엔드).
    position_yaw = atan2(y,x) — AGX IK solver가 위치에 따라 다른 해를
    선택하므로, 이 yaw가 특정 위치에서 IK 성공률을 높일 수 있음."""
    position_yaw = math.atan2(y, x)
    return euler_to_quat(math.radians(-angle_deg), math.radians(90), position_yaw)


def sim_top_down_angle_quat(x: float, y: float, angle_deg: float) -> list:
    """sim(MoveIt2) 전용 top-down + 위치추종 yaw 반영 쿼터니언.

    [2026-08 수정, sim2real 정합] 기존엔 yaw=angle_deg(박스 절대각)만 써서
    물체 위치와 무관했다 — x>0처럼 팔이 이미 몸 가까이 뻗어있는 영역에서
    이 절대각이 자연스러운 reach 방향과 어긋나면, 7DOF redundancy 때문에
    IK가 여분 관절을 뒤틀어 넣는(팔이 꼬이는) branch를 고르는 것으로
    보임. 실물 top_down_angle_quat이 yaw를 position_yaw=atan2(y,x)로
    고정해 이 문제가 구조적으로 발생하지 않는 것과 동일한 메커니즘을
    적용: yaw = position_yaw + angle_deg. pitch=180 자체는 sim IK에서
    이미 검증된 값이라 그대로 유지하고, yaw 계산만 위치추종으로 바꿨다
    (pitch=90 실물 구조를 sim에 그대로 옮기는 건 sim IK 백엔드에서
    미검증이라 하지 않음 — grasp-kinematics-design 스킬 문서 참고).

    [2026-08 추가 수정, +180도 보정] 위 수정 배포 후에도 x>0에서 팔이
    꼬인 branch를 고르는 현상이 실측 재현됨 (override 테스트로 확인).
    원인: euler_to_quat(roll=0, pitch=180, yaw)는 로컬 기준축을
    Ry(180°)로 먼저 뒤집는다((x,y,z)->(-x,y,-z)) — 즉 로컬 +X축이
    yaw 회전 전에 이미 -X로 뒤집혀 있어서, 실제 그리퍼가 향하는 수평
    방향은 요청한 yaw가 아니라 yaw+180°가 된다. 그 결과 position_yaw를
    그대로 넣으면 그리퍼가 물체 방향이 아니라 정반대(로봇 쪽으로 접혀
    들어가는 방향)를 향하도록 명령한 셈이었다. +180°(π) 보정으로 실제
    향하는 방향이 position_yaw와 일치하도록 상쇄한다.
    """
    position_yaw = math.atan2(y, x)
    return euler_to_quat(
        0.0, math.pi,
        position_yaw + math.radians(angle_deg) + math.pi)


def sim_box_aligned_quat(angle_deg: float) -> list:
    """sim(MoveIt2) 전용, align 단계 전용 박스-정렬 쿼터니언
    (position_yaw 미포함).

    [2026-08 추가, 근본 원인 규명] sim_top_down_angle_quat은
    yaw = position_yaw + angle_deg + 180으로 계산한다. approach처럼
    angle_deg=0 고정("yaw 자유", 아무 twist나 상관없고 자세만 좋으면
    됨)일 때는 이게 맞는 공식이지만, align은 angle_deg에 실제 박스 edge
    절대각(perception이 tf로 base_link 기준까지 변환해서 보내는 값,
    0~90도)이 들어온다 — 이 값에 position_yaw(물체 방향 bearing, 물체의
    실제 회전과 무관한 값)를 또 더하면, 최종 그리퍼 방향이 박스 edge와
    정확히 position_yaw만큼 어긋난다. 실측으로 확진됨(2026-08-15):
    박스 (0.233,0.250), 인식각도=12.9도인데 정렬 후 모서리를 잡는
    현상 재현 -> position_yaw=atan2(0.250,0.233)=47.0도를 역산해보니
    사용자가 육안으로 본 "약 45도 어긋남"과 정확히 일치.

    align은 이 함수(position_yaw 미포함, 순수 박스각도+180)를 써야
    한다. YawCandidateSelector의 cost 비교(현재 orientation과의 차이)는
    여전히 유효하다 — current_quat은 approach가 만든 실제 자세(이미
    position_yaw가 반영된 좋은 자세)이므로, cost = |position_yaw -
    candidate| 형태로 자연스럽게 계산되어 "박스에 정확히 정렬되면서도
    approach 자세에서 회전량이 가장 적은" 후보를 고르게 된다 — 정렬
    정확도와 팔 자세 자연스러움을 동시에 만족.

    실물 backend(top_down_angle_quat)는 애초에 이 문제가 없다 — roll에
    angle_deg(박스정렬), yaw에 position_yaw(자세)가 서로 다른 축으로
    분리돼 있어서 섞이지 않는다. 이 함수는 sim 전용 보정이다.
    """
    return euler_to_quat(0.0, math.pi, math.radians(angle_deg) + math.pi)


def side_quat_for(pos: dict, approach_offset_deg: float = 0.0) -> list:
    """로봇(base_link 원점) -> 물체 위치 방향을 바라보는 '옆에서 수평 그립' 쿼터니언.
    탑다운 : math.radians(-angle_deg), math.radians(90), position_yaw

    approach_offset_deg: position_yaw 기준 접근 방향 오프셋 (도). 0이면 정면 직선 접근.
    """
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    position_yaw = math.atan2(y, x)

    yaw = position_yaw + math.radians(approach_offset_deg)
    pitch = 0.0
    roll = math.radians(-90.0)

    return euler_to_quat(roll, pitch, yaw)


def side_reachability_check(pos: dict) -> tuple:
    """side 그립 시도 전, 물체 위치가 도달 가능 영역인지 사전 판정한다.

    Returns:
        (True, '') 이면 도달 가능. (False, reason) 이면 거부 사유 포함.
    """
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    dist = math.sqrt(x * x + y * y)
    effective_dist = dist - SIDE_TCP_OFFSET
    if effective_dist < SIDE_MIN_DIST:
        return (False,
                f'물체가 로봇 베이스로부터 너무 가깝습니다 '
                f'(물체거리 {dist:.3f}m, TCP오프셋 적용 후 {effective_dist:.3f}m, '
                f'side 그립 최소 거리 {SIDE_MIN_DIST}m). '
                f'이 위치는 옆에서 수평으로 잡는 자세로 도달할 수 없는 영역입니다. '
                f'물체를 더 멀리 옮기거나 top(위에서 잡기) 방식을 사용하세요.')
    return (True, '')


def auto_grasp_quat(pos: dict, label: str) -> list:
    """라벨 힌트 -> 없으면 위치 기반 휴리스틱(y가 x보다 훨씬 크면 side)으로
    top-down 또는 side 쿼터니언을 결정한다."""
    hint = LABEL_GRASP_HINT.get(label, None)
    if hint:
        entry = GRASP_DIR_MAP[hint]
        if entry == SIDE_TAG:
            return side_quat_for(pos)
        return entry
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    if abs(y) > abs(x) * 1.5:
        return side_quat_for(pos)
    return QUAT_TOP_DOWN


def resolve_grasp_quat(grasp_dir, pos: dict, label: str = '',
                       side_approach_offset_deg: float = 0.0) -> tuple:
    """[신설] planning_node.py의 on_command 안에 pick/place/move 세 곳에
    거의 동일하게 반복돼 있던 "grasp_dir 문자열 -> (quat, is_side)" 해석
    로직을 하나로 통합한 것. 기존에는 이 세 곳이 서로 미묘하게 달랐다
    (move 분기만 기본값이 QUAT_HOME이고 pick/place는 None이었음 — 실질적
    동작 차이는 없었지만 유지보수 시 혼동 요인이었다). 이제 이 함수
    하나만 보고 판단하면 된다.

    Args:
        grasp_dir: 사용자가 명시한 자세 문자열('top', 'side', ...) 또는
            None/'auto' (라벨/위치 기반 자동 판단에 위임)
        pos: {'x', 'y', 'z'} 물체 또는 목표 위치
        label: 물체 라벨 (auto 판단 시에만 사용, 없으면 위치 휴리스틱만 적용)
        side_approach_offset_deg: side 그립 시 position_yaw 기준 접근 방향 오프셋 (도).

    Returns:
        (quat: list[4], is_side: bool)
    """
    if grasp_dir and grasp_dir != 'auto':
        quat = GRASP_DIR_MAP.get(grasp_dir, QUAT_HOME)
        if quat == SIDE_TAG:
            return side_quat_for(pos, side_approach_offset_deg), True
        return quat, False
    quat = auto_grasp_quat(pos, label)
    is_side = (
        LABEL_GRASP_HINT.get(label) in ('side', 'side_front')
        or (label not in LABEL_GRASP_HINT
            and abs(pos.get('y', 0.0)) > abs(pos.get('x', 0.0)) * 1.5)
    )
    return quat, is_side


class YawCandidateSelector:
    """_pick_best_yaw_candidate의 순수 비교 로직만 분리한 것 (원본 로직
    그대로 이관 — candidates_deg 구성, cost 계산 방식 모두 동일).

    ROS가 필요한 두 지점(IK 조회, FK로 현재 orientation 조회, backend별
    top_down quat 계산)은 콜백으로 주입받는다. planning_node.py는 기존
    self._compute_ik_joints / self._fk_gripper_pos / self._top_down_quat_for
    를 그대로 넘기면 되고, 테스트 코드는 이 세 콜백만 가짜로 바꿔치기하면
    ROS 없이 순수 비교 로직만 검증할 수 있다.
    """

    def __init__(self, ik_check_fn, quat_for_angle_fn):
        """
        Args:
            ik_check_fn: (x: float, y: float, z: float, quat: list) -> dict|None
                해당 pose의 IK 해가 있으면 joint dict, 없으면 None.
                (planning_node에서는 self._compute_ik_joints 래퍼)
            quat_for_angle_fn: (position: dict, angle_deg: float) -> list
                backend(sim/real)에 맞는 top-down quat 생성.
                (planning_node에서는 self._top_down_quat_for 래퍼)
        """
        self._ik_check_fn = ik_check_fn
        self._quat_for_angle_fn = quat_for_angle_fn

    # [2026-08 추가] 선택된 후보의 cost(quat_angle_diff)가 이 값을
    # 넘으면 "가까운 후보가 전부 IK 실패해서 어쩔 수 없이 큰 회전으로
    # 넘어갔다"는 경고를 로그로 남긴다 (동작은 그대로, 관측성만 추가).
    LARGE_REORIENT_WARN_DEG = 90.0

    def pick_best(self, position: dict, angle_deg: float,
                  current_quat, log_fn=None, ik_check_z: float = None) -> tuple:
        """box mod-90 대칭(짧은변/긴변) × 그리퍼 mod-180 대칭(좌우 대칭
        그리퍼라 손끝 방향 반대로 잡아도 물리적으로 같은 그립)을 조합한
        네 후보(angle_deg%90, +90, +180, +270) 중 IK로 도달 가능하면서
        current_quat과의 순수 회전각 차이(quat_angle_diff)가 가장 작은
        쪽을 선택한다. IK는 도달가능성 확인용으로만 쓰고, 후보 선택
        기준(cost)은 FK 기반 기하학적 회전량이다 (7DOF redundancy 때문에
        IK 이동량은 실제 회전량과 무관할 수 있음 — grasp_kinematics.py
        모듈 docstring 참고).

        [2026-08 확장, 2후보→4후보] 기존엔 mod-90 두 후보만 시도했는데,
        "가까운(cost 작은) 후보의 IK가 실패하면 남은 먼(cost 큰, 사실상
        180도 반대) 후보로 넘어가면서 approach 대비 큰 폭으로 그리퍼가
        도는" 사고가 실측 확인됐다(2026-08-15, x=0.039,y=-0.476,
        angle=81.1도 케이스: 81.1도 IK 실패 -> 171.1도만 성공해서 그걸로
        결정, cost=171.1도). 그런데 171.1도의 "그리퍼 mod-180 대칭 짝"인
        351.1도(=171.1+180)는 quat_angle_diff가 wrap돼서 cost가 겨우
        8.9도임에도 애초에 후보에 없어서 시도조차 못 됐다. mod-90 두
        값 각각에 +180을 더한 두 후보를 추가해 이 사각지대를 없앤다.
        cost 최소화 로직 자체가 이미 "먼(뒤집힌) 후보는 cost가 커서
        후순위"로 처리하므로, 후보를 늘려도 불필요하게 큰 회전이
        선호되는 방향으로 동작이 바뀌지 않는다 — 오히려 이전엔 없던
        선택지(작은 cost 후보)가 추가되는 것이라 순수 개선.

        Args:
            current_quat: 현재 그리퍼 orientation quat (FK로 조회한 값).
                None이면 조회 실패로 간주하고 원래 각도로 폴백.
            log_fn: 선택 optional logger(str) -> None. 생략 가능.
            ik_check_z: IK 사전검사에 쓸 z (실제 이동 목표 높이와 맞춰야
                함 — approach_z 등). 생략 시 position['z'] 사용(하위호환).

        Returns:
            (best_deg, best_quat)
        """
        base = angle_deg % 90
        candidates_deg = [base, base + 90, base + 180, base + 270]
        if current_quat is None:
            if log_fn:
                log_fn('[ik_check] 현재 orientation FK 조회 실패, 원래 각도로 진행')
            deg = angle_deg % 90
            return deg, self._quat_for_angle_fn(position, deg)

        check_z = ik_check_z if ik_check_z is not None else position.get('z', 0.0)
        best_deg = candidates_deg[0]
        best_quat = self._quat_for_angle_fn(position, best_deg)
        best_cost = float('inf')
        found_any = False
        for cand_deg in candidates_deg:
            cand_quat = self._quat_for_angle_fn(position, cand_deg)
            sol = self._ik_check_fn(position['x'], position['y'], check_z, cand_quat)
            if sol is None:
                continue
            cost = quat_angle_diff(current_quat, cand_quat)
            if log_fn:
                log_fn(f'[ik_check] 후보 {cand_deg:.1f}\xb0 -> '
                       f'orientation 각도차={math.degrees(cost):.1f}\xb0')
            if cost < best_cost:
                best_cost = cost
                best_deg = cand_deg
                best_quat = cand_quat
                found_any = True
        if not found_any:
            if log_fn:
                log_fn('[ik_check] 네 후보 모두 IK 조회 실패, 원래 각도 유지')
            deg = angle_deg % 90
            return deg, self._quat_for_angle_fn(position, deg)
        if log_fn:
            log_fn(f'[ik_check] 선택된 각도: {best_deg:.1f}\xb0')
            if math.degrees(best_cost) > self.LARGE_REORIENT_WARN_DEG:
                log_fn(f'[ik_check] ⚠ 선택된 후보의 회전량이 큼 '
                       f'({math.degrees(best_cost):.1f}\xb0 > '
                       f'{self.LARGE_REORIENT_WARN_DEG:.0f}\xb0) -- '
                       f'더 가까운 후보들이 전부 IK 실패해서 어쩔 수 없이 '
                       f'선택됨. approach 대비 큰 폭으로 그리퍼가 돌 수 있음.')
        return best_deg, best_quat
