#!/usr/bin/env python3
"""
planning_node.py  (IK 안정화 패치 + 박스 각도보정 재도입판)

[변경점 vs 이전 버전]
1. _move_joints_sequence 중복 정의 제거
2. _move() 재시도 로직 추가 (max_attempts=3)
   - OMPL 랜덤 샘플링 특성상 같은 목표라도 재시도 시 성공률 크게 향상
3. 이동 단계별 tolerance 분리
   - approach/lift/transit: tolerance_orientation=0.15 (느슨하게 → 플래너 성공률 ↑)
   - descend/ascend: tolerance_orientation=0.05 (파지 직전/직후는 정밀하게)
4. POSES_FILE 경로를 XDG_DATA_HOME 기반으로 변경 (Jetson/PC 모두 호환)
5. force_reset 타이밍 개선: move_to_pose 직전에만 호출

[2026-07 각도보정 — 1차 시도 실패 -> 제거 -> 2차 시도로 재도입]
- 1차 시도(실패, 제거됨): approach 이후 joint5 값을 읽어 joint7 자리에
  넣는 방식. 직렬 링크 로봇은 특정 조인트 하나만 움직여도 그 이후 링크
  전체가 같이 움직이므로 "그리퍼 yaw만 살짝 돌리기"가 조인트 스페이스
  제어로는 애초에 성립 불가능했음. 게다가 approach 중간에 자세를 바꾸는
  방식이라, MoveIt2/펌웨어가 매번 새로 IK를 풀면서 조인트1~4가 20도
  이상 튀는 위험한 동작(엘보 플립)까지 관찰되어 폐기.

- 2차 시도(현재, 채택): 실물 컨트롤러(agx_arm_ctrl_single_node,
  pyAgxArm 펌웨어)를 대상으로 직접 orientation sweep 실험을 수행한 결과:
    * roll=180°,pitch=0°(우리가 "top-down"이라 가정했던 자세)는 여러
      위치에서 광범위하게 NO_SOLUTION.
    * 반대로 pitch=90°,yaw=0° 고정, roll을 0°~-90°로 스윕한 전 구간은
      전부 IK 성공. 실측 tf 결과는 항상 roll≈180°,pitch≈0°(즉 실제로는
      정확히 top-down 자세)이면서, yaw만 "실측yaw = 요청roll - 90°"
      관계로 정확히 선형 대응함을 확인 (2026-07 검증, 15° 간격 스윕).
    * 즉 이 로봇 펌웨어에서 "그리퍼를 top-down으로 유지한 채 접근축
      기준 yaw만 돌리는" 자세를 안정적으로 얻으려면, 요청 쿼터니언을
      (roll=-angle_deg, pitch=90°, yaw=0)로 구성해야 한다.
  이 자세는 approach 진입 전에 딱 한 번만 계산해서 pick 시퀀스
  (approach→descend→lift) 내내 동일하게 유지한다. 시퀀스 중간에 자세를
  바꾸는 로직이 전혀 없으므로, 1차 시도 때 있었던 "중간 전환 -> IK 분기
  튐" 문제가 구조적으로 재발하지 않는다.
  perception_node가 계산해 보내는 angle_base_deg(0~90도 정규화)를
  그대로 사용한다.
"""

import json
import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient

try:
    from agx_arm_msgs.msg import AgxArmStatus
    HAVE_ARM_STATUS_MSG = True
except ImportError:
    HAVE_ARM_STATUS_MSG = False

# arm_status/motion_status 값 -> 사람이 읽을 이름 (로그용)
ARM_STATUS_NAMES = {
    0: "NORMAL", 1: "EMERGENCY_STOP", 2: "NO_SOLUTION",
    3: "SINGULARITY_POINT", 4: "TARGET_POS_EXCEEDS_LIMIT",
    5: "JOINT_COMMUNICATION_ERR", 6: "JOINT_BRAKE_NOT_RELEASED",
    7: "COLLISION_OCCURRED",
}
MOTION_STATUS_NAMES = {0: "SUCCESS", 1: "FAILED", 255: "UNKNOWN"}

# 실물 모드에서 _move()가 도착 확인을 기다리는 최대 시간(초)
REAL_MOVE_TIMEOUT = 8.0
REAL_MOVE_POLL_SEC = 0.1
REAL_MOVE_SETTLE_SEC = 0.3  # 퍼블리시 직후 arm_status가 갱신되기까지 최소 대기

# ── 그리퍼 상수 ───────────────────────────────────────────────────────────────
GRIPPER_OPEN  = 0.08
GRIPPER_CLOSE = 0.01
GRIPPER_FORCE = 1.5
SIM_GRIPPER_OPEN  = [0.08]
SIM_GRIPPER_CLOSE = [0.01]

# ── 시퀀스 오프셋 (미터) ──────────────────────────────────────────────────────
APPROACH_Z = 0.13
DESCEND_Z  = 0.03
LIFT_Z     = 0.23

# ── side 그립 전용 (대규모 IK 그리드 전수조사 결과) ──────────────────────────
# [검증 내역 - 연구실 PC compute_ik 전수조사, 2026-06-30]
#   x: -0.45~0.45 (0.02m 간격), y: -0.40~0.40 (0.02m 간격),
#   z: 0.09 / 0.15 / 0.25 (낮음/중간/높음 대표 높이), 총 5,598개 좌표 스캔.
#   전체 평균 OK 비율: 59.4%
#
#   거리(dist = sqrt(x²+y²))를 기준으로 분석한 결과, yaw(각도)와 거의 무관하게
#   "원점으로부터의 수평거리"가 도달 가능 여부를 가장 잘 설명함:
#     dist < 0.30  : 거의 100% FAIL (0%)
#     dist = 0.30  : 25% OK
#     dist = 0.32  : threshold로 썼을 때 정확도 95~96% (z=0.09/0.15 기준)
#     dist >= 0.35 : 90%대 이상 OK
#     dist >= 0.55 : 100% OK
#
#   SIDE_MIN_DIST=0.32를 "거부 기준"으로 사용 시:
#     z=0.09: 오탐(위험) 2.8%, 기회손실 1.7%, 정확도 95.5%
#     z=0.15: 오탐 2.8%, 기회손실 1.0%, 정확도 96.1%
#     z=0.25: 오탐 0.8%, 기회손실 7.1%, 정확도 92.1%
#   → 가장 균형 잡힌 임계값으로 채택. (재현 데이터: ik_side_reachable_map.txt)
SIDE_MIN_DIST   = 0.32   # side 그립 사전 검사 임계값: 이 거리 미만이면 IK 시도 없이 즉시 거부
SIDE_PITCH_DEG  = 90      # side 그립 손목 pitch (그리퍼를 눕히는 각도)

# ── top 그립 각도보정 전용 상수 ───────────────────────────────────────────────
# [검증 내역 - 실물 agx_arm_ctrl_single_node 대상 orientation sweep, 2026-07]
#   위치 (0.3,0.3,0.15) 고정, pitch=90°,yaw=0° 고정, roll을 0~-90도로
#   15도 간격 스윕 -> 전 구간 IK 성공(arm_status=NORMAL). 실측 tf 결과가
#   항상 top-down 자세(roll≈180°,pitch≈0°)이며, 실측yaw = 요청roll - 90°
#   관계로 정확히 선형 대응함을 확인. 이 결과를 이용해 top 그립 시
#   "top-down 유지 + 박스각도만큼 yaw 회전"을 아래 함수로 구현한다.
TOP_ANGLE_PITCH_DEG = 90.0   # 고정값 (실측상 항상 top-down으로 귀결됨)


def _top_down_angle_quat(angle_deg: float) -> list:
    """
    박스 각도(angle_deg, base_link 기준 0~90도)를 반영한 top-down 쿼터니언.

    roll=-angle_deg, pitch=90, yaw=0 으로 요청하면, 실물 펌웨어 기준
    실제로는 top-down 자세(roll≈180,pitch≈0)를 유지하면서 접근축 기준
    yaw만 angle_deg만큼 회전한 자세로 귀결됨 (2026-07 실측 검증).
    """
    return _euler_to_quat(math.radians(-angle_deg), math.radians(TOP_ANGLE_PITCH_DEG), 0.0)


# ── 그리퍼 TCP 오프셋 (gripper_flange → 손가락 중간 지점까지 거리) ───────────
# URDF: gripper_joint origin xyz="0 0 0.1358" → 그리퍼 전체 길이 13.58cm
# 물체를 실제로 쥐는 지점은 손가락 끝이 아니라 손가락 중간(절반)이므로
# top/side 모두 절반값을 TCP 오프셋으로 사용한다.
#   (이전에 TOP을 0.136으로 둔 적이 있었으나, 그건 PLACE_DROP_Z 계산에서
#    APPROACH_Z가 같이 더해져 너무 낮아진 게 원인이었음 -> place는 별도
#    PLACE_DROP_Z + TOP_TCP_OFFSET 조합으로 따로 보정하므로 pick 쪽은
#    절반값을 써도 무방함)
TOP_TCP_OFFSET  = 0.068   # 미터 (그리퍼 길이의 절반)
SIDE_TCP_OFFSET = 0.068   # 미터 (그리퍼 길이의 절반, 검증됨)

# place(내려놓기) 시 그리퍼 손가락 끝이 바닥에서 떨어진 높이.
PLACE_DROP_Z = 0.08   # 미터, place_pos.z 기준 손가락 끝 절대 높이

# ── 홈 자세 ───────────────────────────────────────────────────────────────────
HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, 0.75

# ── 타이밍 ────────────────────────────────────────────────────────────────────
MOVE_DELAY    = 6.0
GRIPPER_DELAY = 4.0

# ── MoveIt2 플래너 설정 ────────────────────────────────────────────────────────
# approach/lift/transit: 방향 정밀도 낮춰 플래너 성공률 ↑
TOL_POS_LOOSE   = 0.01
TOL_ORI_LOOSE   = 0.15   # 약 8.6도 — 수평 이동 단계
# descend/ascend: 파지 직전·직후는 정밀하게
TOL_POS_TIGHT   = 0.008
TOL_ORI_TIGHT   = 0.05   # 약 2.9도
# 재시도 횟수 (OMPL 랜덤 샘플링 특성상 재시도로 성공률 크게 개선)
MOVE_MAX_ATTEMPTS = 3
MOVE_TIMEOUT      = 30.0  # 단일 시도 타임아웃

# ── 사전 정의 쿼터니언 (xyzw, base_link 기준) ─────────────────────────────────
QUAT_TOP_DOWN   = [0.008,  0.999,  0.023,  0.037]
QUAT_TOP_DOWN_R = [0.999, -0.008, -0.037,  0.023]
QUAT_TOP_DOWN_L = [0.708,  0.697, -0.010,  0.043]
QUAT_TOP_DOWN_RR= [-0.643, 0.653,  0.041,  0.011]
QUAT_HOME       = [0.0, 0.0, 0.0, 1.0]
QUAT_SIDE_FRONT = [0.481, -0.527,  0.427,  0.556]

GRASP_DIR_MAP = {
    'top':         QUAT_TOP_DOWN,
    'top_down':    QUAT_TOP_DOWN,
    'top_down_r':  QUAT_TOP_DOWN_R,
    'top_down_l':  QUAT_TOP_DOWN_L,
    'top_down_rr': QUAT_TOP_DOWN_RR,
    'side':        'DYNAMIC',   # 물체 위치에 따라 _side_quat_for()로 동적 계산
    'side_front':  'DYNAMIC',
    'side_left':   'DYNAMIC',
    'side_right':  'DYNAMIC',
}

# side 계열 식별용 마커 (쿼터니언 비교 대신 이 태그로 판별)
SIDE_TAG = 'DYNAMIC'


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> list:
    """ZYX 오일러각(roll, pitch, yaw, 라디안) -> 쿼터니언(xyzw)."""
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    ]


def _side_quat_for(pos: dict) -> list:
    """
    로봇(base_link 원점) -> 물체 위치 방향을 바라보는 '옆에서 수평 그립' 쿼터니언.

    [검증 내역 - 연구실 PC compute_ik 전수조사(5,598점), 2026-06-30]
      roll=0, pitch=90도, yaw=atan2(y,x) 조합으로 x=-0.45~0.45, y=-0.4~0.4,
      z=0.09/0.15/0.25 전 영역 스캔. dist=sqrt(x²+y²) < 0.32 는 거의 항상 FAIL.
      실제 거부 판단은 _side_reachability_check()에서 수행한다.
    """
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    yaw = math.atan2(y, x)
    pitch = math.radians(SIDE_PITCH_DEG)
    roll = 0.0
    return _euler_to_quat(roll, pitch, yaw)


def _side_reachability_check(pos: dict) -> tuple:
    """
    side 그립 시도 전, 물체 위치가 도달 가능 영역인지 사전 판정한다.

    IK를 직접 풀지 않고 거리(dist) 임계값만으로 판단하는 빠른 휴리스틱이며,
    5,598점 전수조사로 검증된 SIDE_MIN_DIST(0.32m) 기준을 사용한다.
    오탐(실제로는 되는데 거부, false negative) 약 1~7% 가능성이 있지만,
    반대로 ABORTED를 반복하며 30초씩 허비하는 것보다 훨씬 빠르고 안전하다.

    주의: SIDE_MIN_DIST는 "실제 로봇이 이동하는 목표 좌표(gripper_flange)"
    기준으로 검증된 값이다. _pick_sequence/_place_sequence는 물체 좌표에서
    SIDE_TCP_OFFSET만큼 당긴 지점으로 이동하므로, 검사도 그 오프셋이
    적용된 지점(= 실제 이동 목표) 기준으로 해야 정확하다.

    Returns:
        (ok: bool, reason: str)
        ok=True  -> 진행 가능
        ok=False -> reason에 거부 사유 메시지 (사용자에게 그대로 노출 가능)
    """
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    dist = math.sqrt(x*x + y*y)
    # 오프셋 적용 후 실제 이동 목표 지점까지의 거리
    effective_dist = dist - SIDE_TCP_OFFSET
    if effective_dist < SIDE_MIN_DIST:
        return (False,
                f'물체가 로봇 베이스로부터 너무 가깝습니다 '
                f'(물체거리 {dist:.3f}m, TCP오프셋 적용 후 {effective_dist:.3f}m, '
                f'side 그립 최소 거리 {SIDE_MIN_DIST}m). '
                f'이 위치는 옆에서 수평으로 잡는 자세로 도달할 수 없는 영역입니다. '
                f'물체를 더 멀리 옮기거나 top(위에서 잡기) 방식을 사용하세요.')
    return (True, '')


LABEL_GRASP_HINT = {
    'bottle':   'side_front',
    'cup':      'top_down',
    'book':     'side_front',
    'box':      'top_down',
    'ball':     'top_down',
    'scissors': 'side_front',
    'remote':   'top_down',
}


def _auto_grasp_quat(pos: dict, label: str) -> list:
    hint = LABEL_GRASP_HINT.get(label, None)
    if hint:
        entry = GRASP_DIR_MAP[hint]
        if entry == SIDE_TAG:
            return _side_quat_for(pos)
        return entry
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    if abs(y) > abs(x) * 1.5:
        return _side_quat_for(pos)
    return QUAT_TOP_DOWN


class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        self.use_moveit2 = self.declare_parameter('use_moveit2', True).value

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self._cb = ReentrantCallbackGroup()

        self.latest_joint_state = None
        self.create_subscription(
            JointState, '/feedback/joint_states',
            lambda msg: setattr(self, 'latest_joint_state', msg), 10)

        self.latest_arm_status = None
        self._arm_status_available = False
        if HAVE_ARM_STATUS_MSG:
            try:
                self.create_subscription(
                    AgxArmStatus, '/feedback/arm_status',
                    lambda msg: setattr(self, 'latest_arm_status', msg), 10)
                self._arm_status_available = True
            except Exception as e:
                self.get_logger().warn(
                    f'/feedback/arm_status 구독 생성 실패 (typesupport 문제로 추정): {e} | '
                    f'실물 모드 _move()가 도착 확인 없이 예전처럼 즉시 성공 처리됩니다.')
        else:
            self.get_logger().warn(
                'agx_arm_msgs.msg.AgxArmStatus import 실패 — 실물 모드 _move()가 '
                '도착 확인 없이 예전처럼 즉시 성공 처리됩니다. '
                '(ros2 interface show agx_arm_msgs/msg/AgxArmStatus 로 실제 타입 확인 필요)')

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects,
            qos_best_effort, callback_group=self._cb)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command,
            qos_best_effort, callback_group=self._cb)

        self.pub_move    = self.create_publisher(PoseStamped, '/control/move_p', qos_reliable)
        self.pub_gripper = self.create_publisher(JointState,  '/control/joint_states', qos_reliable)
        self.pub_result  = self.create_publisher(String,      '/pick_result', qos_reliable)

        self.moveit2 = self.moveit2_gripper = None
        if self.use_moveit2:
            from pymoveit2 import MoveIt2
            cb_arm     = ReentrantCallbackGroup()
            cb_gripper = ReentrantCallbackGroup()
            self.moveit2 = MoveIt2(
                node=self,
                joint_names=['joint1','joint2','joint3','joint4','joint5','joint6','joint7'],
                base_link_name='base_link',
                end_effector_name='gripper_flange',
                group_name='arm',
                callback_group=cb_arm,
                use_move_group_action=True,
                ignore_new_calls_while_executing=True,
            )
            # 기본값(0.5초, 5회)이 너무 짧아 side 자세처럼 까다로운 IK에서
            # OMPL이 충분히 탐색 못 하고 ABORTED 나는 경우가 많음 → 넉넉하게 상향
            self.moveit2.allowed_planning_time = 5.0
            self.moveit2.num_planning_attempts = 20
            self.get_logger().info(
                'MoveIt2 ENABLED (allowed_planning_time=5.0s, num_planning_attempts=20)')
        # 그리퍼 액션 클라이언트 (moveit2_gripper 대신 직접 액션 호출)
        self._gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
        )

        self.latest_objects = []
        self._box_angle_deg = None
        self.busy = False
        self.lock = threading.Lock()
        if self.use_moveit2:
            time.sleep(3.0)
        self.get_logger().info('PlanningNode 준비 완료')

    # ── 토픽 콜백 ─────────────────────────────────────────────────────────────
    def on_objects(self, msg):
        try:
            self.latest_objects = json.loads(msg.data).get('objects', [])
        except Exception:
            pass

    def on_command(self, msg):
        self.get_logger().info(f'[CMD] 수신: {msg.data[:80]}')
        with self.lock:
            if self.busy:
                self.get_logger().warn('작업 중. 명령 무시.')
                return
            self.busy = True

        try:
            cmd = json.loads(msg.data)
        except Exception:
            with self.lock:
                self.busy = False
            return

        action = cmd.get('action', 'pick')

        if action == 'pick':
            label = cmd.get('target_label', '')
            pos, angle_deg = self._find_object_with_angle(label)
            self._box_angle_deg = angle_deg
            if pos is None:
                self.get_logger().warn(f"'{label}' 못 찾음.")
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            is_side = False
            if grasp_dir:
                quat = GRASP_DIR_MAP.get(grasp_dir, None)
                if quat == SIDE_TAG:
                    is_side = True
                    quat = _side_quat_for(pos)
            else:
                quat = _auto_grasp_quat(pos, label)
                x, y = pos.get('x', 0.0), pos.get('y', 0.0)
                if LABEL_GRASP_HINT.get(label) in ('side', 'side_front'):
                    is_side = True
                elif abs(y) > abs(x) * 1.5 and label not in LABEL_GRASP_HINT:
                    is_side = True
            self.get_logger().info(
                f'PICK 시작: {label} @ {pos} | 자세: {grasp_dir or "auto"} '
                f'({"side" if is_side else "top"}) {[round(v,3) for v in quat]} | '
                f'박스각도={angle_deg}')
            t = threading.Thread(target=self._pick_sequence, args=(pos, quat, is_side))
            t.daemon = False
            t.start()

        elif action == 'place':
            pos = cmd.get('place_pos')
            if pos is None:
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            is_side = False
            if grasp_dir:
                quat = GRASP_DIR_MAP.get(grasp_dir, None)
                if quat == SIDE_TAG:
                    is_side = True
                    quat = _side_quat_for(pos)
            else:
                # ── 2026-07 수정 ──────────────────────────────────────────
                # 예전엔 여기서 QUAT_TOP_DOWN(roll≈180,pitch≈0) 고정값을 썼는데,
                # 이 자세가 실물 펌웨어에서 워크스페이스 넓은 범위에 걸쳐
                # NO_SOLUTION 나는 것으로 이미 확인됨(오늘 pick 각도보정 검증
                # 과정에서 발견). place도 pick과 동일하게 검증된
                # (roll=-angle, pitch=90, yaw=0) 공식을 사용하도록 교체.
                # 마지막으로 집었던 물체의 박스각도(self._box_angle_deg)를
                # 그대로 이어받아, 잡은 자세 그대로 내려놓게 한다
                # (물체를 든 채로 억지로 비틀지 않기 위함). pick 이력이 없으면
                # angle=0으로 처리.
                angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
                quat = _top_down_angle_quat(angle_deg)
                self.get_logger().info(
                    f'[place] 박스각도 반영 쿼터니언 사용: angle={angle_deg}° -> '
                    f'quat={[round(v,3) for v in quat]}')
            self.get_logger().info(
                f'PLACE 시작 @ {pos} | 자세: {"side" if is_side else "top"} '
                f'{[round(v,3) for v in quat]}')
            threading.Thread(
                target=self._place_sequence, args=(pos, quat, is_side), daemon=True).start()

        elif action == 'move':
            pos = cmd.get('target_pos')
            if pos is None:
                self.get_logger().warn("'move' 명령에 target_pos 없음.")
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            if grasp_dir:
                quat = GRASP_DIR_MAP.get(grasp_dir, QUAT_HOME)
                if quat == SIDE_TAG:
                    quat = _side_quat_for(pos)
            else:
                quat = QUAT_HOME
            self.get_logger().info(f'MOVE 시작 @ {pos} | 자세: {quat}')
            threading.Thread(
                target=self._move_sequence, args=(pos, quat), daemon=True).start()

        elif action == 'move_joints':
            joints = cmd.get('joints', {})
            if not joints:
                self.get_logger().warn("'move_joints' 명령에 joints 없음.")
                with self.lock:
                    self.busy = False
                return
            self.get_logger().info(f'MOVE_JOINTS 시작: {joints}')
            t = threading.Thread(target=self._move_joints_sequence, args=(joints,))
            t.daemon = False
            t.start()

        elif action == 'home':
            self.get_logger().info('HOME 시작')
            threading.Thread(target=self._home_sequence, daemon=True).start()

        else:
            self.get_logger().warn(f"알 수 없는 action: '{action}'")
            with self.lock:
                self.busy = False

    # ── 유틸 ──────────────────────────────────────────────────────────────────
    def _find_object(self, label):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d')
        return None

    def _find_object_with_angle(self, label):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d'), obj.get('angle_base_deg', None)
        return None, None

    # ── 시퀀스 ────────────────────────────────────────────────────────────────
    def _pick_sequence(self, pos, quat, is_side=False):
        try:
            # ── top 그립 + 박스각도 있음: approach 진입 전에 딱 한 번만
            # "top-down 유지 + yaw=박스각도" 쿼터니언을 계산하고, 이후
            # approach/descend/lift 내내 이 값을 그대로 사용한다.
            # (시퀀스 중간에 자세를 바꾸지 않음 — 1차 시도 실패의 핵심 원인이었던
            #  "중간 전환 -> IK 재계산 -> 조인트 튐"을 구조적으로 피하기 위함)
            if not is_side:
                angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
                quat = _top_down_angle_quat(angle_deg)
                self.get_logger().info(
                    f'[top] 박스각도 반영 쿼터니언 고정: angle={angle_deg}° -> '
                    f'quat={[round(v,3) for v in quat]} (approach~lift 내내 동일 유지)')

            # side 그립은 IK를 시도하기 전에 먼저 도달 가능 영역인지 검사한다.
            # (전수조사 결과 dist<0.32는 거의 항상 IK 실패 -> ABORTED 3회 재시도로
            #  30초씩 허비하는 대신 즉시 거부하고 사유를 알려준다)
            if is_side:
                ok, reason = _side_reachability_check(pos)
                if not ok:
                    self.get_logger().warn(f'PICK 거부: {reason}')
                    self._publish_result('rejected', reason)
                    return

            # top 계열: 물체 바로 위에서 z 방향 TCP 오프셋만큼 띄워 수직 접근
            # side 계열: 물체→로봇 방향으로 TCP 오프셋만큼 당겨서 손가락 중간이
            #            물체 중심에 오도록 보정 (사전 검사는 원래 좌표 기준으로 이미 통과함)
            if is_side:
                dx, dy = pos['x'], pos['y']
                dist = math.sqrt(dx*dx + dy*dy) or 1.0
                # 물체 방향 단위벡터의 반대로 SIDE_TCP_OFFSET만큼 당김
                px = dx - (dx / dist) * SIDE_TCP_OFFSET
                py = dy - (dy / dist) * SIDE_TCP_OFFSET
                offset_desc = f'side(TCP오프셋 {SIDE_TCP_OFFSET}m 적용)'
            else:
                px = pos['x']
                py = pos['y']
                offset_desc = 'top(z)'

            if is_side:
                # side는 수직 낙하 충돌 위험이 없는 수평 접근이므로
                # approach 단계부터 바로 물체 실제 높이로 이동한다 (descend 생략 효과)
                approach_z = pos['z']
                descend_z  = pos['z']
                lift_z     = pos['z'] + LIFT_Z      # top과 동일한 상승폭
            else:
                pz = pos['z']
                approach_z = pz + APPROACH_Z + TOP_TCP_OFFSET
                descend_z  = pz + DESCEND_Z + TOP_TCP_OFFSET
                lift_z     = pz + LIFT_Z + TOP_TCP_OFFSET

            self.get_logger().info(
                f'1/5: 접근 (approach) | {offset_desc}')
            ok = self._move(px, py, approach_z, quat, tol_ori=TOL_ORI_LOOSE)
            if not ok:
                raise RuntimeError('approach 이동 실패 (재시도 초과)')

            self.get_logger().info('2/5: 그리퍼 열기')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)

            if is_side:
                self.get_logger().info('3/5: 내려가기 (descend) — side는 approach와 동일 높이, 재확인만')
            else:
                self.get_logger().info('3/5: 내려가기 (descend) — tight tolerance')
            tol = TOL_ORI_TIGHT if not is_side else TOL_ORI_LOOSE
            ok = self._move(px, py, descend_z, quat, tol_ori=tol)
            if not ok:
                raise RuntimeError('descend 이동 실패 (재시도 초과)')

            self.get_logger().info('4/5: 그리퍼 닫기')
            self._gripper(SIM_GRIPPER_CLOSE, GRIPPER_CLOSE)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('5/5: 들어올리기 (lift) — joint-space (빙글 방지)')
            if not is_side:
                lift_quat = _top_down_angle_quat(0.0)
                self.get_logger().info(
                    f'[top] lift 각도보정 해제: quat={[round(v,3) for v in lift_quat]}로 전환')
            else:
                lift_quat = quat

            # ── 2026-07 수정 ──────────────────────────────────────────────
            # 특정 위치에서는 완전한 LIFT_Z 높이가 워크스페이스 경계/특이점
            # 근처라 간헐적으로 NO_SOLUTION 나는 것이 실측으로 확인됨(동일
            # 목표가 어떨 땐 성공, 어떨 땐 6연속 실패). 근본 원인(정확한
            # 안전 범위)은 side 그립처럼 전수조사가 필요하지만, 당장은
            # 완전한 높이가 막히면 더 낮은 높이로 단계적으로 낮춰가며
            # 재시도해서 최소한 바닥에서는 띄우는 것을 우선한다.
            lift_candidates = [lift_z, pos['z'] + 0.15 + (0 if is_side else TOP_TCP_OFFSET),
                               pos['z'] + 0.08 + (0 if is_side else TOP_TCP_OFFSET)]
            ok = False
            for i, lz in enumerate(lift_candidates):
                if i > 0:
                    self.get_logger().warn(
                        f'  lift 높이 낮춰서 재시도: {lz:.3f}m (원래 목표 {lift_z:.3f}m)')
                ok = self._move(px, py, lz, lift_quat, tol_ori=TOL_ORI_LOOSE)
                if ok:
                    break
            if not ok:
                raise RuntimeError('lift 이동 실패 (모든 높이 재시도 초과)')

            self.get_logger().info('✅ PICK 완료')
            self._publish_result('success', 'pick_complete')
        except Exception as e:
            self.get_logger().error(f'PICK 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _place_sequence(self, pos, quat, is_side=False):
        try:
            # ── 2026-07 수정 ──────────────────────────────────────────────
            # on_command에서 grasp_dir을 명시적으로 'top'/'top_down' 등으로
            # 보내면 GRASP_DIR_MAP의 옛날 QUAT_TOP_DOWN 상수로 빠지는 구멍이
            # 있었음(기본값(grasp_dir 생략) 케이스만 고쳤던 게 원인).
            # _pick_sequence와 동일하게, top 계열이면 여기서 무조건
            # 각도보정 쿼터니언으로 덮어써서 이 구멍을 원천 차단한다.
            if not is_side:
                angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
                quat = _top_down_angle_quat(angle_deg)
                self.get_logger().info(
                    f'[place-top] 각도보정 쿼터니언 고정: angle={angle_deg}° -> '
                    f'quat={[round(v,3) for v in quat]}')

            if is_side:
                ok, reason = _side_reachability_check(pos)
                if not ok:
                    self.get_logger().warn(f'PLACE 거부: {reason}')
                    self._publish_result('rejected', reason)
                    return
                dx, dy = pos['x'], pos['y']
                dist = math.sqrt(dx*dx + dy*dy) or 1.0
                px = dx - (dx / dist) * SIDE_TCP_OFFSET
                py = dy - (dy / dist) * SIDE_TCP_OFFSET
                target_z = pos['z'] + APPROACH_Z
            else:
                px = pos['x']
                py = pos['y']
                # PLACE_DROP_Z는 "그리퍼 손가락 끝"이 바닥에서 떨어진 높이.
                # 실제 이동 목표는 gripper_flange 기준이므로 TOP_TCP_OFFSET(그리퍼
                # 길이)을 더해야 한다. 안 더하면 flange가 손가락 끝 위치까지
                # 내려가버려서 그리퍼가 바닥을 뚫고 들어가는 문제가 생김.
                target_z = pos['z'] + PLACE_DROP_Z + TOP_TCP_OFFSET

            self.get_logger().info('1/2: place 이동')
            ok = self._move(px, py, target_z, quat, tol_ori=TOL_ORI_LOOSE)
            if not ok:
                raise RuntimeError('place 이동 실패 (재시도 초과)')

            self.get_logger().info('2/2: 그리퍼 열기')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('✅ PLACE 완료')
            self._publish_result('success', 'place_complete')
        except Exception as e:
            self.get_logger().error(f'PLACE 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _move_sequence(self, pos, quat):
        try:
            self.get_logger().info(
                f"1/1: 이동 → ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            ok = self._move(pos['x'], pos['y'], pos['z'], quat,
                            tol_ori=TOL_ORI_LOOSE)
            if not ok:
                raise RuntimeError('move 이동 실패 (재시도 초과)')
            self.get_logger().info('✅ MOVE 완료')
            self._publish_result('success', 'move_complete')
        except Exception as e:
            self.get_logger().error(f'MOVE 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _publish_joint_positions(self, positions, joint_names=None):
        joint_names = joint_names or ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = joint_names
        js.position = [float(p) for p in positions]
        self.pub_gripper.publish(js)

    def _move_joints_real_wait(self, positions, joint_names=None) -> bool:
        """
        실물 모드에서 joint-space 이동(/control/joint_states 발행) 후
        /feedback/arm_status로 실제 도착을 확인한다 (home, move_joints 공용).

        예전엔 발행만 하고 3초 고정 대기 후 무조건 성공 처리했는데, 이게
        "간헐적으로 그리퍼만 열리고 팔은 안 움직임" 문제의 원인이었다 —
        CAN 버스 혼잡/타이밍 문제로 조인트 이동 메시지만 조용히 씹혀도
        확인할 방법이 없었음. _move()와 동일한 패턴으로 확인+재시도 추가.
        """
        if not self._arm_status_available:
            self._publish_joint_positions(positions, joint_names)
            time.sleep(3.0)  # 확인 불가 시 예전 방식(고정 대기)으로 폴백
            return True

        for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
            self.get_logger().info(
                f'  → [실물] joint 이동 시도 {attempt}/{MOVE_MAX_ATTEMPTS}: '
                f'{[round(p,3) for p in positions]}')
            self._publish_joint_positions(positions, joint_names)

            time.sleep(REAL_MOVE_SETTLE_SEC)
            deadline = time.time() + REAL_MOVE_TIMEOUT
            last_status = None
            while time.time() < deadline:
                status = self.latest_arm_status
                if status is not None:
                    last_status = status
                    if status.motion_status == 0:  # SUCCESS
                        return True
                    if status.motion_status == 1:  # FAILED
                        break
                time.sleep(REAL_MOVE_POLL_SEC)

            if last_status is not None:
                arm_s = ARM_STATUS_NAMES.get(last_status.arm_status, str(last_status.arm_status))
                motion_s = MOTION_STATUS_NAMES.get(last_status.motion_status, str(last_status.motion_status))
                self.get_logger().warn(
                    f'  [실물] joint 이동 실패/타임아웃: arm_status={arm_s} motion_status={motion_s}')
            else:
                self.get_logger().warn('  [실물] arm_status 수신 안 됨 (타임아웃)')

            if attempt < MOVE_MAX_ATTEMPTS:
                time.sleep(0.5)

        self.get_logger().error(
            f'[실물] joint 이동 최종 실패: {[round(p,3) for p in positions]} '
            f'[{MOVE_MAX_ATTEMPTS}회 모두 실패]')
        return False

    def _move_joints_sequence(self, joints: dict):
        """joint space 직접 이동. _move_joints_sequence 중복 정의 제거됨."""
        try:
            joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
            key_map = {'j1':'joint1','j2':'joint2','j3':'joint3','j4':'joint4',
                       'j5':'joint5','j6':'joint6','j7':'joint7'}
            try:
                js_msg = self.latest_joint_state
                name_to_pos = dict(zip(js_msg.name, js_msg.position))
                positions = [float(name_to_pos.get(jn, 0.0)) for jn in joint_names]
            except Exception:
                positions = [0.0] * 7
            for k, v in joints.items():
                jname = key_map.get(k, k)
                if jname in joint_names:
                    idx = joint_names.index(jname)
                    positions[idx] = float(v)
            self.get_logger().info(f'joint positions: {[round(p,3) for p in positions]}')
            if self.use_moveit2:
                self.moveit2.move_to_configuration(positions)
                time.sleep(0.5)
                deadline = time.time() + MOVE_TIMEOUT
                while time.time() < deadline:
                    if (not self.moveit2._MoveIt2__is_motion_requested and
                            not self.moveit2._MoveIt2__is_executing):
                        break
                    time.sleep(0.1)
            else:
                ok = self._move_joints_real_wait(positions)
                if not ok:
                    raise RuntimeError('move_joints 이동 실패 (재시도 초과)')
            self._publish_result('success', 'joint_move_complete')
        except Exception as e:
            self.get_logger().error(f'MOVE_JOINTS 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _home_sequence(self):
        try:
            self.get_logger().info('1/2: 홈 위치로 이동 (joint space, all-zero)')
            if self.use_moveit2:
                self.moveit2.force_reset_executing_state()
                self.moveit2.move_to_configuration([0.0] * 7)
                time.sleep(0.5)
                deadline = time.time() + MOVE_TIMEOUT
                while time.time() < deadline:
                    if (not self.moveit2._MoveIt2__is_motion_requested and
                            not self.moveit2._MoveIt2__is_executing):
                        break
                    time.sleep(0.1)
            else:
                ok = self._move_joints_real_wait([0.0] * 7)
                if not ok:
                    raise RuntimeError('home 조인트 이동 실패 (재시도 초과)')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('✅ HOME 완료')
            self._publish_result('success', 'home_complete')
        except Exception as e:
            self.get_logger().error(f'HOME 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    # ── 핵심: 재시도 포함 _move() ─────────────────────────────────────────────
    def _move(self, x: float, y: float, z: float, quat: list,
              tol_pos: float = TOL_POS_LOOSE,
              tol_ori: float = TOL_ORI_LOOSE) -> bool:
        """
        MoveIt2로 목표 pose 이동. 실패 시 MOVE_MAX_ATTEMPTS 회 재시도.
        성공하면 True, 모든 시도 실패 시 False 반환.

        tol_ori 파라미터로 단계별 tolerance 분리:
          - approach/lift/transit → TOL_ORI_LOOSE (0.15 rad)
          - descend/ascend       → TOL_ORI_TIGHT  (0.05 rad)
        """
        if not self.use_moveit2:
            # 실물 모드: PoseStamped 발행 후 /feedback/arm_status로 실제
            # 도착(성공/실패)을 확인할 때까지 대기한다.
            # (예전엔 발행만 하고 바로 True를 반환했는데, 그러면 다음 단계
            #  명령이 로봇이 아직 이동 중인 위로 덮어써지는 레이스 컨디션이
            #  발생함 — 그리퍼가 회전 도중에 닫히거나, lift 실패를 못
            #  알아채는 원인이 됐음. 2026-07 수정.)
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.header.stamp    = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            if not self._arm_status_available:
                # 상태 확인 불가 — 예전 방식(발행만 하고 성공 처리)으로 폴백
                self.pub_move.publish(pose)
                return True

            for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
                self.get_logger().info(
                    f'  → [실물] move 시도 {attempt}/{MOVE_MAX_ATTEMPTS} '
                    f'({x:.3f},{y:.3f},{z:.3f}) quat={[round(v,3) for v in quat]}')
                pose.header.stamp = self.get_clock().now().to_msg()
                self.pub_move.publish(pose)

                time.sleep(REAL_MOVE_SETTLE_SEC)
                deadline = time.time() + REAL_MOVE_TIMEOUT
                last_status = None
                while time.time() < deadline:
                    status = self.latest_arm_status
                    if status is not None:
                        last_status = status
                        if status.motion_status == 0:  # SUCCESS
                            return True
                        if status.motion_status == 1:  # FAILED
                            break
                    time.sleep(REAL_MOVE_POLL_SEC)

                if last_status is not None:
                    arm_s = ARM_STATUS_NAMES.get(last_status.arm_status, str(last_status.arm_status))
                    motion_s = MOTION_STATUS_NAMES.get(last_status.motion_status, str(last_status.motion_status))
                    self.get_logger().warn(
                        f'  [실물] 이동 실패/타임아웃: arm_status={arm_s} motion_status={motion_s}')
                else:
                    self.get_logger().warn('  [실물] arm_status 수신 안 됨 (타임아웃)')

                if attempt < MOVE_MAX_ATTEMPTS:
                    time.sleep(0.5)

            self.get_logger().error(
                f'[실물] _move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) '
                f'[{MOVE_MAX_ATTEMPTS}회 모두 실패]')
            return False

        for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
            self.get_logger().info(
                f'  → move 시도 {attempt}/{MOVE_MAX_ATTEMPTS} '
                f'({x:.3f},{y:.3f},{z:.3f}) tol_ori={tol_ori:.3f}')

            # 이전 상태 리셋 (재시도 시 필수)
            self.moveit2.force_reset_executing_state()

            ready = self.moveit2._MoveIt2__move_action_client.server_is_ready()
            if not ready:
                self.get_logger().warn('  action server not ready, 잠시 대기...')
                time.sleep(1.0)
                continue

            self.moveit2.move_to_pose(
                position=[x, y, z],
                quat_xyzw=quat,
                tolerance_position=tol_pos,
                tolerance_orientation=tol_ori,
                cartesian=False,
            )

            time.sleep(0.5)
            deadline = time.time() + MOVE_TIMEOUT
            motion_done = False
            while time.time() < deadline:
                req = self.moveit2._MoveIt2__is_motion_requested
                exe = self.moveit2._MoveIt2__is_executing
                if not req and not exe:
                    motion_done = True
                    break
                time.sleep(0.2)

            # motion_suceeded 로 실제 IK/실행 성공 여부 확인
            # (req/exe 가 False 가 됐다고 성공인 것은 아님 — ABORTED 시에도 False 가 됨)
            success = motion_done and self.moveit2.motion_suceeded
            error_code = self.moveit2.get_last_execution_error_code()
            err_str = f' [error_code={error_code.val}]' if (not success and error_code) else ''

            self.get_logger().info(
                f'  [sim] move → {x:.3f} {y:.3f} {z:.3f} | '
                f'quat={quat} | {"✅ 성공" if success else "❌ 실패/타임아웃"}{err_str}')

            if success:
                return True

            if attempt < MOVE_MAX_ATTEMPTS:
                self.get_logger().warn(
                    f'  재시도 전 1초 대기... ({attempt}/{MOVE_MAX_ATTEMPTS})')
                time.sleep(1.0)

        self.get_logger().error(
            f'_move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) '
            f'[{MOVE_MAX_ATTEMPTS}회 모두 실패]')
        return False

    def _gripper(self, sim_joints, real_width):
        """
        그리퍼 제어.
        시뮬(use_moveit2=True):
          /gripper_controller/follow_joint_trajectory 액션 직접 호출
          SRDF: open=[0.05,-0.05], close=[0.0,0.0]
        실물(use_moveit2=False):
          /control/joint_states 퍼블리시 (agx_arm_ros 공식)
        """
        if self.use_moveit2:
            if float(sim_joints[0]) > 0.04:
                positions = [0.05, -0.05]  # open
                label = 'open'
            else:
                positions = [0.0, 0.0]     # close
                label = 'close'

            traj = JointTrajectory()
            traj.joint_names = ['gripper_joint1', 'gripper_joint2']
            pt = JointTrajectoryPoint()
            pt.positions = positions
            pt.time_from_start = Duration(sec=2, nanosec=0)
            traj.points = [pt]

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj

            if not self._gripper_action.wait_for_server(timeout_sec=2.0):
                self.get_logger().warn('[gripper] action server 없음, 스킵')
                return

            future = self._gripper_action.send_goal_async(goal)
            # 비동기 — 완료 대기 없이 반환 (GRIPPER_DELAY로 대기)
            self.get_logger().info(
                f'[sim gripper] {label} → {positions} (follow_joint_trajectory)')
        else:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name         = ['gripper']
            msg.position     = [float(real_width)]
            msg.effort       = [GRIPPER_FORCE]
            self.pub_gripper.publish(msg)
            self.get_logger().info(
                f'[gripper] width={real_width:.3f}m force={GRIPPER_FORCE}N → /control/joint_states')

    def _publish_result(self, status, reason=''):
        msg = String()
        msg.data = json.dumps({'status': status, 'reason': reason},
                              ensure_ascii=False)
        self.pub_result.publish(msg)


def main():
    rclpy.init()
    node = PlanningNode()
    executor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
