#!/usr/bin/env python3
# 서브에이전트로 수정함
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
[2026-07 복원] 별도 세션에서 project knowledge의 예전 아키텍처를 참고해
_seeded_ik/_best_yaw_solution/_move_joint_target(관절한계 게이트 +
joint-goal 직접이동) 및 _register_floor_collision(PlanningScene ACM
직접 조작)을 새로 도입했으나, approach 자체가 전혀 안 움직이는 회귀가
발생해(게이트가 후보 전부를 거부, ACM 조작이 SRDF 기반 자기충돌 허용
목록을 깨뜨림) 실측 검증된 이전 상태로 전량 복원함. 복원된 구조:
pre-rotate(joint1 사전회전) + _pick_best_yaw_candidate(2후보 IK 이동량
비교) + _move(pose-goal, planner_chain=['PTP','LIN']→OMPL 안전폴백).
바닥/자기충돌은 SRDF(nero.srdf의 disable_collisions)와 joint_limits.yaml
레벨에서 처리하며, 런타임에 PlanningScene ACM을 직접 재작성하지 않는다.
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
from std_srvs.srv import Empty as EmptySrv
import tf2_ros
from geometry_msgs.msg import Vector3Stamped
from rclpy.duration import Duration as RclpyDuration
try:
    from agx_arm_msgs.msg import AgxArmStatus
    HAVE_ARM_STATUS_MSG = True
except ImportError:
    HAVE_ARM_STATUS_MSG = False
# [신설] 그립 각도/자세 선택 로직 분리 모듈 (ROS 비의존, 순수 함수).
# 상세: grasp_kinematics.py 모듈 docstring 참고 (폐기된 접근/채택 이유/한계).
from .grasp_kinematics import (
    SIDE_MIN_DIST, SIDE_PITCH_DEG, SIDE_TCP_OFFSET, TOP_ANGLE_PITCH_DEG,
    QUAT_TOP_DOWN, QUAT_TOP_DOWN_R, QUAT_TOP_DOWN_L, QUAT_TOP_DOWN_RR,
    QUAT_HOME, QUAT_SIDE_FRONT, SIDE_TAG, GRASP_DIR_MAP, LABEL_GRASP_HINT,
    quat_angle_diff, top_down_angle_quat, sim_top_down_angle_quat,
    side_quat_for, side_reachability_check, auto_grasp_quat,
    resolve_grasp_quat, YawCandidateSelector,
)
# [신설] 폐루프(closed-loop) place 검증 -- 그리퍼를 열고 물러난 뒤 실제로
# 목표 위치에 물체가 안착했는지 perception으로 재확인한다.
# 상세: placement_verification.py 모듈 docstring 참고.
from .placement_verification import verify_placement
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
# ── 안전장치: 조인트 급회전("빙글 도는 문제") 감지 ────────────────────────────
JOINT_JUMP_MAX_DEG_PER_SEC = 150.0
JOINT_JUMP_MIN_DT_SEC = 0.02  # 이보다 짧은 간격의 샘플은 노이즈로 보고 무시
# ── 그리퍼 상수 ───────────────────────────────────────────────────────────────
GRIPPER_OPEN  = 0.08
GRIPPER_CLOSE = 0.01
GRIPPER_FORCE = 1.5
SIM_GRIPPER_OPEN  = [0.08]
SIM_GRIPPER_CLOSE = [0.01]
# ── 시퀀스 오프셋 (미터) ──────────────────────────────────────────────────────
APPROACH_Z = 0.10
DESCEND_Z  = 0.0
LIFT_Z     = 0.10
# [이관] SIDE_MIN_DIST, SIDE_PITCH_DEG, TOP_ANGLE_PITCH_DEG -> grasp_kinematics.py (import 참고)
# [이관] top_down_angle_quat, sim_top_down_angle_quat ->
# grasp_kinematics.top_down_angle_quat / sim_top_down_angle_quat
# ── 그리퍼 TCP 오프셋 (gripper_flange → 손가락 중간 지점까지 거리) ───────────
TOP_TCP_OFFSET  = 0.1358  # 미터 (GRIPPER_FLANGE_TO_FINGERTIP과 동일 실측값, 2026-07 확정)
# ── descend z 안전 하한 (그리퍼 끝단 기준) ─────────────────────────────
GRIPPER_FLANGE_TO_FINGERTIP = 0.1358
DESCEND_MIN_FINGERTIP_Z = 0.03
DESCEND_MIN_FLANGE_Z = DESCEND_MIN_FINGERTIP_Z + GRIPPER_FLANGE_TO_FINGERTIP
# [이관] SIDE_TCP_OFFSET -> grasp_kinematics.SIDE_TCP_OFFSET (import 참고)
# place(내려놓기) 시 그리퍼 손가락 끝이 바닥에서 떨어진 높이.
PLACE_DROP_Z = 0.01   # 미터, place_pos.z 기준 손가락 끝 절대 높이
# ── 홈 자세 ───────────────────────────────────────────────────────────────────
HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, 0.75
# ── 타이밍 ────────────────────────────────────────────────────────────────────
MOVE_DELAY    = 6.0
GRIPPER_DELAY = 2.0  # [2026-07 수정] 4.0->2.0, 체감 속도 개선
# [신설] place 후 폐루프 검증 전 perception이 새 프레임을 관측하도록
# 주는 여유 시간. 그리퍼가 lift로 물러난 직후라 시야는 이미 열려있지만,
# 토픽 발행 주기/처리 지연을 감안해 약간의 여유를 둔다.
VERIFY_SETTLE_SEC = 0.8
# [신설] pre-rotate와 verify-vantage 재사용 -- joint1 명령각과 실측 물체
# 방향(atan2) 사이 오프셋. 스캔 실측 2개 지점(-155°→46°, 145°→-39°)의
# 평균 약 188°(범위 176~201°)로 확인됨. 카메라가 팔 끝에 top-down으로
# 장착되어 "정면"의 의미가 base 기준과 반대에 가깝게 어긋나는 것으로 추정.
BEARING_OFFSET_DEG = 188.0
# ── MoveIt2 플래너 설정 ────────────────────────────────────────────────────────
TOL_POS_LOOSE   = 0.01
TOL_ORI_LOOSE   = 0.15   # 약 8.6도 — 수평 이동 단계
TOL_POS_TIGHT   = 0.003
TOL_ORI_TIGHT   = 0.03   # 약 1.7도 (0.05->0.03, 실측 6~7도 편차 확인되어 완만히 강화)
MOVE_MAX_ATTEMPTS = 3
MOVE_TIMEOUT      = 30.0  # 단일 시도 타임아웃
# ── 반경 기반 워크스페이스 필터 (2026-07 추가) ────────────────────────────
MIN_REACH_R_M = 0.20   # 이보다 가까우면 top-down으로는 물리적으로 안 닿는 "도넛 구멍" -> 사전배제
BOUNDARY_R_M  = 0.30   # 이 반경 안쪽이면 tolerance 완화 폴백 필요할 수 있는 "경계 링"
# descend용 tolerance 폴백 단계 (x/y축=top-down 유지정도만 완화, z축=그립각도는 항상 고정)
DESCEND_TOL_ORI_FALLBACK = [
    (TOL_ORI_TIGHT, TOL_ORI_TIGHT, TOL_ORI_TIGHT),   # 1차: 원래대로 (완전 top-down)
    (0.10, 0.10, TOL_ORI_TIGHT),                      # 2차: 앞뒤/틸트만 살짝 완화 (~5.7도)
    (0.20, 0.20, TOL_ORI_TIGHT),                      # 3차: 더 완화 (~11.5도) -- 그립각도는 여전히 고정
]
# ── (A) 조인트1 스윕 탐색 기본값 (나중에 실제 값으로 교체 가능) ─────────────
JOINT1_LOWER = -2.70526
JOINT1_UPPER = 2.70526
DEFAULT_SCAN_BASE_JOINTS = {
    'joint1': 0.0, 'joint2': -0.8026, 'joint3': 0.0,
    'joint4': 1.8273, 'joint5': 0.0, 'joint6': 0.0, 'joint7': 1.3,
}
DEFAULT_SCAN_STEP_DEG = 60.0
SCAN_STEP_SETTLE_SEC = 2.0
SCAN_DEDUP_DIST_M = 0.05
# [이관] QUAT_TOP_DOWN 계열, GRASP_DIR_MAP, SIDE_TAG, LABEL_GRASP_HINT,
# euler_to_quat, quat_angle_diff, side_quat_for, side_reachability_check,
# auto_grasp_quat -> 전부 grasp_kinematics.py로 이관 (파일 상단 import 참고).
# 호출부는 grasp_kinematics.resolve_grasp_quat() 하나로 통합됨
# (기존 on_command 3곳의 거의 동일한 중복 분기 로직 대체).
class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        # [2026-07 추가] pick 시점 카메라 orientation 로그용
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        from moveit_msgs.srv import GetPositionFK
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk')
        from moveit_msgs.srv import GetStateValidity
        self._state_validity_client = self.create_client(
            GetStateValidity, '/check_state_validity')
        self.use_moveit2 = self.declare_parameter('use_moveit2', True).value
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self._cb = ReentrantCallbackGroup()
        self.latest_joint_state = None
        self._prev_joint_snapshot = None   # (dict{name:pos}, monotonic_time)
        self._safety_tripped = False
        self._emergency_stop_client = self.create_client(EmptySrv, 'emergency_stop')
        self.create_subscription(
            JointState, '/feedback/joint_states', self._on_joint_state_safety_check, 10)
        self.latest_joint_state_sim = None
        self.create_subscription(
            JointState, '/joint_states',
            lambda msg: setattr(self, 'latest_joint_state_sim', msg), 10)
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
            self.moveit2.allowed_planning_time = 5.0
            self.moveit2.num_planning_attempts = 20
            self.get_logger().info(
                'MoveIt2 ENABLED (allowed_planning_time=5.0s, num_planning_attempts=20)')
        self._gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
        )
        self.latest_objects = []
        self._latest_objects_stamp = 0.0
        self._scanned_boxes = []  # [2026-07 추가] scan_box 결과 저장
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
            self._latest_objects_stamp = time.time()
        except Exception:
            pass
    def on_command(self, msg):
        self.get_logger().info(f'[CMD] 수신: {msg.data[:80]}')
        if self._safety_tripped:
            try:
                cmd_peek = json.loads(msg.data)
            except Exception:
                cmd_peek = {}
            if cmd_peek.get('action') != 'clear_safety':
                self.get_logger().error(
                    '🚨 안전 잠금 상태라 명령을 거부합니다. '
                    '로봇 상태 육안 확인 후 {"action":"clear_safety"} 로 해제하세요.')
                return
            else:
                self._safety_tripped = False
                with self.lock:
                    self.busy = False
                self.get_logger().warn('✅ 안전 잠금 해제됨 (clear_safety).')
                self._publish_result('success', 'safety_cleared')
                return
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
            use_from_scan = cmd.get('from_scan', False)
            override_pos = cmd.get('override_pos')
            override_angle_deg = cmd.get('override_angle_deg')
            skip_reacquire = override_pos is not None
            if skip_reacquire:
                pos = override_pos
                angle_deg = override_angle_deg if override_angle_deg is not None else 0.0
                self.get_logger().info(
                    f'[override] 좌표/각도 직접 지정 -> pos={pos} angle_deg={angle_deg} '
                    f'(perception 재조회 전부 스킵)')
            elif use_from_scan:
                box_index = cmd.get('box_index')
                _label_matches = [b for b in self._scanned_boxes if b.get('label') == label]
                if box_index is not None:
                    # [신규] 명시적 인덱스 지정 -- 큐 순서(맨 앞)가 아니라
                    # 사용자가(Claude가) remaining_scanned_boxes에서 본
                    # 특정 항목을 정확히 집는다. 이미 다른 박스 위에
                    # 쌓인 좌표를 "다음 큐 항목"으로 착각해 잘못 집는
                    # 사고(2026-07-22 실측)가 구조적으로 발생하지 않는다.
                    if 0 <= box_index < len(_label_matches):
                        _scan_match = _label_matches[box_index]
                    else:
                        _scan_match = None
                        self.get_logger().warn(
                            f"[from_scan] box_index={box_index} 범위 초과 "
                            f"('{label}' 매칭 {len(_label_matches)}개)")
                else:
                    # 하위호환: 인덱스 미지정 시 기존처럼 큐 맨 앞
                    _scan_match = _label_matches[0] if _label_matches else None
                if _scan_match is not None:
                    pos = {'x': _scan_match['x'], 'y': _scan_match['y'], 'z': _scan_match['z']}
                    angle_deg = _scan_match.get('angle_base_deg') or 0.0
                    skip_reacquire = True
                    self.get_logger().info(
                        f'[from_scan] 저장된 좌표 사용 -> pos={pos} angle_deg={angle_deg}')
                    try:
                        self._scanned_boxes.remove(_scan_match)
                    except ValueError:
                        pass
                else:
                    pos, angle_deg = None, None
                    self.get_logger().warn(f"[from_scan] 저장된 '{label}' 박스 없음.")
            else:
                pos, angle_deg = self._find_object_with_angle(label)
            self._box_angle_deg = angle_deg
            if pos is None:
                self.get_logger().warn(f"'{label}' 못 찾음.")
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            # [단순화] GRASP_DIR_MAP 조회 + SIDE_TAG 체크 + 라벨/위치 휴리스틱을
            # grasp_kinematics.resolve_grasp_quat() 하나로 통합 (동작 동일).
            quat, is_side = resolve_grasp_quat(grasp_dir, pos, label)
            self.get_logger().info(
                f'PICK 시작: {label} @ {pos} | 자세: {grasp_dir or "auto"} '
                f'({"side" if is_side else "top"}) {[round(v,3) for v in quat]} | '
                f'박스각도={angle_deg}')
            t = threading.Thread(target=self._pick_sequence, args=(pos, quat, is_side, label, skip_reacquire, angle_deg))
            t.daemon = False
            t.start()
        elif action == 'place':
            pos = cmd.get('place_pos')
            if pos is None:
                with self.lock:
                    self.busy = False
                return
            # [2026-07 추가] 반경 사전 필터 -- 도넛 구멍(r < MIN_REACH_R_M)은
            # top-down을 완화해도 물리적으로 안 닿는 구간이라, OMPL/IK를
            # 시도하기 전에 즉시 거부한다. (경계 링은 여기서 거르지 않고
            # descend 단계의 tolerance 폴백 체인이 처리하도록 넘긴다.)
            _r = math.hypot(pos.get('x', 0.0), pos.get('y', 0.0))
            if _r < MIN_REACH_R_M:
                self.get_logger().warn(
                    f'PLACE 거부: 목표 반경 r={_r:.3f}m < {MIN_REACH_R_M}m '
                    f'(물리적 도달 불가 구간 -- top-down 완화로도 해결 안 됨)')
                self._publish_result('rejected', f'place_pos too close to base (r={_r:.3f}m)')
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            # [단순화] place는 grasp_dir 미지정 시 pick과 달리 라벨 휴리스틱이
            # 아니라 "스캔된 박스 각도(top-down)"가 기본값이라, 이 분기만은
            # resolve_grasp_quat의 auto 경로를 그대로 쓸 수 없다 (의도된
            # 차이 — nero_robot_place_reachability memory: 명시적 grasp_dir
            # 없이는 top-down 유지가 기본 안전값). explicit 지정 시에만
            # resolve_grasp_quat로 통합.
            if grasp_dir:
                quat, is_side = resolve_grasp_quat(grasp_dir, pos, label='')
            else:
                is_side = False
                angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
                quat = self._top_down_quat_for(pos, angle_deg)
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
            # [단순화] move도 place와 동일 이유로 auto 기본값은 top-down 유지.
            # (원본 버그 후보였던 기본값 불일치 — move만 QUAT_HOME을 fallback으로
            # 썼던 것 — 는 resolve_grasp_quat 도입으로 place와 동일하게 통일됨.
            # 실질 동작은 원래도 is_side=False 분기에서 top_down으로 덮어써져
            # 차이가 없었으므로 회귀 아님.)
            if grasp_dir:
                quat, is_side = resolve_grasp_quat(grasp_dir, pos, label='')
            else:
                is_side = False
                quat = None
            if not is_side:
                angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
                quat = self._top_down_quat_for(pos, angle_deg)
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
        elif action == 'scan_box':
            base_joints = cmd.get('base_joints', DEFAULT_SCAN_BASE_JOINTS)
            step_deg = cmd.get('step_deg', DEFAULT_SCAN_STEP_DEG)
            scan_label = cmd.get('target_label', 'box')
            self.get_logger().info(
                f'SCAN_BOX 시작: base_joints={base_joints} step_deg={step_deg} label={scan_label}')
            t = threading.Thread(
                target=self._scan_box_sequence, args=(base_joints, step_deg, scan_label))
            t.daemon = False
            t.start()
        elif action == 'home':
            self.get_logger().info('HOME 시작')
            threading.Thread(target=self._home_sequence, daemon=True).start()
        else:
            self.get_logger().warn(f"알 수 없는 action: '{action}'")
            with self.lock:
                self.busy = False
    # ── 안전장치: 조인트 급회전 감지 ──────────────────────────────────────────
    def _on_joint_state_safety_check(self, msg: JointState):
        self.latest_joint_state = msg
        now = time.monotonic()
        current = dict(zip(msg.name, msg.position))
        if self._prev_joint_snapshot is not None:
            prev, prev_time = self._prev_joint_snapshot
            dt = now - prev_time
            if dt >= JOINT_JUMP_MIN_DT_SEC:
                for jname, pos in current.items():
                    if jname not in prev or jname == 'gripper':
                        continue
                    delta_deg = abs(math.degrees(pos - prev[jname]))
                    speed_deg_s = delta_deg / dt
                    if speed_deg_s > JOINT_JUMP_MAX_DEG_PER_SEC and not self._safety_tripped:
                        self.get_logger().error(
                            f'🚨 [안전정지] {jname} 급회전 감지: '
                            f'{delta_deg:.1f}° in {dt*1000:.0f}ms '
                            f'({speed_deg_s:.0f}°/s > 임계값 {JOINT_JUMP_MAX_DEG_PER_SEC}°/s)')
                        self._trigger_emergency_stop(
                            reason=f'{jname} {speed_deg_s:.0f}°/s 급회전 감지')
        self._prev_joint_snapshot = (current, now)
    def _trigger_emergency_stop(self, reason: str):
        """emergency_stop 서비스 호출 + 이후 모든 이동 명령 거부하는 안전 잠금."""
        self._safety_tripped = True
        with self.lock:
            self.busy = True  # 이후 on_command가 새 명령을 아예 안 받도록 잠금
        self.get_logger().error(f'🚨🚨🚨 EMERGENCY STOP 발동: {reason}')
        try:
            if self._emergency_stop_client.wait_for_service(timeout_sec=1.0):
                self._emergency_stop_client.call_async(EmptySrv.Request())
                self.get_logger().error('emergency_stop 서비스 호출 완료.')
            else:
                self.get_logger().error(
                    'emergency_stop 서비스 응답 없음! agx_arm_ctrl_single_node 상태 즉시 확인 필요.')
        except Exception as e:
            self.get_logger().error(f'emergency_stop 서비스 호출 중 예외: {e}')
        self.get_logger().error(
            '⚠ 안전 잠금 상태입니다. 로봇 상태를 육안으로 확인한 뒤, '
            '"clear_safety" 액션으로 잠금을 해제해야 다음 명령을 받습니다.')
        self._publish_result('emergency_stop', reason)
    # ── 유틸 ──────────────────────────────────────────────────────────────────
    def _find_object(self, label):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d')
        return None
    OBJECT_CONF_MIN = 0.35  # 이 아래는 그림자/오검출로 간주해 후보에서 제외
    def _find_object_with_angle(self, label):
        BOX_MIN_Z = 0.015
        candidates = [
            obj for obj in self.latest_objects
            if obj.get('label') == label
            and obj.get('confidence', 0.0) >= self.OBJECT_CONF_MIN
            and obj.get('center_3d', {}).get('z', 0.0) >= BOX_MIN_Z
        ]
        if not candidates:
            all_matching = [o for o in self.latest_objects if o.get('label') == label]
            self.get_logger().warn(
                f'[find_object] candidates 없음. label={label} '
                f'전체 매칭={len(all_matching)}개, '
                f'confidences={[round(o.get("confidence",0),3) for o in all_matching]}')
            return None, None
        def _dist(obj):
            c = obj.get('center_3d', {})
            return math.hypot(c.get('x', 0.0), c.get('y', 0.0))
        best = min(candidates, key=_dist)
        return best.get('center_3d'), best.get('angle_base_deg', None)
    # ── top-down 쿼터니언 백엔드 분기 ────────────────────────────────────────
    def _top_down_quat_for(self, pos, angle_deg):
        """
        top 그립(side 아님)용 쿼터니언을 self.use_moveit2 여부로 분기해서 반환.
        """
        if self.use_moveit2:
            return sim_top_down_angle_quat(angle_deg)
        return top_down_angle_quat(pos['x'], pos['y'], angle_deg)
    # ── 시퀀스 ────────────────────────────────────────────────────────────────
    def _pick_sequence(self, pos, quat, is_side=False, label='', skip_reacquire=False, angle_deg=0.0):
        try:
            if is_side:
                ok, reason = side_reachability_check(pos)
                if not ok:
                    self.get_logger().warn(f'PICK 거부: {reason}')
                    self._publish_result('rejected', reason)
                    return
            if is_side:
                dx, dy = pos['x'], pos['y']
                dist = math.sqrt(dx*dx + dy*dy) or 1.0
                px = dx - (dx / dist) * SIDE_TCP_OFFSET
                py = dy - (dy / dist) * SIDE_TCP_OFFSET
                offset_desc = f'side(TCP오프셋 {SIDE_TCP_OFFSET}m 적용)'
            else:
                px = pos['x']
                py = pos['y']
                offset_desc = 'top(z)'
            if is_side:
                approach_z = pos['z']
                descend_z  = pos['z']
                lift_z     = pos['z'] + LIFT_Z      # top과 동일한 상승폭
            else:
                pz = pos['z']
                approach_z = pz + APPROACH_Z + TOP_TCP_OFFSET
                descend_z  = pz + DESCEND_Z + TOP_TCP_OFFSET
                descend_z  = max(descend_z, DESCEND_MIN_FLANGE_Z)
                lift_z     = pz + LIFT_Z + TOP_TCP_OFFSET
            # [2026-07 추가] approach 진입 전 joint1만 먼저 물체 방향으로
            # 회전. 스캔이 끝난 자세(예: joint1=145°)에서 바로 approach를
            # 시도하면, 물체 방향까지의 큰 차이를 approach의 나머지
            # 조인트(top-down 자세 유지)와 동시에 풀어야 해서 IK가 훨씬
            # 어려워지는 것이 실측+로그 분석으로 확인됨. joint1만 먼저
            # 물체 쪽으로 돌려 "쉬운 시작 상태"를 만든다.
            # BEARING_OFFSET_DEG: joint1 명령각과 실측 물체 방향(atan2)
            # 사이 오프셋. 스캔 실측 2개 지점(-155°→46°, 145°→-39°)의
            # 평균 약 188°(범위 176~201°)로 확인됨. 카메라가 팔 끝에
            # top-down으로 장착되어 "정면"의 의미가 base 기준과 반대에
            # 가깝게 어긋나는 것으로 추정 — 완벽한 정렬이 목적이 아니라
            # IK 부담을 줄이는 사전 회전이므로 이 정도 오차는 허용.
            target_bearing = math.atan2(pos['y'], pos['x']) - math.radians(BEARING_OFFSET_DEG)
            target_bearing = math.atan2(math.sin(target_bearing), math.cos(target_bearing))
            js = self.latest_joint_state_sim
            if js is not None:
                joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
                current = dict(zip(js.name, js.position))
                pre_target = [current.get(jn, 0.0) for jn in joint_names]
                pre_target[0] = target_bearing
                self.get_logger().info(
                    f'[pre-rotate] joint1을 물체 방향 근처'
                    f'({math.degrees(target_bearing):.1f}\xb0)로 사전 회전...')
                if self.use_moveit2:
                    self.moveit2.force_reset_executing_state()
                    self.moveit2.max_velocity = 0.3
                    self.moveit2.max_acceleration = 0.3
                    self.moveit2.pipeline_id = ''
                    self.moveit2.planner_id = ''
                    self.moveit2.move_to_configuration(pre_target)
                    time.sleep(0.5)
                    deadline = time.time() + MOVE_TIMEOUT
                    while time.time() < deadline:
                        if (not self.moveit2._MoveIt2__is_motion_requested and
                                not self.moveit2._MoveIt2__is_executing):
                            break
                        time.sleep(0.1)
                    # [2026-08 수정, 코드리뷰 발견] 이전엔 motion_suceeded를
                    # 확인 안 하고 무조건 approach로 넘어갔음 -- pre-rotate가
                    # 조용히 실패해도 원인 불명의 "approach가 왜 이렇게 크게
                    # 도나"로만 보였음. 하드게이트(중단)는 걸지 않음 --
                    # pre-rotate는 순수 최적화라 실패해도 approach 자체의
                    # 재시도/OMPL 폴백이 이어받을 수 있음(handoff 6절 원칙).
                    if not self.moveit2.motion_suceeded:
                        self.get_logger().warn(
                            '[pre-rotate] 이동 실패/중단 (motion_suceeded=False) -- '
                            '사전회전 안 된 자세에서 approach 시작함. '
                            'approach 자체의 재시도/OMPL 폴백에 맡기고 계속 진행.')
                else:
                    self._move_joints_real_wait(pre_target)
            else:
                self.get_logger().warn('[pre-rotate] latest_joint_state_sim=None, 사전 회전 스킵')
            self.get_logger().info(
                f'1/6: 접근 (approach) | {offset_desc}')
            if not is_side:
                _approach_angle_deg = 0.0
                quat = self._top_down_quat_for(pos, _approach_angle_deg)
                # [2026-07 수정] yaw tolerance=3.14(완전 자유)가 너무
                # 넓어서, IK가 gripper_flange와 link6이 충돌하는 극단적인
                # yaw 해를 고르는 경우가 실측 확인됨. 1.6(±91.7도)으로
                # 좁혀 자기충돌 유발 여지를 줄인다.
                approach_tol_ori = (TOL_ORI_TIGHT, TOL_ORI_TIGHT, 1.6)
                backend = 'sim(pitch180)' if self.use_moveit2 else 'real(pitch90)'
                self.get_logger().info(
                    f'[top:{backend}] approach는 yaw 자유(위치만 정렬) -> '
                    f'quat={[round(v,3) for v in quat]}')
                ok = self._move(px, py, approach_z, quat, tol_ori=approach_tol_ori, planner_chain=['PTP', 'LIN'])
            else:
                ok = self._move(px, py, approach_z, quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['PTP', 'LIN'])
            if not ok:
                raise RuntimeError('approach 이동 실패 (재시도 초과)')
            self.get_logger().info('2/6: 그리퍼 열기')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)
            if not is_side:
                if skip_reacquire:
                    angle_deg_refined = angle_deg
                    self.get_logger().info(
                        f'[override] align 각도 재조회 스킵, override 값 사용: {angle_deg_refined}\xb0')
                else:
                    angle_deg_refined = None
                    for _retry in range(3):
                        _, angle_deg_refined = self._find_object_with_angle(label)
                        if angle_deg_refined is not None:
                            break
                        time.sleep(0.3)
                if angle_deg_refined is not None:
                    best_deg, quat = self._pick_best_yaw_candidate(
                        pos, angle_deg_refined, ik_check_z=approach_z)
                    self.get_logger().info(
                        f'[top] 3/6: 정렬(align) 재조회각도={angle_deg_refined}\xb0 -> '
                        f'선택각도={best_deg:.1f}\xb0 quat={[round(v,3) for v in quat]}')
                    ok = self._move(px, py, approach_z, quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['PTP', 'LIN'])
                    if not ok:
                        raise RuntimeError('align 이동 실패 (재시도 초과)')
                    _current_top_angle_deg = best_deg
                else:
                    self.get_logger().warn('[top] 각도 재조회 실패, yaw=0 유지')
                    _current_top_angle_deg = _approach_angle_deg
                time.sleep(0.3)
                try:
                    _refreshed_pos = pos if skip_reacquire else \
                        self._find_object_with_angle(label)[0]
                    _cam_tf = self.tf_buffer.lookup_transform(
                        'base_link', 'camera_color_optical_frame',
                        rclpy.time.Time(), timeout=RclpyDuration(seconds=0.2))
                    _cq = _cam_tf.transform.rotation
                    _cam_yaw = math.degrees(math.atan2(
                        2*(_cq.w*_cq.z + _cq.x*_cq.y), 1 - 2*(_cq.y*_cq.y + _cq.z*_cq.z)))
                    _p = _refreshed_pos if _refreshed_pos is not None else pos
                    _dist = math.hypot(_p.get('x', 0.0), _p.get('y', 0.0))
                    _bearing = math.degrees(math.atan2(_p.get('y', 0.0), _p.get('x', 0.0)))
                    self.get_logger().info(
                        f'[calib_debug] align완료후 cam_yaw={_cam_yaw:.1f} | '
                        f'obj_pos=({_p.get("x",0):.3f},{_p.get("y",0):.3f},{_p.get("z",0):.3f}) | '
                        f'obj_dist={_dist:.3f}m obj_bearing={_bearing:.1f} | '
                        f'rel_bearing={_bearing - _cam_yaw:.1f}')
                except Exception as e:
                    self.get_logger().warn(f'[calib_debug] align완료후 tf 조회 실패: {e}')
                if _refreshed_pos is not None:
                    px = _refreshed_pos.get('x', px)
                    py = _refreshed_pos.get('y', py)
                    self.get_logger().info(
                        f'[top] descend 목표 위치 갱신: ({px:.4f}, {py:.4f})')
                    # [2026-08 수정, 코드리뷰 발견] real 백엔드는
                    # top_down_angle_quat(x,y,angle_deg)가 yaw=atan2(y,x)로
                    # 위치에 의존하는데, 위에서 px,py만 갱신하고 quat은
                    # 갱신 전 pos 기준 값이 그대로 남아있었음. sim은 quat이
                    # 위치와 무관해 문제가 안 드러났지만, real에서는 descend
                    # 커맨드의 위치와 orientation이 서로 다른 시점 값으로
                    # 섞여서 나가는 버그였음. 갱신된 위치로 quat도 재계산.
                    quat = self._top_down_quat_for(
                        {'x': px, 'y': py, 'z': _refreshed_pos.get('z', pos.get('z', 0.0))},
                        _current_top_angle_deg)
            
            if is_side:
                self.get_logger().info('4/6: 내려가기 (descend) — side는 approach와 동일 높이, 재확인만')
            else:
                self.get_logger().info('4/6: 내려가기 (descend) — tight tolerance (+ 실패 시 완화 폴백)')
            if not is_side:
                ok = self._move_descend_with_tilt_fallback(px, py, descend_z, quat)
            else:
                ok = self._move(px, py, descend_z, quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['LIN'])
            if not ok:
                raise RuntimeError('descend 이동 실패 (재시도 초과, tolerance 완화 폴백까지 실패)')
            self.get_logger().info('5/6: 그리퍼 닫기')
            self._gripper(SIM_GRIPPER_CLOSE, GRIPPER_CLOSE)
            time.sleep(GRIPPER_DELAY)
            self.get_logger().info('6/6: 들어올리기 (lift) — joint-space (빙글 방지)')
            if not is_side:
                lift_quat = quat
            else:
                lift_quat = quat
            lift_candidates = [lift_z, pos['z'] + 0.06 + (0 if is_side else TOP_TCP_OFFSET),
                   pos['z'] + 0.03 + (0 if is_side else TOP_TCP_OFFSET)]
            ok = False
            for i, lz in enumerate(lift_candidates):
                if i > 0:
                    self.get_logger().warn(
                        f'  lift 높이 낮춰서 재시도: {lz:.3f}m (원래 목표 {lift_z:.3f}m)')
                ok = self._move(px, py, lz, lift_quat, tol_ori=TOL_ORI_TIGHT, planner_chain=['LIN'])
                if ok:
                    break
            if not ok:
                raise RuntimeError('lift 이동 실패 (모든 높이 재시도 초과)')
            self.get_logger().info('✅ PICK 완료')
            self._publish_result(
                'success', 'pick_complete',
                extra={'remaining_scanned_boxes': self._scanned_boxes})
        except Exception as e:
            self.get_logger().error(f'PICK 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
    def _place_sequence(self, pos, quat, is_side=False):
        """[2026-08 재설계] 실패 시 좌표를 자동으로 시프트해가며 재시도한다.
        근거: nero_robot_place_reachability memory (2026-07-30~08-03 실측).
          - approach/descend 실패는 특정 (x,y) 국소 도달불가 영역(singularity)일
            수 있다. 같은 좌표를 그냥 재시도해봐야 소용없다.
          - +y 먼저 시도, 실패하면 -y도 시도 (한 방향만 믿으면 안 됨 -- 08-03
            사례는 +y도 실패, -y만 성공했음).
          - z가 비정상적으로 높으면(≳0.2m) x/y가 아니라 z를 0.05~0.1m씩
            낮춰야 하는 별도의 height ceiling 문제였음.
        이 재시도는 이전에는 Claude가 place_object 실패 응답을 보고 매번
        직접 판단해서 순차 호출하던 것 -- 서버가 대신 처리하므로 Claude는
        place_object를 한 번만 호출하면 된다 (토큰/지연시간 절감).
        """
        try:
            # [신설] 폐루프 검증용 -- place 시작 전(그리퍼가 아직 아무것도
            # 안 놓은 상태) 장면을 스냅샷. 쌓기 시나리오에서 베이스 박스를
            # "새로 놓인 물체"로 착각하지 않으려면 이 스냅샷이 필수다
            # (placement_verification.py 모듈 docstring "쌓기 시 오탐 방지"
            # 참고). approach/descend 재시도 중에는 장면이 안 바뀌므로
            # 후보 좌표를 여러 번 시도해도 이 스냅샷 하나를 계속 재사용한다.
            before_objects = list(self.latest_objects)
            if is_side:
                # side는 좌표 시프트 재시도 대상이 아님(원본과 동일) --
                # 도달불가 판정은 여기서 즉시 'rejected'로 보고하고 종료.
                ok, reason = side_reachability_check(pos)
                if not ok:
                    self.get_logger().warn(f'PLACE 거부: {reason}')
                    self._publish_result('rejected', reason)
                    return
            candidates = [dict(pos)]
            if not is_side:
                shifted_plus = dict(pos); shifted_plus['y'] = pos['y'] + 0.05
                shifted_minus = dict(pos); shifted_minus['y'] = pos['y'] - 0.05
                candidates += [shifted_plus, shifted_minus]
                if pos['z'] > 0.20:
                    for dz in (0.05, 0.10, 0.15):
                        lowered = dict(pos); lowered['z'] = pos['z'] - dz
                        candidates.append(lowered)

            last_err = None
            for i, cand in enumerate(candidates):
                try:
                    verification = self._try_place_at(cand, quat, is_side, before_objects)
                    if i > 0:
                        self.get_logger().warn(
                            f'  PLACE: 원래 목표 {pos} 실패 -> 대체 좌표 {cand}로 성공')
                    self.get_logger().info('✅ PLACE 완료')
                    # [신설] verified=False여도 여기서 재시도하지 않는다 --
                    # 그리퍼는 이미 물체를 놓고 물러난 상태라, 좌표를 바꿔서
                    # 다시 움직여봐야 애초에 잡고 있는 물체가 없어 의미가
                    # 없다. 검증 결과는 그대로 Claude에게 보고해서 판단을
                    # 맡긴다 (재감지 후 다시 pick&place 하거나, 사용자에게
                    # 알리거나).
                    self._publish_result(
                        'success', 'place_complete',
                        extra={
                            'actual_place_pos': cand,
                            'placement_verified': verification['verified'],
                            'verification_reason': verification['reason'],
                        })
                    return
                except RuntimeError as e:
                    last_err = e
                    self.get_logger().warn(
                        f'  PLACE 후보 {i+1}/{len(candidates)} {cand} 실패: {e}')
                    continue
            self.get_logger().error(f'PLACE 오류: 모든 후보 좌표 소진 ({last_err})')
            self._publish_result('failed', f'모든 대체 좌표 실패: {last_err}')
        finally:
            with self.lock:
                self.busy = False

    def _move_to_joint_config(self, target_positions, label=''):
        """[신설] joint-space 목표로 이동하고 완료까지 대기, 성공 시 True.
        scan_box/pre-rotate/home과 동일 패턴(pipeline_id/planner_id 리셋
        포함 -- 이거 빠뜨리면 직전 Pilz 이동 설정이 남아 매번 즉시
        ABORTED되는 버그가 재발한다, 2026-08 scan_box에서 실측 확인된
        바로 그 문제). pose 기반 _move()와 달리 IK를 풀 필요가 없어서
        NO_IK_SOLUTION으로 실패할 위험이 없다 -- verify-vantage처럼
        "반드시 도달 가능해야 하는" 이동에 적합하다.
        """
        if not self.use_moveit2:
            return self._move_joints_real_wait(target_positions)
        self.moveit2.force_reset_executing_state()
        self.moveit2.pipeline_id = ''
        self.moveit2.planner_id = ''
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3
        self.moveit2.move_to_configuration(target_positions)
        time.sleep(0.5)
        deadline = time.time() + MOVE_TIMEOUT
        while time.time() < deadline:
            if (not self.moveit2._MoveIt2__is_motion_requested and
                    not self.moveit2._MoveIt2__is_executing):
                break
            time.sleep(0.1)
        ok = self.moveit2.motion_suceeded
        if not ok:
            self.get_logger().warn(f'  [{label}] joint-space 이동 실패/중단')
        return ok

    def _try_place_at(self, pos, quat, is_side=False, before_objects=None):
        """실제 approach->descend->open->lift 시퀀스 1회 시도.
        기존 _place_sequence 본문과 100% 동일 (분리만 함). 실패 시
        self._publish_result를 직접 부르지 않고 RuntimeError만 던진다 --
        결과 보고는 바깥의 _place_sequence 재시도 루프가 전담한다.

        before_objects: place 시작 전 장면 스냅샷 (폐루프 검증용, 쌓기
        시나리오에서 베이스 박스 오탐 방지 -- placement_verification.py
        모듈 docstring 참고).
        """
        if not is_side:
            place_angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
        if is_side:
            ok, reason = side_reachability_check(pos)
            if not ok:
                raise RuntimeError(reason)
            dx, dy = pos['x'], pos['y']
            dist = math.sqrt(dx*dx + dy*dy) or 1.0
            px = dx - (dx / dist) * SIDE_TCP_OFFSET
            py = dy - (dy / dist) * SIDE_TCP_OFFSET
            approach_z = pos['z'] + APPROACH_Z
            descend_z = pos['z'] + APPROACH_Z
            lift_z = pos['z'] + APPROACH_Z + LIFT_Z
        else:
            px = pos['x']
            py = pos['y']
            approach_z = pos['z'] + APPROACH_Z + TOP_TCP_OFFSET
            descend_z = pos['z'] + PLACE_DROP_Z + TOP_TCP_OFFSET
            descend_z = max(descend_z, DESCEND_MIN_FLANGE_Z)
            lift_z = pos['z'] + LIFT_Z + TOP_TCP_OFFSET
        # [2026-07 삭제] place pre-rotate 제거. approach가 이미
        # yaw 자유(±91.7°)로 위치만 맞추면 되는 이동이라, 그 전에
        # joint1을 미리 돌려놓는 별도 스텝이 굳이 필요 없다고 판단됨
        # (pick 쪽 pre-rotate는 그대로 유지 -- approach의 목표
        # orientation이 top-down 고정이라 사정이 다름).
        self.get_logger().info('1/4: 접근 (approach)')
        if not is_side:
            entry_quat = self._top_down_quat_for(pos, 0.0)
            approach_tol_ori = (TOL_ORI_TIGHT, TOL_ORI_TIGHT, 1.6)
            ok = self._move(px, py, approach_z, entry_quat, tol_ori=approach_tol_ori, planner_chain=['PTP', 'LIN'])
        else:
            entry_quat = quat
            ok = self._move(px, py, approach_z, entry_quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['PTP', 'LIN'])
        if not ok:
            raise RuntimeError('place approach 이동 실패 (재시도 초과)')
        # align 단계(재이동)는 여전히 없음 -- approach 도착 orientation과
        # descend orientation을 어차피 동일하게 맞출 거라 별도 이동이
        # 불필요했고, 오히려 재시도로 인한 지연의 주범이었음.
        if not is_side:
            place_quat = entry_quat  # [2026-08 수정] descend를 approach와 동일 각도(0°)로 통일
                           # — 회전+하강 동시요구로 인한 덤블링/IK실패 방지.
                           # trade-off: 박스 각도 정렬 없이 항상 축정렬로 place됨
        else:
            place_quat = quat
        self.get_logger().info('3/4: 내려가기 (descend)')
        if not is_side:
            ok = self._move_descend_with_tilt_fallback(px, py, descend_z, place_quat)
        else:
            ok = self._move(px, py, descend_z, place_quat, tol_ori=TOL_ORI_TIGHT, planner_chain=['LIN'])
        if not ok:
            raise RuntimeError('place descend 이동 실패 (재시도 초과, tolerance 완화 폴백까지 실패)')
        self.get_logger().info('4/4: 그리퍼 열기')
        self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
        time.sleep(GRIPPER_DELAY)
        self.get_logger().info('  들어올리기 (lift) — 다음 동작 여유 확보')
        lift_candidates = [lift_z, pos['z'] + 0.06 + (0 if is_side else TOP_TCP_OFFSET),
                           pos['z'] + 0.03 + (0 if is_side else TOP_TCP_OFFSET)]
        ok = False
        achieved_lift_z = lift_z
        for i, lz in enumerate(lift_candidates):
            if i > 0:
                self.get_logger().warn(
                    f'  place lift 높이 낮춰서 재시도: {lz:.3f}m (원래 목표 {lift_z:.3f}m)')
            ok = self._move(px, py, lz, place_quat, tol_ori=TOL_ORI_TIGHT, planner_chain=['LIN'])
            if ok:
                achieved_lift_z = lz
                break
        if not ok:
            self.get_logger().warn('  place 후 lift 모든 높이 재시도 실패 (place 자체는 성공, 무시하고 진행)')

        # [신설] 폐루프 검증 -- 그리퍼가 물러났으니 이제 카메라 시야가
        # 안 가려진 상태. perception이 새로 관측할 시간을 잠깐 준 뒤,
        # 목표 위치에 실제로 물체가 있는지 확인한다. 이 검사는 place
        # 시퀀스를 실패시키지 않는다(로봇 동작 자체는 이미 물리적으로
        # 완료됨) -- 대신 결과에 verified 여부를 실어서 Claude가
        # 다음 판단(재확인/재조정/사용자 보고)을 할 수 있게 한다.
        time.sleep(VERIFY_SETTLE_SEC)
        age = (time.time() - self._latest_objects_stamp
               if self._latest_objects_stamp > 0 else None)
        verification = verify_placement(
            pos, self.latest_objects, before_objects=before_objects, obj_age_sec=age)
        self.get_logger().info(f'  [verify] {verification["reason"]}')

        # [2026-08 재설계] 첫 검증이 애매하면(True가 아니면) 스캔에서 이미
        # 검증된 자세(DEFAULT_SCAN_BASE_JOINTS, conf 0.85~0.95로 안정적으로
        # 박스를 잡아온 실측 근거 있음)를 재사용해서 다시 본다. joint1만
        # 이번 물체 방향으로 역산해서 바꾸고 나머지 관절은 스캔값 그대로
        # 유지한다.
        #
        # [1차 시도, 폐기] pose 기반 _move()로 반경 바깥쪽 xy 오프셋을 주는
        # 방식을 먼저 시도했으나, 실측 결과 두 후보 다 NO_IK_SOLUTION
        # (error_code=-31)으로 즉시 실패함 (2026-08). pose 기반 이동은
        # IK를 풀어야 하는데, 그 근처 좌표에 항상 해가 있다는 보장이 없다.
        # joint-space 이동(_move_to_joint_config)은 애초에 IK를 풀 필요가
        # 없어서 이 문제가 원천적으로 발생하지 않는다 -- 도달 가능성이
        # 훨씬 확실한 접근으로 교체함.
        #
        # Claude Code hook으로는 이걸 할 수 없다 -- PostToolUse는 exit
        # code로 이미 반환된 결과를 바꿀 방법이 없고(도구 차단은
        # PreToolUse만 가능), 애초에 hook은 로봇을 직접 움직일 수도 없다.
        # 반드시 서버(planning_node) 안에서, Claude가 다시 판단할 필요
        # 없이 결정론적으로 처리해야 한다.
        if verification['verified'] is not True:
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4',
                           'joint5', 'joint6', 'joint7']
            target_bearing = math.atan2(pos['y'], pos['x']) - math.radians(BEARING_OFFSET_DEG)
            target_bearing = math.atan2(math.sin(target_bearing), math.cos(target_bearing))
            vantage_cfg = dict(DEFAULT_SCAN_BASE_JOINTS)
            vantage_cfg['joint1'] = target_bearing
            target_list = [vantage_cfg.get(jn, 0.0) for jn in joint_names]
            self.get_logger().info(
                f'  [verify] 1차 결과 애매함({verification["verified"]}) -> '
                f'스캔 자세 재사용, joint1={math.degrees(target_bearing):.1f}\xb0로 이동')
            moved = self._move_to_joint_config(target_list, label='verify-vantage')
            if moved:
                time.sleep(VERIFY_SETTLE_SEC)
                age = (time.time() - self._latest_objects_stamp
                       if self._latest_objects_stamp > 0 else None)
                retry_verification = verify_placement(
                    pos, self.latest_objects, before_objects=before_objects, obj_age_sec=age)
                self.get_logger().info(
                    f'  [verify] 재확인 결과: {retry_verification["reason"]}')
                verification = retry_verification
            else:
                self.get_logger().warn(
                    '  [verify] 재확인 자세 이동 실패, 1차 검증 결과 그대로 유지')
        return verification

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
        if not self._arm_status_available:
            self._publish_joint_positions(positions, joint_names)
            time.sleep(3.0)  # 확인 불가 시 예전 방식(고정 대기)으로 폴백
            return True
        for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
            self.get_logger().info(
                f'  → [실물] joint 이동 시도 {attempt}/{MOVE_MAX_ATTEMPTS}: '
                f'{[round(p,3) for p in positions]}')
            self.latest_arm_status = None
            publish_time = time.time()
            self._publish_joint_positions(positions, joint_names)
            time.sleep(REAL_MOVE_SETTLE_SEC)
            deadline = time.time() + REAL_MOVE_TIMEOUT
            last_status = None
            while time.time() < deadline:
                status = self.latest_arm_status
                if status is not None:
                    last_status = status
                    if status.motion_status == 0:  # SUCCESS
                        elapsed = time.time() - publish_time
                        if elapsed < 0.5:
                            self.get_logger().warn(
                                f'  ⚠ [실물] {elapsed*1000:.0f}ms만에 성공 확인됨 — '
                                f'실제 이동이 반영 안 된 이전 상태값일 가능성 있음. '
                                f'로봇이 실제로 움직였는지 육안 확인 권장.')
                        return True
                    if status.motion_status == 1:  # FAILED
                        if status.arm_status != 0:
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
    def _merge_scanned_box(self, obj):
        c3d = obj.get('center_3d', {})
        x, y, z = c3d.get('x'), c3d.get('y'), c3d.get('z')
        if x is None or y is None or z is None:
            return
        conf = obj.get('confidence', 0.0)
        entry = {
            'x': x, 'y': y, 'z': z,
            'confidence': conf,
            'angle_base_deg': obj.get('angle_base_deg'),
            'label': obj.get('label'),
        }
        for i, existing in enumerate(self._scanned_boxes):
            ex, ey, ez = existing['x'], existing['y'], existing['z']
            dist = math.sqrt((x-ex)**2 + (y-ey)**2 + (z-ez)**2)
            if dist < SCAN_DEDUP_DIST_M:
                if conf > existing.get('confidence', 0.0):
                    self._scanned_boxes[i] = entry
                    self.get_logger().info(
                        f'[스캔] 기존 박스 갱신(더 높은 conf): ({x:.3f},{y:.3f},{z:.3f}) conf={conf:.2f}')
                return
        self._scanned_boxes.append(entry)
        self.get_logger().info(
            f'[스캔] 새 박스 추가: ({x:.3f},{y:.3f},{z:.3f}) conf={conf:.2f}')
    def _scan_box_sequence(self, base_joints, step_deg, label):
        try:
            self._scanned_boxes = []
            joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
            step_rad = math.radians(step_deg)
            j1 = JOINT1_LOWER
            n_steps = 0
            while j1 <= JOINT1_UPPER + 1e-6:
                target = [base_joints.get(jn, 0.0) for jn in joint_names]
                target[0] = j1
                self.get_logger().info(
                    f'[스캔] joint1={math.degrees(j1):.1f}\xb0 로 이동 중...')
                if self.use_moveit2:
                    self.moveit2.force_reset_executing_state()
                    # [2026-08 수정] _home_sequence에서 이미 겪었던 것과 동일한
                    # 버그: 직전 명령(pick/place)이 Pilz(PTP/LIN)로 성공하면
                    # self.moveit2.pipeline_id/planner_id에 그 설정이 그대로
                    # 남는다. scan_box는 joint-space move_to_configuration을
                    # 쓰는데, 여기서 리셋을 안 하면 Pilz 설정이 그대로 재사용돼
                    # 매 스텝 100% 즉시 ABORTED되는 것이 실측 확인됨(2026-08,
                    # place 직후 scan_box 재호출 시 6스텝 전부 실패). 원인/수정
                    # 방식은 _home_sequence와 완전히 동일.
                    self.moveit2.pipeline_id = ''
                    self.moveit2.planner_id = ''
                    self.moveit2.max_velocity = 0.3
                    self.moveit2.max_acceleration = 0.3
                    self.moveit2.move_to_configuration(target)
                    time.sleep(0.5)
                    deadline = time.time() + MOVE_TIMEOUT
                    while time.time() < deadline:
                        if (not self.moveit2._MoveIt2__is_motion_requested and
                                not self.moveit2._MoveIt2__is_executing):
                            break
                        time.sleep(0.1)
                    # [2026-08 수정, 코드리뷰 발견] 이전엔 이동 성공 여부를
                    # 확인 안 하고 바로 SCAN_STEP_SETTLE_SEC만큼 대기 후
                    # 검출 결과를 저장했음 -- 이동이 실패/중단됐으면 이
                    # 스텝에서 저장되는 박스는 의도한 joint1 각도가 아닌
                    # 다른(어중간한) 위치에서 관측된 것일 수 있음. TF 기반
                    # 좌표 계산이라 좌표 자체가 틀리진 않겠지만, 그 각도
                    # 구간의 커버리지가 비어있는 걸 놓치기 쉬워 경고만 남김
                    # (하드게이트 없이 계속 진행 -- handoff 6절 원칙).
                    if not self.moveit2.motion_suceeded:
                        self.get_logger().warn(
                            f'[스캔] joint1={math.degrees(j1):.1f}\xb0 이동 실패/중단 '
                            f'(motion_suceeded=False) -- 이 스텝의 검출 결과는 '
                            f'의도한 각도가 아닌 다른 위치에서 관측된 것일 수 있음.')
                else:
                    self._move_joints_real_wait(target)
                time.sleep(SCAN_STEP_SETTLE_SEC)
                for obj in self.latest_objects:
                    if obj.get('label') != label:
                        continue
                    self._merge_scanned_box(obj)
                n_steps += 1
                j1 += step_rad
            self.get_logger().info(
                f'[스캔] 완료 ({n_steps}스텝), 누적 박스 {len(self._scanned_boxes)}개: '
                f'{[(round(b["x"],3), round(b["y"],3), round(b["z"],3)) for b in self._scanned_boxes]}')
            self._publish_result(
                'success', f'scan_complete:{len(self._scanned_boxes)}',
                extra={'boxes': self._scanned_boxes})
        except Exception as e:
            self.get_logger().error(f'SCAN 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
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
                # [2026-08 수정] scan_box/home과 동일한 버그 -- 직전 명령이
                # Pilz(PTP/LIN)로 성공하면 pipeline_id/planner_id가 남아서
                # 이 joint-space 이동이 항상 즉시 ABORTED될 수 있음.
                self.moveit2.force_reset_executing_state()
                self.moveit2.pipeline_id = ''
                self.moveit2.planner_id = ''
                self.moveit2.max_velocity = 0.3
                self.moveit2.max_acceleration = 0.3
                self.moveit2.move_to_configuration(positions)
                time.sleep(0.5)
                deadline = time.time() + MOVE_TIMEOUT
                while time.time() < deadline:
                    if (not self.moveit2._MoveIt2__is_motion_requested and
                            not self.moveit2._MoveIt2__is_executing):
                        break
                    time.sleep(0.1)
                # [2026-08 수정, 코드리뷰 발견] 이전엔 motion_suceeded를
                # 확인 안 하고 무조건 'success'를 퍼블리시했음 -- 이동이
                # 실제로 ABORTED돼도 MCP/사용자에게는 성공으로 보고돼서,
                # 그 뒤 명령들이 로봇이 도달 못 한 잘못된 전제(joint값)
                # 위에서 계산되는 위험이 있었음. 다른 액션들(home 등)과
                # 동일하게 명시적으로 확인.
                if not self.moveit2.motion_suceeded:
                    raise RuntimeError('move_joints 이동 실패 (motion_suceeded=False)')
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
                ok = False
                for attempt in range(1, 4):
                    self.moveit2.force_reset_executing_state()
                    # [2026-07 수정] pipeline_id/planner_id를 명시적으로
                    # OMPL로 리셋하지 않으면, 직전 이동이 Pilz(PTP/LIN)로
                    # 성공했을 경우 그 설정이 그대로 남아 조인트 스페이스
                    # 홈 이동에 Pilz LIN이 그대로 쓰이면서 매번 동일하게
                    # 즉시 ABORTED되는 현상이 실측 확인됨(handoff 3.2절과
                    # 같은 "공유 상태, 세팅 안 하면 이전 값 잔류" 패턴).
                    self.moveit2.pipeline_id = ''
                    self.moveit2.planner_id = ''
                    self.moveit2.max_velocity = 0.3
                    self.moveit2.max_acceleration = 0.3
                    self.moveit2.move_to_configuration([0.0] * 7)
                    time.sleep(0.5)
                    deadline = time.time() + MOVE_TIMEOUT
                    while time.time() < deadline:
                        if (not self.moveit2._MoveIt2__is_motion_requested and
                                not self.moveit2._MoveIt2__is_executing):
                            break
                        time.sleep(0.1)
                    ok = self.moveit2.motion_suceeded
                    if ok:
                        break
                    self.get_logger().warn(
                        f'  홈 이동 실패/중단(시도 {attempt}/3), 재시도...')
                    time.sleep(0.5)
                if not ok:
                    raise RuntimeError('home 이동 실패 (재시도 초과)')
            else:
                ok = self._move_joints_real_wait([0.0] * 7)
                if not ok:
                    raise RuntimeError('home 조인트 이동 실패 (재시도 초과)')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)
            # [신규, 2026-08] settle delay -- 홈 직후 첫 pick_object가 매번
            # 타임아웃나던 문제의 근본 처방 (nero_robot_pick_after_home_timeout
            # memory, 2026-07-03 실측: get_system_status는 정상인데 첫 시도만
            # 타임아웃, 즉시 재시도하면 100% 성공 -- planning_node/IK서비스가
            # 아직 정착 중인 것으로 추정). busy 락이 풀리기 전에 넣어서, 다음
            # 명령이 이 시간만큼 자연스럽게 대기하도록 한다.
            time.sleep(1.5)
            self.get_logger().info('✅ HOME 완료')
            self._publish_result('success', 'home_complete')
        except Exception as e:
            self.get_logger().error(f'HOME 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
    # ── 핵심: 재시도 포함 _move() ─────────────────────────────────────────────
    def _compute_ik_joints(self, position, quat_xyzw):
        """IK만 조회(실행 없음). 성공 시 dict{joint_name: pos}, 실패 시 None."""
        try:
            from moveit_msgs.srv import GetPositionIK
            if not hasattr(self.moveit2, '_MoveIt2__compute_ik_client'):
                self.moveit2._MoveIt2__init_compute_ik()
            client = self.moveit2._MoveIt2__compute_ik_client
            if not client.service_is_ready():
                return None
            req = GetPositionIK.Request()
            req.ik_request.group_name = self.moveit2._MoveIt2__group_name
            req.ik_request.pose_stamped.header.frame_id = self.moveit2._MoveIt2__base_link_name
            req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
            req.ik_request.pose_stamped.pose.position.x = float(position[0])
            req.ik_request.pose_stamped.pose.position.y = float(position[1])
            req.ik_request.pose_stamped.pose.position.z = float(position[2])
            req.ik_request.pose_stamped.pose.orientation.x = float(quat_xyzw[0])
            req.ik_request.pose_stamped.pose.orientation.y = float(quat_xyzw[1])
            req.ik_request.pose_stamped.pose.orientation.z = float(quat_xyzw[2])
            req.ik_request.pose_stamped.pose.orientation.w = float(quat_xyzw[3])
            req.ik_request.ik_link_name = 'gripper_flange'
            req.ik_request.avoid_collisions = True
            js = self.latest_joint_state_sim
            if js is not None:
                req.ik_request.robot_state.joint_state = js
            future = client.call_async(req)
            _deadline = time.time() + 5.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            if not future.done():
                self.get_logger().warn(
                    '[ik_check] ⚠ compute_ik 타임아웃(1.0s 초과) — '
                    '바닥 콜리전 체크로 느려졌을 가능성')
                return None
            res = future.result()
            from moveit_msgs.msg import MoveItErrorCodes
            if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
                err_val = res.error_code.val if res is not None else 'res=None'
                self.get_logger().warn(
                    f'[ik_check] ⚠ IK 실패, error_code={err_val} '
                    f'(NO_IK_SOLUTION=-31, GOAL_IN_COLLISION=-12 등 확인용)')
                return None
            sol = res.solution.joint_state
            return dict(zip(sol.name, sol.position))
        except Exception as e:
            self.get_logger().warn(f'[ik_check] 조회 실패: {e}')
            return None
    def _pick_best_yaw_candidate(self, position, angle_deg, ik_check_z=None):
        """[2026-08 리팩터] 순수 비교 로직은 grasp_kinematics.YawCandidateSelector
        로 이관됨 (mod-90 후보 구성, FK 기반 회전량 비교 등 전체 설계/이력은
        grasp_kinematics.py 모듈 docstring + YawCandidateSelector.pick_best
        docstring 참고). 이 메서드는 ROS 서비스(IK 조회, FK 조회, backend별
        quat 생성)를 콜백으로 주입하는 얇은 어댑터로만 남는다.
        """
        current_fk = self._fk_gripper_pos()
        current_quat = current_fk[1] if current_fk is not None else None
        check_z = ik_check_z if ik_check_z is not None else position.get('z', 0.0)

        def _ik_check(x, y, z, cand_quat):
            # IK는 도달 가능성 확인용으로만 사용 (해 자체는 cost에 안 씀).
            # 검사 높이는 실제 이동 목표(approach_z 등)와 맞춰야 함
            # (2026-08 코드리뷰 발견 -- ik_check_z 파라미터 도입 배경).
            return self._compute_ik_joints([x, y, z], cand_quat)

        selector = YawCandidateSelector(
            ik_check_fn=_ik_check,
            quat_for_angle_fn=self._top_down_quat_for,
        )
        return selector.pick_best(
            position, angle_deg, current_quat,
            log_fn=lambda msg: self.get_logger().info(msg),
            ik_check_z=check_z,
        )
    def _check_state_validity(self, joint_state=None):
        """[2026-07 추가] START_STATE_IN_COLLISION(-31) 진단용.
        joint_state 생략 시 현재 상태(self.latest_joint_state_sim) 사용.
        (is_valid, contact_pairs, joint_limit_violations) 튜플 반환,
        조회 실패 시 (None, [], []).
        """
        try:
            from moveit_msgs.srv import GetStateValidity
            if not self._state_validity_client.service_is_ready():
                self.get_logger().warn('[validity_debug] service_is_ready()=False')
                return None, [], []
            js = joint_state if joint_state is not None else self.latest_joint_state_sim
            if js is None:
                self.get_logger().warn('[validity_debug] joint_state=None')
                return None, [], []
            req = GetStateValidity.Request()
            req.group_name = 'arm'
            req.robot_state.joint_state = js
            future = self._state_validity_client.call_async(req)
            _deadline = time.time() + 1.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None:
                self.get_logger().warn('[validity_debug] 서비스 타임아웃')
                return None, [], []
            contacts = [
                f'{c.contact_body_1}<->{c.contact_body_2} (depth={c.depth:.4f})'
                for c in res.contacts
            ]
            self.get_logger().warn(
                f'[validity_debug] valid={res.valid} contacts={contacts}')
            return res.valid, contacts, []
        except Exception as e:
            self.get_logger().warn(f'[validity_debug] 조회 실패: {e}')
            return None, [], []
    def _fk_for_joints(self, joint_names, joint_positions):
        """[2026-07 추가, 07 확장] 임의의 조인트값 목록에 대해
        gripper_flange의 base_link 기준 position+orientation을 조회.
        OMPL 궤적 웨이포인트 안전성 검사(position+orientation 둘 다)용.
        반환: (position_tuple(x,y,z), quat_tuple(x,y,z,w)) 또는 실패 시 None.
        [주의] 반환 타입이 튜플 1개(position만)에서 (position, quat) 쌍으로
        바뀜 -- 이 함수를 새로 호출하는 곳을 추가할 때 언패킹 잊지 말 것.
        """
        try:
            from moveit_msgs.srv import GetPositionFK
            from sensor_msgs.msg import JointState as JointStateMsg
            if not self._fk_client.service_is_ready():
                return None
            js = JointStateMsg()
            js.name = list(joint_names)
            js.position = list(joint_positions)
            req = GetPositionFK.Request()
            req.header.frame_id = 'base_link'
            req.header.stamp = self.get_clock().now().to_msg()
            req.fk_link_names = ['gripper_flange']
            req.robot_state.joint_state = js
            future = self._fk_client.call_async(req)
            _deadline = time.time() + 1.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None or not res.pose_stamped:
                return None
            p = res.pose_stamped[0].pose.position
            q = res.pose_stamped[0].pose.orientation
            return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
        except Exception as e:
            self.get_logger().warn(f'[fk_batch] 조회 실패: {e}')
            return None
    def _is_trajectory_reasonable(self, traj, max_ratio=1.8, max_ori_ratio=3.0):
        """[2026-07 추가, 07 확장] OMPL은 랜덤 샘플링 기반이라, 물리적으로는
        유효하지만 그리퍼가 목적지까지 크게 우회하는 경로를 찾을 때가
        있음이 실측 확인됨. execute 전에 궤적의 gripper_flange 경로
        총 길이를 시작-끝 직선거리와 비교해 걸러낸다.
        [2026-07 확장 배경] position 비율만으로는 못 잡는 사고가 실측
        확인됨: position 경로는 거의 직선(비율 1.24)인데, 그 안에서
        joint5/7이 왕복 1.5rad 이상 스윙하며 그리퍼가 순간적으로
        뒤집히는 궤적이 "정상"으로 통과해 실행된 사례 (place descend,
        2026-07 로그 대조로 확인). tol_ori는 시작/끝점에만 걸리고
        중간 웨이포인트의 orientation은 원래 자유였던 게 원인.
        position과 같은 방식(경로길이/직선거리 비율)을 orientation
        각도 거리에도 그대로 적용해서 우회를 잡는다. 하드 게이트가
        아니라 기존 재계획 루프(_move_try_ompl_safe)에 태워지는
        소프트 판정이라는 점은 동일하게 유지.
        max_ori_ratio=3.0은 초기 제안값, 실측 튜닝 필요(3.6절의
        max_ratio=1.8도 실측으로 정한 값이었음 참고).
        """
        points = traj.points
        if len(points) < 3:
            return True
        n = len(points)
        sample_idx = sorted(set([0, n - 1] + [int(i * (n - 1) / 6) for i in range(1, 6)]))
        positions_3d = []
        quats = []
        for idx2 in sample_idx:
            joint_pos = list(points[idx2].positions)
            fk = self._fk_for_joints(traj.joint_names, joint_pos)
            if fk is None:
                return True
            pos, quat = fk
            positions_3d.append(pos)
            quats.append(quat)
        path_length = sum(
            math.dist(positions_3d[i], positions_3d[i + 1])
            for i in range(len(positions_3d) - 1)
        )
        straight_dist = math.dist(positions_3d[0], positions_3d[-1])
        ratio = (path_length / straight_dist) if straight_dist >= 0.01 else 0.0
        reasonable_pos = (straight_dist < 0.01) or (ratio <= max_ratio)
        ori_path = sum(
            quat_angle_diff(quats[i], quats[i + 1])
            for i in range(len(quats) - 1)
        )
        ori_straight = quat_angle_diff(quats[0], quats[-1])
        # 목표 orientation 변화 자체가 거의 없는 이동(approach/descend처럼
        # tol_ori_tight로 orientation을 고정하는 단계)일수록 분모가
        # 0에 가까워 비율이 과민해지므로 최소 5도로 바닥을 둠.
        ori_ratio = ori_path / max(ori_straight, math.radians(5.0))
        reasonable_ori = ori_ratio <= max_ori_ratio
        self.get_logger().info(
            f'  [safety] 궤적 검사: 경로길이={path_length:.3f}m '
            f'직선거리={straight_dist:.3f}m 비율={ratio:.2f} | '
            f'orientation 경로={math.degrees(ori_path):.1f}\xb0 '
            f'직선={math.degrees(ori_straight):.1f}\xb0 비율={ori_ratio:.2f}')
        if not reasonable_ori and reasonable_pos:
            self.get_logger().warn(
                '  [safety] position은 정상이나 orientation 우회(뒤집힘 의심) 감지')
        return reasonable_pos and reasonable_ori
    def _fk_gripper_pos(self):
        """[2026-07 확장] 현재 조인트 상태로 gripper_flange의 실제
        base_link 기준 position+orientation을 조회. 실패 시 None 반환.
        반환: (position_tuple(x,y,z), quat_tuple(x,y,z,w))
        [주의] 예전엔 position 튜플 1개만 반환했음 -- 이 함수를 쓰는
        곳이 새로 생기면 반드시 언패킹할 것 (지금은 호출부 없음).
        """
        try:
            from moveit_msgs.srv import GetPositionFK
            if not self._fk_client.service_is_ready():
                self.get_logger().warn('[fk_debug] service_is_ready()=False')
                return None
            js = self.latest_joint_state_sim
            if js is None:
                self.get_logger().warn('[fk_debug] latest_joint_state=None')
                return None
            req = GetPositionFK.Request()
            req.header.frame_id = 'base_link'
            req.header.stamp = self.get_clock().now().to_msg()
            req.fk_link_names = ['gripper_flange']
            req.robot_state.joint_state = js
            future = self._fk_client.call_async(req)
            _deadline = time.time() + 1.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None or not res.pose_stamped:
                self.get_logger().warn(
                    f'[fk_debug] future.done()={future.done()} res={res} '
                    f'pose_stamped={getattr(res, "pose_stamped", None)}')
                return None
            p = res.pose_stamped[0].pose.position
            q = res.pose_stamped[0].pose.orientation
            return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
        except Exception as e:
            self.get_logger().warn(f'[fk_debug] FK 조회 실패: {e}')
            return None
    def _move_descend_with_tilt_fallback(self, px, py, descend_z, quat) -> bool:
        """
        [2026-07 추가] descend 이동을 시도하되, 실패하면 top-down 유지정도
        (tol_ori의 x/y축)만 단계적으로 완화해서 재시도한다.
        z축(그립 각도 = 박스와의 좌우 정렬)은 절대 풀지 않는다 -- 이게
        풀리면 그리퍼가 박스와 어긋난 방향으로 닫혀서 파지가 틀어진다.
        (참고: tol_ori의 축 순서가 월드축인지 목표 quat 로컬축인지는
        pymoveit2 내부까지 확인 안 했음 -- approach 단계에서 이미
        3번째=yaw축만 따로 풀던 기존 패턴을 그대로 신뢰해서 확장한 것.
        실제로 돌려서 그리퍼가 박스와 어긋나게 닫히면 축 순서 가정이
        틀린 것이니 (0,1)과 (2)를 바꿔서 재검증 필요.)
        """
        for i, tol in enumerate(DESCEND_TOL_ORI_FALLBACK):
            if i > 0:
                self.get_logger().warn(
                    f'  descend tolerance 완화 재시도 {i}/{len(DESCEND_TOL_ORI_FALLBACK)-1}: '
                    f'top-down 여유={math.degrees(tol[0]):.1f}° (그립각도는 그대로 고정)')
            ok = self._move(px, py, descend_z, quat, tol_ori=tol, planner_chain=['LIN'])
            if ok:
                return True
        return False
    def _move(self, x: float, y: float, z: float, quat: list,
              tol_pos: float = TOL_POS_LOOSE,
              tol_ori=TOL_ORI_LOOSE, planner_id: str = '',
              planner_chain: list = None) -> bool:
        """
        MoveIt2로 목표 pose 이동. 실패 시 MOVE_MAX_ATTEMPTS 회 재시도.
        성공하면 True, 모든 시도 실패 시 False 반환.
        """
        if not self.use_moveit2:
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
                self.pub_move.publish(pose)
                return True
            for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
                self.get_logger().info(
                    f'  → [실물] move 시도 {attempt}/{MOVE_MAX_ATTEMPTS} '
                    f'({x:.3f},{y:.3f},{z:.3f}) quat={[round(v,3) for v in quat]}')
                self.latest_arm_status = None
                publish_time = time.time()
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
                            elapsed = time.time() - publish_time
                            if elapsed < 0.5:
                                self.get_logger().warn(
                                    f'  ⚠ [실물] {elapsed*1000:.0f}ms만에 성공 확인됨 — '
                                    f'실제 이동이 반영 안 된 이전 상태값일 가능성 있음. '
                                    f'로봇이 실제로 움직였는지 육안 확인 권장.')
                            return True
                        if status.motion_status == 1:  # FAILED
                            if status.arm_status != 0:
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
            # [2026-08 수정, 코드리뷰 발견] MOVE_MAX_ATTEMPTS회 모두
            # 실패했을 때 return이 없어서 아래 OMPL/Pilz 코드로 그대로
            # fall-through됐음. 실물 모드에서는 self.moveit2가 None이라
            # self._move_try_pilz(...)가 AttributeError를 던지고, 그게
            # 상위 try/except에 잡혀서 'failed'로 나가긴 했지만 원래
            # 실패 원인(arm_status 등)이 뭉개지고 'NoneType has no
            # attribute...'라는 뜻 모를 에러로 대체되는 문제였음.
            # _move_joints_real_wait()는 이 return을 이미 갖고 있었음
            # (여기만 누락).
            self.get_logger().error(
                f'[실물] move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) '
                f'[{MOVE_MAX_ATTEMPTS}회 모두 실패]')
            return False
        chain = list(planner_chain) if planner_chain else ([planner_id] if planner_id else [])
        for chain_planner in chain:
            if self._move_try_pilz(x, y, z, quat, tol_pos, tol_ori, chain_planner):
                return True
            self.get_logger().warn(f'  {chain_planner} 플래너 실패, 다음 플래너로 전환...')
        # [2026-08 추가] fail-fast: OMPL 3회(+내부 재계획 3회)까지 가기
        # 전에, 순수 IK로 "애초에 이 pose에 해가 있는가"만 먼저 빠르게
        # 확인한다. 해가 아예 없으면 OMPL을 반복해도 결과가 달라지지
        # 않는데(실측 로그: lift 실패 케이스 하나에 재시도 예산을 다
        # 태우는 데 약 57초 소요), 이 경우 즉시 실패로 반환해서 상위
        # (_pick_sequence의 lift_candidates 등)가 더 빨리 대체 전략으로
        # 넘어가게 한다.
        # 주의: 이건 "명백히 불가능"만 걸러내는 용도지 OMPL 재시도
        # 로직을 대체하지 않는다 -- IK가 성공해도 OMPL이 충돌 없는
        # 경로 자체를 못 찾아 실패하는 경우는 여전히 있을 수 있으므로,
        # 그 경우는 원래대로 아래 재시도 루프가 담당한다.
        quick_ik = self._compute_ik_joints([x, y, z], quat)
        if quick_ik is None:
            self.get_logger().warn(
                f'  [fail-fast] IK 사전조회 실패 -- 이 pose({x:.3f},{y:.3f},{z:.3f})는 '
                f'해가 없는 것으로 보임. OMPL 재시도 생략.')
            self.get_logger().error(
                f'_move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) [IK 사전조회 단계에서 조기 종료]')
            return False
        for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
            self.get_logger().info(
                f'  → OMPL 시도 {attempt}/{MOVE_MAX_ATTEMPTS} ({x:.3f},{y:.3f},{z:.3f})')
            if self._move_try_pilz(x, y, z, quat, tol_pos, tol_ori, ''):
                return True
            if attempt < MOVE_MAX_ATTEMPTS:
                time.sleep(1.0)
        self.get_logger().error(
            f'_move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) [모든 플래너 실패]')
        return False
    def _move_try_ompl_safe(self, x, y, z, quat, tol_pos, tol_ori, max_ratio=1.8) -> bool:
        self.moveit2.pipeline_id = ''
        self.moveit2.planner_id = ''
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3
        for replan_attempt in range(1, 4):
            self.moveit2.force_reset_executing_state()
            ready = self.moveit2._MoveIt2__move_action_client.server_is_ready()
            if not ready:
                self.get_logger().warn('  action server not ready, 잠시 대기...')
                time.sleep(1.0)
                continue
            future = self.moveit2.plan_async(
                position=[x, y, z], quat_xyzw=quat,
                tolerance_position=tol_pos, tolerance_orientation=tol_ori,
            )
            if future is None:
                return False
            _deadline = time.time() + MOVE_TIMEOUT
            while not future.done() and time.time() < _deadline:
                time.sleep(0.05)
            if not future.done():
                self.get_logger().warn('  [safety] OMPL plan 타임아웃')
                continue
            traj = self.moveit2.get_trajectory(future)
            if traj is None:
                self.get_logger().info('  [sim/OMPL] plan 실패')
                continue
            reasonable = self._is_trajectory_reasonable(traj, max_ratio=max_ratio)
            if not reasonable and replan_attempt < 3:
                self.get_logger().warn(
                    f'  [safety] 비정상 우회 경로 감지, 재계획... '
                    f'({replan_attempt}/3)')
                continue
            if not reasonable:
                self.get_logger().warn(
                    '  [safety] 3회 재계획해도 우회 경로만 나옴, 마지막 결과로 진행')
            time.sleep(0.2)
            self.moveit2.execute(traj)
            time.sleep(0.1)
            deadline = time.time() + MOVE_TIMEOUT
            motion_done = False
            while time.time() < deadline:
                req = self.moveit2._MoveIt2__is_motion_requested
                exe = self.moveit2._MoveIt2__is_executing
                if not req and not exe:
                    motion_done = True
                    break
                time.sleep(0.05)
            success = motion_done and self.moveit2.motion_suceeded
            self.get_logger().info(
                f'  [sim/OMPL] move → {x:.3f} {y:.3f} {z:.3f} | '
                f'{"✅ 성공" if success else "❌ 실패"}')
            if success:
                time.sleep(0.3)
                return True
        return False
    def _move_try_pilz(self, x, y, z, quat, tol_pos, tol_ori, planner_id: str) -> bool:
        """단일 플래너로 1회 시도. 성공 시 True."""
        if not planner_id:
            return self._move_try_ompl_safe(x, y, z, quat, tol_pos, tol_ori)
        self.moveit2.force_reset_executing_state()
        ready = self.moveit2._MoveIt2__move_action_client.server_is_ready()
        if not ready:
            self.get_logger().warn('  action server not ready, 잠시 대기...')
            time.sleep(1.0)
            return False
        if planner_id:
            self.moveit2.pipeline_id = 'pilz_industrial_motion_planner'
            self.moveit2.planner_id = planner_id
        else:
            self.moveit2.pipeline_id = ''
            self.moveit2.planner_id = ''
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3
        self.moveit2.move_to_pose(
            position=[x, y, z],
            quat_xyzw=quat,
            tolerance_position=tol_pos,
            tolerance_orientation=tol_ori,
            cartesian=False,
        )
        time.sleep(0.1)
        deadline = time.time() + MOVE_TIMEOUT
        motion_done = False
        while time.time() < deadline:
            req = self.moveit2._MoveIt2__is_motion_requested
            exe = self.moveit2._MoveIt2__is_executing
            if not req and not exe:
                motion_done = True
                break
            time.sleep(0.05)
        success = motion_done and self.moveit2.motion_suceeded
        error_code = self.moveit2.get_last_execution_error_code()
        err_str = f' [error_code={error_code.val}]' if (not success and error_code) else ''
        label = planner_id if planner_id else 'OMPL'
        self.get_logger().info(
            f'  [sim/{label}] move → {x:.3f} {y:.3f} {z:.3f} | '
            f'{"✅ 성공" if success else "❌ 실패"}{err_str}')
        if not success and error_code and error_code.val == -31:
            self._check_state_validity()
        if success:
            time.sleep(0.3)
        return success
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
    def _publish_result(self, status, reason='', extra=None):
        payload = {'status': status, 'reason': reason}
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
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