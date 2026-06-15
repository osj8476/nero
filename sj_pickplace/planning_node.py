#!/usr/bin/env python3
"""
planning_node.py  (그리퍼 자세 유연화 패치)

[변경점]
- QUAT_DOWN 하드코딩 제거
- _move(x, y, z, quat) 로 자세를 매번 명시적으로 전달
- _auto_grasp_quat(pos, label) : 물체 위치·라벨 기반 자동 자세 선택
    * 기본 : top-down (위에서 아래)
    * 로봇 측면 가까이 있는 물체 : side (수평)
    * 커맨드에 grasp_dir 필드가 있으면 그것을 우선
- pick/place/move 시퀀스 모두 quat 을 인자로 받도록 변경
"""

import json
import math
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

# ── 그리퍼 상수 ───────────────────────────────────────────────────────────────
GRIPPER_OPEN  = 0.08
GRIPPER_CLOSE = 0.01
GRIPPER_FORCE = 1.5
SIM_GRIPPER_OPEN  = [0.05, -0.05]
SIM_GRIPPER_CLOSE = [0.01, -0.01]

# ── 시퀀스 오프셋 (미터) ──────────────────────────────────────────────────────
APPROACH_Z = 0.13
DESCEND_Z  = 0.03
LIFT_Z     = 0.23

# ── 홈 자세 ───────────────────────────────────────────────────────────────────
HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, 0.35

# ── 타이밍 ────────────────────────────────────────────────────────────────────
MOVE_DELAY    = 6.0
GRIPPER_DELAY = 4.0

# ── 사전 정의 쿼터니언 (xyzw, base_link 기준) ─────────────────────────────────
# top-down : 그리퍼가 수직 아래를 향함 (pitch 90°)
QUAT_TOP_DOWN   = [0.0,  0.7071, 0.0, 0.7071]
# side     : 그리퍼가 수평 앞을 향함 (pitch 0°, 기본 자세)
QUAT_SIDE       = [0.0,  0.0,    0.0, 1.0   ]
# side_left: 그리퍼가 왼쪽을 향함 (yaw 90°)
QUAT_SIDE_LEFT  = [0.0,  0.0,    0.7071, 0.7071]
# side_right: 그리퍼가 오른쪽을 향함 (yaw -90°)
QUAT_SIDE_RIGHT = [0.0,  0.0,   -0.7071, 0.7071]

GRASP_DIR_MAP = {
    'top':        QUAT_TOP_DOWN,
    'top_down':   QUAT_TOP_DOWN,
    'side':       QUAT_SIDE,
    'side_front': QUAT_SIDE,
    'side_left':  QUAT_SIDE_LEFT,
    'side_right': QUAT_SIDE_RIGHT,
}

# 물체 라벨별 기본 파지 방향 힌트
LABEL_GRASP_HINT = {
    'bottle':  'side',       # 세워진 병 → 옆에서 잡기
    'cup':     'top',        # 컵 → 위에서
    'book':    'side',       # 책 → 옆에서
    'box':     'top',        # 박스 → 위에서
    'ball':    'top',        # 공 → 위에서
    'scissors':'side',       # 가위 → 옆에서
    'remote':  'top',        # 리모컨 → 위에서
}


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> list:
    """rpy → xyzw 쿼터니언."""
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return [
        round(sr*cp*cy - cr*sp*sy, 6),
        round(cr*sp*cy + sr*cp*sy, 6),
        round(cr*cp*sy - sr*sp*cy, 6),
        round(cr*cp*cy + sr*sp*sy, 6),
    ]


def _auto_grasp_quat(pos: dict, label: str) -> list:
    """물체 위치·라벨로 파지 자세 자동 선택.

    우선순위:
      1. 라벨 힌트 (LABEL_GRASP_HINT)
      2. 물체가 로봇 옆(|y| > |x|)이면 side_left/right
      3. 기본 top_down
    """
    # 1) 라벨 힌트
    hint = LABEL_GRASP_HINT.get(label, None)
    if hint:
        return GRASP_DIR_MAP[hint]

    # 2) 위치 기반
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    if abs(y) > abs(x):
        return QUAT_SIDE_LEFT if y > 0 else QUAT_SIDE_RIGHT

    # 3) 기본
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

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects,
            qos_best_effort, callback_group=self._cb)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command,
            qos_reliable, callback_group=self._cb)

        self.pub_move    = self.create_publisher(PoseStamped, '/control/move_p', qos_reliable)
        self.pub_gripper = self.create_publisher(JointState,  '/control/joint_states', qos_reliable)
        self.pub_result  = self.create_publisher(String, '/pick_result', qos_reliable)

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
                ignore_new_calls_while_executing=False,
            )
            self.moveit2_gripper = MoveIt2(
                node=self,
                joint_names=['gripper_joint1', 'gripper_joint2'],
                base_link_name='base_link',
                end_effector_name='gripper_flange',
                group_name='gripper',
                callback_group=cb_gripper,
                use_move_group_action=True,
                ignore_new_calls_while_executing=False,
            )
            self.get_logger().info('MoveIt2 ENABLED')

        self.latest_objects = []
        self.busy = False
        self.lock = threading.Lock()
        self.get_logger().info('PlanningNode 준비 완료')

    # ── 인식 결과 캐시 ──────────────────────────────────────────────
    def on_objects(self, msg):
        try:
            self.latest_objects = json.loads(msg.data).get('objects', [])
        except Exception:
            pass

    # ── 명령 수신 ───────────────────────────────────────────────────
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
            pos   = self._find_object(label)
            if pos is None:
                self.get_logger().warn(f"'{label}' 못 찾음.")
                with self.lock:
                    self.busy = False
                return
            # grasp_dir : MCP가 명시하면 우선, 없으면 자동 선택
            grasp_dir = cmd.get('grasp_dir', None)
            quat = (GRASP_DIR_MAP.get(grasp_dir, None)
                    if grasp_dir else _auto_grasp_quat(pos, label))
            self.get_logger().info(
                f'PICK 시작: {label} @ {pos} | 자세: {grasp_dir or "auto"} {quat}')
            threading.Thread(
                target=self._pick_sequence, args=(pos, quat), daemon=True).start()

        elif action == 'place':
            pos = cmd.get('place_pos')
            if pos is None:
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            quat = (GRASP_DIR_MAP.get(grasp_dir, None)
                    if grasp_dir else QUAT_TOP_DOWN)
            self.get_logger().info(f'PLACE 시작 @ {pos} | 자세: {quat}')
            threading.Thread(
                target=self._place_sequence, args=(pos, quat), daemon=True).start()

        elif action == 'move':
            pos = cmd.get('target_pos')
            if pos is None:
                self.get_logger().warn("'move' 명령에 target_pos 없음.")
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            quat = (GRASP_DIR_MAP.get(grasp_dir, None)
                    if grasp_dir else QUAT_TOP_DOWN)
            self.get_logger().info(f'MOVE 시작 @ {pos} | 자세: {quat}')
            threading.Thread(
                target=self._move_sequence, args=(pos, quat), daemon=True).start()

        elif action == 'home':
            self.get_logger().info('HOME 시작')
            threading.Thread(target=self._home_sequence, daemon=True).start()

        else:
            self.get_logger().warn(f"알 수 없는 action: '{action}'")
            with self.lock:
                self.busy = False

    # ── 객체 탐색 ───────────────────────────────────────────────────
    def _find_object(self, label):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d')
        return None

    # ── pick 시퀀스 ─────────────────────────────────────────────────
    def _pick_sequence(self, pos, quat):
        try:
            self.get_logger().info('1/5: 접근')
            self._move(pos['x'], pos['y'], pos['z'] + APPROACH_Z, quat)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('2/5: 그리퍼 열기')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('3/5: 내려가기')
            self._move(pos['x'], pos['y'], pos['z'] + DESCEND_Z, quat)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('4/5: 그리퍼 닫기')
            self._gripper(SIM_GRIPPER_CLOSE, GRIPPER_CLOSE)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('5/5: 들어올리기')
            self._move(pos['x'], pos['y'], pos['z'] + LIFT_Z, quat)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('✅ PICK 완료')
            self._publish_result('success', 'pick_complete')
        except Exception as e:
            self.get_logger().error(f'PICK 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    # ── place 시퀀스 ────────────────────────────────────────────────
    def _place_sequence(self, pos, quat):
        try:
            self.get_logger().info('1/2: place 이동')
            self._move(pos['x'], pos['y'], pos['z'] + APPROACH_Z, quat)
            time.sleep(MOVE_DELAY)

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

    # ── move 시퀀스 ─────────────────────────────────────────────────
    def _move_sequence(self, pos, quat):
        try:
            self.get_logger().info(
                f"1/1: 이동 → ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            self._move(pos['x'], pos['y'], pos['z'], quat)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('✅ MOVE 완료')
            self._publish_result('success', 'move_complete')
        except Exception as e:
            self.get_logger().error(f'MOVE 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    # ── home 시퀀스 ─────────────────────────────────────────────────
    def _home_sequence(self):
        try:
            self.get_logger().info(f'1/2: 홈 위치로 이동')
            self._move(HOME_X, HOME_Y, HOME_Z, QUAT_TOP_DOWN)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('2/2: 그리퍼 열기')
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

    # ── 하드웨어 헬퍼 ───────────────────────────────────────────────
    def _move(self, x: float, y: float, z: float, quat: list):
        """end-effector 를 (x,y,z) + quat 자세로 이동."""
        if self.use_moveit2:
            self.moveit2.move_to_pose(
                position=[x, y, z],
                quat_xyzw=quat,
                cartesian=False,
            )
            self.get_logger().info(
                f'[sim] move → {x:.3f} {y:.3f} {z:.3f} | quat={quat}')
        else:
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
            self.pub_move.publish(pose)

    def _gripper(self, sim_joints, real_width):
        if self.use_moveit2:
            self.moveit2_gripper.move_to_configuration(sim_joints)
            self.get_logger().info(f'[sim] gripper {sim_joints}')
        else:
            msg = JointState()
            msg.header.stamp  = self.get_clock().now().to_msg()
            msg.name          = ['gripper']
            msg.position      = [float(real_width)]
            msg.effort        = [GRIPPER_FORCE]
            self.pub_gripper.publish(msg)

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
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
