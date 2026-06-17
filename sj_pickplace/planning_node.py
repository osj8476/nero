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
SIM_GRIPPER_OPEN  = [0.08]
SIM_GRIPPER_CLOSE = [0.01]

# ── 시퀀스 오프셋 (미터) ──────────────────────────────────────────────────────
APPROACH_Z = 0.13
DESCEND_Z  = 0.03
LIFT_Z     = 0.23

# ── 홈 자세 ───────────────────────────────────────────────────────────────────
HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, 0.75   # 실측 홈 좌표

# ── 타이밍 ────────────────────────────────────────────────────────────────────
MOVE_DELAY    = 6.0
GRIPPER_DELAY = 4.0

# ── 사전 정의 쿼터니언 (xyzw, base_link 기준) ─────────────────────────────────
# RViz tf2_echo 실측값 기반 (IK 검증 완료)
#
# top_down  : gripper z → [0,0,-1]  위에서 아래로 파지
QUAT_TOP_DOWN   = [0.008,  0.999,  0.023,  0.037]
# top_down yaw+180
QUAT_TOP_DOWN_R = [0.999, -0.008, -0.037,  0.023]
# top_down yaw-90
QUAT_TOP_DOWN_L = [0.708,  0.697, -0.010,  0.043]
# top_down yaw+90
QUAT_TOP_DOWN_RR= [-0.643, 0.653,  0.041,  0.011]
# home: 홈 대기 자세 (gripper 위를 향함)
QUAT_HOME       = [0.0, 0.0, 0.0, 1.0]  # 실측 홈 자세
# ★ side_front: 앞에서 수평으로 파지 (RViz 실측값 2025-06-15)
QUAT_SIDE_FRONT = [0.481, -0.527,  0.427,  0.556]

GRASP_DIR_MAP = {
    'top':         QUAT_TOP_DOWN,
    'top_down':    QUAT_TOP_DOWN,
    'top_down_r':  QUAT_TOP_DOWN_R,
    'top_down_l':  QUAT_TOP_DOWN_L,
    'top_down_rr': QUAT_TOP_DOWN_RR,
    'side':        QUAT_SIDE_FRONT,   # ★ 실측값으로 교체
    'side_front':  QUAT_SIDE_FRONT,   # ★ 실측값으로 교체
    'side_left':   QUAT_TOP_DOWN_L,
    'side_right':  QUAT_TOP_DOWN_RR,
}

# 물체 라벨별 기본 파지 방향 힌트
LABEL_GRASP_HINT = {
    'bottle':  'side_front',  # 세워진 병 → 앞에서 수평
    'cup':     'top_down',    # 컵 → 위에서 아래로
    'book':    'side_front',  # 책 → 앞에서 수평
    'box':     'top_down',    # 박스 → 위에서 아래로
    'ball':    'top_down',    # 공 → 위에서 아래로
    'scissors':'side_front',  # 가위 → 앞에서 수평
    'remote':  'top_down',    # 리모컨 → 위에서 아래로
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
    """물체 위치·라벨로 파지 자세 자동 선택."""
    hint = LABEL_GRASP_HINT.get(label, None)
    if hint:
        return GRASP_DIR_MAP[hint]
    x, y = pos.get('x', 0.0), pos.get('y', 0.0)
    if abs(y) > abs(x) * 1.5:
        return QUAT_TOP_DOWN_L if y > 0 else QUAT_TOP_DOWN_RR
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

        from sensor_msgs.msg import JointState as _JointState
        self.latest_joint_state = None
        self.create_subscription(
            _JointState, '/feedback/joint_states',
            lambda msg: setattr(self, 'latest_joint_state', msg), 10)

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
                ignore_new_calls_while_executing=True,
            )
            self.moveit2_gripper = MoveIt2(
                node=self,
                joint_names=['gripper'],
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
        # action server 준비 대기
        if self.use_moveit2:
            import time as _t
            _t.sleep(3.0)
        self.get_logger().info('PlanningNode 준비 완료')

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
            pos   = self._find_object(label)
            if pos is None:
                self.get_logger().warn(f"'{label}' 못 찾음.")
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            quat = (GRASP_DIR_MAP.get(grasp_dir, None)
                    if grasp_dir else _auto_grasp_quat(pos, label))
            self.get_logger().info(
                f'PICK 시작: {label} @ {pos} | 자세: {grasp_dir or "auto"} {quat}')
            t = threading.Thread(target=self._pick_sequence, args=(pos, quat))
            t.daemon = False
            t.start()

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
            # grasp_dir 없으면 현재 자세 유지 (QUAT_HOME 기본)
            quat = (GRASP_DIR_MAP.get(grasp_dir, QUAT_HOME)
                    if grasp_dir else QUAT_HOME)
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

    def _find_object(self, label):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d')
        return None

    def _pick_sequence(self, pos, quat):
        try:
            self.get_logger().info('1/5: 접근')
            self._move(pos['x'], pos['y'], pos['z'] + APPROACH_Z, quat)

            self.get_logger().info('2/5: 그리퍼 열기')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('3/5: 내려가기')
            self._move(pos['x'], pos['y'], pos['z'] + DESCEND_Z, quat)

            self.get_logger().info('4/5: 그리퍼 닫기')
            self._gripper(SIM_GRIPPER_CLOSE, GRIPPER_CLOSE)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('5/5: 들어올리기')
            self._move(pos['x'], pos['y'], pos['z'] + LIFT_Z, quat)

            self.get_logger().info('✅ PICK 완료')
            self._publish_result('success', 'pick_complete')
        except Exception as e:
            self.get_logger().error(f'PICK 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _place_sequence(self, pos, quat):
        try:
            self.get_logger().info('1/2: place 이동')
            self._move(pos['x'], pos['y'], pos['z'] + APPROACH_Z, quat)

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
            self._move(pos['x'], pos['y'], pos['z'], quat)

            self.get_logger().info('✅ MOVE 완료')
            self._publish_result('success', 'move_complete')
        except Exception as e:
            self.get_logger().error(f'MOVE 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _move_joints_sequence(self, joints: dict):
        try:
            joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
            key_map = {'j1':'joint1','j2':'joint2','j3':'joint3','j4':'joint4',
                       'j5':'joint5','j6':'joint6','j7':'joint7'}
            # 현재 joint_state에서 기본값 가져오기
            # joint_state에서 arm joint 7개만 추출 (gripper 제외)
            from sensor_msgs.msg import JointState as _JS
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
            self.moveit2.move_to_configuration(positions)
            import time as _t
            _t.sleep(0.5)
            deadline = _t.time() + 30.0
            while _t.time() < deadline:
                if not self.moveit2._MoveIt2__is_motion_requested and \
                   not self.moveit2._MoveIt2__is_executing:
                    break
                _t.sleep(0.1)
            self.get_logger().info('✅ MOVE_JOINTS 완료')
            self._publish_result('success', 'joint_move_complete')
        except Exception as e:
            self.get_logger().error(f'MOVE_JOINTS 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _move_joints_sequence(self, joints: dict):
        try:
            joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
            key_map = {'j1':'joint1','j2':'joint2','j3':'joint3','j4':'joint4',
                       'j5':'joint5','j6':'joint6','j7':'joint7'}
            # 현재 joint_state에서 기본값 가져오기
            # joint_state에서 arm joint 7개만 추출 (gripper 제외)
            from sensor_msgs.msg import JointState as _JS
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
            self.moveit2.move_to_configuration(positions)
            import time as _t
            _t.sleep(0.5)
            deadline = _t.time() + 30.0
            while _t.time() < deadline:
                if not self.moveit2._MoveIt2__is_motion_requested and \
                   not self.moveit2._MoveIt2__is_executing:
                    break
                _t.sleep(0.1)
            self.get_logger().info('✅ MOVE_JOINTS 완료')
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
                import time as _t
                _t.sleep(0.5)
                deadline = _t.time() + 30.0
                while _t.time() < deadline:
                    if not self.moveit2._MoveIt2__is_motion_requested and \
                       not self.moveit2._MoveIt2__is_executing:
                        break
                    _t.sleep(0.1)
            else:
                self._move(HOME_X, HOME_Y, HOME_Z, QUAT_HOME)

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

    def _move(self, x: float, y: float, z: float, quat: list):
        if self.use_moveit2:
            # 이전 상태 리셋
            self.moveit2.force_reset_executing_state()
            # action server 준비 확인
            import time as _t
            ready = self.moveit2._MoveIt2__move_action_client.server_is_ready()
            self.get_logger().info(f'action server ready: {ready}')
            self.get_logger().info(f'mutex locked: {self.moveit2._MoveIt2__execution_mutex.locked()}')
            self.moveit2.move_to_pose(
                position=[x, y, z],
                quat_xyzw=quat,
                tolerance_position=0.01,
                tolerance_orientation=0.05,
                cartesian=False,
            )
            self.get_logger().info(f'move_to_pose 반환됨')
            # executor가 별도 스레드에서 spin 중이므로 단순 sleep으로 완료 대기
            import time as _t
            _t.sleep(0.5)
            deadline = _t.time() + 30.0
            while _t.time() < deadline:
                req = self.moveit2._MoveIt2__is_motion_requested
                exe = self.moveit2._MoveIt2__is_executing
                if not req and not exe:
                    self.get_logger().info(f'move 완료: ({x:.3f},{y:.3f},{z:.3f})')
                    break
                _t.sleep(0.2)
            else:
                self.get_logger().warn(f'move 타임아웃: ({x:.3f},{y:.3f},{z:.3f})')
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
            g_goal = self.moveit2_gripper._MoveIt2__move_action_goal
            self.get_logger().info(f'GRIPPER group_name: [{g_goal.request.group_name}]')
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
    # executor를 별도 스레드에서 spin (테스트 스크립트와 동일한 방식)
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
