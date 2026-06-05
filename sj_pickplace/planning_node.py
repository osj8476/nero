#!/usr/bin/env python3
import json
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

GRIPPER_OPEN  = 0.08
GRIPPER_CLOSE = 0.01
GRIPPER_FORCE = 1.5
SIM_GRIPPER_OPEN  = [0.05, -0.05]
SIM_GRIPPER_CLOSE = [0.01, -0.01]

QUAT_DOWN = [1.0, 0.0, 0.0, 0.0]

APPROACH_Z  = 0.13
DESCEND_Z   = 0.03
LIFT_Z      = 0.23

MOVE_DELAY    = 6.0
GRIPPER_DELAY = 4.0

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
        self.pub_gripper = self.create_publisher(JointState, '/control/joint_states', qos_reliable)
        self.pub_result  = self.create_publisher(String, '/pick_result', qos_reliable)

        self.moveit2         = None
        self.moveit2_gripper = None
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

    def on_objects(self, msg):
        try:
            self.latest_objects = json.loads(msg.data).get('objects', [])
        except:
            pass

    def on_command(self, msg):
        self.get_logger().info(f'[CMD] 수신: {msg.data[:60]}')
        with self.lock:
            if self.busy:
                self.get_logger().warn('작업 중. 명령 무시.')
                return
            self.busy = True

        try:
            cmd = json.loads(msg.data)
        except:
            with self.lock:
                self.busy = False
            return

        action = cmd.get('action', 'pick')
        if action == 'pick':
            label = cmd.get('target_label')
            pos = self._find_object(label)
            if pos is None:
                self.get_logger().warn(f"'{label}' 못 찾음.")
                with self.lock:
                    self.busy = False
                return
            self.get_logger().info(f'PICK 시작: {label} @ {pos}')
            threading.Thread(target=self._pick_sequence, args=(pos,), daemon=True).start()

        elif action == 'place':
            pos = cmd.get('place_pos')
            if pos is None:
                with self.lock:
                    self.busy = False
                return
            self.get_logger().info(f'PLACE 시작 @ {pos}')
            threading.Thread(target=self._place_sequence, args=(pos,), daemon=True).start()

    def _find_object(self, label):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d')
        return None

    def _pick_sequence(self, pos):
        try:
            self.get_logger().info('1/5: 접근')
            self._move(pos['x'], pos['y'], pos['z'] + APPROACH_Z)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('2/5: 그리퍼 열기')
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('3/5: 내려가기')
            self._move(pos['x'], pos['y'], pos['z'] + DESCEND_Z)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('4/5: 그리퍼 닫기')
            self._gripper(SIM_GRIPPER_CLOSE, GRIPPER_CLOSE)
            time.sleep(GRIPPER_DELAY)

            self.get_logger().info('5/5: 들어올리기')
            self._move(pos['x'], pos['y'], pos['z'] + LIFT_Z)
            time.sleep(MOVE_DELAY)

            self.get_logger().info('✅ PICK 완료')
            self._publish_result('success', 'pick_complete')
        except Exception as e:
            self.get_logger().error(f'PICK 오류: {e}')
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False

    def _place_sequence(self, pos):
        try:
            self.get_logger().info('1/2: place 이동')
            self._move(pos['x'], pos['y'], pos['z'] + APPROACH_Z)
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

    def _move(self, x, y, z):
        if self.use_moveit2:
            self.moveit2.move_to_pose(
                position=[x, y, z],
                quat_xyzw=QUAT_DOWN,
                cartesian=False,
            )
            self.get_logger().info(f'[sim] move → {x:.3f} {y:.3f} {z:.3f}')
        else:
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = QUAT_DOWN[0]
            pose.pose.orientation.y = QUAT_DOWN[1]
            pose.pose.orientation.z = QUAT_DOWN[2]
            pose.pose.orientation.w = QUAT_DOWN[3]
            self.pub_move.publish(pose)

    def _gripper(self, sim_joints, real_width):
        if self.use_moveit2:
            self.moveit2_gripper.move_to_configuration(sim_joints)
            self.get_logger().info(f'[sim] gripper {sim_joints}')
        else:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ['gripper']
            msg.position = [float(real_width)]
            msg.effort = [GRIPPER_FORCE]
            self.pub_gripper.publish(msg)

    def _publish_result(self, status, reason=''):
        msg = String()
        msg.data = json.dumps({'status': status, 'reason': reason}, ensure_ascii=False)
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
