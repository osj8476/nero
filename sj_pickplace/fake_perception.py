#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

FAKE_OBJECTS = [
    {"label": "cup",      "center_3d": {"x": 0.25, "y": - 0.20, "z": 0.00}},
    {"label": "bottle",   "center_3d": {"x": -0.32, "y": -0.10, "z": 0.00}},
    {"label": "book", "center_3d": {"x": 0.20, "y": -0.25, "z": 0.00}},
]

ORIGINAL_POSITIONS = {obj["label"]: dict(obj["center_3d"]) for obj in FAKE_OBJECTS}

class FakePerception(Node):
    def __init__(self):
        super().__init__("fake_perception")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pub = self.create_publisher(String, "/detected_objects", qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/detected_markers", 10)

        # 표시 위치 (마커용) - 원래 위치 고정, pick 완료 시만 변경
        self.display_positions = {obj["label"]: dict(obj["center_3d"]) for obj in FAKE_OBJECTS}
        self.picked_label = None

        self.create_timer(1.0, self.publish_objects)
        self.create_timer(0.2, self.publish_markers)  # 5Hz로 부드럽게

        self.create_subscription(String, '/pick_result', self.on_result, 10)
        self.create_subscription(String, '/arm_command', self.on_command, 10)

        self.get_logger().info("Fake perception started.")

    def on_command(self, msg: String):
        try:
            cmd = json.loads(msg.data)
            if cmd.get('action') == 'pick':
                self.picked_label = cmd.get('target_label')
                self.get_logger().info(f"[fp] pick 감지: {self.picked_label}")
        except:
            pass

    def on_result(self, msg: String):
        try:
            result = json.loads(msg.data)
            if result.get('status') == 'success' and result.get('reason') == 'pick_complete':
                label = self.picked_label
                if label and label in self.display_positions:
                    orig = ORIGINAL_POSITIONS[label]
                    # pick 완료 후 들어올린 위치로 고정
                    self.display_positions[label] = {
                        'x': orig['x'],
                        'y': orig['y'],
                        'z': orig['z'] + 0.20
                    }
                    self.picked_label = None
                    self.get_logger().info(f"[fp] {label} pick 완료 → z+0.20")
        except:
            pass

    def publish_objects(self):
        # planning_node에는 항상 원래 위치 전달 (pick 판단용)
        msg = String()
        msg.data = json.dumps({"objects": [
            {"label": obj["label"],
             "bbox": [0,0,0,0],
             "center_2d": {"x": 0, "y": 0},
             "center_3d": dict(ORIGINAL_POSITIONS[obj["label"]]),
             "depth_m": 0.6}
            for obj in FAKE_OBJECTS
        ]}, ensure_ascii=False)
        self.pub.publish(msg)

    def publish_markers(self):
        array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, obj in enumerate(FAKE_OBJECTS):
            label = obj["label"]
            pos = self.display_positions[label]

            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = now
            m.ns = "objects"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = pos['x']
            m.pose.position.y = pos['y']
            m.pose.position.z = pos['z']
            m.pose.orientation.w = 1.0
            m.scale.x = 0.06
            m.scale.y = 0.06
            m.scale.z = 0.06
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 1.0
            array.markers.append(m)

            t = Marker()
            t.header.frame_id = "base_link"
            t.header.stamp = now
            t.ns = "labels"
            t.id = i + 100
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = pos['x']
            t.pose.position.y = pos['y']
            t.pose.position.z = pos['z'] + 0.08
            t.pose.orientation.w = 1.0
            t.scale.z = 0.05
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            t.text = label
            array.markers.append(t)

        self.pub_markers.publish(array)

def main():
    rclpy.init()
    node = FakePerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
