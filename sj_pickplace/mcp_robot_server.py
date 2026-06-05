#!/usr/bin/env python3
"""
mcp_robot_server.py
AgileX Nero + Gripper 픽앤플레이스용 MCP 서버.
- 스레드 안전한 ROS 초기화 (double-check locking)
- /pick_result 구독으로 픽업/플레이스 완료 대기
- place_object Tool 추가
"""

import os
import json
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from mcp.server.fastmcp import FastMCP


PICK_TIMEOUT_SEC = float(os.environ.get("PICK_TIMEOUT_SEC", "30.0"))


class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('mcp_robot_bridge')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_perception = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_cmd = self.create_publisher(String, '/arm_command', qos)
        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self._on_objects, qos_perception)
        self.sub_result = self.create_subscription(
            String, '/pick_result', self._on_pick_result, qos)

        self._objects_lock = threading.Lock()
        self._latest_objects: list = []
        self._last_obj_stamp: float = 0.0

        # 결과 대기용
        self._result_event = threading.Event()
        self._latest_result: dict = {}

        self.get_logger().info('RosBridgeNode 준비 완료')

    def _on_objects(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._objects_lock:
                self._latest_objects = data.get("objects", [])
                self._last_obj_stamp = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패')

    def _on_pick_result(self, msg: String):
        try:
            self._latest_result = json.loads(msg.data)
            self._result_event.set()
        except json.JSONDecodeError:
            self.get_logger().warn('pick_result JSON 파싱 실패')

    def get_objects(self):
        with self._objects_lock:
            return list(self._latest_objects), self._last_obj_stamp

    def publish_arm_command(self, target_label: str, action: str = "pick",
                            place_pos: Optional[dict] = None):
        payload = {
            "target_label": target_label,
            "action": action,
            "timestamp": time.time(),
        }
        if place_pos is not None:
            payload["place_pos"] = place_pos

        self._result_event.clear()
        self._latest_result = {}

        msg = String()
        msg.data = json.dumps(payload)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f'/arm_command 발행: {payload}')
        return payload

    def wait_for_result(self, timeout: float = PICK_TIMEOUT_SEC) -> dict:
        if self._result_event.wait(timeout=timeout):
            return self._latest_result
        return {"status": "timeout"}


# ──────────────────────────────────────────────
# ROS 노드 초기화 (스레드 안전)
# ──────────────────────────────────────────────
_ros_node: Optional[RosBridgeNode] = None
_ros_ready = threading.Event()
_ros_init_lock = threading.Lock()


def _ros_spin_thread():
    global _ros_node
    rclpy.init()
    _ros_node = RosBridgeNode()
    _ros_ready.set()
    try:
        rclpy.spin(_ros_node)
    finally:
        _ros_node.destroy_node()
        rclpy.shutdown()


def _ensure_ros():
    if _ros_ready.is_set():
        return
    with _ros_init_lock:
        if _ros_ready.is_set():  # double-check
            return
        t = threading.Thread(target=_ros_spin_thread, daemon=True)
        t.start()
        _ros_ready.wait(timeout=10.0)
        if _ros_node is None:
            raise RuntimeError("ROS2 bridge 노드 초기화 실패")


# ──────────────────────────────────────────────
# MCP Tools
# ──────────────────────────────────────────────
mcp = FastMCP("agilex-nero-pnp")


@mcp.tool()
def list_detected_objects() -> str:
    """현재 카메라 비전이 인식 중인 물체 목록을 조회한다.
    로봇에게 무언가를 집으라고 시키기 전에 반드시 먼저 호출해서
    실제로 어떤 물체가 있는지, 라벨이 무엇인지 확인하라.

    Returns:
        JSON 문자열. {"objects":[{"label":"cup","center_3d":{...}}], "age_sec":0.18}
        objects 가 빈 배열이면 인식된 물체가 없다는 뜻이다.
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    slim = [{"label": o.get("label", "?"),
             "center_3d": o.get("center_3d", {})} for o in objects]
    return json.dumps({"objects": slim, "age_sec": age}, ensure_ascii=False)


@mcp.tool()
def pick_object(target_label: str) -> str:
    """지정한 물체를 로봇 팔로 집어 올린다 (pick 동작). 완료될 때까지 대기한다.

    Args:
        target_label: 집을 물체 라벨. list_detected_objects() 가 반환한
                       objects 안에 존재하는 label 이어야 한다. 영어 소문자.

    Returns:
        성공: {"status":"success","target_label":"...","elapsed_sec":12.3}
        실패: {"status":"failed"|"rejected"|"timeout","reason":"..."}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()
    objects, _ = _ros_node.get_objects()
    available = {o.get("label", "").lower() for o in objects}

    if not available:
        return json.dumps({
            "status": "rejected",
            "reason": "현재 인식된 물체가 없습니다. 카메라 확인 필요.",
        }, ensure_ascii=False)

    if target_label not in available:
        return json.dumps({
            "status": "rejected",
            "reason": f"'{target_label}' 은(는) 현재 장면에 없습니다. "
                      f"인식된 물체: {sorted(available)}",
        }, ensure_ascii=False)

    t0 = time.time()
    _ros_node.publish_arm_command(target_label, action="pick")
    result = _ros_node.wait_for_result()
    elapsed = round(time.time() - t0, 2)

    return json.dumps({
        **result,
        "target_label": target_label,
        "elapsed_sec": elapsed,
    }, ensure_ascii=False)


@mcp.tool()
def place_object(location: str) -> str:
    """집어들고 있는 물체를 지정한 위치에 내려놓는다.

    Args:
        location: 내려놓을 위치 라벨. 사전 정의된 위치 중 하나:
                  'box_a', 'box_b', 'home', 'left', 'right'.

    Returns:
        성공: {"status":"success","location":"..."}
        실패: {"status":"failed"|"timeout","reason":"..."}
    """
    _ensure_ros()

    # 사전 정의된 place 위치 (실제 환경에 맞게 수정)
    PLACE_LOCATIONS = {
        "box_a":  {"x": 0.35, "y":  0.20, "z": 0.10},
        "box_b":  {"x": 0.35, "y": -0.20, "z": 0.10},
        "home":   {"x": 0.30, "y":  0.00, "z": 0.15},
        "left":   {"x": 0.30, "y":  0.25, "z": 0.10},
        "right":  {"x": 0.30, "y": -0.25, "z": 0.10},
    }

    location = location.strip().lower()
    if location not in PLACE_LOCATIONS:
        return json.dumps({
            "status": "rejected",
            "reason": f"알 수 없는 위치: '{location}'. "
                      f"가능한 위치: {list(PLACE_LOCATIONS.keys())}",
        }, ensure_ascii=False)

    t0 = time.time()
    _ros_node.publish_arm_command(
        target_label=location,
        action="place",
        place_pos=PLACE_LOCATIONS[location],
    )
    result = _ros_node.wait_for_result()
    elapsed = round(time.time() - t0, 2)

    return json.dumps({
        **result,
        "location": location,
        "elapsed_sec": elapsed,
    }, ensure_ascii=False)


@mcp.tool()
def get_system_status() -> str:
    """로봇/비전 브리지의 현재 연결 상태를 점검한다 (헬스체크용)."""
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    return json.dumps({
        "ros_bridge": "up",
        "vision_objects": len(objects),
        "vision_age_sec": age,
    }, ensure_ascii=False)


def main():
    _ensure_ros()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()