#!/usr/bin/env python3
"""
mcp_robot_server.py
====================
AgileX Nero + Gripper 픽앤플레이스 시스템용 MCP(Model Context Protocol) 서버.

[설계 개념]
- FastMCP 로 LLM 에게 노출할 "로봇 제어 API" (Tool) 를 정의한다.
- 같은 프로세스 안에서 rclpy 노드(RosBridgeNode)를 별도 데몬 스레드로 spin 시킨다.
- LLM 이 Tool 을 호출하면 -> RosBridgeNode 가 기존 토픽 라인에 그대로 발행한다:
      pick_object()         -> /arm_command   (std_msgs/String, 기존 planning_node 가 구독)
      list_detected_objects -> /detected_objects 캐시 조회 (perception_node 가 발행)
- 즉 기존 perception_node.py / planning_node.py 는 단 한 줄도 수정하지 않는다.
  command_parser 가 발행하던 /arm_command 페이로드 포맷
  ( {"target_label": "...", "action": "pick"} ) 를 그대로 유지하므로 완전 호환.

[전송 방식]
- 기본 stdio (Claude Desktop / 로컬 클라이언트가 프로세스를 직접 띄우는 경우)
- MCP_TRANSPORT=sse 또는 streamable-http 로 바꾸면 원격 상주 서버로도 동작.
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


# ──────────────────────────────────────────────────────────────────────────────
# 1. ROS2 Bridge Node
#    - /detected_objects 구독 → 최신 장면 캐시
#    - /arm_command 발행      → 기존 planning_node 로 명령 전달
# ──────────────────────────────────────────────────────────────────────────────
class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('mcp_robot_bridge')

        # 기존 시스템과 동일한 QoS (depth=10) 사용 — 호환성 보장
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # 기존 planning_node 가 구독하는 바로 그 토픽
        self.pub_cmd = self.create_publisher(String, '/arm_command', qos)

        # 기존 perception_node 가 발행하는 바로 그 토픽
        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self._on_objects, qos)

        self._objects_lock = threading.Lock()
        self._latest_objects: list = []
        self._last_obj_stamp: float = 0.0

        self.get_logger().info('RosBridgeNode 준비 완료 '
                                '(/detected_objects 구독, /arm_command 발행)')

    # ── 비전 결과 캐시 ──────────────────────────────────────────────
    def _on_objects(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._objects_lock:
                self._latest_objects = data.get("objects", [])
                self._last_obj_stamp = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패 (무시)')

    def get_objects(self) -> tuple[list, float]:
        with self._objects_lock:
            # 얕은 복사로 스레드 안전하게 반환
            return list(self._latest_objects), self._last_obj_stamp

    # ── /arm_command 발행 (기존 페이로드 포맷 유지) ────────────────
    def publish_arm_command(self, target_label: str, action: str = "pick"):
        payload = {"target_label": target_label, "action": action}
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f'/arm_command 발행: {payload}')
        return payload


# ──────────────────────────────────────────────────────────────────────────────
# 2. ROS2 를 백그라운드 데몬 스레드에서 spin
#    (MCP 서버는 메인 스레드/이벤트루프, ROS 는 별도 스레드 → 서로 블로킹 안 함)
# ──────────────────────────────────────────────────────────────────────────────
_ros_node: Optional[RosBridgeNode] = None
_ros_ready = threading.Event()


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
    """ROS 노드가 아직 안 떴으면 띄우고, 준비될 때까지 대기."""
    if not _ros_ready.is_set():
        t = threading.Thread(target=_ros_spin_thread, daemon=True)
        t.start()
        _ros_ready.wait(timeout=10.0)
        if _ros_node is None:
            raise RuntimeError("ROS2 bridge 노드 초기화 실패")


# ──────────────────────────────────────────────────────────────────────────────
# 3. MCP 서버 + Tool 정의
#    LLM 에게는 이 docstring/시그니처가 "공식 API 명세"로 그대로 전달된다.
#    => 프롬프트에 "JSON만 출력해" 같은 억지 지시가 전혀 필요 없어진다.
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP("agilex-nero-pnp")


@mcp.tool()
def list_detected_objects() -> str:
    """현재 카메라 비전(perception_node)이 인식 중인 물체 목록을 조회한다.

    로봇에게 무언가를 집으라고 시키기 전에, 반드시 먼저 이 도구를 호출해서
    실제로 어떤 물체가 장면에 있는지, 그 라벨(영어 소문자)이 무엇인지 확인하라.

    Returns:
        JSON 문자열. 형식:
        {"objects": [{"label": "cup",
                       "center_3d": {"x": 0.31, "y": -0.04, "z": 0.1}}, ...],
         "age_sec": 0.18}
        objects 가 빈 배열이면 현재 인식된 물체가 없다는 뜻이다.
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    slim = [
        {"label": o.get("label", "?"),
         "center_3d": o.get("center_3d", {})}
        for o in objects
    ]
    return json.dumps({"objects": slim, "age_sec": age}, ensure_ascii=False)


@mcp.tool()
def pick_object(target_label: str) -> str:
    """지정한 물체를 로봇 팔로 집어 올린다 (pick 동작).

    내부적으로 planning_node 의 상태머신
    (Approach → Open → Descend → Close → Lift) 을 트리거한다.
    실제 좌표 매칭은 planning_node 가 /detected_objects 와 대조해 수행하므로,
    여기서는 어떤 물체를 집을지 라벨만 넘기면 된다.

    Args:
        target_label: 집을 물체의 라벨. 반드시 list_detected_objects() 가
                       반환한 objects 안에 존재하는 label 중 하나여야 한다.
                       (예: "cup", "bottle", "box"). 영어 소문자.

    Returns:
        실행 결과 JSON 문자열.
        성공 시: {"status": "dispatched", "target_label": "...", "action": "pick"}
        실패 시: {"status": "rejected", "reason": "..."}  (장면에 없는 물체 등)
    """
    _ensure_ros()
    target_label = target_label.strip().lower()

    # ── 안전장치: 실제 장면에 없는 물체는 거부 (LLM 환각 방지) ──
    objects, _ = _ros_node.get_objects()
    available = {o.get("label", "").lower() for o in objects}
    if available and target_label not in available:
        return json.dumps({
            "status": "rejected",
            "reason": f"'{target_label}' 은(는) 현재 장면에 없습니다. "
                      f"인식된 물체: {sorted(available)}",
        }, ensure_ascii=False)

    payload = _ros_node.publish_arm_command(target_label, action="pick")
    return json.dumps({"status": "dispatched", **payload}, ensure_ascii=False)


@mcp.tool()
def get_system_status() -> str:
    """로봇/비전 브리지의 현재 연결 상태를 점검한다 (헬스체크용).

    Returns:
        {"ros_bridge": "up", "vision_objects": 3, "vision_age_sec": 0.2}
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    return json.dumps({
        "ros_bridge": "up",
        "vision_objects": len(objects),
        "vision_age_sec": age,
    }, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 엔트리포인트
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ROS를 백그라운드에서 먼저 시작 (MCP initialize 타임아웃 방지)
    t = threading.Thread(target=_ros_spin_thread, daemon=True)
    t.start()

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        # 로컬 클라이언트(Claude Desktop / command_parser stdio)가
        # 이 프로세스를 직접 spawn 하는 경우
        mcp.run(transport="stdio")
    else:
        # 원격 상주 서버 (sse / streamable-http)
        # 예: MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
