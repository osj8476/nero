#!/usr/bin/env python3
"""
mcp_robot_server.py  (폐루프 패치)
====================================
[변경 요약 vs 구버전]
1. RosBridgeNode 에 /pick_result 구독 추가
   - _on_result() : 결과 수신 → Event 로 대기 중인 도구 함수에 전달
   - wait_for_result(timeout) : 도구 함수에서 호출, 실제 완료까지 블로킹

2. pick_object / place_object / move_to_position / go_home
   - dispatched 즉시 반환 → wait_for_result 로 실제 성공/실패 반환
   - LLM 이 pick 이 끝났는지 polling 없이 알 수 있음

3. pick_and_place
   - time.sleep(28.0) 제거
   - pick 완료 이벤트 수신 후 place 발행 → place 완료 이벤트 반환
   - pick 실패 시 place 발행하지 않고 즉시 실패 반환

4. busy=True 상태일 때 명령이 planning_node 에 의해 무시되므로
   busy 거부 감지를 위한 RESULT_TIMEOUT 을 도구별로 적절히 설정
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

# 각 동작의 최대 허용 시간 (planning_node 상수 기반 + 여유 5초)
# MOVE_DELAY=6.0 * 3회 + GRIPPER_DELAY=4.0 * 2회 = 26초 → pick: 31초
TIMEOUT_PICK  = 31.0
TIMEOUT_PLACE = 16.0   # MOVE_DELAY + GRIPPER_DELAY + 여유
TIMEOUT_MOVE  = 11.0   # MOVE_DELAY + 여유
TIMEOUT_HOME  = 16.0   # MOVE_DELAY + GRIPPER_DELAY + 여유


# ──────────────────────────────────────────────────────────────────────────────
# 1. ROS2 Bridge Node
# ──────────────────────────────────────────────────────────────────────────────
class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('mcp_robot_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # 명령 발행
        self.pub_cmd = self.create_publisher(String, '/arm_command', qos)

        # 장면 인식 구독
        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self._on_objects, qos)

        # ★ 실행 결과 구독 (폐루프 핵심)
        self.sub_result = self.create_subscription(
            String, '/pick_result', self._on_result, qos)

        self._objects_lock = threading.Lock()
        self._latest_objects: list = []
        self._last_obj_stamp: float = 0.0

        # 결과 대기용 — 도구 함수가 Event 를 등록하고 wait()
        self._result_lock   = threading.Lock()
        self._result_event:  Optional[threading.Event] = None
        self._result_payload: Optional[dict]           = None

        self.get_logger().info(
            'RosBridgeNode 준비 완료 '
            '(/detected_objects·/pick_result 구독, /arm_command 발행)')

    # ── 장면 인식 캐시 ──────────────────────────────────────────────
    def _on_objects(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._objects_lock:
                self._latest_objects = data.get("objects", [])
                self._last_obj_stamp = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패')

    def get_objects(self) -> tuple[list, float]:
        with self._objects_lock:
            return list(self._latest_objects), self._last_obj_stamp

    # ── /pick_result 수신 (폐루프 핵심) ────────────────────────────
    def _on_result(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('pick_result JSON 파싱 실패')
            return

        self.get_logger().info(f'/pick_result 수신: {payload}')
        with self._result_lock:
            self._result_payload = payload
            if self._result_event is not None:
                self._result_event.set()

    def wait_for_result(self, timeout: float) -> dict:
        """
        도구 함수에서 호출.
        publish_command() 직후에 사용하면 planning_node 완료까지 블로킹.
        timeout(초) 내 결과 미도달 시 {"status":"timeout"} 반환.
        """
        event = threading.Event()
        with self._result_lock:
            self._result_payload = None
            self._result_event   = event

        arrived = event.wait(timeout=timeout)

        with self._result_lock:
            self._result_event = None
            result = self._result_payload

        if not arrived or result is None:
            return {"status": "timeout",
                    "reason": f"{timeout:.0f}초 내 응답 없음 (planning_node busy 또는 타임아웃)"}
        return result

    # ── 명령 발행 헬퍼 ──────────────────────────────────────────────
    def publish_command(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f'/arm_command 발행: {payload}')


# ──────────────────────────────────────────────────────────────────────────────
# 2. ROS2 백그라운드 스레드
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
    if not _ros_ready.is_set():
        t = threading.Thread(target=_ros_spin_thread, daemon=True)
        t.start()
        _ros_ready.wait(timeout=10.0)
        if _ros_node is None:
            raise RuntimeError("ROS2 bridge 노드 초기화 실패")


# ──────────────────────────────────────────────────────────────────────────────
# 3. MCP Tool 정의
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP("agilex-nero-pnp")


@mcp.tool()
def list_detected_objects() -> str:
    """현재 카메라 비전(perception_node)이 인식 중인 물체 목록을 조회한다.

    로봇에게 무언가를 시키기 전에 반드시 먼저 이 도구를 호출해서
    실제로 어떤 물체가 장면에 있는지 확인하라.

    Returns:
        {"objects": [{"label": "cup", "center_3d": {"x":0.31,"y":-0.04,"z":0.1}}, ...],
         "age_sec": 0.18}
        objects 가 빈 배열이면 현재 인식된 물체가 없다.
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    slim = [{"label": o.get("label", "?"), "center_3d": o.get("center_3d", {})}
            for o in objects]
    return json.dumps({"objects": slim, "age_sec": age}, ensure_ascii=False)


@mcp.tool()
def pick_object(target_label: str) -> str:
    """지정한 물체를 로봇 팔로 집어 올린다. pick 완료까지 블로킹.

    planning_node 의 5단계 시퀀스(Approach→Open→Descend→Close→Lift)가
    완료되거나 실패할 때까지 기다린 뒤 실제 결과를 반환한다.

    Args:
        target_label: 집을 물체 라벨. list_detected_objects() 결과 중 하나.
                      (예: "cup", "bottle"). 영어 소문자.

    Returns:
        성공: {"status": "success", "reason": "pick_complete", "target_label": "..."}
        실패: {"status": "failed"|"rejected"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()

    objects, _ = _ros_node.get_objects()
    available = {o.get("label", "").lower() for o in objects}
    if available and target_label not in available:
        return json.dumps({
            "status": "rejected",
            "reason": f"'{target_label}' 은(는) 현재 장면에 없습니다. "
                      f"인식된 물체: {sorted(available)}",
        }, ensure_ascii=False)

    _ros_node.publish_command({"action": "pick", "target_label": target_label})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PICK)
    result["target_label"] = target_label
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def place_object(x: float, y: float, z: float) -> str:
    """집은 물체를 지정한 좌표(base_link 기준, 미터)에 내려놓는다. place 완료까지 블로킹.

    pick_object() 성공 이후에 호출해야 한다.

    Args:
        x: 전방 거리 (양수 = 로봇 앞, 단위: 미터)
        y: 좌우 거리 (양수 = 왼쪽, 단위: 미터)
        z: 높이 (테이블 위 ≈ 0.0, 단위: 미터)

    Returns:
        성공: {"status": "success", "reason": "place_complete", "place_pos": {...}}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    place_pos = {"x": x, "y": y, "z": z}
    _ros_node.publish_command({"action": "place", "place_pos": place_pos})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PLACE)
    result["place_pos"] = place_pos
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def pick_and_place(target_label: str, x: float, y: float, z: float) -> str:
    """물체를 집은 뒤 지정 좌표에 내려놓는다. 전체 완료까지 블로킹.

    pick 완료를 확인한 뒤 place 를 발행하므로,
    pick 실패 시 place 를 보내지 않고 즉시 실패를 반환한다.

    Args:
        target_label: 집을 물체 라벨 (예: "cup"). 영어 소문자.
        x, y, z: 내려놓을 위치 (base_link 기준, 미터)

    Returns:
        성공: {"status": "success", "reason": "place_complete",
               "target_label": "...", "place_pos": {...}}
        실패: {"status": "failed"|"rejected"|"timeout", "reason": "...",
               "failed_at": "pick"|"place"}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()
    place_pos    = {"x": x, "y": y, "z": z}

    # ── 1) 장면 유효성 검사 ──────────────────────────────────────
    objects, _ = _ros_node.get_objects()
    available = {o.get("label", "").lower() for o in objects}
    if available and target_label not in available:
        return json.dumps({
            "status": "rejected",
            "reason": f"'{target_label}' 은(는) 현재 장면에 없습니다. "
                      f"인식된 물체: {sorted(available)}",
        }, ensure_ascii=False)

    # ── 2) pick 발행 → 완료 대기 ────────────────────────────────
    _ros_node.publish_command({"action": "pick", "target_label": target_label})
    pick_result = _ros_node.wait_for_result(timeout=TIMEOUT_PICK)

    if pick_result.get("status") != "success":
        pick_result["failed_at"] = "pick"
        pick_result["target_label"] = target_label
        return json.dumps(pick_result, ensure_ascii=False)

    # ── 3) pick 성공 → place 발행 → 완료 대기 ──────────────────
    _ros_node.publish_command({"action": "place", "place_pos": place_pos})
    place_result = _ros_node.wait_for_result(timeout=TIMEOUT_PLACE)

    if place_result.get("status") != "success":
        place_result["failed_at"] = "place"
        place_result["target_label"] = target_label
        place_result["place_pos"]    = place_pos
        return json.dumps(place_result, ensure_ascii=False)

    # ── 4) 전체 성공 ─────────────────────────────────────────────
    place_result["target_label"] = target_label
    place_result["place_pos"]    = place_pos
    return json.dumps(place_result, ensure_ascii=False)


@mcp.tool()
def move_to_position(x: float, y: float, z: float) -> str:
    """로봇 팔 끝(end-effector)을 지정 좌표로 이동한다. 완료까지 블로킹.

    물체를 집지 않고 팔만 특정 위치로 보낼 때 사용한다.

    Args:
        x, y, z: 목표 좌표 (base_link 기준, 미터)

    Returns:
        성공: {"status": "success", "reason": "move_complete", "target_pos": {...}}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    target_pos = {"x": x, "y": y, "z": z}
    _ros_node.publish_command({"action": "move", "target_pos": target_pos})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    result["target_pos"] = target_pos
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def go_home() -> str:
    """로봇 팔을 홈 자세로 복귀시키고 그리퍼를 연다. 완료까지 블로킹.

    Returns:
        성공: {"status": "success", "reason": "home_complete"}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    _ros_node.publish_command({"action": "home"})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_HOME)
    return json.dumps(result, ensure_ascii=False)


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
    t = threading.Thread(target=_ros_spin_thread, daemon=True)
    t.start()

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
