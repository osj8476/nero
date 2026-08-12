#!/usr/bin/env python3
"""
mcp_robot_server.py  (경로 패치)

[변경점 vs 이전 버전]
1. POSES_FILE 경로를 XDG_DATA_HOME 기반으로 변경
   - 이전: ~/sj/saved_poses.json  (Jetson Thor 전용 하드코딩)
   - 이후: ~/.local/share/nero_robot/saved_poses.json
   - 환경변수 NERO_POSES_FILE 로 오버라이드 가능 (기존 Jetson 경로 유지 시 사용)
   - Jetson Thor에서 기존 파일 마이그레이션 방법:
       cp ~/sj/saved_poses.json ~/.local/share/nero_robot/saved_poses.json

그 외 로직은 이전 버전(폐루프 패치)과 동일.
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
from sensor_msgs.msg import JointState

from mcp.server.fastmcp import FastMCP

# [신설] stack_boxes(allow_reorder=True) 타임아웃 계산에 필요한 백트래킹
# 예산 상수. planning_node.py의 task_planner.run_stack_plan과 동일한
# 값을 공유하기 위해 import (중복정의 방지).
from .task_planner import DEFAULT_MAX_BACKTRACK_ATTEMPTS

# ── 타임아웃 설정 ──────────────────────────────────────────────────────────────
TIMEOUT_PICK  = 75.0  # [2026-07 수정] IK 후보비교, 재조회 재시도,
# calib_debug tf lookup 등이 추가되며 pick 시퀀스가 예전보다 느려짐
# (실측: 로봇은 성공했는데 MCP가 31초 타임아웃으로 먼저 실패 처리한
# 사례 확인). 넉넉하게 상향.
TIMEOUT_PLACE = 60.0  # [2026-07 수정] place가 approach->align->descend
# 4단계 구조로 바뀌고 재시도까지 겹치면 40초 이상 걸리는 것이 실측
# 확인됨(16초는 MCP가 로봇 완료 전에 먼저 포기해서 중복 명령/재시도
# 루프를 만드는 원인이었음). pick과 비슷한 수준으로 넉넉하게 상향.
TIMEOUT_MOVE  = 11.0
TIMEOUT_HOME  = 16.0
TIMEOUT_STACK_PER_BOX = 90.0  # pick(최대 75s)+place(최대 60s)를 박스당 여유있게 커버

# [2026-07 추가] 박스 실측 치수 지원 전까지 임시 고정값 (추후 A안:
# depth 기반 실측으로 교체 예정). 현재 테스트 환경 박스 높이 기준.
DEFAULT_BOX_HEIGHT_M = 0.05
TIMEOUT_SCAN  = 60.0  # joint1 스윕 시간 고려 (6스텝 * 약 8초)

# ── 포즈 저장 경로 (XDG 기반 — Jetson/PC 모두 호환) ───────────────────────────
# 오버라이드: export NERO_POSES_FILE=~/sj/saved_poses.json  (Jetson 기존 경로 유지 시)
_DEFAULT_POSES_DIR  = os.path.join(
    os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share')),
    'nero_robot'
)
POSES_FILE = os.environ.get(
    'NERO_POSES_FILE',
    os.path.join(_DEFAULT_POSES_DIR, 'saved_poses.json')
)


def _load_poses() -> dict:
    if not os.path.exists(POSES_FILE):
        return {}
    try:
        with open(POSES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_poses(poses: dict):
    os.makedirs(os.path.dirname(POSES_FILE), exist_ok=True)
    with open(POSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(poses, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 1. ROS2 Bridge Node
# ──────────────────────────────────────────────────────────────────────────────
class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('mcp_robot_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub_cmd = self.create_publisher(String, '/arm_command', qos)

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self._on_objects, qos)
        self.sub_result = self.create_subscription(
            String, '/pick_result', self._on_result, qos)
        # [2026-07 수정] /feedback/joint_states는 sim 환경에서 아무도
        # 발행하지 않는 실물 전용 토픽이라(오늘 FK/IK 조회 버그 원인으로
        # 실측 확인됨), get_joint_positions/save_pose/move_joints_relative가
        # 전부 실패하는 문제가 있었음. sim에서 실제로 살아있는
        # /joint_states로 교체.
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, qos)

        self._objects_lock = threading.Lock()
        self._latest_objects: list = []
        self._last_obj_stamp: float = 0.0

        self._joint_lock = threading.Lock()
        self._latest_joint_state = None
        self._last_joint_stamp: float = 0.0

        self._result_lock   = threading.Lock()
        self._result_event:  Optional[threading.Event] = None
        self._result_payload: Optional[dict]           = None

        self.get_logger().info(
            f'RosBridgeNode 준비 완료 | POSES_FILE={POSES_FILE}')

    def _on_objects(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._objects_lock:
                self._latest_objects = data.get('objects', [])
                self._last_obj_stamp = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패')

    def get_objects(self) -> tuple:
        with self._objects_lock:
            return list(self._latest_objects), self._last_obj_stamp

    def _on_joint_state(self, msg: JointState):
        with self._joint_lock:
            self._latest_joint_state = dict(zip(msg.name, msg.position))
            self._last_joint_stamp = time.time()

    def get_joint_state(self):
        with self._joint_lock:
            state = dict(self._latest_joint_state) if self._latest_joint_state else None
            return state, self._last_joint_stamp

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
        event = threading.Event()
        with self._result_lock:
            self._result_payload = None
            self._result_event   = event
        arrived = event.wait(timeout=timeout)
        with self._result_lock:
            self._result_event = None
            result = self._result_payload
        if not arrived or result is None:
            return {'status': 'timeout',
                    'reason': f'{timeout:.0f}초 내 응답 없음 (planning_node busy 또는 타임아웃)'}
        return result

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

# [2026-07 추가] 마지막 scan_for_boxes 결과 캐시. get_scanned_boxes가
# ROS 왕복 없이 즉시 응답할 수 있게 하기 위함 (로봇이 움직이지 않는
# 물체는 재스캔 없이도 이 캐시로 충분히 정확함).
_last_scanned_boxes: list = []
_last_scan_stamp: float = 0.0


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
            raise RuntimeError('ROS2 bridge 노드 초기화 실패')
        time.sleep(3.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. MCP Tool 정의
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP('agilex-nero-pnp')


@mcp.tool()
def list_detected_objects() -> str:
    """현재 카메라 비전(perception_node)이 인식 중인 물체 목록을 조회한다.

    로봇에게 무언가를 시키기 전에 반드시 먼저 이 도구를 호출해서
    실제로 어떤 물체가 장면에 있는지 확인하라.

    ⚠ 중요 — 좌표/치수 해석 시 반드시 지켜야 할 규칙:
    1. center_3d는 물체의 "기하학적 중심점" 좌표다 (바닥면이 아니다).
       즉 center_3d.z는 물체 바닥이 아니라 물체 중앙 높이다.
    2. box_height_m은 "중심에서 바닥까지의 거리"가 아니라
       "직육면체 한 변(높이 방향)의 전체 길이"다. 따라서:
       - 이 물체의 바닥 높이 = center_3d.z - box_height_m / 2
       - 이 물체의 윗면 높이 = center_3d.z + box_height_m / 2
       - 이 물체 "위에" 다른 박스를 쌓으려면, 쌓을 박스의 중심 z를
         (이 물체의 윗면 높이) + (쌓을 박스의 box_height_m / 2) 로 계산해야 한다.
       - box_height_m은 현재 모든 물체에 고정값 0.05m가 채워진다.
         이 값은 신뢰할 수 있는 고정 상수다. 스캔 결과에 물체가
         안 보이거나 애매하더라도, 높이를 재확인하기 위해 재스캔하거나
         go_home으로 복귀하지 마라. 물체 자체가 안 보이는 것과 높이
         불확실성은 별개 문제이며, 후자를 이유로 전자의 조치를 반복
         호출하지 마라.

    Returns:
        {"objects": [{"label": "cup", "center_3d": {"x":0.31,"y":-0.04,"z":0.1},
                       "angle_base_deg": 45.0, "box_height_m": 0.05}, ...],
         "age_sec": 0.18}
        angle_base_deg는 물체(주로 박스류)의 base_link 기준 회전각(0~90도)이다.
        해당 없으면 null.
        objects 가 빈 배열이면 현재 인식된 물체가 없다.
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    slim = [{'label': o.get('label', '?'),
             'center_3d': o.get('center_3d', {}),
             'angle_base_deg': o.get('angle_base_deg', None),
             'box_height_m': DEFAULT_BOX_HEIGHT_M}
            for o in objects]
    return json.dumps({'objects': slim, 'age_sec': age}, ensure_ascii=False)


@mcp.tool()
def get_joint_positions() -> str:
    """현재 로봇 팔의 각 관절(joint) 각도와 그리퍼 위치를 조회한다.

    move_joints로 상대적인 움직임을 만들려면, 먼저 이 도구로
    현재 각도를 확인한 뒤 원하는 만큼 더하거나 뺀 값을 절대각도로 넘겨라.

    Returns:
        {"joints": {"joint1": 0.0, ..., "joint7": 0.0, "gripper": 0.08},
         "age_sec": 0.05}
        joints가 null이면 아직 피드백을 받지 못한 상태다.
    """
    _ensure_ros()
    joints, stamp = _ros_node.get_joint_state()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    return json.dumps({'joints': joints, 'age_sec': age}, ensure_ascii=False)


@mcp.tool()
def save_pose(name: str) -> str:
    """현재 로봇 팔의 관절 자세를 이름을 붙여 저장한다.

    티칭 모드(웹 UI)에서 손으로 자세를 잡은 뒤 호출하면,
    그 순간의 모든 관절 각도와 그리퍼 위치를 기억해둔다.

    Args:
        name: 저장할 자세 이름 (예: "grasp_cup_1")

    Returns:
        성공: {"status": "success", "name": "...", "joints": {...}}
        실패: {"status": "failed", "reason": "..."}
    """
    _ensure_ros()
    joints, stamp = _ros_node.get_joint_state()
    if not joints:
        return json.dumps({'status': 'failed',
                           'reason': '아직 관절 피드백을 받지 못했습니다.'}, ensure_ascii=False)
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    if age > 2.0:
        return json.dumps({
            'status': 'failed',
            'reason': f'관절 피드백이 {age}초 전 값이라 오래됐습니다. 로봇 연결을 확인하세요.',
        }, ensure_ascii=False)

    poses = _load_poses()
    poses[name] = {'joints': joints, 'saved_at': time.time()}
    _save_poses(poses)
    return json.dumps({'status': 'success', 'name': name, 'joints': joints},
                      ensure_ascii=False)


@mcp.tool()
def list_saved_poses() -> str:
    """저장된 모든 자세 이름과 관절 값을 조회한다.

    Returns:
        {"poses": {"grasp_cup_1": {"joint1": 0.1, ...}, ...}}
    """
    poses = _load_poses()
    slim = {name: data.get('joints', {}) for name, data in poses.items()}
    return json.dumps({'poses': slim}, ensure_ascii=False)


@mcp.tool()
def move_to_saved_pose(name: str) -> str:
    """저장된 자세 이름으로 로봇 팔(관절 1~7)을 이동시킨다. 완료까지 블로킹.

    Args:
        name: save_pose 로 저장했던 자세 이름

    Returns:
        성공: {"status": "success", "reason": "joint_move_complete", "joints": {...}}
        실패: {"status": "failed"|"rejected"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    poses = _load_poses()
    if name not in poses:
        return json.dumps({
            'status': 'rejected',
            'reason': f"'{name}' 이라는 자세가 없습니다. 저장된 이름: {sorted(poses.keys())}",
        }, ensure_ascii=False)

    joints = poses[name].get('joints', {})
    key_map = {'joint1': 'j1', 'joint2': 'j2', 'joint3': 'j3', 'joint4': 'j4',
               'joint5': 'j5', 'joint6': 'j6', 'joint7': 'j7'}
    move_joints = {key_map[k]: v for k, v in joints.items() if k in key_map}

    if not move_joints:
        return json.dumps({'status': 'rejected',
                           'reason': '저장된 자세에 팔 관절 값이 없습니다.'}, ensure_ascii=False)

    _ros_node.publish_command({'action': 'move_joints', 'joints': move_joints})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    result['joints'] = move_joints
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def scan_for_boxes(target_label: str = 'box') -> str:
    """로봇 팔을 좌우로 훑으며(joint1 스윕) 현재 시야 밖에 있는
    물체까지 찾아서 좌표를 기억해둔다. list_detected_objects로 물체를
    못 찾았을 때 이 도구를 호출한 뒤, pick_object를 from_scan=True로
    호출하면 스캔 중 발견한 위치로 집으러 갈 수 있다. 완료까지 블로킹
    (수십 초 소요).

    ⚠ 중요: 결과의 boxes 배열에 각 물체의 정확한 좌표(x,y,z)가 이미
    포함되어 있다. 이 물체들은 로봇이 스스로 움직이지 않는 한 위치가
    바뀌지 않으므로, 이후 place_object 등에 좌표가 필요할 때
    list_detected_objects로 다시 확인하지 말고 여기서 받은 boxes 값을
    그대로 사용하라. 특히 pick_object 실행 중에는 로봇 팔(카메라)이
    계속 움직이므로, pick 전후로 list_detected_objects를 다시 부르면
    카메라 각도가 달라져 다른(부정확한) 좌표를 받게 될 수 있다.

    Args:
        target_label: 찾을 물체 라벨 (기본값 'box')

    Returns:
        성공: {"status": "success", "reason": "scan_complete:N",
               "boxes": [{"x":0.22,"y":0.19,"z":0.022,
                          "confidence":0.78,"angle_base_deg":45.0,
                          "label":"box"}, ...]}
               (N=발견한 개수, boxes 각 항목의 x/y/z는 물체 중심좌표)
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    global _last_scanned_boxes, _last_scan_stamp
    _ensure_ros()
    payload = {'action': 'scan_box', 'target_label': target_label.strip().lower()}
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_SCAN)
    if result.get('status') == 'success' and 'boxes' in result:
        _last_scanned_boxes = result['boxes']
        _last_scan_stamp = time.time()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_scanned_boxes() -> str:
    """가장 최근 scan_for_boxes 실행 결과(발견된 물체 좌표 목록)를
    ROS 재조회 없이 즉시 반환한다. scan_for_boxes를 이미 호출해서 물체
    위치를 확인해둔 뒤, place_object 등에 그 좌표가 다시 필요할 때
    list_detected_objects 대신 이 도구를 사용하라 (로봇이 스스로 움직이지
    않는 한 물체 위치는 바뀌지 않으므로 재스캔/재조회가 불필요하다).

    scan_for_boxes를 아직 한 번도 호출하지 않았다면 빈 목록을 반환한다.

    Returns:
        {"boxes": [{"x":0.22,"y":0.19,"z":0.022,"confidence":0.78,
                    "angle_base_deg":45.0,"label":"box"}, ...],
         "age_sec": 12.3}
        age_sec는 마지막 scan_for_boxes 호출 이후 경과 시간(초).
        boxes가 빈 배열이면 아직 스캔한 적이 없다.
    """
    age = round(time.time() - _last_scan_stamp, 2) if _last_scan_stamp > 0 else -1.0
    return json.dumps({'boxes': _last_scanned_boxes, 'age_sec': age}, ensure_ascii=False)


@mcp.tool()
def pick_object(target_label: str, grasp_dir: str = 'auto',
                 from_scan: bool = False, box_index: int = None,
                 x: float = None, y: float = None, z: float = None,
                 angle_deg: float = None) -> str:
    """지정한 물체를 로봇 팔로 집어 올린다. pick 완료까지 블로킹.

    Args:
        target_label: 집을 물체 라벨 (예: "cup", "bottle"). 영어 소문자.
        grasp_dir: 파지 방향.
            "auto"        : 물체 위치·라벨 기반 자동 선택 (기본값)
            "top"         : 위에서 아래로
            "side"        : 앞에서 수평으로
            "side_left"   : 왼쪽에서 수평으로
            "side_right"  : 오른쪽에서 수평으로
        from_scan: True면 scan_for_boxes로 미리 찾아둔 좌표를 사용한다
            (현재 카메라 시야 밖에 있는 물체도 집을 수 있음). scan_for_boxes를
            먼저 호출해서 물체를 찾아둔 경우에만 True로 설정하라.
        box_index: [신규] from_scan=True일 때, get_scanned_boxes()/이전
            pick_object 응답의 remaining_scanned_boxes 배열에서 몇 번째
            항목(0부터 시작)을 집을지 명시적으로 지정한다. 생략하면
            큐 맨 앞 항목을 집는다(하위호환, 아래 경고 참고).

    ⚠ 여러 박스를 다루는 작업(특히 쌓기)에서는 box_index를 반드시
    명시하라. box_index를 생략하면 "큐 맨 앞 항목"을 집는데, 이미 다른
    스캔된 박스의 (x,y) 위에 무언가를 place한 뒤 그 자리를 다시
    "다음 큐 항목"으로 착각해서 엉뚱한(방금 쌓은) 박스를 다시 집어버리는
    사고가 실측 확인됐다(2026-07-22). box_index로 명시하면 이 문제가
    구조적으로 발생하지 않는다.

    ⚠ from_scan=True로 pick하면, 결과의 remaining_scanned_boxes에
    "이번에 집은 것을 제외한 나머지 스캔된 물체들"의 좌표가 이미 담겨
    돌아온다. 이어서 다른 스캔된 물체를 다루려면(예: 방금 집은 물체를
    남은 물체 위에 쌓기), list_detected_objects나 scan_for_boxes를
    다시 호출하지 말고 이 필드를 그대로 사용하라(인덱스는 이 배열
    기준으로 다시 매겨짐 -- 원래 스캔 인덱스가 아님). pick 실행 중에는
    로봇 팔(카메라)이 계속 움직이므로, 재조회하면 다른(부정확한)
    좌표를 받게 될 위험이 있다. 특히 물체를 쥔 채로 scan_for_boxes를
    다시 호출하는 것은 팔이 크게 움직이며 충돌 위험이 있으니 절대
    하지 말 것.

    Returns:
        성공: {"status": "success", "reason": "pick_complete",
               "target_label": "...",
               "remaining_scanned_boxes": [{"x":..,"y":..,"z":..,
                   "confidence":..,"angle_base_deg":..,"label":".."}, ...],
               "align_angle_deg": 37.2}
               (remaining_scanned_boxes는 from_scan=True였을 때만 채워짐.
               align_angle_deg는 top 그립일 때 approach 후 align 단계에서
               재측정한 각도[도, 0~90]. side 그립이면 null.)
        실패: {"status": "failed"|"rejected"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()

    if not from_scan:
        objects, _ = _ros_node.get_objects()
        available = {o.get('label', '').lower() for o in objects}
        if available and target_label not in available:
            return json.dumps({
                'status': 'rejected',
                'reason': f"'{target_label}' 은(는) 현재 장면에 없습니다. "
                          f"인식된 물체: {sorted(available)}. "
                          f"카메라 시야 밖에 있을 수 있으니 scan_for_boxes를 먼저 시도해보라.",
            }, ensure_ascii=False)

    payload = {'action': 'pick', 'target_label': target_label}
    if grasp_dir and grasp_dir != 'auto':
        payload['grasp_dir'] = grasp_dir
    if x is not None and y is not None and z is not None:
        payload['override_pos'] = {'x': x, 'y': y, 'z': z}
        if angle_deg is not None:
            payload['override_angle_deg'] = angle_deg
    elif from_scan:
        payload['from_scan'] = True
        if box_index is not None:
            payload['box_index'] = box_index
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PICK)
    result['target_label'] = target_label
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def place_object(x: float, y: float, z: float, grasp_dir: str = 'auto') -> str:
    """집은 물체를 지정한 좌표(base_link 기준, 미터)에 내려놓는다. place 완료까지 블로킹.

    Args:
        x, y, z: 내려놓을 위치 (base_link 기준, 미터)
        grasp_dir: 내려놓을 때 자세. pick_object 와 동일한 값 사용 권장.

    ⚠ 요청한 좌표가 로봇의 국소 도달불가 지점(singularity)이거나 비정상적
    으로 높은 z일 경우, 서버가 자동으로 근처 좌표(±0.05m y시프트 또는
    z 하향)로 재시도한다. 성공 시 결과의 place_pos가 실제로 놓인 좌표로
    바뀌어 있을 수 있다 — 이 경우 requested_place_pos 필드에 원래 요청
    좌표가 별도로 담긴다. 다음 작업(예: 이 박스 위에 쌓기)의 기준 좌표는
    반드시 place_pos(실제 좌표)를 써야 한다.

    ⚠ [폐루프 검증] status="success"라고 해서 물체가 실제로 목표 위치에
    안착했다는 보장은 아니다. 그리퍼가 열리는 순간 물체가 미끄러지거나
    다른 물체에 부딪혀 튕겨나갈 수 있다. 서버가 lift 직후 카메라로 재확인한
    결과가 placement_verified 필드에 담긴다:
      - true  : 목표 위치 근처에서 물체가 실제로 확인됨. 안심하고 다음
                작업(그 위에 쌓기 등) 진행 가능.
      - false : 목표 위치에서 물체를 못 찾았거나 크게 벗어남
                (verification_reason에 구체적 사유). 이 물체를 기준으로
                후속 작업(쌓기 등)을 계속하지 말고, list_detected_objects로
                실제 위치를 재확인하거나 사용자에게 알려라.
      - null(필드 자체가 없거나 값이 None) : perception 데이터가 오래돼
                판정 불가. 확실하지 않으니 필요하면 list_detected_objects로
                직접 재확인하라.

    Returns:
        성공: {"status": "success", "reason": "place_complete",
               "place_pos": {...},
               "requested_place_pos": {...}  (좌표가 자동 보정된 경우만 포함),
               "placement_verified": true|false|null,
               "verification_reason": "..."}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    place_pos = {'x': x, 'y': y, 'z': z}
    payload = {'action': 'place', 'place_pos': place_pos}
    if grasp_dir and grasp_dir != 'auto':
        payload['grasp_dir'] = grasp_dir
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PLACE)
    # [신규] planning_node가 도달 불가 지점을 감지해 좌표를 자동으로
    # 시프트했을 수 있다 (approach/descend 실패 시 ±0.05m y시프트 또는
    # z 하향 재시도, nero_robot_place_reachability memory 기반). 실제로
    # 어디 놓였는지를 place_pos에 반영해서, Claude가 요청 좌표와 실제
    # 좌표가 다르다는 걸 놓치지 않게 한다.
    if 'actual_place_pos' in result and result['actual_place_pos'] != place_pos:
        result['requested_place_pos'] = place_pos
        result['place_pos'] = result['actual_place_pos']
    else:
        result['place_pos'] = place_pos
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def stack_boxes(box_indices: list, base_x: float, base_y: float, base_z: float,
                 target_label: str = 'box', box_height_m: float = DEFAULT_BOX_HEIGHT_M,
                 grasp_dir: str = 'auto', allow_reorder: bool = False) -> str:
    """스캔해둔 여러 박스를 지정한 좌표 위에 순서대로(맨 아래→맨 위) 쌓는다.
    pick_object와 place_object를 박스 개수만큼 번갈아 호출하는 대신, 서버가
    내부에서 pick→place를 tier마다 반복 처리한다 (MCP round-trip 1회로
    N개 박스를 다 쌓음 -- 기존 방식은 2N회 필요했음). 완료까지 블로킹
    (박스 개수 * 1~2분 소요될 수 있음, allow_reorder=True면 대체 시도
    만큼 더 오래 걸릴 수 있음).

    반드시 scan_for_boxes를 먼저 호출해서 쌓을 박스들의 위치를 찾아둔 뒤
    사용하라.

    Args:
        box_indices: get_scanned_boxes()/이전 pick_object 응답의
            remaining_scanned_boxes에서 몇 번째 항목(0부터, target_label
            매칭 기준으로 다시 매겨진 인덱스)을 쌓을지 순서대로 나열한다.
            예: [2, 0, 1]이면 2번 박스를 base_z에(맨 아래), 0번 박스를
            base_z+box_height_m에, 1번 박스를 base_z+2*box_height_m에
            (맨 위) 쌓는다. base_x/y/z 위치에 이미 놓여있는 박스(바닥
            박스, 새로 집지 않을 것)는 여기 포함하지 마라.
        base_x, base_y, base_z: 맨 아래 박스가 놓일 좌표 (base_link 기준,
            미터). place_object와 동일한 좌표 규칙 (박스 중심 z 기준).
        target_label: 쌓을 물체 라벨 (기본값 'box')
        box_height_m: 층당 높이 증분. 기본값은 고정 박스높이 0.05m
            (CLAUDE.md 확정값, 재스캔으로 재확인하지 말 것).
        grasp_dir: 각 박스를 집고 놓을 자세. 기본 'auto'.
        allow_reorder: True면 어떤 tier에서 지정한 박스의 pick이
            실패/거부됐을 때, 같은 target_label의 스캔된 다른 박스(여기
            box_indices에 없는 것들)로 자동 대체를 시도한다 (place
            단계는 대체하지 않음 -- place 자체가 물리적으로 실패/거부되면
            allow_reorder 값과 무관하게 즉시 중단한다. placement_verified
            =False는 [2026-08-17 변경] 더 이상 중단 사유가 아니다 --
            아래 ⚠ 참고). 기본값 False -- box_indices로 순서를
            명시적으로 지정했을 수 있으므로, 조용히 다른 박스로 바뀌는
            동작은 opt-in이다.

    ⚠ 중간 tier에서 pick/place 자체가 실패하면(물리적으로 이동/파지
    불가) 그 시점에서 멈추고 status="partial"로 그때까지의 결과만
    반환한다. tiers 배열의 마지막 항목을 보고 어디서 멈췄는지 판단하라
    -- 실패한 tier의 박스는 이미 집었을 수도(place 단계 실패) 아예 못
    집었을 수도(pick 단계 실패) 있으니, list_detected_objects로 실제
    상태를 확인 후 필요하면 개별 pick_object/place_object로 이어서
    처리하라.
    allow_reorder=True일 때는 status="partial"/pick 실패가 곧바로 전체
    중단을 의미하지 않을 수 있으니, 먼저 tiers[].substituted(대체된
    박스로 성공했는지)와 skipped_candidates(시도했다가 제외된 후보들)를
    확인하라 -- 실제로는 다른 박스로 대체돼 그 tier가 성공했을 수 있다.

    ⚠ [2026-08-17 변경] placement_verified=False는 더 이상 중단시키지
    않는다 -- 물리적으로 place가 끝났으면(카메라 재확인 결과와 무관하게)
    다음 tier로 계속 진행한다. 즉 status="success"로 끝나도 중간 tier가
    불안정하게 놓였을 수 있다 -- **반드시 tiers[].placement_verified를
    각 tier마다 확인하라.** false가 하나라도 있으면 그 위에 쌓인 박스들은
    불안정한 기반 위에 있을 수 있으니 list_detected_objects로 실제
    상태를 재확인하는 것을 권장한다.

    Returns:
        {"status": "success"|"partial"|"rejected"|"timeout",
         "tiers": [{"tier":0,"stage":"place","status":"success",
                     "box_used":{"x":..,"y":..,"z":..,...},
                     "substituted":false,
                     "place_pos":{"x":..,"y":..,"z":..},
                     "placement_verified":true|false|null,
                     "verification_reason":"..."}, ...],
         "skipped_candidates": [{"tier":0,"box":{...},"reason":"...",
                                   "prefiltered":false}, ...],
         "backtrack_attempts_used": 0,
         "remaining_scanned_boxes": [...]}
        allow_reorder=False(기본값)면 skipped_candidates는 항상 존재하고
        (실패한 tier가 있으면 그 항목만 담김), box_used/substituted 필드가
        추가되는 것 외엔 기존 응답과 동일하다.
    """
    _ensure_ros()
    target_label = target_label.strip().lower()
    if not box_indices:
        return json.dumps({'status': 'rejected', 'reason': 'box_indices가 비어있습니다.'},
                          ensure_ascii=False)
    payload = {
        'action': 'stack',
        'target_label': target_label,
        'box_indices': list(box_indices),
        'base_pos': {'x': base_x, 'y': base_y, 'z': base_z},
        'box_height_m': box_height_m,
        'allow_reorder': allow_reorder,
    }
    if grasp_dir and grasp_dir != 'auto':
        payload['grasp_dir'] = grasp_dir
    _ros_node.publish_command(payload)
    # [2026-08 수정] allow_reorder=True면 서버가 pick 실패 시 fallback
    # 후보로 최대 DEFAULT_MAX_BACKTRACK_ATTEMPTS번 더 시도할 수 있어
    # 그만큼 시간이 더 걸린다. 이걸 타임아웃 계산에 반영하지 않으면,
    # 서버는 백트래킹으로 실제 완료했는데 MCP 클라이언트 쪽이 먼저
    # 타임아웃나서 거짓 "timeout"을 반환하는 문제가 생긴다.
    timeout = TIMEOUT_STACK_PER_BOX * (
        max(1, len(box_indices))
        + (DEFAULT_MAX_BACKTRACK_ATTEMPTS if allow_reorder else 0)
    )
    result = _ros_node.wait_for_result(timeout=timeout)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def move_to_position(x: float, y: float, z: float) -> str:
    """로봇 팔 끝(end-effector)을 지정 좌표로 이동한다. 완료까지 블로킹.

    Args:
        x, y, z: 목표 좌표 (base_link 기준, 미터)

    Returns:
        성공: {"status": "success", "reason": "move_complete", "target_pos": {...}}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    target_pos = {'x': x, 'y': y, 'z': z}
    _ros_node.publish_command({'action': 'move', 'target_pos': target_pos})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    result['target_pos'] = target_pos
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def move_joints(
    j1: float = None, j2: float = None, j3: float = None,
    j4: float = None, j5: float = None, j6: float = None, j7: float = None
) -> str:
    """로봇 팔의 각 관절(joint)을 직접 제어한다. 완료까지 블로킹.

    지정하지 않은 joint는 현재 각도 유지.

    Args:
        j1~j7: 각 관절 절대 각도 (라디안)

    Returns:
        성공: {"status": "success", "reason": "joint_move_complete", "joints": {...}}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    joints = {}
    if j1 is not None: joints['j1'] = j1
    if j2 is not None: joints['j2'] = j2
    if j3 is not None: joints['j3'] = j3
    if j4 is not None: joints['j4'] = j4
    if j5 is not None: joints['j5'] = j5
    if j6 is not None: joints['j6'] = j6
    if j7 is not None: joints['j7'] = j7

    if not joints:
        return json.dumps({'status': 'rejected',
                           'reason': 'joint 값을 하나 이상 지정해야 합니다.'})

    _ros_node.publish_command({'action': 'move_joints', 'joints': joints})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    result['joints'] = joints
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def move_joints_relative(j1: float = 0.0, j2: float = 0.0, j3: float = 0.0,
                          j4: float = 0.0, j5: float = 0.0, j6: float = 0.0,
                          j7: float = 0.0) -> str:
    """현재 관절 위치에서 상대적으로 이동한다 (라디안 단위).

    Args:
        j1~j7: 각 관절의 상대 이동량 (라디안, 기본값 0.0)
    """
    _ensure_ros()
    joints, stamp = _ros_node.get_joint_state()
    if not joints:
        return json.dumps({'status': 'failed', 'reason': '관절 피드백 없음'}, ensure_ascii=False)

    key_map = {'j1': 'joint1', 'j2': 'joint2', 'j3': 'joint3', 'j4': 'joint4',
               'j5': 'joint5', 'j6': 'joint6', 'j7': 'joint7'}
    deltas = {'j1': j1, 'j2': j2, 'j3': j3, 'j4': j4, 'j5': j5, 'j6': j6, 'j7': j7}

    move_joints_cmd = {}
    for k, delta in deltas.items():
        if delta != 0.0:
            jname = key_map[k]
            current = joints.get(jname, 0.0)
            move_joints_cmd[k] = round(current + delta, 6)

    if not move_joints_cmd:
        return json.dumps({'status': 'rejected', 'reason': '모든 delta가 0'}, ensure_ascii=False)

    _ros_node.publish_command({'action': 'move_joints', 'joints': move_joints_cmd})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def go_home() -> str:
    """로봇 팔을 홈 자세로 복귀시키고 그리퍼를 연다. 완료까지 블로킹.

    Returns:
        성공: {"status": "success", "reason": "home_complete"}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    _ros_node.publish_command({'action': 'home'})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_HOME)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_system_status() -> str:
    """로봇/비전 브리지의 현재 연결 상태를 점검한다 (헬스체크용).

    Returns:
        {"ros_bridge": "up", "vision_objects": 3, "vision_age_sec": 0.2,
         "poses_file": "~/.local/share/nero_robot/saved_poses.json"}
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    return json.dumps({
        'ros_bridge': 'up',
        'vision_objects': len(objects),
        'vision_age_sec': age,
        'poses_file': POSES_FILE,
    }, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 엔트리포인트
# ──────────────────────────────────────────────────────────────────────────────
def main():
    t = threading.Thread(target=_ros_spin_thread, daemon=True)
    t.start()

    transport = os.environ.get('MCP_TRANSPORT', 'stdio')
    if transport == 'stdio':
        mcp.run(transport='stdio')
    else:
        mcp.settings.host = os.environ.get('MCP_HOST', '0.0.0.0')
        mcp.settings.port = int(os.environ.get('MCP_PORT', '8000'))
        mcp.run(transport=transport)


if __name__ == '__main__':
    main()
