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
import math
import time
import threading
import logging
from typing import Optional

_logger = logging.getLogger(__name__)

import base64
import requests as _requests

import cv2
import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — PointStamped/Vector3Stamped 변환 등록용

from mcp.server.fastmcp import FastMCP

# [2026-08 추가, 프로토타입] YOLO가 못 잡은 물체(ground_object/infer_grasp의
# VLM 폴백 경로)에도 face normal yaw를 적용하기 위해 순수 함수만 재사용.
# perception_node.py는 별도 프로세스(ROS 노드)지만 같은 sj_pickplace 패키지
# 안의 순수 함수 import라 문제 없음 (perception_node_sim.py가 이미
# _compute_box_angle_base/_dedup_3d를 같은 방식으로 재사용 중).
from .perception_node import _fit_plane_normal
from .segmentation_backend import NoOpSegmentationBackend
from .point_cloud import Intrinsics as _PCIntrinsics, mask_depth_to_pointcloud
from . import geometry_3d as _geometry_3d
from .grasp_types import GeometryResult as _GeometryResult

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

# [2026-07 추가] 박스 실측 치수 지원 전까지 임시 고정값 (추후 A안:
# depth 기반 실측으로 교체 예정). 현재 테스트 환경 박스 높이 기준.
DEFAULT_BOX_HEIGHT_M = 0.05
TIMEOUT_SCAN  = 60.0  # joint1 스윕 시간 고려 (6스텝 * 약 8초)

# ── VLM 추론 서버 주소 (vlm_grasp_server.py) ──────────────────────────────────
# 오버라이드: export VLM_SERVER_URL=http://<host>:8003  (예: YOLO/VLM을 별도
# 머신에서 돌리는 분리 배포 시 — perception_node.py의 BOX_SERVER_URL과 동일 패턴)
VLM_SERVER_URL = os.environ.get('VLM_SERVER_URL', 'http://127.0.0.1:8003').rstrip('/')

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
        # infer_grasp용 카메라 이미지 버퍼 — camera device 직접 접근 없음,
        # perception_node가 발행하는 ROS2 topic만 구독한다.
        self.sub_image = self.create_subscription(
            Image, '/camera/color/image_raw', self._on_image, qos)
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_raw', self._on_depth, qos)
        self.sub_caminfo = self.create_subscription(
            CameraInfo, '/camera/camera_info', self._on_camera_info, qos)

        # TF 버퍼 — ground_object에서 camera_color_optical_frame → base_link 변환
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._objects_lock = threading.Lock()
        self._latest_objects: list = []
        self._last_obj_stamp: float = 0.0

        self._image_lock = threading.Lock()
        self._latest_image: Optional[np.ndarray] = None  # BGR uint8
        self._last_image_stamp: float = 0.0

        self._depth_lock = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None  # float32 meters
        self._last_depth_stamp: float = 0.0

        self._caminfo_lock = threading.Lock()
        self._cam_fx: Optional[float] = None
        self._cam_fy: Optional[float] = None
        self._cam_cx: Optional[float] = None
        self._cam_cy: Optional[float] = None
        self._cam_width: int = 640
        self._cam_height: int = 480

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

    def _on_image(self, msg: Image):
        try:
            channels = len(msg.data) // (msg.height * msg.width)
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            if msg.encoding in ('rgb8', 'RGB8'):
                arr = arr[:, :, ::-1]  # RGB→BGR
            with self._image_lock:
                self._latest_image = arr.copy()
                self._last_image_stamp = time.time()
        except Exception as e:
            self.get_logger().warn(f'_on_image 처리 실패: {e}')

    def get_image(self) -> tuple:
        with self._image_lock:
            img = self._latest_image.copy() if self._latest_image is not None else None
            return img, self._last_image_stamp

    def _on_depth(self, msg: Image):
        try:
            if msg.encoding != "32FC1":
                return
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            with self._depth_lock:
                self._latest_depth = arr.copy()
                self._last_depth_stamp = time.time()
        except Exception as e:
            self.get_logger().warn(f'_on_depth 처리 실패: {e}')

    def get_depth(self) -> tuple:
        with self._depth_lock:
            d = self._latest_depth.copy() if self._latest_depth is not None else None
            return d, self._last_depth_stamp

    def _on_camera_info(self, msg: CameraInfo):
        with self._caminfo_lock:
            k = msg.k
            self._cam_fx = k[0]
            self._cam_fy = k[4]
            self._cam_cx = k[2]
            self._cam_cy = k[5]
            self._cam_width  = msg.width
            self._cam_height = msg.height

    def get_cam_intrinsics(self) -> tuple:
        with self._caminfo_lock:
            return (self._cam_fx, self._cam_fy,
                    self._cam_cx, self._cam_cy,
                    self._cam_width, self._cam_height)

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
_ros_lock = threading.Lock()

# [2026-07 추가] 마지막 scan_for_boxes 결과 캐시. get_scanned_boxes가
# ROS 왕복 없이 즉시 응답할 수 있게 하기 위함 (로봇이 움직이지 않는
# 물체는 재스캔 없이도 이 캐시로 충분히 정확함).
_last_scanned_boxes: list = []
_last_scan_stamp: float = 0.0

# analyze_scene 결과 캐시 — hierarchical grounding에서 parent bbox 재사용
_last_scene_objects: list = []   # [{"label":..., "bbox":[...], "source":...}, ...]
_last_scene_stamp: float = 0.0
_scene_cache_lock = threading.Lock()


# ── ground_object 헬퍼 ────────────────────────────────────────────────────────

def _px2cam(u: float, v: float, depth_m: float,
            fx: float, fy: float, cx: float, cy: float) -> Optional[dict]:
    """픽셀 좌표 + depth → camera_color_optical_frame 3D 점."""
    if not (0.05 < depth_m < 3.0):
        return None
    return {
        "x": round((u - cx) * depth_m / fx, 4),
        "y": round((v - cy) * depth_m / fy, 4),
        "z": round(float(depth_m), 4),
    }


def _sample_depth_robust(depth: np.ndarray, x1: int, y1: int,
                          x2: int, y2: int) -> Optional[float]:
    """bbox 영역에서 valid depth (0.05–3.0m)의 median. 실패 시 None."""
    h, w = depth.shape
    x1c, x2c = max(0, x1), min(w, x2)
    y1c, y2c = max(0, y1), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return None
    valid = depth[y1c:y2c, x1c:x2c]
    valid = valid[(valid > 0.05) & (valid < 3.0)]
    if len(valid) < 3:
        # center window 폴백
        cy_px = (y1c + y2c) // 2
        cx_px = (x1c + x2c) // 2
        win = 15
        patch = depth[max(0, cy_px - win):min(h, cy_px + win),
                      max(0, cx_px - win):min(w, cx_px + win)]
        valid = patch[(patch > 0.05) & (patch < 3.0)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def _cam_to_base(cam_xyz: dict) -> Optional[dict]:
    """camera_color_optical_frame 점 → base_link 좌표. 실패 시 None."""
    try:
        pt = PointStamped()
        pt.header.frame_id = "camera_color_optical_frame"
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x = cam_xyz["x"]
        pt.point.y = cam_xyz["y"]
        pt.point.z = cam_xyz["z"]
        result = _ros_node.tf_buffer.transform(
            pt, "base_link", timeout=Duration(seconds=0.3))
        return {
            "x": round(result.point.x, 3),
            "y": round(result.point.y, 3),
            "z": round(result.point.z, 3),
        }
    except Exception:
        return None


# [2026-08 추가, 프로토타입 -- 실기 미검증] YOLO가 못 잡은 물체(VLM
# grounding으로만 bbox를 얻은 경우)용 평면적합 face normal yaw.
# perception_node._compute_face_normal_yaw와 같은 수학(_fit_plane_normal
# 재사용)이지만, 이 파일 자체의 depth/intrinsics/tf 접근자(_px2cam,
# get_depth, get_cam_intrinsics, tf_buffer)로 다시 감싼 것 -- 프로세스가
# 달라서(mcp_robot_server는 perception_node의 ROS 노드 인스턴스에 접근
# 불가) 로직만 재사용하고 배선은 이 파일 자체 것을 쓴다.
FACE_NORMAL_GRID_STEP     = int(os.environ.get("FACE_NORMAL_GRID_STEP", "8"))
FACE_NORMAL_MIN_POINTS    = int(os.environ.get("FACE_NORMAL_MIN_POINTS", "25"))
FACE_NORMAL_MAX_POINTS    = int(os.environ.get("FACE_NORMAL_MAX_POINTS", "400"))
FACE_NORMAL_PLANARITY_MIN   = float(os.environ.get("FACE_NORMAL_PLANARITY_MIN", "0.7"))
FACE_NORMAL_VERTICALITY_MIN = float(os.environ.get("FACE_NORMAL_VERTICALITY_MIN", "0.5"))


def _compute_face_normal_yaw_from_bbox(bbox_px: list, depth: np.ndarray,
                                        fx: float, fy: float, cx: float, cy: float) -> tuple:
    """bbox_px([x1,y1,x2,y2]) 영역의 depth를 점군으로 역투영해 평면을 맞추고,
    그 평면 normal의 mod-180 방위각을 base_link 기준으로 반환한다.

    perception_node._compute_face_normal_yaw와 동일한 설계(평면성/수직성
    2단 게이트, verticality 낮으면 -- 즉 카메라가 물체 윗면처럼 수평에
    가까운 면을 보고 있으면 -- None 반환)이지만, 이 노드가 이미 들고 있는
    depth/intrinsics(_ros_node.get_depth/get_cam_intrinsics)와 _px2cam으로
    다시 구현했다. 값이 없으면(신뢰 불가) 무조건 None -- angle_base_deg 같은
    필드가 YOLO 못 잡은 물체엔 원래 없었으니, 실패 시에도 기존 응답 스키마를
    깨지 않는다(단순히 필드가 null로 채워짐).

    Returns:
        (yaw_deg, confidence, n_points) -- perception_node 버전과 동일 계약.
    """
    if depth is None or fx is None:
        return None, 0.0, 0

    h, w = depth.shape[:2]
    x1 = max(0, int(bbox_px[0])); y1 = max(0, int(bbox_px[1]))
    x2 = min(w, int(bbox_px[2])); y2 = min(h, int(bbox_px[3]))
    if x2 - x1 < FACE_NORMAL_GRID_STEP or y2 - y1 < FACE_NORMAL_GRID_STEP:
        return None, 0.0, 0

    points_cam = []
    for py in range(y1, y2, FACE_NORMAL_GRID_STEP):
        for px in range(x1, x2, FACE_NORMAL_GRID_STEP):
            depth_m = float(depth[py, px])
            pt = _px2cam(float(px), float(py), depth_m, fx, fy, cx, cy)
            if pt is not None:
                points_cam.append((pt['x'], pt['y'], pt['z']))
            if len(points_cam) >= FACE_NORMAL_MAX_POINTS:
                break
        if len(points_cam) >= FACE_NORMAL_MAX_POINTS:
            break

    if len(points_cam) < FACE_NORMAL_MIN_POINTS:
        return None, 0.0, len(points_cam)

    normal_cam, planarity = _fit_plane_normal(points_cam)
    if normal_cam is None or planarity < FACE_NORMAL_PLANARITY_MIN:
        return None, round(planarity, 3), len(points_cam)

    try:
        v = Vector3Stamped()
        v.header.frame_id = "camera_color_optical_frame"
        v.header.stamp = rclpy.time.Time().to_msg()
        v.vector.x, v.vector.y, v.vector.z = [float(c) for c in normal_cam]
        v_base = _ros_node.tf_buffer.transform(v, "base_link", timeout=Duration(seconds=0.3))
        nx, ny = v_base.vector.x, v_base.vector.y
    except Exception:
        return None, round(planarity, 3), len(points_cam)

    verticality = math.hypot(nx, ny)
    confidence = round(planarity * verticality, 3)
    if verticality < FACE_NORMAL_VERTICALITY_MIN:
        return None, confidence, len(points_cam)

    yaw_deg = math.degrees(math.atan2(ny, nx)) % 180.0
    return round(yaw_deg, 1), confidence, len(points_cam)


def _rotate_vec_to_base(vec) -> Optional[tuple]:
    """카메라 좌표계 방향벡터(위치 아님) 하나를 base_link로 "회전만" 변환.
    _compute_face_normal_yaw_from_bbox와 동일 Vector3Stamped 패턴 -- 여러
    지점에서 반복되던 걸 함수로 뽑음."""
    if vec is None:
        return None
    try:
        v = Vector3Stamped()
        v.header.frame_id = "camera_color_optical_frame"
        v.header.stamp = rclpy.time.Time().to_msg()
        v.vector.x, v.vector.y, v.vector.z = [float(c) for c in vec]
        v_base = _ros_node.tf_buffer.transform(v, "base_link", timeout=Duration(seconds=0.3))
        return (v_base.vector.x, v_base.vector.y, v_base.vector.z)
    except Exception:
        return None


def _geometry_to_base_link(geometry, depth_quality: float = 1.0):
    """geometry_3d.compute_geometry()가 camera_color_optical_frame 점군에서
    낸 GeometryResult(축/normal이 전부 카메라 좌표계)를 base_link 기준으로
    다시 감싼다. 점 수만 개를 전부 재변환하는 대신, 이미 계산된 축/normal
    (벡터 3~4개)과 centroid(점 1개)만 변환한다 -- 점군 자체를 base_link로
    옮긴 뒤 geometry_3d를 다시 돌리는 것과 수학적으로 동일하지만 훨씬 싸다
    (회전 변환은 선형이라 축소환 순서를 바꿔도 결과가 같음).

    실패(TF 불가 등)하면 valid=False인 빈 GeometryResult를 반환한다 --
    호출부가 무조건 뭔가를 받되, 유효성은 반드시 .valid로 확인해야 한다."""
    from dataclasses import replace as _dc_replace

    if geometry is None or not geometry.valid:
        return _GeometryResult(valid=False)

    centroid_base = _cam_to_base({'x': geometry.centroid[0], 'y': geometry.centroid[1],
                                  'z': geometry.centroid[2]})
    major_base = _rotate_vec_to_base(geometry.major_axis)
    minor_base = _rotate_vec_to_base(geometry.minor_axis)
    third_base = _rotate_vec_to_base(geometry.third_axis)
    normal_base = _rotate_vec_to_base(geometry.plane_normal) if geometry.plane_normal else None

    if centroid_base is None or major_base is None:
        return _GeometryResult(valid=False, point_count=geometry.point_count,
                               filtered_point_count=geometry.filtered_point_count)

    def _yaw(vec):
        if vec is None:
            return None
        h = math.hypot(vec[0], vec[1])
        if h < 0.5:
            return None
        return round(math.degrees(math.atan2(vec[1], vec[0])) % 180.0, 1)

    return _dc_replace(
        geometry,
        centroid=(centroid_base['x'], centroid_base['y'], centroid_base['z']),
        major_axis=major_base, minor_axis=minor_base, third_axis=third_base,
        plane_normal=normal_base,
        major_axis_yaw_deg=_yaw(major_base),
        normal_yaw_deg=_yaw(normal_base),
    )


def _crop_b64(img_bgr: np.ndarray, bbox_norm: list, padding: float = 0.07) -> tuple:
    """parent bbox 영역을 crop하고 base64 JPEG + crop pixel bbox 반환.
    padding: bbox 주변에 추가할 여유 비율 (기본 7%)"""
    h, w = img_bgr.shape[:2]
    x1n, y1n, x2n, y2n = bbox_norm
    pw = (x2n - x1n) * padding
    ph = (y2n - y1n) * padding
    x1n = max(0.0, x1n - pw)
    y1n = max(0.0, y1n - ph)
    x2n = min(1.0, x2n + pw)
    y2n = min(1.0, y2n + ph)
    x1, y1 = max(0, int(x1n * w)), max(0, int(y1n * h))
    x2, y2 = min(w, int(x2n * w)), min(h, int(y2n * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError(f'crop 너무 작음: {x2-x1}x{y2-y1}px')
    ok, buf = cv2.imencode('.jpg', img_bgr[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError('crop JPEG 인코딩 실패')
    return base64.b64encode(buf.tobytes()).decode(), [x1, y1, x2, y2]


def _restore_bbox(child_norm: list, crop_norm: list) -> list:
    """crop 기준 child bbox → 원본 이미지 normalized 좌표 복원.
    crop_norm: 원본 이미지에서의 실제 crop 영역 [x1,y1,x2,y2] (padding 포함)"""
    cx1, cy1, cx2, cy2 = child_norm
    px1, py1, px2, py2 = crop_norm
    pw, ph = px2 - px1, py2 - py1
    return [
        round(px1 + cx1 * pw, 4),
        round(py1 + cy1 * ph, 4),
        round(px1 + cx2 * pw, 4),
        round(py1 + cy2 * ph, 4),
    ]


def _label_matches_parent(candidate: str, query: str) -> bool:
    """candidate label이 query parent_label에 의미적으로 매칭되는지 확인.
    'yellow drawer' ↔ 'drawer' 같은 core word overlap 허용.
    과도한 fuzzy matching은 배제."""
    c = candidate.strip().lower()
    q = query.strip().lower()
    if c == q:
        return True
    q_words = set(q.split())
    c_words = set(c.split())
    common = q_words & c_words
    # query 단어의 절반 이상이 candidate에 포함될 때만 허용
    return len(common) >= max(1, len(q_words) // 2)


def _is_bbox_oversized(bbox_norm: list, threshold: float = 0.8) -> bool:
    """bbox 면적이 이미지의 threshold 이상이면 True (too coarse 경고용)."""
    x1, y1, x2, y2 = bbox_norm
    return (x2 - x1) * (y2 - y1) > threshold


def _make_child_target(target_label: str, parent_label: str) -> str:
    """child grounding용 target_label 생성.
    handle/knob/pull/grip 류는 내부 부품과 구분되도록 상세 설명 포함."""
    target_lower = target_label.strip().lower()
    if any(kw in target_lower for kw in ('handle', 'knob', 'pull', 'grip')):
        return (
            f"Front-facing {target_label} of the {parent_label}. "
            f"The physical handle that a person grabs to open/pull — "
            f"a horizontal bar or protruding grip on the EXTERIOR FRONT FACE of the {parent_label}. "
            f"NOT: internal rails, drawer slides, brackets, hinges, screws, "
            f"or any mechanical component inside the drawer cavity."
        )
    return f"{target_label} (part of {parent_label})"


def _ground_hierarchical(target_label: str, parent_label: str,
                          ground_url: str, vlm_timeout: float,
                          min_conf: float) -> str:
    """parent object crop → child part grounding → depth → TF → 3D (2D fallback).

    Parent bbox 우선순위:
      1. analyze_scene 캐시 (가장 최근 scene 결과)
      2. YOLO detection
      3. VLM /ground_object(parent_label) — 최후 수단
    """

    # ── A. 이미지 획득 ─────────────────────────────────────────────────────────
    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'success': False, 'reason': 'image_unavailable'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'success': False, 'reason': f'image_stale — {img_age:.1f}s'})

    h, w = img_bgr.shape[:2]

    def _encode(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    # ── B. Parent bbox 확보 ────────────────────────────────────────────────────
    # Priority 1: analyze_scene 캐시 (가장 최근 perception 결과)
    # Priority 2: YOLO detection
    # Priority 3: VLM /ground_object(parent_label) — 최후 수단

    _logger.info(f'[ground_object] target={target_label!r}  parent={parent_label!r}')

    parent_source = None
    parent_bbox_norm = None

    # Priority 1: analyze_scene 캐시 — VLM parent grounding 호출 없이 재사용
    with _scene_cache_lock:
        scene_objs = list(_last_scene_objects)
        scene_age  = time.time() - _last_scene_stamp

    if scene_objs and scene_age < 60.0:
        scene_matched = [o for o in scene_objs
                         if _label_matches_parent(str(o.get('label', '')), parent_label)]
        if scene_matched:
            # 면적 큰 순 우선 — parent는 child를 포함하는 더 넓은 영역이어야 함
            scene_matched.sort(
                key=lambda o: (
                    (o.get('bbox', [0, 0, 1, 1])[2] - o.get('bbox', [0, 0, 1, 1])[0]) *
                    (o.get('bbox', [0, 0, 1, 1])[3] - o.get('bbox', [0, 0, 1, 1])[1])
                ),
                reverse=True,
            )
            for sc in scene_matched:
                b = sc.get('bbox', [])
                if isinstance(b, list) and len(b) == 4:
                    cand = [float(v) for v in b]
                    area = (cand[2] - cand[0]) * (cand[3] - cand[1])
                    if area < 0.01:
                        _logger.warning(f'[parent] scene cache bbox too small {cand} (area={area:.4f}) — trying next candidate')
                        continue
                    if _is_bbox_oversized(cand):
                        _logger.warning(f'[parent] scene cache bbox oversized {cand} — trying next candidate')
                        continue
                    parent_bbox_norm = cand
                    parent_source = 'scene'
                    _logger.info(
                        f'[parent] reused existing analyze_scene bbox '
                        f'({sc.get("label")}): {cand}'
                    )
                    break
            # 모두 oversized여도 YOLO/VLM보다 scene 결과가 더 신뢰할 수 있으므로 최후 수단으로 사용
            if parent_bbox_norm is None and scene_matched:
                b = scene_matched[0].get('bbox', [])
                if isinstance(b, list) and len(b) == 4:
                    parent_bbox_norm = [float(v) for v in b]
                    parent_source = 'scene'
                    _logger.warning(
                        f'[parent] all scene candidates oversized — '
                        f'using smallest ({scene_matched[0].get("label")}): {parent_bbox_norm}'
                    )

    # Priority 2: YOLO
    if parent_bbox_norm is None:
        objects, _ = _ros_node.get_objects()
        yolo_matched = [o for o in objects
                        if _label_matches_parent(str(o.get('label', '')), parent_label)]
        if yolo_matched:
            b = yolo_matched[0].get('bbox', [])
            if isinstance(b, list) and len(b) == 4:
                cand = [float(v) for v in b]
                if _is_bbox_oversized(cand):
                    _logger.warning(f'[parent] YOLO bbox oversized {cand}')
                parent_bbox_norm = cand
                parent_source = 'yolo'
                _logger.info(f'[parent] using YOLO bbox ({yolo_matched[0].get("label")}): {cand}')
    else:
        # scene 경로에서는 objects 미조회 — VLM fallback용으로 lazy 조회
        objects = None

    # Priority 3: VLM parent grounding — 최후 수단
    if parent_bbox_norm is None:
        _logger.info(f'[parent] no scene/YOLO match — calling VLM ground_object({parent_label!r})')
        if objects is None:
            objects, _ = _ros_node.get_objects()
        try:
            full_b64 = _encode(img_bgr)
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'encode_failed — {e}'})

        yolo_detections = [
            {'label': o.get('label', '?'), 'bbox': o.get('bbox', []),
             'confidence': round(float(o.get('confidence', 0.0)), 3)}
            for o in objects
        ]
        try:
            resp = _requests.post(ground_url,
                                  json={'full_image_b64': full_b64,
                                        'target_label': parent_label,
                                        'detections': yolo_detections,
                                        'timestamp': time.time()},
                                  timeout=vlm_timeout)
        except _requests.exceptions.ConnectionError:
            return json.dumps({'success': False, 'reason': 'vlm_server_unavailable'})
        except _requests.exceptions.Timeout:
            return json.dumps({'success': False, 'reason': f'vlm_timeout — {vlm_timeout}s 초과'})
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'vlm_request_failed — {e}'})

        if resp.status_code != 200:
            return json.dumps({'success': False,
                               'reason': f'vlm_http_error — {resp.status_code}'})
        try:
            pdata = resp.json()
        except Exception:
            return json.dumps({'success': False, 'reason': 'vlm_invalid_json'})

        if not pdata.get('found', False):
            return json.dumps({'success': False, 'reason': 'parent_not_found'})

        p_conf = float(pdata.get('confidence', 0.0))
        if p_conf < min_conf:
            return json.dumps({'success': False,
                               'reason': f'parent_low_confidence — {p_conf:.2f} < {min_conf}'})

        raw_bbox = pdata.get('bbox_norm', [])
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            return json.dumps({'success': False, 'reason': 'parent_not_found'})
        parent_bbox_norm = [float(v) for v in raw_bbox]
        parent_source = 'vlm'
        if _is_bbox_oversized(parent_bbox_norm):
            _logger.warning(f'[parent] VLM returned oversized bbox {parent_bbox_norm}')
        _logger.info(f'[parent] VLM ground_object result: {parent_bbox_norm}')

    _logger.info(f'[parent source] {parent_source}')
    _logger.info(f'[parent bbox]   {parent_bbox_norm}')

    # ── C. Parent bbox 검증 ────────────────────────────────────────────────────
    parent_bbox_norm = [max(0.0, min(1.0, v)) for v in parent_bbox_norm]
    px1, py1, px2, py2 = parent_bbox_norm
    if px1 >= px2 or py1 >= py2 or (px2 - px1) * (py2 - py1) < 0.01:
        return json.dumps({'success': False, 'reason': 'invalid_parent_bbox'})

    # ── D. Parent crop (7% padding) ────────────────────────────────────────────
    try:
        crop_b64, crop_px = _crop_b64(img_bgr, parent_bbox_norm)
    except Exception as e:
        return json.dumps({'success': False, 'reason': f'crop_failed — {e}'})

    # ── E. Child grounding (crop 이미지에서) ──────────────────────────────────
    # _make_child_target: handle/knob/grip류는 상세 설명(내부 부품 제외) 포함
    child_label_ctx = _make_child_target(target_label, parent_label)
    _logger.info(f'[crop bbox]      px={crop_px}')
    _logger.info(f'[child target]   {child_label_ctx!r}')

    def _call_ground(image_b64: str, label: str) -> Optional[dict]:
        """VLM /ground_object 호출 → response dict 또는 None."""
        try:
            r = _requests.post(ground_url,
                               json={'full_image_b64': image_b64,
                                     'target_label': label,
                                     'detections': [],
                                     'timestamp': time.time()},
                               timeout=vlm_timeout)
        except _requests.exceptions.ConnectionError:
            return {'_error': 'vlm_server_unavailable'}
        except _requests.exceptions.Timeout:
            return {'_error': f'vlm_timeout — {vlm_timeout}s 초과'}
        except Exception as e:
            return {'_error': f'vlm_request_failed — {e}'}
        if r.status_code != 200:
            return {'_error': f'vlm_http_error — {r.status_code}'}
        try:
            return r.json()
        except Exception:
            return {'_error': 'vlm_invalid_json'}

    # 1차 시도: parent crop
    cdata = _call_ground(crop_b64, child_label_ctx)
    _error = cdata.get('_error') if isinstance(cdata, dict) else None
    if _error:
        return json.dumps({'success': False, 'reason': _error})

    child_found = (
        cdata.get('found', False)
        and isinstance(cdata.get('bbox_norm'), list)
        and len(cdata['bbox_norm']) == 4
        and float(cdata.get('confidence', 0.0)) >= min_conf
    )

    if not child_found:
        _logger.warning(
            f'[child] crop grounding failed '
            f'(found={cdata.get("found")}, conf={cdata.get("confidence")}, '
            f'raw={cdata}) — fallback to full image'
        )
        # fallback: 전체 원본 이미지로 한 번만 재시도
        try:
            full_b64_fb = _encode(img_bgr)
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'encode_failed — {e}'})

        cdata = _call_ground(full_b64_fb, child_label_ctx)
        _error = cdata.get('_error') if isinstance(cdata, dict) else None
        if _error:
            return json.dumps({'success': False, 'reason': _error})

        child_found = (
            cdata.get('found', False)
            and isinstance(cdata.get('bbox_norm'), list)
            and len(cdata['bbox_norm']) == 4
            and float(cdata.get('confidence', 0.0)) >= min_conf
        )
        if not child_found:
            _logger.error(
                f'[child] full-image fallback also failed '
                f'(found={cdata.get("found")}, conf={cdata.get("confidence")}, '
                f'raw={cdata})'
            )
            return json.dumps({'success': False, 'reason': 'child_not_found_in_parent'})

        # fallback 성공: child bbox는 이미 원본 좌표이므로 _restore_bbox 불필요
        child_bbox_in_crop = [float(v) for v in cdata['bbox_norm']]
        child_conf = float(cdata.get('confidence', 0.0))
        _logger.info(f'[child raw bbox (full-image fallback)] {child_bbox_in_crop}')

        ox1, oy1, ox2, oy2 = child_bbox_in_crop
        u_c = ((ox1 + ox2) / 2) * w
        v_c = ((oy1 + oy2) / 2) * h
        center_px = [int(round(u_c)), int(round(v_c))]
        child_bbox_px = [int(ox1 * w), int(oy1 * h), int(ox2 * w), int(oy2 * h)]
        _logger.info(f'[restored bbox (full-image fallback)] {child_bbox_in_crop}')
        _logger.info(f'[localization source] full-image VLM fallback')

        # depth → TF pipeline 공통 처리로 이동
        depth_arr_fb, _ = _ros_node.get_depth()
        fx_fb, fy_fb, cx_fb, cy_fb, _, _ = _ros_node.get_cam_intrinsics()

        def _result_2d_fb():
            return json.dumps({
                'success':             True,
                'label':               cdata.get('label', target_label),
                'source':              'vlm',
                'grounding':           'hierarchical',
                'center_px':           center_px,
                'bbox_approx':         child_bbox_px,
                'camera_point':        None,
                'base_link_point':     None,
                'confidence':          round(child_conf, 3),
                'position_confidence': '2d_only',
                'description':         cdata.get('description', ''),
                'inference_ms':        cdata.get('inference_ms', 0.0),
                'parent_label':        parent_label,
                'parent_source':       parent_source,
            }, ensure_ascii=False)

        if depth_arr_fb is None or fx_fb is None:
            _logger.info('[depth] unavailable — 2d_only')
            return _result_2d_fb()
        depth_m_fb = _sample_depth_robust(depth_arr_fb,
                                          child_bbox_px[0], child_bbox_px[1],
                                          child_bbox_px[2], child_bbox_px[3])
        if depth_m_fb is None:
            _logger.info('[depth] invalid — 2d_only')
            return _result_2d_fb()
        cam_xyz_fb = _px2cam(u_c, v_c, depth_m_fb, fx_fb, fy_fb, cx_fb, cy_fb)
        if cam_xyz_fb is None:
            return _result_2d_fb()
        base_xyz_fb = _cam_to_base(cam_xyz_fb)
        if base_xyz_fb is None:
            return _result_2d_fb()
        _logger.info(f'[3d] {base_xyz_fb}  [localization] 3d (full-image fallback)')
        return json.dumps({
            'success':             True,
            'label':               cdata.get('label', target_label),
            'source':              'vlm',
            'grounding':           'hierarchical',
            'center_px':           center_px,
            'bbox_approx':         child_bbox_px,
            'camera_point':        cam_xyz_fb,
            'base_link_point':     base_xyz_fb,
            'confidence':          round(child_conf, 3),
            'position_confidence': 'approximate',
            'description':         cdata.get('description', ''),
            'inference_ms':        cdata.get('inference_ms', 0.0),
            'depth_m':             round(depth_m_fb, 3),
            'parent_label':        parent_label,
            'parent_source':       parent_source,
        }, ensure_ascii=False)

    child_conf = float(cdata.get('confidence', 0.0))
    child_bbox_in_crop = [float(v) for v in cdata['bbox_norm']]
    _logger.info(f'[child raw bbox (crop)]  {child_bbox_in_crop}')

    # ── F. Child bbox 원본 이미지 좌표 복원 ──────────────────────────────────
    # crop_px는 padding 포함된 실제 crop 영역 [bx1,by1,bx2,by2] (pixel)
    bx1, by1, bx2, by2 = crop_px
    crop_norm = [bx1 / w, by1 / h, bx2 / w, by2 / h]
    child_bbox_orig = _restore_bbox(child_bbox_in_crop, crop_norm)
    _logger.info(f'[restored bbox]  {child_bbox_orig}')

    ox1, oy1, ox2, oy2 = child_bbox_orig
    u = ((ox1 + ox2) / 2) * w
    v = ((oy1 + oy2) / 2) * h
    center_px = [int(round(u)), int(round(v))]
    child_bbox_px = [int(ox1 * w), int(oy1 * h), int(ox2 * w), int(oy2 * h)]

    # ── 2D fallback 응답 구성 ──────────────────────────────────────────────────
    def _result_2d():
        _logger.info('[localization] 2d_only')
        return json.dumps({
            'success':             True,
            'label':               cdata.get('label', target_label),
            'source':              'vlm',
            'grounding':           'hierarchical',
            'center_px':           center_px,
            'bbox_approx':         child_bbox_px,
            'camera_point':        None,
            'base_link_point':     None,
            'confidence':          round(child_conf, 3),
            'position_confidence': '2d_only',
            'description':         cdata.get('description', ''),
            'inference_ms':        cdata.get('inference_ms', 0.0),
            'parent_label':        parent_label,
            'parent_source':       parent_source,
        }, ensure_ascii=False)

    # ── G. Depth 샘플링 ────────────────────────────────────────────────────────
    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, cam_w, cam_h = _ros_node.get_cam_intrinsics()

    if depth_arr is None or fx is None:
        _logger.info('[depth] unavailable — 2d_only')
        return _result_2d()

    depth_m = _sample_depth_robust(depth_arr,
                                    child_bbox_px[0], child_bbox_px[1],
                                    child_bbox_px[2], child_bbox_px[3])
    if depth_m is None:
        _logger.info('[depth] invalid — 2d_only')
        return _result_2d()

    _logger.info(f'[depth]          {depth_m:.3f}m')

    # ── H. Camera XYZ → base_link TF ─────────────────────────────────────────
    cam_xyz = _px2cam(u, v, depth_m, fx, fy, cx, cy)
    if cam_xyz is None:
        return _result_2d()

    base_xyz = _cam_to_base(cam_xyz)
    if base_xyz is None:
        return _result_2d()

    _logger.info(f'[3d]             {base_xyz}')
    _logger.info('[localization]   3d')
    return json.dumps({
        'success':             True,
        'label':               cdata.get('label', target_label),
        'source':              'vlm',
        'grounding':           'hierarchical',
        'center_px':           center_px,
        'bbox_approx':         child_bbox_px,
        'camera_point':        cam_xyz,
        'base_link_point':     base_xyz,
        'confidence':          round(child_conf, 3),
        'position_confidence': 'approximate',
        'description':         cdata.get('description', ''),
        'inference_ms':        cdata.get('inference_ms', 0.0),
        'depth_m':             round(depth_m, 3),
        'parent_label':        parent_label,
        'parent_source':       parent_source,
    }, ensure_ascii=False)


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


def _is_ros_alive() -> bool:
    """브리지 노드가 살아있는지 확인."""
    try:
        return _ros_node is not None and _ros_node.context.ok()
    except Exception:
        return False


def _ensure_ros():
    global _ros_ready
    with _ros_lock:
        if _is_ros_alive():
            return
        # 노드가 죽었으면 재초기화 (ROS 재시작 후 자동 복구)
        _ros_ready.clear()
        t = threading.Thread(target=_ros_spin_thread, daemon=True)
        t.start()
        _ros_ready.wait(timeout=10.0)
        if not _is_ros_alive():
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

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ① 알려진 물체 위치 확인         → 이 도구 (YOLO, 빠름)
    ② YOLO 목록에 없는 특정 물체    → ground_object(label)
    ③ 화면 전체 장면 파악 + 배치 공간 → analyze_scene()
    ④ 파지 방법 판단                → infer_grasp(label)
    ⑤ 배치 가능 공간 탐색           → find_placement()

    YOLO 결과에 target object가 있으면 ground_object()를 호출하지 마라.
    YOLO 결과만으로 해결 가능한 작업에 VLM을 호출하지 마라.

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

    [2026-08-26 확장] 이전엔 target_label과 정확히 일치하는 물체만 기억하고
    스윕 중 스쳐지나간 나머지는 버렸다. 이제 라벨 무관하게 스윕 중 보인
    물체를 전부(YOLO dual 백엔드가 인식하는 box + COCO 클래스 전체)
    기억한다 -- 한 번 스캔으로 box든 cup이든 bottle이든 나중에
    pick_object(from_scan=True, target_label=<원하는 라벨>)로 바로 꺼내
    쓸 수 있다. target_label 인자는 더 이상 결과를 필터링하지 않는다
    (하위호환용으로 남아있을 뿐).

    Args:
        target_label: [하위호환용, 현재는 결과에 영향 없음] 과거엔 필터링에
            썼으나 이제 스윕 중 보인 모든 라벨을 기억한다.

    Returns:
        성공: {"status": "success", "reason": "scan_complete:N",
               "boxes": [{"x":0.22,"y":0.19,"z":0.022,
                          "confidence":0.78,"angle_base_deg":45.0,
                          "label":"box"}, {"x":..,"label":"cup",...}, ...]}
               (N=발견한 개수, 라벨 다양할 수 있음. 각 항목의 x/y/z는
               물체 중심좌표)
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
                 angle_deg: float = None,
                 side_approach_deg: float = None) -> str:
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
        side_approach_deg: [2026-08-26 기준 변경] side 그립 시 접근 방향 --
            물체 방위각(position_yaw = atan2(y,x)) 기준 상대각(도), top-down의
            angle_rel과 동일 개념. 생략(또는 0)이면 물체 정면(가장 자연스럽고
            도달 가능성이 높은 방향)에서 접근. +/-로 그 방위각 대비 회전한
            방향에서 접근한다 (world 절대각이 아님 -- 이전엔 world 절대각을
            받았으나, 물체 위치가 바뀔 때마다 "정면"에 해당하는 값을 매번
            다시 계산해야 해서 상대각 기준으로 변경).
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
    if side_approach_deg is not None:
        payload['side_approach_deg'] = side_approach_deg
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
def slide_object(target_label: str, grasp_dir: str, x: float = None, y: float = None,
                  z: float = None, side_approach_deg: float = None,
                  slide_distance_m: float = None, from_scan: bool = False,
                  box_index: int = None) -> str:
    """문고리/서랍 손잡이 등을 잡아서 "당겨 여는" 동작. pick_object와 달리
    들어올리지 않는다 -- 접근각 반대 방향으로 slide_distance_m만큼 직선
    이동한 뒤 그리퍼를 놓는다. [2026-08-26 추가]

    ⚠ grasp_dir는 "side" 또는 "pinch"만 가능하다. top-down은 애초에
    수평 당김 동작과 기하학적으로 안 맞아서 미지원(요청하면 rejected).

    side/pinch의 접근 시퀀스(TCP오프셋 보정, standoff에서 직선 진입,
    접근각 후보 자동 탐색, 사전회전 없음)를 pick_object와 완전히
    동일하게 재사용한다 -- 차이는 마지막 단계뿐이다(들어올리기 대신
    당겨서 놓기).

    Args:
        target_label: 손잡이/물체 라벨 (예: "handle", "drawer").
        grasp_dir: "side" 또는 "pinch" (필수, 다른 값은 거부됨).
        x, y, z: override 좌표 (base_link 기준, 미터). 지정하면 인지
            재조회 없이 이 좌표로 바로 접근한다.
        side_approach_deg: 접근각 -- 물체 방위각(atan2(y,x)) 기준
            상대각(도). 생략(또는 0)이면 정면 접근.
        slide_distance_m: 당길 거리(미터). 생략 시 서버 기본값(0.08m).
        from_scan: True면 scan_for_boxes로 미리 찾아둔 좌표 사용.
        box_index: from_scan=True일 때 remaining_scanned_boxes에서
            몇 번째 항목을 쓸지 (pick_object와 동일 규칙).

    Returns:
        성공: {"status": "success", "reason": "slide_complete"}
        거부: {"status": "rejected", "reason": "..."} (grasp_dir가
              side/pinch가 아니거나 접근 불가 영역)
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    payload = {'action': 'slide', 'target_label': target_label, 'grasp_dir': grasp_dir}
    if side_approach_deg is not None:
        payload['side_approach_deg'] = side_approach_deg
    if slide_distance_m is not None:
        payload['slide_distance_m'] = slide_distance_m
    if x is not None and y is not None and z is not None:
        payload['override_pos'] = {'x': x, 'y': y, 'z': z}
    elif from_scan:
        payload['from_scan'] = True
        if box_index is not None:
            payload['box_index'] = box_index
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PICK)
    result['target_label'] = target_label
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def place_object(x: float, y: float, z: float, grasp_dir: str = 'auto',
                  side_approach_deg: float = None) -> str:
    """집은 물체를 지정한 좌표(base_link 기준, 미터)에 내려놓는다. place 완료까지 블로킹.

    Args:
        x, y, z: 내려놓을 위치 (base_link 기준, 미터)
        grasp_dir: 내려놓을 때 자세. pick_object 와 동일한 값 사용 권장.
        side_approach_deg: [2026-08-26 추가] grasp_dir이 side/pinch일 때
            접근각 -- 물체 방위각(atan2(y,x)) 기준 상대각(도). 생략(또는 0)
            이면 정면 접근. pick_object와 동일한 접근각 후보 자동 탐색이
            place에도 적용된다.

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
        if side_approach_deg is not None:
            payload['side_approach_deg'] = side_approach_deg
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
def move_to_position(x: float, y: float, z: float, grasp_dir: str = 'auto',
                      side_approach_deg: float = None,
                      quat_override: list = None,
                      apply_side_tcp_offset: bool = False) -> str:
    """로봇 팔 끝(end-effector)을 지정 좌표로 이동한다. 완료까지 블로킹.
    pick_object와 달리 그리퍼를 닫지 않는다 -- 자세(orientation) 후보를
    실제로 뭔가 집지 않고 미리 확인해볼 때 씀. side 그립 후보 검증용으로
    [2026-08-26] 추가됨 -- 자세한 배경은 grasp-kinematics-design 스킬 참고.

    ⚠ pick_object의 side 그립과 달리 이 도구는 기본적으로
    side_reachability_check(최소거리 0.32m)나 SIDE_TCP_OFFSET 보정을
    거치지 않는다 -- 순수하게 지정한 (x,y,z)로 지정한 자세로 이동만
    해본다. apply_side_tcp_offset=True로 켜지 않는 한 실제 pick 최종
    도달 지점과 다를 수 있다.

    Args:
        x, y, z: 목표 좌표 (base_link 기준, 미터)
        grasp_dir: 자세. 생략(기본 'auto')하면 기존 동작과 동일(top-down,
            마지막 스캔 각도 기반). "top"/"side"/"side_left"/"side_right" 등
            pick_object와 동일한 값 사용 가능.
        side_approach_deg: [2026-08-26 기준 변경] side 그립일 때 접근 방향 --
            물체 방위각(atan2(y,x)) 기준 상대각(도, world 절대각 아님).
            생략(또는 0)이면 정면 접근.
        quat_override: [x,y,z,w] 쿼터니언을 직접 지정해서 grasp_dir 계산을
            완전히 건너뛴다 (여러 후보 자세를 직접 실험할 때 사용).
            지정하면 grasp_dir/side_approach_deg는 무시된다.
        apply_side_tcp_offset: [2026-08-26 추가] True + side 계열
            grasp_dir이면, pick_object와 동일하게 SIDE_TCP_OFFSET을
            (position_yaw+side_approach_deg 방향으로) 적용한 뒤 이동한다
            -- 실제 pick 시 그리퍼 손끝이 도달할 지점을 그리퍼를 닫지
            않고 미리 확인할 때 켜라. quat_override 지정 시에는 무시됨
            (원본 좌표 그대로 이동).

    Returns:
        성공: {"status": "success", "reason": "move_complete", "target_pos": {...}}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    target_pos = {'x': x, 'y': y, 'z': z}
    payload = {'action': 'move', 'target_pos': target_pos}
    if quat_override is not None:
        payload['quat_override'] = list(quat_override)
    elif grasp_dir and grasp_dir != 'auto':
        payload['grasp_dir'] = grasp_dir
        if side_approach_deg is not None:
            payload['side_approach_deg'] = side_approach_deg
        if apply_side_tcp_offset:
            payload['apply_side_tcp_offset'] = True
    _ros_node.publish_command(payload)
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


def _vlm_ground_bbox_for_grasp(target_label: str, img_bgr: np.ndarray, objects: list):
    """[2026-08-26 추가] infer_grasp 전용 -- YOLO에 없는 물체를 VLM
    /ground_object로 찾아 bbox_norm만 얻는다 (ground_object() MCP 도구와
    동일 엔드포인트/페이로드 재사용, 3D 좌표 계산은 생략하고 bbox만 필요).

    Returns:
        (bbox_norm: list[4] 또는 None, confidence: float, error_reason: str 또는 None)
    """
    GROUND_URL  = f"{VLM_SERVER_URL}/ground_object"
    VLM_TIMEOUT = 45.0

    def _encode(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode(img_bgr)
    except Exception as e:
        return None, 0.0, f'encode_failed — {e}'

    yolo_detections = [
        {'label': o.get('label', '?'), 'bbox': o.get('bbox', []),
         'confidence': round(float(o.get('confidence', 0.0)), 3)}
        for o in objects
    ]
    payload = {
        'full_image_b64': full_b64,
        'target_label':   target_label,
        'detections':     yolo_detections,
        'timestamp':      time.time(),
    }
    try:
        resp = _requests.post(GROUND_URL, json=payload, timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return None, 0.0, 'vlm_server_unavailable'
    except _requests.exceptions.Timeout:
        return None, 0.0, f'vlm_timeout — {VLM_TIMEOUT}s 초과'
    except Exception as e:
        return None, 0.0, f'vlm_request_failed — {e}'

    if resp.status_code != 200:
        return None, 0.0, f'vlm_http_error — {resp.status_code}'
    try:
        gdata = resp.json()
    except Exception:
        return None, 0.0, 'vlm_invalid_json'

    if not gdata.get('found', False):
        return None, 0.0, 'target_not_found'

    conf = float(gdata.get('confidence', 0.0))
    bbox_norm = gdata.get('bbox_norm')
    if not bbox_norm or len(bbox_norm) != 4:
        return None, conf, 'no_valid_bbox_from_vlm'
    return bbox_norm, conf, None


@mcp.tool()
def infer_grasp(target_label: str) -> str:
    """지정한 물체에 대해 VLM grasp 타입을 추론한다 (명시적 온디맨드 호출).

    VLM은 "어떻게 잡을 것인가?"만 판단한다. "어디 있는가?"는 1차로 YOLO가
    담당한다. /detected_objects 에서 target_label과 일치하는 물체를 찾고,
    있으면 그 bbox로 crop해서 VLM_SERVER_URL/infer_grasp 에 POST한다.

    [2026-08-26 추가] YOLO 목록에 없는 라벨(예: dual-yolo box+coco 25종
    밖의 물체)이면 더 이상 바로 에러 내지 않고, ground_object()와 동일한
    VLM /ground_object 엔드포인트로 bbox를 자동으로 확보해서 계속
    진행한다 — 응답의 grounding_source 필드로 어느 쪽이었는지 확인 가능
    ("yolo"=정밀 bbox, "vlm"=근사 bbox라 crop이 다소 부정확할 수 있음).
    이제 ground_object로 미리 위치를 확보할 필요 없이 바로 호출해도 된다.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    "컵 어떻게 잡아?"  → infer_grasp("cup")
    "컵 어디 있어?"   → list_detected_objects()  ← VLM 불필요

    Args:
        target_label: 추론할 물체의 YOLO 탐지 라벨 (예: "box", "cup")

    Returns:
        성공: {"status": "success", "object": "...", "grasp_type": "TOP|SIDE|PINCH",
               "orientation": "HORIZONTAL|VERTICAL", "confidence": 0.95,
               "reason": "...", "inference_ms": 1200.0,
               "bbox_px": [x1,y1,x2,y2], "image_age_sec": 0.1,
               "grounding_source": "yolo"|"vlm", "grounding_confidence": 0.85|null,
               "approach_direction": "FRONT|LEFT|RIGHT|BACK",
               "suggested_side_approach_deg": 0.0,
               "face_normal_yaw_deg": 37.2|null, "face_normal_confidence": 0.81}
               (grasp_type이 SIDE/PINCH면 suggested_side_approach_deg를
               pick_object/slide_object의 side_approach_deg에 그대로 넣어라
               -- 대략적인 시작 각도일 뿐, 틀려도 서버의 접근각 후보 탐색이
               안전망 역할을 한다. 구버전 VLM 서버는 이 필드를 안 주므로
               항상 FRONT/0.0으로 채워진다. face_normal_yaw_deg는 [2026-08
               추가, 프로토타입] depth 평면적합 기반 -- 카메라가 마침 물체
               옆면을 보고 있을 때만 값이 나오고(윗면 위주로 보고 있으면
               null) suggested_side_approach_deg보다 더 정밀할 수 있지만
               아직 모션 결정에는 연결 안 됨, 참고용으로만 볼 것.)
        실패: {"status": "error", "reason": "..."}
    """
    _ensure_ros()

    VLM_URL     = f"{VLM_SERVER_URL}/infer_grasp"
    VLM_TIMEOUT = 30.0   # Qwen2.5-VL inference 시간 여유 확보

    # ── 1. 최신 카메라 이미지 가져오기 (YOLO 폴백 시에도 필요해서 먼저 확보) ──
    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'status': 'error',
                           'reason': 'image_unavailable — /camera/color/image_raw 수신 없음'})

    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'status': 'error',
                           'reason': f'image_stale — 이미지가 {img_age:.1f}초 경과 (>3s)'})

    h, w = img_bgr.shape[:2]

    # ── 2. 감지된 물체 목록에서 target_label 찾기 (YOLO 우선) ──────────────
    objects, obj_stamp = _ros_node.get_objects()
    label_lower = target_label.strip().lower()
    matched = [o for o in objects if str(o.get('label', '')).lower() == label_lower] if objects else []

    grounding_source = 'yolo'
    grounding_conf = None
    if matched:
        obj = matched[0]
        bbox_norm = obj.get('bbox')   # [x_min, y_min, x_max, y_max] 0~1 정규화
    else:
        # [2026-08-26 추가] YOLO 목록에 없으면(라벨을 아예 모르거나 confidence
        # 미달) 바로 에러 내지 않고 ground_object()와 동일한 VLM
        # /ground_object 엔드포인트로 bbox를 확보해서 계속 진행한다.
        # 이전엔 여기서 무조건 'object_not_found'였는데, "book"처럼
        # dual-yolo(box+coco 25종) 밖의 라벨은 infer_grasp 자체를 못 쓰는
        # 문제가 있었다 — VLM(ground_object)은 보는데 infer_grasp만 못 보는
        # 불일치가 실측 확인됨.
        bbox_norm, grounding_conf, ground_err = _vlm_ground_bbox_for_grasp(
            target_label, img_bgr, objects or [])
        if bbox_norm is None:
            available = [o.get('label', '') for o in objects] if objects else []
            return json.dumps({'status': 'error',
                               'reason': f'not_found_by_yolo_or_vlm — yolo_labels={available}, '
                                         f'vlm_ground_error={ground_err}'})
        grounding_source = 'vlm'
        obj = {'label': target_label, 'confidence': grounding_conf}

    # ── 3. bbox crop 생성 ────────────────────────────────────────────────────
    if bbox_norm and len(bbox_norm) == 4:
        x1 = int(bbox_norm[0] * w)
        y1 = int(bbox_norm[1] * h)
        x2 = int(bbox_norm[2] * w)
        y2 = int(bbox_norm[3] * h)
        # 클램프 + 최소 크기 보장
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return json.dumps({'status': 'error',
                               'reason': f'bbox_too_small — crop 크기 {x2-x1}x{y2-y1}px'})
        crop_bgr = img_bgr[y1:y2, x1:x2]
        bbox_px  = [x1, y1, x2, y2]
    else:
        # bbox 정보가 없으면 전체 이미지를 crop으로 사용
        crop_bgr = img_bgr.copy()
        bbox_px  = [0, 0, w, h]

    # ── 4. base64 JPEG 인코딩 ───────────────────────────────────────────────
    def _encode_b64(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode_b64(img_bgr)
        crop_b64 = _encode_b64(crop_bgr)
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'encode_failed — {e}'})

    # ── 5. VLM 서버 POST ────────────────────────────────────────────────────
    payload = {
        'full_image_b64': full_b64,
        'crop_image_b64': crop_b64,
        'object_label':   obj.get('label', target_label),
        'bbox':           bbox_norm if bbox_norm else [0, 0, 1, 1],
        'timestamp':      time.time(),
    }

    try:
        resp = _requests.post(VLM_URL, json=payload, timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_server_unavailable — {VLM_SERVER_URL} 연결 거부'})
    except _requests.exceptions.Timeout:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'})
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'vlm_request_failed — {e}'})

    if resp.status_code != 200:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_http_error — status {resp.status_code}: {resp.text[:200]}'})

    # ── 6. 응답 파싱 및 검증 ────────────────────────────────────────────────
    try:
        data = resp.json()
    except Exception:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_invalid_json — {resp.text[:200]}'})

    grasp_type  = str(data.get('grasp_type', '')).upper()
    orientation = str(data.get('orientation', '')).upper()
    confidence  = data.get('confidence', 0.0)

    if grasp_type not in ('TOP', 'SIDE', 'PINCH'):
        return json.dumps({'status': 'error',
                           'reason': f'vlm_invalid_grasp_type — {grasp_type!r}'})
    if orientation not in ('HORIZONTAL', 'VERTICAL'):
        orientation = 'VERTICAL' if grasp_type == 'PINCH' else 'HORIZONTAL'
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    # [2026-08-26 추가] VLM이 준 정성적 접근방향(FRONT/LEFT/RIGHT/BACK,
    # 카메라 시점 기준)을 pick_object/slide_object의 side_approach_deg
    # (position_yaw 기준 상대각, 도)로 쓸 대략적인 시작값으로 변환한다.
    # 정밀한 각도가 아니라 "후보 탐색의 1순위 추측값" 용도 -- 틀려도
    # planning_node의 접근각 후보 스윕(±15~90°)이 안전망 역할을 한다.
    # VLM 서버가 이 필드를 아직 안 보내는 구버전이면 기본 FRONT(0°)로
    # 처리해서 하위호환된다.
    _APPROACH_DEG_MAP = {'FRONT': 0.0, 'LEFT': 90.0, 'RIGHT': -90.0, 'BACK': 180.0}
    approach_direction = str(data.get('approach_direction', 'FRONT')).upper()
    if approach_direction not in _APPROACH_DEG_MAP:
        approach_direction = 'FRONT'
    suggested_side_approach_deg = _APPROACH_DEG_MAP[approach_direction]

    # [2026-08 추가, 프로토타입 -- 실기 미검증] SIDE/PINCH일 때 접근각
    # 추정에 참고할 만한 depth 평면적합 방위각. suggested_side_approach_deg
    # (VLM의 정성적 FRONT/LEFT/RIGHT/BACK 추측)보다 더 정밀할 수 있지만,
    # 카메라가 마침 물체 옆면을 보고 있어야만 값이 나온다(윗면 위주로 보고
    # 있으면 null) -- 항상 나오는 값이 아니므로 suggested_side_approach_deg
    # 를 대체하지 않고 나란히 노출만 한다.
    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, _cw, _ch = _ros_node.get_cam_intrinsics()
    face_yaw_deg, face_conf, _face_n = _compute_face_normal_yaw_from_bbox(
        bbox_px, depth_arr, fx, fy, cx, cy)

    return json.dumps({
        'status':           'success',
        'object':           data.get('object', target_label),
        'grasp_type':       grasp_type,
        'orientation':      orientation,
        'confidence':       round(confidence, 3),
        'reason':           data.get('reason', ''),
        'inference_ms':     data.get('inference_ms', 0.0),
        'bbox_px':          bbox_px,
        'image_age_sec':    img_age,
        # [2026-08-26 추가] bbox 출처 -- 'yolo'면 정밀, 'vlm'이면 ground_object
        # 경유 근사 bbox라 crop이 다소 부정확할 수 있음(참고용으로 노출).
        'grounding_source': grounding_source,
        'grounding_confidence': round(grounding_conf, 3) if grounding_conf is not None else None,
        # [2026-08-26 추가] side/pinch일 때 pick_object(side_approach_deg=)에
        # 그대로 넣을 수 있는 대략적 시작 각도. approach_direction은 VLM
        # 원본 응답(FRONT/LEFT/RIGHT/BACK), 구버전 VLM 서버면 항상 FRONT/0.0.
        'approach_direction': approach_direction,
        'suggested_side_approach_deg': suggested_side_approach_deg,
        # [2026-08 추가, 프로토타입] depth 평면적합 기반 mod-180 방위각 --
        # 신뢰 불가(카메라가 물체 윗면 위주로 보고 있음/평면성 낮음)면 null.
        # pick_object 등 모션 결정에는 아직 연결 안 됨, 참고용.
        'face_normal_yaw_deg': face_yaw_deg,
        'face_normal_confidence': face_conf,
    }, ensure_ascii=False)


@mcp.tool()
def analyze_scene() -> str:
    """현재 카메라 화면에 보이는 모든 물체를 label + bbox로 반환한다.

    VLM이 카메라 이미지를 독립적으로 보고 물체를 탐지한다.
    YOLO 결과와 병합해서 전체 scene을 반환한다.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ○ 사용: "화면에 뭐가 있어?" / "YOLO가 못 찾은 것도 알려줘" / 전체 장면 파악
           "어디 놓을 수 있어?" / "빈 공간 찾아줘" / placement 영역 탐색
    ✗ 금지: 단일 물체 위치 확인   → list_detected_objects() 또는 ground_object()
    ✗ 금지: YOLO에 이미 있는 물체를 다시 찾기 위한 호출
    ✗ 금지: VLM 실패 시 robot movement 명령

    source 태깅:
    - "both" = YOLO와 VLM 모두 탐지
    - "vlm"  = VLM만 탐지 (예: 선반·바구니·펜처럼 YOLO가 없는 물체)
    - "yolo" = YOLO만 탐지 (VLM이 못 찾은 경우)

    ─── VLM 실패 시 절대 금지 ──────────────────────────────────────────────────
    status가 "error"이면:
    - YOLO 좌표로 "빈 공간" / placement 위치 추론 금지
    - robot movement 명령 금지
    반드시: "시각적 scene reasoning을 수행할 수 없습니다. (이유: {reason})"

    Returns:
        성공: {"status":"success",
               "yolo_detected":["cup","box"],
               "objects":[{"label":"cup","bbox":[x1,y1,x2,y2],"source":"both"},
                          {"label":"pen","bbox":[...],"source":"vlm"},
                          {"label":"box","bbox":[...],"source":"yolo"},...],
               "placement_regions":[{"bbox":[x1,y1,x2,y2],"confidence":0.8},...],
               "inference_ms":8000.0,"image_age_sec":0.1}
        실패: {"status":"error","reason":"..."}
    """
    _ensure_ros()

    ANALYZE_URL = f"{VLM_SERVER_URL}/analyze_scene"
    VLM_TIMEOUT = 45.0

    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'status': 'error',
                           'reason': 'image_unavailable — /camera/color/image_raw 수신 없음'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'status': 'error',
                           'reason': f'image_stale — {img_age:.1f}s 경과 (>3s)'})

    yolo_objects, _ = _ros_node.get_objects()
    detections = [{'label': o.get('label', '?'), 'bbox': o.get('bbox', [])} for o in yolo_objects]

    def _encode_b64(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode_b64(img_bgr)
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'encode_failed — {e}'})

    try:
        resp = _requests.post(ANALYZE_URL,
                              json={'full_image_b64': full_b64, 'detections': detections,
                                    'timestamp': time.time()},
                              timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_server_unavailable — {VLM_SERVER_URL} 연결 거부'})
    except _requests.exceptions.Timeout:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'})
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'vlm_request_failed — {e}'})

    if resp.status_code != 200:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_http_error — status {resp.status_code}: {resp.text[:200]}'})

    try:
        data = resp.json()
    except Exception:
        return json.dumps({'status': 'error', 'reason': f'vlm_invalid_json — {resp.text[:200]}'})

    yolo_label_set = {str(o.get('label', '')).lower() for o in yolo_objects}
    vlm_objects    = data.get('objects', [])
    vlm_label_set  = {str(o.get('label', '')).lower() for o in vlm_objects}

    # VLM 탐지 물체: source="both"(YOLO도 탐지) 또는 source="vlm"(VLM만 탐지)
    objects_merged = [
        {**o, 'source': 'both' if str(o.get('label', '')).lower() in yolo_label_set else 'vlm'}
        for o in vlm_objects
    ]
    # YOLO-only 물체(VLM이 못 찾은 것): source="yolo"로 추가
    for yo in yolo_objects:
        if str(yo.get('label', '')).lower() not in vlm_label_set:
            objects_merged.append({
                'label':  yo.get('label', '?'),
                'bbox':   yo.get('bbox', []),
                'source': 'yolo',
            })

    # analyze_scene 결과를 캐시에 저장 — hierarchical grounding의 parent bbox 재사용용
    global _last_scene_objects, _last_scene_stamp
    with _scene_cache_lock:
        _last_scene_objects = [o.copy() for o in objects_merged]
        _last_scene_stamp   = time.time()

    return json.dumps({
        'status':            'success',
        'yolo_detected':     [o.get('label', '?') for o in yolo_objects],
        'objects':           objects_merged,
        'placement_regions': data.get('placement_regions', []),
        'inference_ms':      data.get('inference_ms', 0.0),
        'image_age_sec':     img_age,
    }, ensure_ascii=False)


@mcp.tool()
def ground_object(target_label: str, parent_label: str = None) -> str:
    """YOLO에 없는 물체를 VLM으로 찾아 depth + TF로 3D 좌표를 추정한다.

    동작 순서 (일반):
      1. YOLO /detected_objects에 target_label이 있으면 그 좌표를 바로 반환
         (source=yolo, grounding=detected, position_confidence=precise)
      2. YOLO에 없으면 VLM /ground_object 호출 → approximate bbox/center_norm 획득
      3. bbox 영역의 depth를 robust median으로 샘플링
      4. pixel_to_camera_xyz → camera_color_optical_frame 3D 점
      5. TF → base_link 3D 좌표 반환
         (source=vlm, grounding=approximate, position_confidence=approximate)

    ─── Hierarchical Grounding (parent_label 지정 시) ────────────────────────
    parent object crop → child part grounding → depth → TF → 3D (2D fallback).

    Use normal grounding when:
    - target이 독립적인 물체인 경우
    - 전체 scene에서 충분히 찾을 수 있는 크기

    Use hierarchical grounding when:
    - target이 큰 물체의 일부 part인 경우
    - handle, knob, button, rim, switch, opening 등 작은 조작 대상
    - 전체 이미지에서 바로 찾으면 놓칠 가능성이 높은 경우

    Examples:
      ground_object("pen")
      ground_object("drawer handle", parent_label="yellow drawer")
      ground_object("cup rim", parent_label="cup")

    라우팅 규칙:
      "은색 선반이 어디 있어?"       → ground_object("silver shelf")
      "서랍 손잡이 위치 알려줘."     → ground_object("drawer handle", parent_label="yellow drawer")
      "컵을 어떻게 잡아?"            → infer_grasp("cup")  (위치 아님)
      "화면에 뭐가 있어?"            → analyze_scene()

    Args:
        target_label: 찾을 물체 설명 (예: "silver shelf", "drawer handle")
        parent_label: [선택] hierarchical grounding 시 parent 물체 라벨
                     (예: "yellow drawer"). None이면 기존 동작 유지.

    Returns:
        YOLO에 있을 때:
          {"label":"cup", "source":"yolo", "grounding":"detected",
           "center_px":[u,v], "camera_point":{x,y,z}, "base_link_point":{x,y,z},
           "confidence":0.95, "position_confidence":"precise"}
        VLM grounding 성공 시:
          {"label":"silver shelf", "source":"vlm", "grounding":"approximate",
           "center_px":[u,v], "bbox_approx":[x1,y1,x2,y2],
           "camera_point":{x,y,z}, "base_link_point":{x,y,z},
           "confidence":0.78, "position_confidence":"approximate",
           "face_normal_yaw_deg":37.2|null, "face_normal_confidence":0.81}
          (face_normal_yaw_deg는 [2026-08 추가, 프로토타입] depth 평면적합
          기반 mod-180 방위각 -- YOLO는 원래 이 필드가 없었으니 VLM경로
          에서만 시도하는 신규 정보. 카메라가 물체 옆면을 볼 때만 값이
          나오고, 윗면을 보고 있거나 평면성이 낮으면 null. 아직
          pick_object 등 모션 결정에는 연결 안 됨 -- 참고용으로만 볼 것.)
        Hierarchical grounding 3D 성공 시:
          {"success":true, "label":"drawer handle", "source":"vlm",
           "grounding":"hierarchical", "center_px":[u,v], "bbox_approx":[x1,y1,x2,y2],
           "camera_point":{x,y,z}, "base_link_point":{x,y,z},
           "confidence":0.78, "position_confidence":"approximate",
           "parent_label":"yellow drawer", "parent_source":"yolo"|"vlm"}
        Hierarchical grounding 2D fallback (depth/TF 실패):
          {"success":true, ..., "camera_point":null, "base_link_point":null,
           "position_confidence":"2d_only"}
        실패 시:
          {"success":false, "reason":"target_not_found"|"parent_not_found"|
                            "invalid_parent_bbox"|"child_not_found_in_parent"|
                            "invalid_depth"|"tf_unavailable"|...}
    """
    _ensure_ros()

    GROUND_URL  = f"{VLM_SERVER_URL}/ground_object"
    VLM_TIMEOUT = 45.0
    MIN_CONF    = 0.3

    # ── Hierarchical grounding 위임 ───────────────────────────────────────────
    if parent_label is not None:
        return _ground_hierarchical(target_label, parent_label,
                                    GROUND_URL, VLM_TIMEOUT, MIN_CONF)

    # ── 1. YOLO에서 먼저 검색 ────────────────────────────────────────────────
    objects, _ = _ros_node.get_objects()
    label_lower = target_label.strip().lower()
    matched = [o for o in objects if str(o.get('label', '')).lower() == label_lower]

    if matched:
        obj = matched[0]
        c3d = obj.get('center_3d', {})
        # YOLO는 이미 base_link 좌표를 가지고 있음
        bbox_norm = obj.get('bbox', [])
        img_bgr, _ = _ros_node.get_image()
        center_px = None
        if img_bgr is not None and len(bbox_norm) == 4:
            h, w = img_bgr.shape[:2]
            cx = int((bbox_norm[0] + bbox_norm[2]) / 2 * w)
            cy = int((bbox_norm[1] + bbox_norm[3]) / 2 * h)
            center_px = [cx, cy]
        return json.dumps({
            'label':              obj.get('label', target_label),
            'source':             'yolo',
            'grounding':          'detected',
            'center_px':          center_px,
            'camera_point':       None,
            'base_link_point':    c3d,
            'confidence':         round(float(obj.get('confidence', 1.0)), 3),
            'position_confidence': 'precise',
        }, ensure_ascii=False)

    # ── 2. YOLO 미감지 → VLM grounding ──────────────────────────────────────
    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'success': False,
                           'reason': 'image_unavailable'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'success': False,
                           'reason': f'image_stale — {img_age:.1f}s'})

    # 이미지 인코딩
    def _encode(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode(img_bgr)
    except Exception as e:
        return json.dumps({'success': False, 'reason': f'encode_failed — {e}'})

    yolo_detections = [
        {'label': o.get('label', '?'),
         'bbox':  o.get('bbox', []),
         'confidence': round(float(o.get('confidence', 0.0)), 3)}
        for o in objects
    ]

    payload = {
        'full_image_b64': full_b64,
        'target_label':   target_label,
        'detections':     yolo_detections,
        'timestamp':      time.time(),
    }
    try:
        resp = _requests.post(GROUND_URL, json=payload, timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return json.dumps({'success': False,
                           'reason': 'vlm_server_unavailable'})
    except _requests.exceptions.Timeout:
        return json.dumps({'success': False,
                           'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'})
    except Exception as e:
        return json.dumps({'success': False, 'reason': f'vlm_request_failed — {e}'})

    if resp.status_code != 200:
        return json.dumps({'success': False,
                           'reason': f'vlm_http_error — {resp.status_code}'})

    try:
        gdata = resp.json()
    except Exception:
        return json.dumps({'success': False, 'reason': 'vlm_invalid_json'})

    if not gdata.get('found', False):
        return json.dumps({'success': False, 'reason': 'target_not_found'})

    conf = float(gdata.get('confidence', 0.0))
    if conf < MIN_CONF:
        return json.dumps({'success': False,
                           'reason': f'low_grounding_confidence — {conf:.2f} < {MIN_CONF}'})

    # ── 3. bbox_norm → pixel 좌표 변환 ──────────────────────────────────────
    h, w = img_bgr.shape[:2]
    bbox_norm = gdata.get('bbox_norm', [])
    center_norm = gdata.get('center_norm', [])

    if len(bbox_norm) == 4:
        bx1 = int(bbox_norm[0] * w)
        by1 = int(bbox_norm[1] * h)
        bx2 = int(bbox_norm[2] * w)
        by2 = int(bbox_norm[3] * h)
        bbox_px = [bx1, by1, bx2, by2]
    else:
        bbox_px = None

    # bbox_norm에서 직접 계산 (VLM center_norm은 부정확할 수 있어 사용 안 함)
    if bbox_px is not None:
        u = (bbox_px[0] + bbox_px[2]) / 2.0
        v = (bbox_px[1] + bbox_px[3]) / 2.0
    elif len(center_norm) == 2:
        u = center_norm[0] * w
        v = center_norm[1] * h
    else:
        return json.dumps({'success': False, 'reason': 'no_valid_bbox_from_vlm'})

    center_px = [int(round(u)), int(round(v))]

    # ── 4. depth 샘플링 (없으면 시각적 결과만 반환) ─────────────────────────
    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, cam_w, cam_h = _ros_node.get_cam_intrinsics()

    if depth_arr is None or fx is None:
        # depth 없음 → 3D 좌표 없이 시각적 결과만 반환
        return json.dumps({
            'label':              gdata.get('label', target_label),
            'source':             'vlm',
            'grounding':          'visual_only',
            'center_px':          center_px,
            'bbox_approx':        bbox_px,
            'camera_point':       None,
            'base_link_point':    None,
            'confidence':         round(conf, 3),
            'position_confidence': 'visual_only — depth 없음, 3D 좌표 불가',
            'description':        gdata.get('description', ''),
            'inference_ms':       gdata.get('inference_ms', 0.0),
        }, ensure_ascii=False)

    if bbox_px is not None:
        depth_m = _sample_depth_robust(depth_arr, bbox_px[0], bbox_px[1],
                                        bbox_px[2], bbox_px[3])
    else:
        win = 20
        depth_m = _sample_depth_robust(depth_arr,
                                        center_px[0] - win, center_px[1] - win,
                                        center_px[0] + win, center_px[1] + win)

    if depth_m is None:
        return json.dumps({'success': False, 'reason': 'invalid_depth'})

    # ── 5. camera XYZ ────────────────────────────────────────────────────────
    cam_xyz = _px2cam(u, v, depth_m, fx, fy, cx, cy)
    if cam_xyz is None:
        return json.dumps({'success': False, 'reason': 'invalid_depth'})

    # ── 6. TF → base_link ────────────────────────────────────────────────────
    base_xyz = _cam_to_base(cam_xyz)
    if base_xyz is None:
        return json.dumps({'success': False, 'reason': 'tf_unavailable'})

    # [2026-08 추가, 프로토타입 -- 실기 미검증] YOLO 못 잡은 물체는 원래
    # angle_base_deg 자체가 없었다 -- 여기서라도 평면적합으로 방위각을
    # 시도한다. bbox_px가 있어야(면적 있는 영역이어야) 의미 있으므로
    # center_norm만 온 경우(bbox_px is None)는 스킵.
    face_yaw_deg, face_conf, _face_n = (
        _compute_face_normal_yaw_from_bbox(bbox_px, depth_arr, fx, fy, cx, cy)
        if bbox_px is not None else (None, 0.0, 0)
    )

    return json.dumps({
        'label':              gdata.get('label', target_label),
        'source':             'vlm',
        'grounding':          'approximate',
        'center_px':          center_px,
        'bbox_approx':        bbox_px,
        'camera_point':       cam_xyz,
        'base_link_point':    base_xyz,
        'confidence':         round(conf, 3),
        'position_confidence': 'approximate',
        'description':        gdata.get('description', ''),
        'inference_ms':       gdata.get('inference_ms', 0.0),
        'depth_m':            round(depth_m, 3),
        # [2026-08 추가, 프로토타입] mod-180 각도, 신뢰 불가 시 null.
        # side_approach_deg 계산의 참고용 -- planning_node.py엔 아직 미연결.
        'face_normal_yaw_deg': face_yaw_deg,
        'face_normal_confidence': face_conf,
    }, ensure_ascii=False)


@mcp.tool()
def estimate_object_geometry(target_label: str) -> str:
    """[2026-08 추가, 프로토타입 -- 실기 미검증] 물체의 3D 형상(RANSAC 평면 +
    PCA 주축)을 실측 depth로 계산한다. pick_object의 자세 결정에는 아직
    자동 연결되지 않았다 -- 이 값을 보고 판단에 참고하거나, pick_object가
    받는 grasp_candidate 필드를 직접 구성할 때 쓰기 위한 읽기 전용 도구다.

    파이프라인: YOLO bbox(현재 segmentation은 NoOp -- bbox를 그대로 사각형
    마스크로 씀) → depth 역투영 → point cloud → RANSAC(우세 평면) + PCA
    (주축) → base_link 좌표계로 변환.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ○ 사용: side/pinch 그립 전 물체의 실제 긴 축 방향을 확인하고 싶을 때,
           또는 infer_grasp의 grasp_relation("perpendicular_to_long_axis" 등)이
           실제로 어느 각도를 가리키는지 검증하고 싶을 때.
    ✗ 대체 아님: infer_grasp(어떻게 잡을지)나 ground_object(어디 있는지)를
           대체하지 않는다 -- 이 도구는 "물체가 실제로 어떤 모양으로 놓여
           있는지"만 답한다.

    Args:
        target_label: /detected_objects에 있는 라벨 (YOLO 검출 필요 --
            아직 VLM-only 물체는 지원 안 함, ground_object의 bbox_approx를
            나중에 여기 연결할 수 있음).

    Returns:
        성공: {"status":"success", "label":"...",
               "geometry_confidence":0.87,
               "major_axis_yaw_deg":37.2, "normal_yaw_deg":12.0|null,
               "plane_inlier_ratio":0.71, "point_count":812,
               "major_axis":[x,y,z], "plane_normal":[x,y,z]|null,
               "extents":[L,W,H]}
               (major_axis_yaw_deg/normal_yaw_deg는 mod-180 -- 부호 모호성은
               해소 안 됨, side grasp 후보 생성 시 양쪽 다 고려해야 함.
               None이면 그 축/면이 충분히 수직이 아니라 방위각이 무의미하다는
               뜻 -- geometry_3d.py의 verticality 게이트 참고.)
        실패: {"status":"error"|"low_confidence", "reason":"..."}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()

    objects, _ = _ros_node.get_objects()
    matched = [o for o in objects if str(o.get('label', '')).lower() == target_label]
    if not matched:
        return json.dumps({'status': 'error',
                           'reason': f"'{target_label}' 이(가) /detected_objects에 없습니다 "
                                     f"(YOLO 검출 필요 -- VLM-only 물체는 아직 미지원)."})
    obj = matched[0]
    bbox_norm = obj.get('bbox')

    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'status': 'error', 'reason': 'image_unavailable'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'status': 'error', 'reason': f'image_stale — {img_age:.1f}s'})

    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, _cw, _ch = _ros_node.get_cam_intrinsics()
    if depth_arr is None or fx is None:
        return json.dumps({'status': 'error', 'reason': 'depth_or_intrinsics_unavailable'})

    h, w = img_bgr.shape[:2]
    if not bbox_norm or len(bbox_norm) != 4:
        return json.dumps({'status': 'error', 'reason': 'invalid_bbox'})
    bbox_px = [int(bbox_norm[0] * w), int(bbox_norm[1] * h),
               int(bbox_norm[2] * w), int(bbox_norm[3] * h)]

    seg = NoOpSegmentationBackend().segment(img_bgr, bbox_px)
    if seg is None:
        return json.dumps({'status': 'error', 'reason': 'segmentation_failed'})

    points_cam = mask_depth_to_pointcloud(
        depth_arr, seg.mask, _PCIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
        max_points=2000)
    if len(points_cam) < 20:
        return json.dumps({'status': 'error',
                           'reason': f'insufficient_points — {len(points_cam)}개 (최소 20)'})

    geo_cam = _geometry_3d.compute_geometry(points_cam, min_inliers=20)
    if not geo_cam.valid:
        return json.dumps({'status': 'error', 'reason': 'geometry_fit_failed'})

    geo = _geometry_to_base_link(geo_cam)
    if not geo.valid:
        return json.dumps({'status': 'error', 'reason': 'tf_unavailable'})

    return json.dumps({
        'status': 'success',
        'label': target_label,
        'geometry_confidence': geo.geometry_confidence,
        'major_axis_yaw_deg': geo.major_axis_yaw_deg,
        'normal_yaw_deg': geo.normal_yaw_deg,
        'plane_inlier_ratio': round(geo.plane_inlier_ratio, 3),
        'point_count': geo.point_count,
        'major_axis': list(geo.major_axis) if geo.major_axis else None,
        'plane_normal': list(geo.plane_normal) if geo.plane_normal else None,
        'extents': list(geo.extents) if geo.extents else None,
    }, ensure_ascii=False)


@mcp.tool()
def find_placement() -> str:
    """카메라 화면에서 물체를 놓을 수 있는 빈 공간을 추론한다.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ○ 사용: "빈 공간에 놓아" / "정리해" / "어디 두면 좋을지 봐줘" / "선반에 놓아"
    ✗ 금지: "컵 옆에 놓아" → YOLO 3D 좌표 상대 계산으로 해결 (VLM 불필요)
    ✗ 금지: target placement가 명확한 좌표로 이미 알려진 경우

    VLM은 normalized bbox로 배치 가능 영역만 반환한다.
    실제 robot 좌표는 반환된 bbox → depth → camera coords → TF → base_link 순으로 변환한다.
    VLM placement는 approximate임을 반드시 명시한다.

    ─── VLM 실패 시 절대 금지 ──────────────────────────────────────────────────
    status가 "error"이면:
    - YOLO 좌표로 placement 위치 추측 금지
    - robot movement 명령 금지
    반드시: "시각적 scene reasoning을 수행할 수 없습니다. (이유: {reason})"

    Returns:
        성공: {"status":"success",
               "placement_regions":[{"bbox":[x1,y1,x2,y2],"confidence":0.8},...],
               "inference_ms":8000.0,"image_age_sec":0.1}
        실패: {"status":"error","reason":"..."}
    """
    _ensure_ros()

    PLACEMENT_URL = f"{VLM_SERVER_URL}/find_placement"
    VLM_TIMEOUT = 45.0

    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'status': 'error',
                           'reason': 'image_unavailable — /camera/color/image_raw 수신 없음'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'status': 'error',
                           'reason': f'image_stale — {img_age:.1f}s 경과 (>3s)'})

    yolo_objects, _ = _ros_node.get_objects()
    detections = [{'label': o.get('label', '?'), 'bbox': o.get('bbox', [])} for o in yolo_objects]

    def _encode_b64(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode_b64(img_bgr)
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'encode_failed — {e}'})

    try:
        resp = _requests.post(PLACEMENT_URL,
                              json={'full_image_b64': full_b64, 'detections': detections,
                                    'timestamp': time.time()},
                              timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_server_unavailable — {VLM_SERVER_URL} 연결 거부'})
    except _requests.exceptions.Timeout:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'})
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'vlm_request_failed — {e}'})

    if resp.status_code != 200:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_http_error — status {resp.status_code}: {resp.text[:200]}'})

    try:
        data = resp.json()
    except Exception:
        return json.dumps({'status': 'error', 'reason': f'vlm_invalid_json — {resp.text[:200]}'})

    return json.dumps({
        'status':            'success',
        'placement_regions': data.get('placement_regions', []),
        'inference_ms':      data.get('inference_ms', 0.0),
        'image_age_sec':     img_age,
    }, ensure_ascii=False)


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
