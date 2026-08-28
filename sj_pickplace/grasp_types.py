#!/usr/bin/env python3
"""
grasp_types.py

[신설, 2026-08] NERO grasp pose pipeline 재구조화 — VLM(semantic) / Geometry(metric)
/ Learned Grasp Detector / IK 각 단계가 주고받는 공용 데이터 구조.

배경: 기존 파이프라인은 VLM이 grasp_type(TOP/SIDE/PINCH)만 판단하고, 실제 접근각은
grasp_kinematics.py의 atan2(y,x) 기반 공식 + planning_node.py의 블라인드/Hough 후보
탐색이 담당했다(자세한 문제의식은 docs/wiki 또는 세션 기록 참고). 이 모듈은 그 사이에
"물체의 실제 3D 형상에서 유도한 6-DoF grasp 후보"라는 중간 표현을 추가하기 위한
타입 정의다.

ROS 비의존 순수 dataclass만 정의한다(grasp_kinematics.py와 동일한 모듈 경계 원칙 --
ROS 서비스 호출/실행은 이 모듈 책임이 아니다. pytest로 단위테스트 가능해야 한다).
"""
from dataclasses import dataclass
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# VLM Semantic Grasp Intent
# ═══════════════════════════════════════════════════════════════════════

# VLM에게 자유 텍스트를 그대로 downstream geometry에 넣지 않기 위한 허용 enum.
# vlm_grasp_server.py의 응답이 이 집합 밖의 값을 내면 안전값/unknown으로
# 강등하고, downstream은 unknown을 "구조화된 힌트 없음"으로 취급해 기존
# fallback 경로(grasp_type만으로 판단)로 넘어가야 한다.
GRASP_TYPES = {"TOP", "SIDE", "PINCH"}

GRASP_RELATIONS = {
    "perpendicular_to_long_axis",
    "along_long_axis",
    "perpendicular_to_surface",
    "along_surface_normal",
    "from_top",
    "unknown",
}

ACTIONS = {"PICK", "PULL", "PUSH", "SLIDE"}

ACTION_DIRECTIONS = {
    "along_long_axis",
    "perpendicular_to_long_axis",
    "along_surface_normal",
    "opposite_approach",
    "unknown",
}


@dataclass
class GraspIntent:
    """VLM의 semantic 판단 결과. metric 각도/좌표를 절대 담지 않는다 --
    이 dataclass에 도(degree)나 quaternion 필드를 추가하면 그건 설계
    위반이다(VLM은 "무엇을 어떻게"만 말하고 "몇 도"는 절대 말하지 않는다).

    target_part/grasp_relation/action/action_direction은 [2026-08 확장]
    기존 grasp_type/orientation/confidence(하위호환 유지)에 추가된 필드다.
    구버전 VLM 서버(vlm_grasp_server.py 확장 전 배포본)는 이 필드들을 안
    주므로 전부 기본값(None/unknown)으로 채워지고, downstream(특히
    grasp_pose_generator.generate_action_vector)은 grasp_relation/
    action_direction이 "unknown"이면 반드시 안전한 기존 동작(action=PICK
    가정, geometry 후보만으로 진행)으로 폴백해야 한다 -- 이 필드가
    unknown인데 임의로 방향을 추측하지 말 것.
    """
    grasp_type: str                      # TOP | SIDE | PINCH
    orientation: str = "HORIZONTAL"      # HORIZONTAL | VERTICAL (기존 필드, 유지)
    confidence: float = 0.0
    target_part: Optional[str] = None    # 예: "handle". None이면 object 전체
    grasp_relation: str = "unknown"      # GRASP_RELATIONS 중 하나
    action: str = "PICK"                 # ACTIONS 중 하나
    action_direction: str = "unknown"    # ACTION_DIRECTIONS 중 하나
    reason: str = ""
    source: str = "vlm"                  # vlm | label_fallback | unknown

    @classmethod
    def from_vlm_response(cls, data: dict) -> "GraspIntent":
        """vlm_grasp_server.py의 /infer_grasp 원본 JSON(dict)에서 생성.
        지원 안 하는 값은 자유 텍스트를 그대로 믿지 않고 안전값으로 강등한다
        (모듈 docstring "자유 텍스트를 그대로 downstream에 넣지 않는다" 원칙,
        기존 mcp_robot_server.infer_grasp()의 grasp_type/orientation 검증
        로직과 동일한 패턴 -- 중복 유지 대신 여기로 통합해도 됨).
        """
        grasp_type = str(data.get("grasp_type", "")).upper()
        if grasp_type not in GRASP_TYPES:
            grasp_type = "TOP"  # 안전 기본값 -- 호출부가 별도로 실패 처리해야 함

        orientation = str(data.get("orientation", "HORIZONTAL")).upper()
        if orientation not in ("HORIZONTAL", "VERTICAL"):
            orientation = "VERTICAL" if grasp_type == "PINCH" else "HORIZONTAL"

        grasp_relation = str(data.get("grasp_relation", "unknown")).strip()
        if grasp_relation not in GRASP_RELATIONS:
            grasp_relation = "unknown"

        action = str(data.get("action", "PICK")).upper()
        if action not in ACTIONS:
            action = "PICK"

        action_direction = str(data.get("action_direction", "unknown")).strip()
        if action_direction not in ACTION_DIRECTIONS:
            action_direction = "unknown"

        target_part = data.get("target_part")
        if target_part is not None:
            target_part = str(target_part).strip() or None

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return cls(
            grasp_type=grasp_type,
            orientation=orientation,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            target_part=target_part,
            grasp_relation=grasp_relation,
            action=action,
            action_direction=action_direction,
            reason=str(data.get("reason", "")),
            source="vlm",
        )


# ═══════════════════════════════════════════════════════════════════════
# 3D Geometry Result (RANSAC + PCA/OBB) — geometry_3d.py가 채움
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GeometryResult:
    """geometry_3d.py가 point cloud에서 추출한 형상 정보.
    좌표계는 입력 point cloud와 동일(호출부 책임 -- 보통 카메라 좌표계 입력,
    필요시 base_link로 변환은 벡터 회전으로 별도 처리, geometry_3d.py 참고).
    """
    valid: bool = False
    centroid: Optional[Tuple[float, float, float]] = None

    # PCA
    major_axis: Optional[Tuple[float, float, float]] = None   # 최대 분산 방향(단위벡터)
    minor_axis: Optional[Tuple[float, float, float]] = None
    third_axis: Optional[Tuple[float, float, float]] = None
    eigenvalues: Optional[Tuple[float, float, float]] = None  # 내림차순
    extents: Optional[Tuple[float, float, float]] = None      # OBB 각 축 길이(입력 단위, 보통 미터)

    # RANSAC (dominant plane 1순위 -- 여러 평면 지원은 fit_dominant_planes가
    # list[PlaneResult]로 반환, GeometryResult엔 1순위만 요약)
    plane_normal: Optional[Tuple[float, float, float]] = None
    plane_inlier_count: int = 0
    plane_inlier_ratio: float = 0.0

    # 품질 지표
    point_count: int = 0
    filtered_point_count: int = 0
    geometry_confidence: float = 0.0

    # 파생값(로깅/디버깅 편의) -- major_axis/plane_normal을 수평면에 투영한
    # mod-180 방위각(도). None이면 수직성 부족 등으로 방위각이 무의미하다는 뜻
    # (perception_node._compute_face_normal_yaw의 verticality 게이트와 동일 원칙).
    major_axis_yaw_deg: Optional[float] = None
    normal_yaw_deg: Optional[float] = None

    source: str = "ransac_pca"   # 실험 비교용 -- source 기록 원칙(비교실험 A/B/C/D)


# ═══════════════════════════════════════════════════════════════════════
# Grasp Candidate
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GraspCandidate:
    """6-DoF grasp 후보 하나. Learned detector 출력이든 geometry fallback
    이든 동일한 타입으로 표현해서 candidate generator/semantic filtering이
    출처를 몰라도 동작하게 한다.

    position/quaternion은 base_link 프레임 기준(planning_node.py 기존 관례와
    동일). quaternion은 스펙 예시의 Tuple[float,float,float,float] 대신
    **list[4] xyzw**를 쓴다 -- grasp_kinematics.py 전체(euler_to_quat 등)가
    이미 이 표현을 쓰고 있어서, 여기서만 다른 타입을 쓰면 변환 지점마다
    실수 유발 위험이 커진다(기존 코드베이스 관례 우선).
    """
    position: Tuple[float, float, float]
    quaternion: list                              # [x,y,z,w], base_link 기준
    approach_vector: Tuple[float, float, float]    # 접근 방향(단위벡터, base_link)
    grasp_axis: Optional[Tuple[float, float, float]] = None  # 그리퍼 손끝이 닫히는 축

    geometry_score: float = 0.0
    semantic_score: float = 0.0
    learned_grasp_score: float = 0.0
    reachability_score: float = 0.0
    total_score: float = 0.0

    is_side: bool = False           # grasp_kinematics.py의 is_side 관례와 연결(TCP offset 분기용)
    grasp_dir_hint: str = "auto"    # resolve_grasp_quat 호환용 문자열(top/side/pinch)
    source: str = "unknown"         # learned_grasp | ransac_pca | face_normal | hough | blind_sweep
    debug_reason: str = ""
