#!/usr/bin/env python3
"""
segmentation_backend.py

[신설, 2026-08] Segmentation 단계 backend abstraction.

실제 pixel-precise segmentation 모델(SAM/SAM2/YOLO-seg/Grounded-SAM 등)은
이번 라운드에서 붙이지 않는다 — 사용자 판단 보류(SAM 연동은 별도 결정
예정). 지금은 YOLO bbox를 그대로 "마스크"로 취급하는
NoOpSegmentationBackend만 제공한다.

이래도 point_cloud.py 이후 단계는 그대로 동작한다 — bbox 영역을 사각형
마스크로 보는 것뿐이라 배경/인접 물체 픽셀이 섞일 수 있다는 한계는 있지만,
이건 원래 파이프라인이 bbox 영역 depth를 그대로 쓰던 것(perception_node.
_multi_point_3d_centroid, mcp_robot_server._sample_depth_robust)과 동일한
수준의 한계이지 새로 생긴 문제가 아니다.

향후 실제 segmentation을 붙이려면 SegmentationBackend를 구현한 새 클래스
(예: SamSegmentationBackend)를 추가하고 교체하면 된다 — 이 인터페이스를
쓰는 point_cloud.py/grasp_pose_generator.py 쪽 코드는 변경 불필요
(backend abstraction의 목적 자체가 이거다 — architecture 문서 28번 참고).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray           # bool 배열, 이미지와 동일 H,W. True=object 픽셀
    bbox_px: list               # [x1,y1,x2,y2], mask의 bounding box(디버깅/재사용용)
    source: str = "bbox"        # bbox(NoOp) | sam | yolo_seg | grounded_sam
    confidence: float = 1.0


class SegmentationBackend(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, bbox_px: list,
                target_part: Optional[str] = None) -> Optional[SegmentationResult]:
        """image(HxWx3, BGR/RGB 무관 -- 마스크만 만들 뿐 색공간 안 씀) +
        bbox_px([x1,y1,x2,y2] 픽셀좌표)로 마스크를 만든다.
        target_part는 SAM 등 prompt 기반 backend가 참고할 수 있는 힌트
        (예: "handle") -- NoOp는 무시한다.
        실패(bbox 무효 등) 시 None."""
        raise NotImplementedError


class NoOpSegmentationBackend(SegmentationBackend):
    """YOLO bbox를 그대로 사각형 마스크로 반환. 실제 segmentation 없음 --
    현재 기본/유일 backend(2026-08, SAM 등은 별도 판단 보류)."""

    def segment(self, image: np.ndarray, bbox_px: list,
                target_part: Optional[str] = None) -> Optional[SegmentationResult]:
        if image is None or not bbox_px or len(bbox_px) != 4:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_px]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return SegmentationResult(mask=mask, bbox_px=[x1, y1, x2, y2],
                                   source="bbox", confidence=1.0)


def get_default_backend() -> SegmentationBackend:
    """현재 기본 backend를 반환한다 -- 호출부(perception_node.py 등)는 이
    함수를 통해서만 backend를 얻어야 나중에 기본값 교체가 한 곳으로
    끝난다."""
    return NoOpSegmentationBackend()
