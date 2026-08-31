#!/usr/bin/env python3
"""
vlm_grasp_server.py — Qwen2.5-VL-3B-Instruct 기반 grasp type 추론 서버
===========================================================================
- ON-DEMAND: /infer_grasp 호출 시에만 추론 (카메라 스트림 상시 소비 없음)
- 카메라 device 직접 접근 없음 — base64 이미지만 수신
- 실제 로봇 동작 없음 — JSON 응답만 반환
- grasp_type: TOP | SIDE | PINCH

실행:
    python vlm_grasp_server.py --port 8003
    python vlm_grasp_server.py --port 8003 --debug --debug-dir /tmp/vlm_debug
    python vlm_grasp_server.py --port 8003 --model Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ── 요청 / 응답 스키마 ──────────────────────────────────────────────────────
class GraspRequest(BaseModel):
    full_image_b64: str           # 전체 카메라 프레임 (base64 JPEG)
    crop_image_b64: str           # bbox crop 이미지 (base64 JPEG)
    object_label: str             # YOLO 탐지 라벨
    bbox: list                    # [x_min, y_min, x_max, y_max] 정규화 0~1
    timestamp: Optional[float] = None

class GraspResponse(BaseModel):
    object: str
    grasp_type: str               # TOP | SIDE | PINCH
    orientation: str              # HORIZONTAL | VERTICAL
    confidence: float
    reason: str
    inference_ms: float
    # [2026-08 확장] semantic grasp intent 필드 -- 전부 기본값이 있어 구버전
    # 클라이언트(mcp_robot_server.py 구버전)에게 필드가 늘어난 것 자체는
    # 문제 없다(pydantic이 JSON 응답에 그대로 추가 필드를 실어보냄, 클라
    # 이언트가 무시하면 그만). VLM이 이 필드들을 안 채우거나 무효한 값을
    # 내면 _validate()가 안전값(target_part=None, grasp_relation="unknown",
    # action="PICK", action_direction="unknown")으로 강등한다 -- 자유
    # 텍스트를 그대로 downstream geometry에 흘리지 않는다는 원칙
    # (sj_pickplace/grasp_types.py GraspIntent와 동일 enum 집합, 이 서버는
    # 독립 배포 스크립트라 import 대신 로컬에 동일 정의를 둔다).
    target_part: Optional[str] = None
    grasp_relation: str = "unknown"
    action: str = "PICK"
    action_direction: str = "unknown"

class SceneRequest(BaseModel):
    full_image_b64: str
    detections: list = []         # YOLO 감지 목록 (label+bbox 맥락 제공용)
    timestamp: Optional[float] = None

class SceneResponse(BaseModel):
    objects: list = []            # [{"label":"cup","bbox":[x1,y1,x2,y2]}, ...]
    inference_ms: float = 0.0

class PlacementRequest(BaseModel):
    full_image_b64: str
    detections: list = []         # YOLO 감지 목록 (겹침 방지용)
    timestamp: Optional[float] = None

class PlacementResponse(BaseModel):
    placement_regions: list = []  # [{"bbox":[x1,y1,x2,y2],"confidence":0.8}, ...]
    inference_ms: float = 0.0

class GroundRequest(BaseModel):
    full_image_b64: str
    target_label: str             # 찾을 물체 라벨 (예: "silver shelf")
    detections: list = []         # YOLO 감지 목록 (맥락 제공용)
    timestamp: Optional[float] = None

class GroundResponse(BaseModel):
    found: bool = False
    label: str = ""
    bbox_norm: list = []          # [x_min, y_min, x_max, y_max] 정규화 0-1
    center_norm: list = []        # [cx, cy] 정규화 0-1
    confidence: float = 0.0
    description: str = ""
    inference_ms: float = 0.0

ALLOWED_GRASP  = {"TOP", "SIDE", "PINCH"}
ALLOWED_ORIENT = {"HORIZONTAL", "VERTICAL"}

# [2026-08 추가] sj_pickplace/grasp_types.py의 GRASP_RELATIONS/ACTIONS/
# ACTION_DIRECTIONS와 동일 집합 -- 이 파일은 독립 배포 스크립트(4090 랩탑,
# ROS 워크스페이스 밖)라 import 대신 로컬에 중복 정의한다. 한쪽만 바꾸면
# 어긋날 위험이 있으니, grasp_types.py를 고칠 땐 이쪽도 같이 확인할 것.
ALLOWED_RELATION = {
    "perpendicular_to_long_axis", "along_long_axis",
    "perpendicular_to_surface", "along_surface_normal", "from_top", "unknown",
}
ALLOWED_ACTION = {"PICK", "PULL", "PUSH", "SLIDE"}
ALLOWED_ACTION_DIRECTION = {
    "along_long_axis", "perpendicular_to_long_axis",
    "along_surface_normal", "opposite_approach", "unknown",
}

# ── 시스템 프롬프트 ──────────────────────────────────────────────────────────
_SCENE_SYSTEM = """\
Look at the camera image independently and detect every distinct physical object that is actually visible.

Rules:
- Detect objects based ONLY on what you see in the image. Do NOT copy, reuse, or reference any object names or coordinates from this prompt.
- Include manipulable objects: cups, boxes, scissors, books, phones, speakers, bags, markers, pens, pencils, bottles, containers, wallets, cables, and any other physical item.
- Exclude background surfaces: tables, floors, walls, the robot arm.
- Do NOT rely on any prior detector results.

Return ONLY minified JSON with this schema — no markdown, no extra text:
{"objects":[{"label":"<actual object name from image>","bbox":[x_min,y_min,x_max,y_max]},{"label":"<another object>","bbox":[x_min,y_min,x_max,y_max]}]}
bbox values are normalized 0.0-1.0. List ALL objects you actually see. No descriptions, no confidence scores."""

_PLACEMENT_SYSTEM = """\
Find 1-3 empty flat regions where an object can be placed on the surface.
Return ONLY this minified single-line JSON (no markdown, no extra text):
{"placement_regions":[{"bbox":[0.6,0.1,0.95,0.9],"confidence":0.8}]}
bbox: normalized [x_min,y_min,x_max,y_max] 0.0-1.0.
If no empty area is visible, return: {"placement_regions":[]}
No labels. No reasons. Do NOT overlap any existing object."""

_GROUNDING_SYSTEM = """\
You are a visual scene grounding module for a robot.

Analyze the camera image carefully.

YOLO detections are provided as context, but they are NOT a complete list.
Your task is to visually locate the REQUESTED TARGET OBJECT in the image,
even if YOLO did not detect it.

Return ONLY valid JSON — no markdown, no extra text:
{
  "found": true,
  "label": "<object name>",
  "bbox_norm": [x_min, y_min, x_max, y_max],
  "center_norm": [cx, cy],
  "confidence": 0.85,
  "description": "<brief visual description>"
}

Rules:
- bbox_norm and center_norm use normalized coordinates: 0.0 = left/top, 1.0 = right/bottom
- If the requested object is NOT visible in the image, return:
  {"found": false, "label": "", "confidence": 0.0, "description": "not found"}
- Do NOT invent objects that are not clearly visible
- Do NOT estimate metric 3D coordinates
- Do NOT output robot actions
- The bbox is an APPROXIMATE visual region, not a precise detector bbox"""

_SYSTEM = """\
You are a robotic grasp reasoning module.

Analyze the target object using:
  Image 1: full scene (environment context, obstacles, accessibility)
  Image 2: cropped target object (shape, orientation, graspable surfaces)

IMPORTANT: Base your decision on the ACTUAL visible shape and aspect ratio
in Image 2, not on assumptions from the object's name/category. The same
object type can appear in different orientations (e.g. a book can be lying
flat OR standing upright on its spine) — look at the crop before deciding.

Determine the most suitable grasp configuration for a parallel-jaw gripper.

Available grasp types (typical cases, not fixed rules — verify against
the actual image):
  TOP   — gripper descends from above; typical for objects lying flat/wide
          when viewed from the camera (e.g. a book lying on a table)
  SIDE  — horizontal approach from the side; typical for tall/upright
          objects (a standing book, cup, bottle, vase)
  PINCH — gripper rotated ~90° vertical; typical for thin/elongated items
          held between fingertips (pen, knife, fork, spoon, scissors,
          toothbrush, remote, phone, credit card) regardless of orientation

You do NOT compute metric angles, coordinates, or quaternions — a separate
geometry module measures the object's actual 3D shape. Your job is only to
describe the grasp SEMANTICALLY, relative to the object's own shape/surface.

grasp_relation (how the gripper relates to the object's geometry — pick one):
  perpendicular_to_long_axis — object is elongated (handle, bottle, pen); grip
                                across its short dimension, not along its length
  along_long_axis             — grip along the object's long dimension (rare —
                                only when the long axis itself is graspable)
  perpendicular_to_surface    — approach straight into a flat face
  along_surface_normal        — same idea as perpendicular_to_surface, for TOP grasps
  from_top                    — no strong shape cue, just come from above
  unknown                     — none of the above clearly applies

action (what happens after the grasp — pick one):
  PICK — lift the object up and away
  PULL — grasp then pull (e.g. a drawer/door handle)
  PUSH — grasp then push
  SLIDE — grasp then slide sideways without lifting

action_direction (only meaningful when action != PICK — pick one):
  along_long_axis             — move along the object's long axis
  perpendicular_to_long_axis  — move perpendicular to the object's long axis
  along_surface_normal        — move along the surface normal (e.g. straight out)
  opposite_approach           — retreat back the way the gripper came in
  unknown                     — action is PICK, or direction is unclear

target_part: if the grasp target is a specific part of a larger object (e.g.
  "handle" of a drawer), name it. Otherwise null.

Rules:
- Do NOT generate joint angles, trajectories, or motor commands
- Do NOT estimate metric 3D coordinates or quaternions
- Do NOT move the robot
- If you are not confident about grasp_relation/action/action_direction, use
  "unknown" rather than guessing — a wrong confident answer is worse than
  "unknown" because downstream code trusts these values directly.
- Return ONLY valid JSON — no markdown, no extra text

Required JSON format:
{
  "object": "<label>",
  "grasp_type": "<TOP|SIDE|PINCH>",
  "orientation": "<HORIZONTAL|VERTICAL>",
  "confidence": <0.0-1.0>,
  "reason": "<one concise sentence>",
  "target_part": "<part name or null>",
  "grasp_relation": "<perpendicular_to_long_axis|along_long_axis|perpendicular_to_surface|along_surface_normal|from_top|unknown>",
  "action": "<PICK|PULL|PUSH|SLIDE>",
  "action_direction": "<along_long_axis|perpendicular_to_long_axis|along_surface_normal|opposite_approach|unknown>"
}"""


# ── JSON 파싱 / 검증 ─────────────────────────────────────────────────────────
def _extract_json(text: str):
    """dict 또는 list를 반환. Qwen2.5-VL이 [{...}] 배열을 직접 출력하는 경우도 처리.
    전체 JSON 파싱 실패 시 개별 객체 항목을 정규식으로 추출하는 fallback 사용."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # 배열 우선 시도 (Qwen2.5-VL native format: [{...}])
    arr_start = text.find("[")
    obj_start = text.find("{")
    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        try:
            val, _ = json.JSONDecoder().raw_decode(text, arr_start)
            return val
        except json.JSONDecodeError:
            # 전체 배열 파싱 실패 → 개별 항목 정규식 fallback
            items = []
            for m in re.finditer(r'\{[^{}]*"label"\s*:\s*"([^"]+)"[^{}]*\}', text):
                try:
                    items.append(json.loads(m.group(0)))
                except json.JSONDecodeError:
                    pass
            if items:
                print(f"[_extract_json] fallback partial parse: {len(items)} items", flush=True)
                return items
    if obj_start != -1:
        val, _ = json.JSONDecoder().raw_decode(text, obj_start)
        return val
    raise ValueError(f"JSON 블록 없음: {text!r}")


def _validate(data: dict, label: str, elapsed: float) -> GraspResponse:
    grasp   = str(data.get("grasp_type", "")).upper()
    orient  = str(data.get("orientation", "HORIZONTAL")).upper()
    conf    = float(data.get("confidence", 0.0))

    if grasp not in ALLOWED_GRASP:
        raise ValueError(f"grasp_type 불허: {grasp!r}")
    if orient not in ALLOWED_ORIENT:
        orient = "VERTICAL" if grasp == "PINCH" else "HORIZONTAL"
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"confidence 범위 오류: {conf}")

    # [2026-08 추가] semantic intent 필드 -- 자유 텍스트를 그대로 믿지
    # 않는다. 허용 집합 밖이면 "unknown"으로 강등(모델 자체 실패로 보고
    # 전체를 fallback시키지 않는다 -- grasp_type/orientation만 있어도
    # 기존 downstream은 정상 동작하므로, 이 필드들은 "있으면 좋고 없어도
    # 무방한" 확장이어야 한다).
    target_part = data.get("target_part")
    if target_part is not None:
        target_part = str(target_part).strip() or None

    relation = str(data.get("grasp_relation", "unknown")).strip()
    if relation not in ALLOWED_RELATION:
        relation = "unknown"

    action = str(data.get("action", "PICK")).upper()
    if action not in ALLOWED_ACTION:
        action = "PICK"

    action_direction = str(data.get("action_direction", "unknown")).strip()
    if action_direction not in ALLOWED_ACTION_DIRECTION:
        action_direction = "unknown"

    return GraspResponse(
        object=str(data.get("object", label)),
        grasp_type=grasp,
        orientation=orient,
        confidence=round(conf, 3),
        reason=str(data.get("reason", "")),
        inference_ms=round(elapsed * 1000, 1),
        target_part=target_part,
        grasp_relation=relation,
        action=action,
        action_direction=action_direction,
    )


# _run_vlm에서 이미지를 480x360으로 리사이즈하므로 픽셀 좌표 정규화 기준값
_VLM_IMG_W = 480.0
_VLM_IMG_H = 360.0

# ── scene / placement 검증 ────────────────────────────────────────────────────
def _validate_scene(data, elapsed: float) -> SceneResponse:
    """dict({"objects":[...]}) 또는 list([{"label":..,"bbox"/"bbox_2d":..}]) 모두 허용.
    bbox가 픽셀 좌표(>1.0)이면 _VLM_IMG_W/_VLM_IMG_H 기준으로 정규화."""
    if isinstance(data, list):
        raw_objects = data
    else:
        raw_objects = data.get("objects", [])
        if not isinstance(raw_objects, list):
            raw_objects = []

    valid = []
    for o in raw_objects:
        if not isinstance(o, dict):
            continue
        label = str(o.get("label", "")).strip()
        if not label:
            continue
        # bbox / bbox_2d 둘 다 허용
        bbox = o.get("bbox") or o.get("bbox_2d", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bbox = [float(v) for v in bbox]
        # 픽셀 좌표 → 정규화 변환 (값이 1.0 초과이면 픽셀로 판단)
        if any(v > 1.0 for v in bbox):
            bbox = [
                bbox[0] / _VLM_IMG_W, bbox[1] / _VLM_IMG_H,
                bbox[2] / _VLM_IMG_W, bbox[3] / _VLM_IMG_H,
            ]
        bbox = [round(max(0.0, min(1.0, v)), 4) for v in bbox]
        valid.append({"label": label, "bbox": bbox})

    return SceneResponse(objects=valid, inference_ms=round(elapsed * 1000, 1))


def _validate_placement(data, elapsed: float) -> PlacementResponse:
    """list([{"bbox"/"bbox_2d":...}]) 또는 dict({"placement_regions":[...]}) 모두 허용.
    bbox가 픽셀 좌표(>1.0)이면 _VLM_IMG_W/_VLM_IMG_H 기준으로 정규화."""
    if isinstance(data, list):
        regions = data
    else:
        regions = data.get("placement_regions", [])
        if not isinstance(regions, list):
            regions = []

    valid = []
    for r in regions:
        if not isinstance(r, dict):
            continue
        bbox = r.get("bbox") or r.get("bbox_2d", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bbox = [float(v) for v in bbox]
        if any(v > 1.0 for v in bbox):
            bbox = [
                bbox[0] / _VLM_IMG_W, bbox[1] / _VLM_IMG_H,
                bbox[2] / _VLM_IMG_W, bbox[3] / _VLM_IMG_H,
            ]
        bbox = [round(max(0.0, min(1.0, v)), 4) for v in bbox]
        conf = round(float(r.get("confidence", 0.8)), 3)
        valid.append({"bbox": bbox, "confidence": conf})

    return PlacementResponse(placement_regions=valid, inference_ms=round(elapsed * 1000, 1))


def _validate_ground(data: dict, target_label: str, elapsed: float) -> GroundResponse:
    """bbox_norm / bbox_2d / bbox 키를 모두 허용. 픽셀 좌표(>1.0)이면 자동 정규화."""
    found = bool(data.get("found", False))
    conf  = float(data.get("confidence", 0.0))
    if not 0.0 <= conf <= 1.0:
        conf = 0.0
    # bbox_norm → bbox_2d → bbox 순서로 fallback
    bbox = data.get("bbox_norm") or data.get("bbox_2d") or data.get("bbox", [])
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = []
    else:
        bbox = [float(v) for v in bbox]
        # 픽셀 좌표이면 정규화
        if any(v > 1.0 for v in bbox):
            bbox = [
                bbox[0] / _VLM_IMG_W, bbox[1] / _VLM_IMG_H,
                bbox[2] / _VLM_IMG_W, bbox[3] / _VLM_IMG_H,
            ]
        bbox = [round(max(0.0, min(1.0, v)), 4) for v in bbox]
    # center_norm: 명시 값 우선, 없으면 bbox 중심에서 유도
    center = data.get("center_norm", [])
    if not isinstance(center, list) or len(center) != 2:
        center = []
    else:
        center = [float(v) for v in center]
        if any(v > 1.0 for v in center):
            center = [center[0] / _VLM_IMG_W, center[1] / _VLM_IMG_H]
        center = [round(max(0.0, min(1.0, v)), 4) for v in center]
    if found and len(bbox) == 4 and not center:
        center = [round((bbox[0] + bbox[2]) / 2, 4), round((bbox[1] + bbox[3]) / 2, 4)]
    if found and (len(bbox) != 4 or len(center) != 2):
        found = False
        conf  = 0.0
        bbox, center = [], []
    return GroundResponse(
        found=found,
        label=str(data.get("label", target_label)) if found else "",
        bbox_norm=bbox,
        center_norm=center,
        confidence=round(conf, 3),
        description=str(data.get("description", "")),
        inference_ms=round(elapsed * 1000, 1),
    )


def _fallback_scene(detections: list, elapsed: float) -> SceneResponse:
    objects = [{"label": d.get("label", "?"), "bbox": d.get("bbox", [])} for d in detections]
    return SceneResponse(objects=objects, inference_ms=round(elapsed * 1000, 1))


# ── 라벨 기반 폴백 (모델 없거나 추론 실패 시) ────────────────────────────────
_SIDE_LABELS  = {"cup", "bottle", "wine glass", "vase", "bowl"}
_PINCH_LABELS = {"pen", "pencil", "knife", "fork", "spoon", "scissors",
                 "toothbrush", "baseball bat", "remote", "cell phone",
                 "tie", "umbrella"}

def _fallback(label: str, elapsed: float) -> GraspResponse:
    ll = label.lower()
    if ll in _SIDE_LABELS:
        g, o = "SIDE", "VERTICAL"
    elif ll in _PINCH_LABELS:
        g, o = "PINCH", "VERTICAL"
    else:
        g, o = "TOP", "HORIZONTAL"
    return GraspResponse(
        object=label, grasp_type=g, orientation=o,
        confidence=0.4,
        reason=f"VLM unavailable — label-based heuristic for '{label}'",
        inference_ms=round(elapsed * 1000, 1),
    )


# ── 디버그 저장 ───────────────────────────────────────────────────────────────
def _save_debug(debug_dir: str, counter: list,
                full_img: Image.Image, crop_img: Image.Image,
                req: GraspRequest, result: GraspResponse) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    counter[0] += 1
    prefix = os.path.join(debug_dir, f"{counter[0]:04d}_{int(time.time())}")
    full_img.save(f"{prefix}_full.jpg")
    crop_img.save(f"{prefix}_crop.jpg")
    with open(f"{prefix}_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "label": req.object_label,
            "bbox": req.bbox,
            "result": result.dict(),
        }, f, ensure_ascii=False, indent=2)


# ── FastAPI 앱 빌드 ───────────────────────────────────────────────────────────
def build_app(port: int, model_name: str, debug: bool, debug_dir: str) -> FastAPI:
    app = FastAPI(title="VLM Grasp Inference Server")

    model = None
    processor = None
    _lock    = threading.Lock()
    _counter = [0]

    # 모델 로드
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[vlm :{port}] loading {model_name} on {device} …")
        t0 = time.time()

        # [2026-08 랩탑 실측] float16으로 로드하면 퇴화 생성("!!!!" 등) 발생 --
        # bfloat16으로 고쳐서 해결됨. GPU가 bfloat16을 지원해야 함(Ampere+).
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name)
        print(f"[vlm :{port}] ready — {time.time()-t0:.1f}s | {device}")

    except Exception as e:
        print(f"[vlm :{port}] [WARN] 모델 로드 실패: {e}")
        print(f"[vlm :{port}] /health 동작, /infer_grasp 는 fallback 반환")

    def _infer(full_img: Image.Image, crop_img: Image.Image, label: str) -> str:
        """Qwen2.5-VL 추론. 호출 전 _lock 획득 필요."""
        import torch

        # base64 URI 방식으로 이미지 전달 (qwen_vl_utils 불필요)
        def _to_b64_uri(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        messages = [{
            "role": "user",
            "content": [
                {"type": "text",  "text": "Full scene image:"},
                {"type": "image", "image": _to_b64_uri(full_img)},
                {"type": "text",  "text": f"Cropped target object '{label}':"},
                {"type": "image", "image": _to_b64_uri(crop_img)},
                {"type": "text",  "text": _SYSTEM + f'\n\nTarget label: "{label}"'},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # process_vision_info 사용 시도 (qwen_vl_utils 있으면 더 안정적)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
        except ImportError:
            inputs = processor(
                text=[text],
                images=[full_img, crop_img],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return processor.decode(new_tokens, skip_special_tokens=True)

    def _run_vlm(full_img: Image.Image, system_prompt: str, det_str: str,
                  max_new_tokens: int, tag: str, include_yolo: bool = True) -> str:
        """공통 VLM 추론 헬퍼. 호출 전 _lock 획득 필요."""
        import torch

        # visual token 억제: 480x360으로 resize (640x480 → ~937→870 input tokens)
        MAX_W, MAX_H = 480, 360
        if full_img.width > MAX_W or full_img.height > MAX_H:
            full_img = full_img.resize((MAX_W, MAX_H), Image.LANCZOS)

        def _to_b64_uri(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        if include_yolo:
            user_prompt = f"YOLO detected: {det_str}\n\n{system_prompt}"
        else:
            user_prompt = system_prompt

        messages = [{"role": "user", "content": [
            {"type": "text",  "text": "Camera image:"},
            {"type": "image", "image": _to_b64_uri(full_img)},
            {"type": "text",  "text": user_prompt},
        ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)
        except ImportError:
            inputs = processor(
                text=[text], images=[full_img],
                padding=True, return_tensors="pt",
            ).to(model.device)

        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        raw_out = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        generated = output_ids[0][input_len:].shape[0]
        print(f"[vlm :{port}] {tag} input={input_len} generated={generated} chars={len(raw_out)}", flush=True)
        print(f"[vlm :{port}] {tag} raw={raw_out!r}", flush=True)
        return raw_out

    def _infer_scene(full_img: Image.Image, detections: list) -> str:
        """object label+bbox만 반환. YOLO context 없이 VLM 독립 탐지. 호출 전 _lock 획득 필요."""
        return _run_vlm(full_img, _SCENE_SYSTEM, "[]", max_new_tokens=512, tag="analyze_scene", include_yolo=False)

    def _infer_placement(full_img: Image.Image, detections: list) -> str:
        """빈 배치 영역만 반환. 호출 전 _lock 획득 필요."""
        det_str = json.dumps([{"label": d.get("label"), "bbox": d.get("bbox")} for d in detections])
        return _run_vlm(full_img, _PLACEMENT_SYSTEM, det_str, max_new_tokens=128, tag="find_placement")

    def _infer_ground(full_img: Image.Image, target_label: str, detections: list) -> str:
        """특정 물체를 이미지에서 찾아 approximate bbox를 반환. 호출 전 _lock 획득 필요."""
        import torch

        # _run_vlm과 동일: 480x360으로 resize해 _VLM_IMG_W/H 기준 픽셀 좌표 정합성 확보
        MAX_W, MAX_H = 480, 360
        if full_img.width > MAX_W or full_img.height > MAX_H:
            full_img = full_img.resize((MAX_W, MAX_H), Image.LANCZOS)

        def _to_b64_uri(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        det_str = json.dumps(detections, ensure_ascii=False)
        user_prompt = (
            f"YOLO already detected: {det_str}\n\n"
            f"Target object to locate in the image: \"{target_label}\"\n\n"
            + _GROUNDING_SYSTEM
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text",  "text": "Camera image:"},
                {"type": "image", "image": _to_b64_uri(full_img)},
                {"type": "text",  "text": user_prompt},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)
        except ImportError:
            inputs = processor(
                text=[text], images=[full_img],
                padding=True, return_tensors="pt",
            ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return processor.decode(new_tokens, skip_special_tokens=True)

    # ── 엔드포인트 ────────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "port": port,
            "model": model_name,
            "model_loaded": model is not None,
        }

    @app.post("/infer_grasp", response_model=GraspResponse)
    def infer_grasp(req: GraspRequest):
        t0    = time.time()
        label = req.object_label.strip() or "object"

        # 이미지 디코드
        try:
            full_img = Image.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
            crop_img = Image.open(io.BytesIO(base64.b64decode(req.crop_image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코드 실패: {e}")

        if model is None or processor is None:
            result = _fallback(label, time.time() - t0)
        else:
            with _lock:   # 동시 요청 직렬화
                try:
                    raw = _infer(full_img, crop_img, label)
                    data = _extract_json(raw)
                    result = _validate(data, label, time.time() - t0)
                except Exception as e:
                    print(f"[vlm :{port}] 추론 실패 ({label}): {e}")
                    result = _fallback(label, time.time() - t0)

        if debug:
            _save_debug(debug_dir, _counter, full_img, crop_img, req, result)

        print(f"[vlm :{port}] {label} → {result.grasp_type}/{result.orientation} "
              f"conf={result.confidence:.2f} {result.inference_ms:.0f}ms")
        return result

    @app.post("/analyze_scene", response_model=SceneResponse)
    def analyze_scene(req: SceneRequest):
        t0 = time.time()
        try:
            full_img = Image.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코드 실패: {e}")

        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="vlm_model_not_loaded")

        with _lock:
            try:
                raw    = _infer_scene(full_img, req.detections)
                data   = _extract_json(raw)
                result = _validate_scene(data, time.time() - t0)
            except Exception as e:
                elapsed_ms = round((time.time() - t0) * 1000, 1)
                print(f"[vlm :{port}] analyze_scene 실패: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"vlm_inference_failed|{type(e).__name__}:{e}|elapsed={elapsed_ms}ms",
                )

        print(f"[vlm :{port}] analyze_scene — {result.inference_ms:.0f}ms objects={len(result.objects)}")
        return result

    @app.post("/find_placement", response_model=PlacementResponse)
    def find_placement(req: PlacementRequest):
        t0 = time.time()
        try:
            full_img = Image.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코드 실패: {e}")

        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="vlm_model_not_loaded")

        with _lock:
            try:
                raw    = _infer_placement(full_img, req.detections)
                data   = _extract_json(raw)
                result = _validate_placement(data, time.time() - t0)
            except Exception as e:
                elapsed_ms = round((time.time() - t0) * 1000, 1)
                print(f"[vlm :{port}] find_placement 실패: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"vlm_inference_failed|{type(e).__name__}:{e}|elapsed={elapsed_ms}ms",
                )

        print(f"[vlm :{port}] find_placement — {result.inference_ms:.0f}ms regions={len(result.placement_regions)}")
        return result

    @app.post("/ground_object", response_model=GroundResponse)
    def ground_object_endpoint(req: GroundRequest):
        t0 = time.time()

        try:
            full_img = Image.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코드 실패: {e}")

        if model is None or processor is None:
            return GroundResponse(
                found=False, label="", confidence=0.0,
                description="VLM model not loaded",
                inference_ms=round((time.time() - t0) * 1000, 1),
            )

        with _lock:
            try:
                raw  = _infer_ground(full_img, req.target_label, req.detections)
                print(f"[vlm :{port}] ground_object raw={repr(raw)}", flush=True)
                data = _extract_json(raw)
                result = _validate_ground(data, req.target_label, time.time() - t0)
            except Exception as e:
                print(f"[vlm :{port}] ground_object 실패 ({req.target_label}): {e}")
                result = GroundResponse(
                    found=False, label="", confidence=0.0,
                    description=f"inference_error: {e}",
                    inference_ms=round((time.time() - t0) * 1000, 1),
                )

        status = "found" if result.found else "not_found"
        print(f"[vlm :{port}] ground_object({req.target_label}) → {status} "
              f"conf={result.confidence:.2f} {result.inference_ms:.0f}ms")
        return result

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",      type=str, default="127.0.0.1")
    ap.add_argument("--port",      type=int, default=8003)
    ap.add_argument("--model",     default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--debug",     action="store_true")
    ap.add_argument("--debug-dir", default="/tmp/vlm_grasp_debug")
    args = ap.parse_args()

    application = build_app(args.port, args.model, args.debug, args.debug_dir)
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")
