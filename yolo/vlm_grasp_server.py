#!/usr/bin/env python3
"""
vlm_grasp_server.py — grasp / scene / placement / grounding 추론 서버
===========================================================================

⚠️⚠️  연구실 PC / 다른 워킹카피에서 이 서버를 띄우는 사람에게  ⚠️⚠️
-----------------------------------------------------------------------------
2026-09-01 이 파일은 **완전히 다른 구조로 리팩터됐다.**
  구버전  : 이 프로세스가 transformers로 Qwen2.5-VL-3B 웨이트를 직접 로드
            (`--model Qwen/Qwen2.5-VL-3B-Instruct`), GPU 필요, 로딩 수십 초.
  신버전  : 웨이트를 안 올린다. 별도 vLLM OpenAI 서버(8005, Qwen3-VL-8B)에
            HTTP로 요청만 던지는 얇은 어댑터.

  → **반드시 `git pull` 로 최신본을 받아 이 파일을 쓸 것.** 캐시된 옛
    파일이나 옛 배포본(`*.pyc`, 복사해둔 사본, 다른 워킹카피의 stale
    체크아웃)을 실행하지 마라. 옛 파일은 `--vllm-url` 인자를 모르고
    (argparse 에러), transformers/torch import부터 다르다.
  → 신버전을 쓰려면 먼저 vLLM 서버가 떠 있어야 한다:
        ~/vllm-venv/serve_qwen3vl.sh      (Thor, tmux)
    안 떠 있으면 /infer_grasp 는 라벨 휴리스틱 fallback, scene/placement/
    ground 는 503 을 준다.
  → 요청/응답 JSON 스키마·엔드포인트는 구/신 동일하다. 클라이언트
    (`mcp_robot_server.py`)는 손댈 필요 없다.
-----------------------------------------------------------------------------

이 서버는 **로컬에서 VLM 웨이트를 직접 로드하지 않는다.** 대신 별도로
떠 있는 vLLM OpenAI 호환 서버(기본 http://127.0.0.1:8005/v1, Qwen3-VL-8B)를
HTTP로 호출하는 얇은 어댑터다.

- 요청/응답 스키마는 기존(transformers 직접 로드) 버전과 100% 동일 →
  `mcp_robot_server.py` 등 클라이언트 무변경.
- ON-DEMAND: 엔드포인트 호출 시에만 vLLM에 요청 (카메라 스트림 상시 소비 없음)
- 카메라 device 직접 접근 없음 — base64 이미지만 수신
- 실제 로봇 동작 없음 — JSON 응답만 반환
- grasp_type: TOP | SIDE | PINCH

레버 1 (출력 토큰 감축, 2026-09-01):
- /infer_grasp 는 vLLM guided JSON(`response_format=json_schema`)으로
  구조화 필드만 생성. `reason` 필드는 구조화 필드 + visual_analysis 에서
  서버가 합성한다(클라이언트 하위호환 유지).

grasp 정확도 개선 (2026-09-01, 3B bf16 회귀 후 grip 형태 오판 대응):
- (a) crop 이미지를 `_fit`(강제 480x360, 종횡비 파괴) 대신 `_pad_square`로
  종횡비 보존 + 정사각 패딩. orientation/grasp_type 판단의 핵심 신호 복원.
- (b) `_GRASP_SCHEMA` 맨 앞에 `visual_analysis`(짧은 crop 묘사) 필드 추가
  → CoT-lite. `GRASP_MAX_TOKENS` 128 → 224.
- (c) 시스템 프롬프트에 텍스트 few-shot 5개(crop 실루엣 → enum).
- (d) 스키마 필드 순서 재배열(핵심 3필드를 앞, confidence/target_part 뒤).

실행:
    # 먼저 vLLM 서버를 띄운다 (별도 tmux):
    #   ~/vllm-venv/serve_qwen3vl.sh
    # 그다음:
    python vlm_grasp_server.py --port 8003
    python vlm_grasp_server.py --port 8003 --vllm-url http://127.0.0.1:8005/v1
    python vlm_grasp_server.py --port 8003 --debug --debug-dir /tmp/vlm_debug

환경변수:
    VLLM_BASE_URL   기본 http://127.0.0.1:8005/v1
    VLLM_MODEL      기본 qwen3-vl  (serve_qwen3vl.sh 의 --served-model-name)
    VLLM_TIMEOUT    기본 60 (초)
    GRASP_MAX_TOKENS 기본 224
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

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ── vLLM OpenAI 서버 설정 ────────────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8005/v1").rstrip("/")
VLLM_MODEL    = os.environ.get("VLLM_MODEL", "qwen3-vl")
VLLM_TIMEOUT  = float(os.environ.get("VLLM_TIMEOUT", "60"))
GRASP_MAX_TOKENS = int(os.environ.get("GRASP_MAX_TOKENS", "224"))

_session = requests.Session()


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
    approach_direction: str = "FRONT"   # FRONT | LEFT | RIGHT | BACK (카메라 시점)
    confidence: float
    reason: str                   # 구조화 필드에서 서버가 합성 (자유문장 아님)
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
    objects: list = []            # [{"label":"cup","bbox":[x1,y1,x2,y2],
                                  #   "container_type":"none","is_open":false}, ...]
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

class OpenVocabRequest(BaseModel):
    full_image_b64: str
    labels: list                  # 찾을 라벨 문자열 목록 (open-vocab 프롬프트)
    conf: float = 0.25            # confidence threshold. YOLO-World 는 낮추면
                                  # 오검출을 뿌린다(레포 vlm_grounding_dino.py
                                  # 메모와 동일 성격) -- 호출부는 보통 top-1
                                  # 만 쓰므로 0.25 정도가 무난. 필요 시 조정.
    timestamp: Optional[float] = None

class OpenVocabResponse(BaseModel):
    detections: list = []         # [{"label":..,"bbox":[x1,y1,x2,y2],"confidence":..}]
    backend: str = "yoloworld"
    inference_ms: float = 0.0

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
ALLOWED_APPROACH = {"FRONT", "LEFT", "RIGHT", "BACK"}
ALLOWED_CONTAINER = {"basket", "bin", "tray", "box", "drawer", "shelf", "bowl", "none"}

# [2026-08 추가] sj_pickplace/grasp_types.py의 GRASP_RELATIONS/ACTIONS/
# ACTION_DIRECTIONS와 동일 집합 -- 이 파일은 독립 배포 스크립트(ROS
# 워크스페이스 밖)라 import 대신 로컬에 중복 정의한다. 한쪽만 바꾸면
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

# ── guided JSON 스키마 (vLLM response_format) ────────────────────────────────
# grasp만 스키마를 강제한다 (레버 1: 구조화 필드만 생성, 자유문장 reason 제거).
# scene/placement/ground 는 배열/픽셀좌표 처리가 까다로워 기존 자유생성 +
# _extract_json 유지.
# 필드 순서 = xgrammar 생성 순서. (b) 맨 앞 visual_analysis 로 CoT-lite 유도
# (3B는 숙고 스텝 없이 enum부터 뱉으면 grasp_type/orientation 오판이 잦음),
# (d) 핵심 3필드(grasp_type/orientation/approach)를 덜 중요한 필드보다 앞으로,
# confidence/target_part는 뒤로.
_GRASP_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_analysis":    {"type": "string"},
        "grasp_type":         {"type": "string", "enum": sorted(ALLOWED_GRASP)},
        "orientation":        {"type": "string", "enum": sorted(ALLOWED_ORIENT)},
        "approach_direction": {"type": "string", "enum": ["FRONT", "LEFT", "RIGHT", "BACK"]},
        "grasp_relation":     {"type": "string", "enum": sorted(ALLOWED_RELATION)},
        "action":             {"type": "string", "enum": sorted(ALLOWED_ACTION)},
        "action_direction":   {"type": "string", "enum": sorted(ALLOWED_ACTION_DIRECTION)},
        "confidence":         {"type": "number"},
        "target_part":        {"type": ["string", "null"]},
    },
    "required": ["visual_analysis", "grasp_type", "orientation", "approach_direction",
                 "grasp_relation", "action", "action_direction", "confidence"],
    "additionalProperties": False,
}

# ── 시스템 프롬프트 ──────────────────────────────────────────────────────────
_SCENE_SYSTEM = """\
Look at the camera image independently and detect every distinct physical object that is actually visible.

Rules:
- Detect objects based ONLY on what you see in the image. Do NOT copy, reuse, or reference any object names or coordinates from this prompt.
- Include manipulable objects: cups, boxes, scissors, books, phones, speakers, bags, markers, pens, pencils, bottles, containers, wallets, cables, and any other physical item.
- ALSO include placement receptacles that an object could be put into or onto: baskets, bins, trays, boxes, drawers, shelves, racks, bowls.
- Exclude background surfaces: tables, floors, walls, the robot arm.
- Do NOT rely on any prior detector results.

For each object also report:
- "container_type": one of "basket","bin","tray","box","drawer","shelf","bowl","none".
  Use "none" for anything that is not a receptacle you could place an item into/onto.
- "is_open": true if it is a receptacle with an accessible opening facing roughly toward the camera (so the robot could drop something in), false otherwise. Use false for "none".

Return ONLY minified JSON with this schema — no markdown, no extra text:
{"objects":[{"label":"<name>","bbox":[x_min,y_min,x_max,y_max],"container_type":"none","is_open":false}]}
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

HARD RULE: if the target is a cup, mug, glass, tumbler, bowl, bottle, vase,
jar or can, grasp_type is SIDE and orientation is VERTICAL — never TOP —
unless it is visibly tipped over on its side. A top-down grasp on an open
container closes on the rim or empty air.

Determine the most suitable grasp for a parallel-jaw gripper. A JSON schema
is enforced on your answer. The FIRST field, "visual_analysis", MUST be a
brief (<= 15 words) factual phrase: the object's shape in Image 2 and which
surface a gripper would close on (e.g. "thin elongated tool, closed on its
narrow shaft"). The user message states the crop's measured aspect
(wider-than-tall / taller-than-wide / near-square) — trust that measurement
over your own eyeballing. Then fill every other field CONSISTENTLY with it.
Output the JSON as a single line with no newlines and no indentation.

grasp_type (typical cases, not fixed rules — verify against the image):
  TOP   — gripper descends from above; solid objects lying flat/wide with a
          usable flat top face (box, book lying flat, phone flat on a table)
  SIDE  — horizontal approach onto a side wall; tall/upright objects
          (standing book, bottle, vase) AND any open-top container
          (cup, mug, glass, bowl, can) even when its crop looks near-square —
          a TOP grasp there just closes on the rim or empty air.
  PINCH — gripper rotated ~90° vertical; thin/elongated items held
          between fingertips (pen, knife, fork, spoon, scissors,
          toothbrush, remote, phone standing on edge, credit card)

orientation: follow the crop's measured aspect — WIDER-than-tall crop →
  HORIZONTAL, TALLER-than-wide crop → VERTICAL. Only pick the opposite if the
  object is plainly rotated diagonally inside the crop. For a near-square crop,
  use the object's own long axis (a mug/can standing up = VERTICAL).
  NOTE: PINCH does NOT force VERTICAL — a pen lying flat is PINCH + HORIZONTAL.

approach_direction — from which side the gripper should come in, in the
  CAMERA's view of Image 1:
  FRONT — from the near side, toward the camera (use this when unsure)
  LEFT / RIGHT — from the object's left / right
  BACK  — from the far side

You do NOT compute metric angles, coordinates, or quaternions — a separate
geometry module measures the object's actual 3D shape. Describe the grasp
SEMANTICALLY, relative to the object's own shape/surface.

grasp_relation (how the gripper relates to the object's geometry):
  perpendicular_to_long_axis — elongated object (handle, bottle, pen); grip
                                across its short dimension, not along its length
  along_long_axis            — grip along the long dimension (rare)
  perpendicular_to_surface   — approach straight into a flat face
  along_surface_normal       — same idea, for TOP grasps
  from_top                   — no strong shape cue, just come from above
  unknown                    — none of the above clearly applies

action (what happens after the grasp):
  PICK  — lift the object up and away
  PULL  — grasp then pull (e.g. a drawer/door handle)
  PUSH  — grasp then push
  SLIDE — grasp then slide sideways without lifting

action_direction (only meaningful when action != PICK):
  along_long_axis | perpendicular_to_long_axis | along_surface_normal |
  opposite_approach | unknown

target_part: if the grasp target is a specific part of a larger object
  (e.g. "handle" of a drawer), name it. Otherwise null.

Worked examples (shape + measured crop aspect → fields):
  bottle, cylindrical side wall, crop taller-than-wide
     → grasp_type=SIDE, orientation=VERTICAL, grasp_relation=perpendicular_to_long_axis
  book lying flat, flat rectangular top face, crop wider-than-tall
     → grasp_type=TOP, orientation=HORIZONTAL, grasp_relation=from_top
  scissors / pen / knife / spoon, thin elongated tool, crop wider-than-tall
     → grasp_type=PINCH, orientation=HORIZONTAL, grasp_relation=perpendicular_to_long_axis
  scissors / pen standing up, thin elongated tool, crop taller-than-wide
     → grasp_type=PINCH, orientation=VERTICAL, grasp_relation=perpendicular_to_long_axis
  mug/can standing upright, curved side wall + opening on top, crop near-square
     → grasp_type=SIDE, orientation=VERTICAL, grasp_relation=perpendicular_to_long_axis
  cube / small box, flat top face, crop near-square
     → grasp_type=TOP, orientation=HORIZONTAL, grasp_relation=from_top

If you are not confident about grasp_relation / action / action_direction,
use "unknown" / "PICK" rather than guessing — downstream code trusts these
values directly, so a wrong confident answer is worse than "unknown"."""


# ── 이미지 유틸 ──────────────────────────────────────────────────────────────
# _fit()이 이미지를 480x360으로 리사이즈하므로 픽셀 좌표 정규화 기준값
_VLM_IMG_W = 480.0
_VLM_IMG_H = 360.0
_MAX_W, _MAX_H = 480, 360


def _fit(img: Image.Image) -> Image.Image:
    """visual token 억제: 480x360으로 resize (640x480 → ~937→870 input tokens).
    _validate_scene/_validate_placement/_validate_ground 의 픽셀→정규화 변환이
    정확히 480x360 기준이므로 종횡비 무시하고 강제 리사이즈한다."""
    if img.width > _MAX_W or img.height > _MAX_H:
        img = img.resize((_MAX_W, _MAX_H), Image.LANCZOS)
    return img


def _pad_square(img: Image.Image, size: int = 448) -> Image.Image:
    """(a) infer_grasp crop 전용. `_fit`(강제 480x360, 종횡비 파괴)은 쓰지
    않는다 — grasp_type/orientation 판단은 crop 속 물체의 종횡비(세로로 긴가
    가로로 긴가)가 핵심이라, 눌러서 왜곡하면 세로 병이 정사각처럼 보여 SIDE↔TOP
    을 계속 틀린다. 종횡비를 보존한 채 긴 변을 size에 맞추고(작은 crop은 업스케일
    해 3B가 디테일을 보게 함) 정사각으로 회색 레터박스 패딩한다. 이 엔드포인트는
    픽셀 좌표를 반환하지 않으므로 _validate_scene류의 480x360 정규화 제약과 무관.
    size=448 은 Qwen2.5-VL 패치 격자(28px)의 배수."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _img_to_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── vLLM 호출 ────────────────────────────────────────────────────────────────
def _chat(messages: list, *, max_tokens: int, schema: Optional[dict] = None,
          tag: str = "out") -> tuple:
    """vLLM OpenAI /chat/completions 호출. (content_str, usage_dict) 반환.
    실패 시 예외를 그대로 올린다 — 호출부가 fallback/HTTP 에러로 매핑한다."""
    body = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": tag, "schema": schema},
        }
    r = _session.post(f"{VLLM_BASE_URL}/chat/completions", json=body, timeout=VLLM_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    content = (j["choices"][0]["message"].get("content") or "").strip()
    usage = j.get("usage", {}) or {}
    return content, usage


def _vllm_alive() -> bool:
    try:
        r = _session.get(f"{VLLM_BASE_URL}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── YOLO-World (open-vocab 검출) ─────────────────────────────────────────────
# [2026-09-02 추가] YOLO가 못 잡는 라벨을 8B VLM /ground_object(2~4초, chat형
# 이라 좌표 정밀도 낮음) 대신 open-vocab 검출기로 빠르게(~30~50ms) 잡는다.
# 이 서버는 원래 "로컬 웨이트 안 올리는 얇은 어댑터"지만, 사용자 판단으로
# 경량 perception 모델은 Thor 에 co-locate 하기로 함(2026-09-02). ultralytics
# 미설치/로드 실패해도 서버는 정상 기동하고 이 엔드포인트만 503 을 준다.
_YW_MODEL_PATH = os.environ.get("YOLOWORLD_MODEL", "yolov8l-worldv2.pt")
_yw_model = None
_yw_lock = threading.Lock()
_yw_failed = False
_yw_last_classes: list = []


def _yoloworld():
    global _yw_model, _yw_failed
    if _yw_model is not None or _yw_failed:
        return _yw_model
    with _yw_lock:
        if _yw_model is not None or _yw_failed:
            return _yw_model
        try:
            import numpy as _np
            import torch as _torch
            from ultralytics import YOLOWorld
            mdl = YOLOWorld(_YW_MODEL_PATH)
            # YOLOWorld 는 CPU 로 로드되고 predict 때 GPU 로 옮겨진다. 그
            # 순서면 첫 set_classes 가 CLIP 을 CPU 에 캐시해서 이후 class
            # 스위칭이 device mismatch 로 죽는다. 로드 직후 GPU 로 강제
            # 이동해서, 이후 모든 set_classes(get_text_pe)가 CLIP 을 GPU 에
            # 빌드/캐시하도록 한다.
            _dev = "cuda" if _torch.cuda.is_available() else "cpu"
            try:
                mdl.model.to(_dev)
            except Exception:
                pass
            mdl.set_classes(["object"])                       # CLIP 캐시를 _dev 에
            mdl.predict(_np.zeros((64, 64, 3), _np.uint8), verbose=False)  # backend 워밍업
            _yw_model = mdl
            print(f"[yoloworld] loaded + warmed on {_dev}: {_YW_MODEL_PATH}", flush=True)
        except Exception as e:
            _yw_failed = True
            print(f"[yoloworld] load 실패: {type(e).__name__}: {e}", flush=True)
        return _yw_model


def _yoloworld_detect(img: Image.Image, labels: list, conf: float) -> list:
    """(label, [x1,y1,x2,y2] 정규화, confidence) 목록. 실패 시 예외."""
    global _yw_last_classes
    model = _yoloworld()
    if model is None:
        raise RuntimeError("yoloworld_unavailable")
    import numpy as np
    arr = np.array(img)[:, :, ::-1]  # RGB->BGR
    h, w = arr.shape[:2]
    classes = [str(x).strip() for x in labels if str(x).strip()]
    if not classes:
        raise ValueError("empty_labels")
    if classes != _yw_last_classes:
        # ultralytics YOLO-World 8.4 device 이슈 대응(로더에서 모델을 미리
        # GPU 로 올려 CLIP 캐시가 GPU 에 잡히게 한 것과 한 세트):
        #   - set_classes 후 txt_feats 를 모델 device 로 이동
        #   - predictor 폐기 → 새 클래스로 AutoBackend 재초기화
        # (안 하면 self.model 과 persistent predictor.model 사이 클래스/
        #  device 불일치로 predict 가 죽는다.)
        wm = model.model
        model.set_classes(classes)
        try:
            dev = next(wm.parameters()).device
            if getattr(wm, "txt_feats", None) is not None:
                wm.txt_feats = wm.txt_feats.to(dev)
        except Exception:
            pass
        model.predictor = None
        _yw_last_classes = classes
    res = model.predict(arr, conf=conf, verbose=False)[0]
    out = []
    for b in res.boxes:
        xa, ya, xb, yb = [float(v) for v in b.xyxy[0].tolist()]
        out.append({
            "label": classes[int(b.cls[0])],
            "bbox": [round(xa / w, 4), round(ya / h, 4),
                     round(xb / w, 4), round(yb / h, 4)],
            "confidence": round(float(b.conf[0]), 3),
        })
    return out


# ── JSON 파싱 / 검증 ─────────────────────────────────────────────────────────
def _extract_json(text: str):
    """dict 또는 list를 반환. VLM이 [{...}] 배열을 직접 출력하는 경우도 처리.
    전체 JSON 파싱 실패 시 개별 객체 항목을 정규식으로 추출하는 fallback 사용."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # 배열 우선 시도 (native format: [{...}])
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


def _synth_reason(grasp: str, orient: str, relation: str, action: str) -> str:
    """레버 1: 자유문장 reason 생성을 없앤 대신, 구조화 필드에서 짧은
    설명 문자열을 서버가 합성한다 (클라이언트가 로그/표시용으로 읽음)."""
    parts = [f"{grasp} grasp"]
    if orient:
        parts.append(orient.lower())
    if relation and relation != "unknown":
        parts.append(relation.replace("_", " "))
    if action and action != "PICK":
        parts.append(f"then {action.lower()}")
    return ", ".join(parts)


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

    approach = str(data.get("approach_direction", "FRONT")).upper()
    if approach not in ALLOWED_APPROACH:
        approach = "FRONT"

    # [2026-08 추가] semantic intent 필드 -- 자유 텍스트를 그대로 믿지
    # 않는다. 허용 집합 밖이면 안전값으로 강등(모델 자체 실패로 보고
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

    # (b) CoT-lite: 모델이 crop을 실제로 묘사한 문장. 로그/표시용으로만 쓰이므로
    # (downstream geometry는 enum만 신뢰) 그대로 reason 에 덧붙인다.
    visual = str(data.get("visual_analysis", "")).strip().replace("\n", " ")[:200]

    return GraspResponse(
        object=label,
        grasp_type=grasp,
        orientation=orient,
        approach_direction=approach,
        confidence=round(conf, 3),
        reason=(_synth_reason(grasp, orient, relation, action)
                + (f" — {visual}" if visual else "")),
        inference_ms=round(elapsed * 1000, 1),
        target_part=target_part,
        grasp_relation=relation,
        action=action,
        action_direction=action_direction,
    )


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

        # [2026-09-02] 배치 리셉터클 태그 -- 허용집합 밖/누락이면 안전값.
        ctype = str(o.get("container_type", "none")).strip().lower()
        if ctype not in ALLOWED_CONTAINER:
            ctype = "none"
        is_open = bool(o.get("is_open", False)) if ctype != "none" else False

        valid.append({"label": label, "bbox": bbox,
                      "container_type": ctype, "is_open": is_open})

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


# ── 라벨 기반 폴백 (vLLM 서버 없거나 추론 실패 시) ──────────────────────────
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
        approach_direction="FRONT",
        confidence=0.4,
        reason=f"VLM unavailable — label-based heuristic for '{label}'",
        inference_ms=round(elapsed * 1000, 1),
    )


# ── 디버그 저장 ───────────────────────────────────────────────────────────────
def _save_debug(debug_dir: str, counter: list, lock: threading.Lock,
                full_img: Image.Image, crop_img: Image.Image,
                req: GraspRequest, result: GraspResponse) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    with lock:
        counter[0] += 1
        idx = counter[0]
    prefix = os.path.join(debug_dir, f"{idx:04d}_{int(time.time())}")
    full_img.save(f"{prefix}_full.jpg")
    crop_img.save(f"{prefix}_crop.jpg")
    with open(f"{prefix}_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "label": req.object_label,
            "bbox": req.bbox,
            "result": result.model_dump(),
        }, f, ensure_ascii=False, indent=2)


# ── FastAPI 앱 빌드 ───────────────────────────────────────────────────────────
def build_app(port: int, debug: bool, debug_dir: str) -> FastAPI:
    app = FastAPI(title="VLM Grasp Inference Server (vLLM adapter)")

    _counter  = [0]
    _dbg_lock = threading.Lock()

    print(f"[vlm :{port}] adapter → vLLM {VLLM_BASE_URL} (model={VLLM_MODEL})")
    if _vllm_alive():
        print(f"[vlm :{port}] vLLM reachable — ready")
    else:
        print(f"[vlm :{port}] [WARN] vLLM 서버 미응답 — /health 동작, "
              f"/infer_grasp 는 fallback 반환, scene/placement/ground 는 503")

    def _infer_grasp_vlm(full_img: Image.Image, crop_img: Image.Image, label: str) -> str:
        """grasp 추론 — guided JSON. crop은 종횡비 보존(_pad_square) + 측정된
        crop aspect를 텍스트로 같이 전달, 맨 앞 visual_analysis 로 CoT-lite."""
        cw, ch = crop_img.size
        ar = cw / ch if ch else 1.0
        if   ar >= 1.18: aspect = f"WIDER-than-tall ({cw}x{ch}px, {ar:.2f}:1)"
        elif ar <= 0.85: aspect = f"TALLER-than-wide ({cw}x{ch}px, 1:{1/ar:.2f})"
        else:            aspect = f"NEAR-SQUARE ({cw}x{ch}px, {ar:.2f}:1)"

        full_img = _fit(full_img)
        crop_img = _pad_square(crop_img, 448)   # (a) 종횡비 보존
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": [
                {"type": "text",      "text": "Image 1 — full scene (context, obstacles, approach direction):"},
                {"type": "image_url", "image_url": {"url": _img_to_uri(full_img)}},
                {"type": "text",      "text": f"Image 2 — cropped target object '{label}'. "
                                              "Grey borders are square padding, ignore them. "
                                              f"Measured crop aspect: {aspect}."},
                {"type": "image_url", "image_url": {"url": _img_to_uri(crop_img)}},
                {"type": "text",      "text": f'Target label: "{label}". Measured crop aspect: {aspect}.'},
            ]},
        ]
        out, usage = _chat(messages, max_tokens=GRASP_MAX_TOKENS,
                           schema=_GRASP_SCHEMA, tag="grasp")
        print(f"[vlm :{port}] infer_grasp prompt_tok={usage.get('prompt_tokens')} "
              f"completion_tok={usage.get('completion_tokens')} raw={out!r}", flush=True)
        return out

    def _run_vlm(full_img: Image.Image, system_prompt: str, det_str: str,
                 max_new_tokens: int, tag: str, include_yolo: bool = True) -> str:
        """공통 단일 이미지 VLM 추론 헬퍼 (scene / placement)."""
        full_img = _fit(full_img)
        if include_yolo:
            user_prompt = f"YOLO detected: {det_str}\n\n{system_prompt}"
        else:
            user_prompt = system_prompt
        messages = [{"role": "user", "content": [
            {"type": "text",      "text": "Camera image:"},
            {"type": "image_url", "image_url": {"url": _img_to_uri(full_img)}},
            {"type": "text",      "text": user_prompt},
        ]}]
        out, usage = _chat(messages, max_tokens=max_new_tokens, tag=tag)
        print(f"[vlm :{port}] {tag} prompt_tok={usage.get('prompt_tokens')} "
              f"completion_tok={usage.get('completion_tokens')} chars={len(out)}", flush=True)
        print(f"[vlm :{port}] {tag} raw={out!r}", flush=True)
        return out

    def _infer_scene(full_img: Image.Image, detections: list) -> str:
        """object label+bbox만 반환. YOLO context 없이 VLM 독립 탐지."""
        return _run_vlm(full_img, _SCENE_SYSTEM, "[]", max_new_tokens=512,
                        tag="analyze_scene", include_yolo=False)

    def _infer_placement(full_img: Image.Image, detections: list) -> str:
        """빈 배치 영역만 반환."""
        det_str = json.dumps([{"label": d.get("label"), "bbox": d.get("bbox")} for d in detections])
        return _run_vlm(full_img, _PLACEMENT_SYSTEM, det_str, max_new_tokens=128,
                        tag="find_placement")

    def _infer_ground(full_img: Image.Image, target_label: str, detections: list) -> str:
        """특정 물체를 이미지에서 찾아 approximate bbox를 반환."""
        full_img = _fit(full_img)
        det_str = json.dumps(detections, ensure_ascii=False)
        user_prompt = (
            f"YOLO already detected: {det_str}\n\n"
            f"Target object to locate in the image: \"{target_label}\"\n\n"
            + _GROUNDING_SYSTEM
        )
        messages = [{"role": "user", "content": [
            {"type": "text",      "text": "Camera image:"},
            {"type": "image_url", "image_url": {"url": _img_to_uri(full_img)}},
            {"type": "text",      "text": user_prompt},
        ]}]
        out, usage = _chat(messages, max_tokens=256, tag="ground_object")
        print(f"[vlm :{port}] ground_object prompt_tok={usage.get('prompt_tokens')} "
              f"completion_tok={usage.get('completion_tokens')}", flush=True)
        return out

    # ── 엔드포인트 ────────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "port": port,
            "model": VLLM_MODEL,
            "vllm_url": VLLM_BASE_URL,
            "model_loaded": _vllm_alive(),
            "yoloworld_loaded": _yw_model is not None,
        }

    @app.post("/detect_open_vocab", response_model=OpenVocabResponse)
    def detect_open_vocab(req: OpenVocabRequest):
        """open-vocab 검출 (YOLO-World). YOLO가 못 잡는 라벨의 정밀 bbox 를
        8B VLM 대신 ~30~50ms 로 얻기 위한 티어. ultralytics 미설치/로드 실패
        시 503 -- 호출부는 그때만 VLM /ground_object 로 폴백하면 된다."""
        t0 = time.time()
        try:
            img = Image.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코드 실패: {e}")
        try:
            dets = _yoloworld_detect(img, req.labels, req.conf)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=f"open_vocab_unavailable|{e}")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"open_vocab_failed|{type(e).__name__}:{e}")
        elapsed = round((time.time() - t0) * 1000, 1)
        print(f"[vlm :{port}] detect_open_vocab labels={req.labels} → {len(dets)} dets {elapsed:.0f}ms")
        return OpenVocabResponse(detections=dets, backend="yoloworld", inference_ms=elapsed)

    @app.post("/infer_grasp", response_model=GraspResponse)
    def infer_grasp(req: GraspRequest):
        t0    = time.time()
        label = req.object_label.strip() or "object"

        try:
            full_img = Image.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
            crop_img = Image.open(io.BytesIO(base64.b64decode(req.crop_image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코드 실패: {e}")

        try:
            raw    = _infer_grasp_vlm(full_img, crop_img, label)
            data   = _extract_json(raw)
            result = _validate(data, label, time.time() - t0)
        except Exception as e:
            print(f"[vlm :{port}] 추론 실패 ({label}): {type(e).__name__}: {e}")
            result = _fallback(label, time.time() - t0)

        if debug:
            _save_debug(debug_dir, _counter, _dbg_lock, full_img, crop_img, req, result)

        print(f"[vlm :{port}] {label} → {result.grasp_type}/{result.orientation} "
              f"approach={result.approach_direction} conf={result.confidence:.2f} "
              f"{result.inference_ms:.0f}ms")
        return result

    @app.post("/analyze_scene", response_model=SceneResponse)
    def analyze_scene(req: SceneRequest):
        t0 = time.time()
        try:
            full_img = Image.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코드 실패: {e}")

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
    ap.add_argument("--host",       type=str, default="127.0.0.1")
    ap.add_argument("--port",       type=int, default=8003)
    ap.add_argument("--vllm-url",   default=None,
                    help="vLLM OpenAI base URL (default: $VLLM_BASE_URL or http://127.0.0.1:8005/v1)")
    ap.add_argument("--vllm-model", default=None,
                    help="vLLM served model name (default: $VLLM_MODEL or qwen3-vl)")
    ap.add_argument("--model",      default=None,
                    help="(deprecated) 무시됨 — 이 서버는 로컬 웨이트를 로드하지 않는다. "
                         "vLLM 모델명은 --vllm-model 사용.")
    ap.add_argument("--debug",      action="store_true")
    ap.add_argument("--debug-dir",  default="/tmp/vlm_grasp_debug")
    args = ap.parse_args()

    if args.model:
        print(f"[vlm] --model {args.model!r} 은 이 버전에서 무시됨 "
              f"(vLLM 어댑터 — 로컬 로드 없음). --vllm-model 참고.")
    if args.vllm_url:
        VLLM_BASE_URL = args.vllm_url.rstrip("/")
    if args.vllm_model:
        VLLM_MODEL = args.vllm_model

    application = build_app(args.port, args.debug, args.debug_dir)
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")
