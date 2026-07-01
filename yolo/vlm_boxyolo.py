#!/usr/bin/env python3
"""
vlm_boxyolo.py  -  박스 전용 커스텀 YOLO 추론 서버
=================================================================================
- vlm_yoloworld.py / vlm_moondream.py와 API 스키마 100% 호환 (perception_node 코드 변경 없음)
- 차이점: 오픈보캡(YOLO-World/CLIP) 대신, 커스텀 학습된 단일/소수 클래스 YOLO 가중치를 로드.
  → set_classes() 호출이나 텍스트 인코더가 없어서 구조가 더 가볍고 빠름.

[병합 이력 — 2026-07 두 사본 통합]
- (구버전 A 유래) 아직 박스 학습 weight가 없는 상태에서도 서버가 켜질 수 있게
  플레이스홀더 감지 + "box" 클래스 미포함 시 경고 로그 유지.
- (구버전 B 유래, ★중요★ 탐지 정확도에 직접 영향) RGB→BGR 변환 추가
  (PIL이 RGB로 디코딩 → YOLO는 BGR 기대하므로 필수) + half=False 고정
  (FP16 모드에서 탐지 누락 문제 발생했던 것 수정).

[중요] 박스 학습 weight가 없는 상태에서도 이 서버는 켜질 수 있음.
       기본값은 COCO 사전학습 yolov8n.pt라서 "box" 클래스가 실제로는 안 잡힘(정상 동작).
       → 지금은 서버 기동 / health체크 / perception_node 통신 구조만 검증하는 용도.
       학습 끝나면:
           python vlm_boxyolo.py --port 8002 --model runs/detect/box_yolo_v2/weights/best.pt

실행:
    python vlm_boxyolo.py --port 8002
    python vlm_boxyolo.py --port 8002 --model runs/detect/box_yolo_v2/weights/best.pt --conf 0.4

요구 패키지: ultralytics, fastapi, uvicorn, pydantic, pillow, opencv-python, torch
"""

import argparse
import base64
import io
import os
import time
from typing import List

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


class DetectRequest(BaseModel):
    image_b64: str
    labels: List[str]


class Detection(BaseModel):
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0


class DetectResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float


def build_app(port: int, model_path: str, conf_threshold: float,
              iou_threshold: float, imgsz: int) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[server :{port}] device={device}")
    if device == "cuda":
        print(f"[server :{port}] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[server :{port}] [WARN] CPU 모드 (느림)")

    is_placeholder = not os.path.exists(model_path) and not model_path.endswith(".pt")
    print(f"[server :{port}] loading model: {model_path}"
          + ("  (placeholder, COCO weight 자동 다운로드)" if not os.path.exists(model_path) else ""))

    from ultralytics import YOLO
    model = YOLO(model_path)
    names_map: dict = model.names  # {idx: class_name}, 학습된 weight면 보통 {0: 'box'}
    print(f"[server :{port}] model classes: {names_map}")
    print(f"[server :{port}] conf_threshold: {conf_threshold}")
    if "box" not in [v.lower() for v in names_map.values()]:
        print(f"[server :{port}] [NOTICE] 현재 weight에 'box' 클래스가 없음 → "
              f"실제 학습 완료된 best.pt로 --model 교체 필요 (지금은 플레이스홀더 동작)")

    app = FastAPI(title=f"boxyolo-{port}")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "port": port,
            "device": device,
            "model": model_path,
            "classes": names_map,
            "backend": "custom-yolo",
        }

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        try:
            img_bytes = base64.b64decode(req.image_b64)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # PIL은 RGB로 디코딩 → YOLO는 BGR 기대 → 변환 필수
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(400, f"image decode failed: {e}")

        t0 = time.time()
        H, W = img_np.shape[:2]

        try:
            results = model.predict(
                img_np, conf=conf_threshold, iou=iou_threshold,
                imgsz=imgsz, device=device, verbose=False,
                half=False,  # FP16 비활성화 (탐지 누락 문제 방지)
            )
        except Exception as e:
            raise HTTPException(500, f"inference failed: {e}")

        wanted = set(l.strip().lower() for l in req.labels if l.strip())
        out: List[Detection] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls_idx = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                ci = int(cls_idx[i])
                label = results[0].names.get(ci, None)
                if label is None:
                    continue
                if wanted and label.lower() not in wanted:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                out.append(Detection(
                    label=label.lower(),
                    x_min=float(x1 / W), y_min=float(y1 / H),
                    x_max=float(x2 / W), y_max=float(y2 / H),
                    confidence=float(confs[i]),
                ))

        return DetectResponse(detections=out, inference_ms=(time.time() - t0) * 1000.0)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("BOX_MODEL", "yolov8n.pt"),
        help="커스텀 학습 weight 경로 (예: runs/detect/box_yolo_v2/weights/best.pt). "
             "기본값은 임시 COCO weight (통신 테스트용).",
    )
    parser.add_argument("--conf", type=float, default=float(os.environ.get("BOX_CONF", "0.25")))
    parser.add_argument("--iou", type=float, default=float(os.environ.get("BOX_IOU", "0.5")))
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("BOX_IMGSZ", "640")))
    args = parser.parse_args()

    app = build_app(args.port, args.model, args.conf, args.iou, args.imgsz)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
