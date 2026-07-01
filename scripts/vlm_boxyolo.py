#!/usr/bin/env python3
"""
vlm_boxyolo.py  -  박스 전용 커스텀 YOLO 추론 서버
=================================================================================
[변경 이력]
- RGB→BGR 변환 추가 (PIL이 RGB로 디코딩 → YOLO는 BGR 기대하므로 필수)
- half=False 고정 (FP16 모드에서 탐지 누락 문제 해결)
- 기존 vlm_yoloworld.py와 API 스키마 100% 호환

실행:
    python vlm_boxyolo.py --port 8002 --model runs/detect/box_yolo_v2/weights/best.pt
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

    print(f"[server :{port}] loading model: {model_path}")
    from ultralytics import YOLO
    model = YOLO(model_path)
    names_map: dict = model.names
    print(f"[server :{port}] model classes: {names_map}")
    print(f"[server :{port}] conf_threshold: {conf_threshold}")

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
    parser.add_argument("--model", type=str,
                        default=os.environ.get("BOX_MODEL", "yolov8n.pt"))
    parser.add_argument("--conf", type=float,
                        default=float(os.environ.get("BOX_CONF", "0.25")))
    parser.add_argument("--iou", type=float,
                        default=float(os.environ.get("BOX_IOU", "0.5")))
    parser.add_argument("--imgsz", type=int,
                        default=int(os.environ.get("BOX_IMGSZ", "640")))
    args = parser.parse_args()

    app = build_app(args.port, args.model, args.conf, args.iou, args.imgsz)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
