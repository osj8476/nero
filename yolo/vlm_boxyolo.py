#!/usr/bin/env python3
"""
vlm_boxyolo.py  -  듀얼 모델 YOLO 추론 서버
==============================================================
- "box"     → 커스텀 weight (best.pt)  conf=0.75
- COCO 25종 → yolov8n.pt              conf=0.25
API 스키마는 vlm_yoloworld.py 와 동일 (DetectRequest / DetectResponse).

실행:
    python vlm_boxyolo.py --port 8002 --model best.pt
    python vlm_boxyolo.py --port 8002 --model best.pt --model-coco yolov8n.pt \
        --conf-box 0.75 --conf-coco 0.25
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

# yolov8n.pt 에서 허용할 COCO 라벨 목록
ALLOWED_COCO: set = {
    "person", "umbrella", "tie",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple",
    "book", "clock", "vase", "scissors", "toothbrush",
    "laptop", "remote", "keyboard", "cell phone",
    "sports ball", "baseball bat", "baseball glove",
}


class DetectRequest(BaseModel):
    image_b64: str
    labels: List[str]
    want_mask: bool = False          # True 면 SAM 인스턴스 마스크도 반환 (--sam 필요)


class Detection(BaseModel):
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0
    mask_b64: str = None             # SAM 마스크 PNG(1채널) base64. want_mask=True 이고 SAM 로드시만.


class DetectResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float


def _parse(results, wanted: set, W: int, H: int) -> List[Detection]:
    out: List[Detection] = []
    if not results or results[0].boxes is None:
        return out
    boxes = results[0].boxes
    xyxy   = boxes.xyxy.cpu().numpy()
    cls_idx = boxes.cls.cpu().numpy().astype(int)
    confs  = boxes.conf.cpu().numpy()
    names  = results[0].names
    for i in range(len(xyxy)):
        lbl = names.get(int(cls_idx[i]), None)
        if lbl is None or lbl.lower() not in wanted:
            continue
        x1, y1, x2, y2 = xyxy[i]
        out.append(Detection(
            label=lbl.lower(),
            x_min=float(x1/W), y_min=float(y1/H),
            x_max=float(x2/W), y_max=float(y2/H),
            confidence=float(confs[i]),
        ))
    return out


def build_app(port: int, box_model_path: str, coco_model_path: str,
              conf_box: float, conf_coco: float,
              iou: float, imgsz: int, sam_path: str = None) -> FastAPI:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[server :{port}] device={device}")
    if device == "cuda":
        print(f"[server :{port}] GPU: {torch.cuda.get_device_name(0)}")

    from ultralytics import YOLO
    print(f"[server :{port}] loading box model  : {box_model_path}")
    box_model  = YOLO(box_model_path)
    print(f"[server :{port}] loading COCO model : {coco_model_path}")
    coco_model = YOLO(coco_model_path)

    sam_model = None
    if sam_path and os.path.exists(sam_path):
        from ultralytics import SAM
        print(f"[server :{port}] loading SAM        : {sam_path}")
        sam_model = SAM(sam_path)
    print(f"[server :{port}] ready | conf_box={conf_box} conf_coco={conf_coco} "
          f"iou={iou} imgsz={imgsz} sam={'on' if sam_model else 'off'}")

    def _mask_b64(img_bgr, dets: "List[Detection]", W: int, H: int):
        """dets 의 bbox 를 프롬프트로 SAM 실행 -> 각 det.mask_b64 채움."""
        if sam_model is None or not dets:
            return
        bboxes = [[d.x_min * W, d.y_min * H, d.x_max * W, d.y_max * H] for d in dets]
        try:
            r = sam_model.predict(img_bgr, bboxes=bboxes, device=device, verbose=False)
        except Exception as e:
            print(f"[server :{port}] SAM failed: {e}"); return
        if not r or r[0].masks is None:
            return
        m = r[0].masks.data.cpu().numpy()          # (n, h, w) 0/1
        for i, d in enumerate(dets):
            if i >= len(m):
                break
            mi = (m[i] > 0.5).astype(np.uint8) * 255
            if mi.shape != (H, W):
                mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
            ok, buf = cv2.imencode(".png", mi)
            if ok:
                d.mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    app = FastAPI(title=f"boxyolo-dual-{port}")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "port": port,
            "device": device,
            "model": box_model_path,
            "model_coco": coco_model_path,
            "backend": "dual-yolo",
            "sam": sam_model is not None,
        }

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        try:
            img_bytes = base64.b64decode(req.image_b64)
            img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np    = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(400, f"image decode failed: {e}")

        wanted     = {l.strip().lower() for l in req.labels if l.strip()}
        box_wanted  = wanted & {"box"}
        coco_wanted = wanted & ALLOWED_COCO

        if not box_wanted and not coco_wanted:
            return DetectResponse(detections=[], inference_ms=0.0)

        H, W = img_np.shape[:2]
        t0   = time.time()
        out: List[Detection] = []

        try:
            if box_wanted:
                res = box_model.predict(
                    img_np, conf=conf_box, iou=iou,
                    imgsz=imgsz, device=device, verbose=False, half=False,
                )
                out += _parse(res, box_wanted, W, H)

            if coco_wanted:
                res = coco_model.predict(
                    img_np, conf=conf_coco, iou=iou,
                    imgsz=imgsz, device=device, verbose=False, half=False,
                )
                out += _parse(res, coco_wanted, W, H)

        except Exception as e:
            raise HTTPException(500, f"inference failed: {e}")

        if req.want_mask:
            _mask_b64(img_np, out, W, H)

        return DetectResponse(detections=out,
                              inference_ms=(time.time() - t0) * 1000.0)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int, default=8002)
    parser.add_argument("--host",       type=str, default="127.0.0.1")
    parser.add_argument("--model",      type=str,
                        default=os.environ.get("BOX_MODEL", "best.pt"),
                        help="박스 전용 커스텀 weight 경로")
    parser.add_argument("--model-coco", type=str,
                        default=os.environ.get("COCO_MODEL", "yolov8n.pt"),
                        help="COCO 일반 weight 경로")
    parser.add_argument("--conf-box",   type=float,
                        default=float(os.environ.get("BOX_CONF",  "0.75")))
    parser.add_argument("--conf-coco",  type=float,
                        default=float(os.environ.get("COCO_CONF", "0.25")))
    parser.add_argument("--iou",        type=float,
                        default=float(os.environ.get("BOX_IOU",   "0.5")))
    parser.add_argument("--imgsz",      type=int,
                        default=int(os.environ.get("BOX_IMGSZ",   "640")))
    parser.add_argument("--sam",        type=str,
                        default=os.environ.get("SAM_MODEL", ""),
                        help="SAM weight 경로 (예: mobile_sam.pt). 주면 want_mask 요청에 마스크 반환.")
    args = parser.parse_args()

    app = build_app(
        args.port, args.model, args.model_coco,
        args.conf_box, args.conf_coco, args.iou, args.imgsz,
        sam_path=(args.sam or None),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
