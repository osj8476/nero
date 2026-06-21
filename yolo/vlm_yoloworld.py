"""
vlm_yoloworld.py  -  YOLO-World 기반 VLM 서버 (vlm_moondream.py 드롭인 대체)
=================================================================================
- API 스키마, 포트 사용법, 응답 포맷 모두 vlm_moondream.py와 100% 호환
- perception_node.py / realsense_demo.py / run_cluster.ps1 한 줄도 안 바꿔도 동작

[Moondream2 대비 핵심 차이]
- 라벨 N개를 단일 forward로 동시 검출 (Moondream은 N회 루프)
  → 다중 객체 latency 폭증 + 오인식 문제 구조적 해결
- 단일 RTX 4090에서 100+ FPS 가능 → 클러스터 N대 불필요

실행 (PowerShell):
    # 가장 단순:
    python vlm_yoloworld.py --port 8000

    # 모델 크기 선택 (s=빠름, m=균형, l=정확, x=최대정확):
    python vlm_yoloworld.py --port 8000 --model yolov8l-worldv2.pt

    # confidence threshold 조정 (작은 객체나 적은 검출 시 낮추기):
    python vlm_yoloworld.py --port 8000 --conf 0.05

요구 패키지 (기존 venv에 추가):
    pip install ultralytics

(기존 fastapi, uvicorn, pydantic, pillow, opencv-python, torch는 이미 설치됨)

[모델 자동 다운로드]
첫 실행 시 ultralytics가 Hugging Face/CDN에서 자동으로 받아옵니다.
- yolov8s-worldv2.pt : ~25 MB
- yolov8m-worldv2.pt : ~58 MB
- yolov8l-worldv2.pt : ~90 MB  (RTX 4090 권장)
- yolov8x-worldv2.pt : ~141 MB
"""

import argparse
import base64
import io
import os
import time
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# ultralytics는 lazy import (서버 시작 시점에 로드)


# ──────────────────────────────────────────────
# Request / Response Schema
# vlm_moondream.py와 완전 동일하게 유지
# ──────────────────────────────────────────────
class DetectRequest(BaseModel):
    image_b64: str
    labels: List[str]


class Detection(BaseModel):
    label: str
    x_min: float  # 0~1 정규화 좌표 (Moondream과 동일)
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0


class DetectResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float


# ──────────────────────────────────────────────
# 서버 빌더
# ──────────────────────────────────────────────
def build_app(
    port: int,
    model_name: str,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[server :{port}] device={device}")
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[server :{port}] GPU: {gpu} | VRAM: {vram:.1f} GB")
    else:
        print(f"[server :{port}] [WARN] CPU mode (very slow)")

    print(f"[server :{port}] loading YOLO-World ({model_name})...")
    from ultralytics import YOLOWorld
    model = YOLOWorld(model_name)
    if device == "cuda":
        # half precision은 ultralytics가 predict() 호출 시 자동 처리 가능
        # 명시적으로 fuse하면 약간 더 빠름
        try:
            model.fuse()
        except Exception:
            pass
    # ── Startup 시점에 CLIP 의존성 fail-fast 체크 ──
    # set_classes()가 처음 호출될 때 CLIP을 import한다.
    # CLIP이 없으면 매 요청마다 실패하고 무한 재시도하므로,
    # 여기서 한 번 호출해서 의존성 문제면 즉시 죽이고 명확한 에러 표시.
    try:
        model.set_classes(["object"])  # dummy 라벨로 의존성만 체크
        print(f"[server :{port}] CLIP text encoder OK")
    except ImportError as e:
        print(f"\n[server :{port}] ❌ CLIP 의존성 누락:\n  {e}")
        print(f"[server :{port}] 다음 중 하나를 실행하세요:")
        print(f"    pip install 'ultralytics==8.2.103'   # 다운그레이드 (가장 빠름)")
        print(f"    pip install ftfy regex tqdm && pip install git+https://github.com/ultralytics/CLIP.git")
        raise SystemExit(1)

    print(f"[server :{port}] model ready "
          f"(conf={conf_threshold}, iou={iou_threshold}, imgsz={imgsz})")

    # 라벨 캐싱: 같은 라벨 리스트면 set_classes() 호출 생략
    # (set_classes()는 CLIP 텍스트 인코더를 돌려서 prompt embedding을 만드는데,
    #  매 호출마다 하면 latency가 100ms+ 추가됨)
    state = {"current_labels_key": ("object",)}  # dummy 등록된 상태로 시작

    # 공백 포함 라벨 → YOLO-World 친화적 단일토큰 치환 테이블
    # YOLO-World 텍스트 인코더(CLIP)가 공백 있는 라벨을 두 토큰으로 쪼개
    # 엉뚱한 클래스에 매핑하는 문제를 방지
    LABEL_REMAP = {
        "cell phone": "cellphone",
        "cell-phone": "cellphone",
        "mobile phone": "cellphone",
        "hot dog": "hotdog",
        "traffic light": "trafficlight",
        "fire hydrant": "firehydrant",
        "stop sign": "stopsign",
        "sports ball": "sportsball",
        "baseball bat": "baseballbat",
        "tennis racket": "tennisracket",
        "wine glass": "wineglass",
        "dining table": "diningtable",
        "potted plant": "pottedplant",
        "teddy bear": "teddybear",
    }

    def _ensure_labels(labels: List[str]):
        raw_cleaned = [l.strip().lower() for l in labels if l.strip()]
        # 문제 라벨 치환
        remapped = [LABEL_REMAP.get(l, l) for l in raw_cleaned]
        # 역매핑: yolo내부라벨 → 원본라벨 (응답 시 원래 이름으로 복원)
        remap_back: dict = {yolo: orig for orig, yolo in zip(raw_cleaned, remapped)}
        cleaned = sorted(set(remapped))
        key = tuple(cleaned)
        if state["current_labels_key"] != key:
            print(f"[server :{port}] set_classes({cleaned})  (remap: {remap_back})")
            try:
                model.set_classes(list(cleaned))
                state["current_labels_key"] = key
                state["remap_back"] = remap_back
            except Exception as e:
                state["current_labels_key"] = key
                raise HTTPException(500, f"set_classes failed: {e}")
        else:
            remap_back = state.get("remap_back", {})
        return cleaned, remap_back

    app = FastAPI(title=f"yoloworld-{port}")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "port": port,
            "device": device,
            "model": model_name,
            "backend": "yolo-world-v2",
        }

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        try:
            img_bytes = base64.b64decode(req.image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"image decode failed: {e}")

        if not req.labels:
            return DetectResponse(detections=[], inference_ms=0.0)

        # 라벨 등록 (캐싱)
        active_labels, remap_back = _ensure_labels(req.labels)
        if not active_labels:
            return DetectResponse(detections=[], inference_ms=0.0)

        # ── 추론 ──
        t0 = time.time()
        img_np = np.array(img)  # PIL → numpy (H, W, 3) RGB
        H, W = img_np.shape[:2]

        try:
            results = model.predict(
                img_np,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
                half=(device == "cuda"),
            )
        except Exception as e:
            raise HTTPException(500, f"inference failed: {e}")

        # ── 결과 파싱 ──
        # [YOLO-World 버그 우회]
        # predict() 후 boxes.cls는 COCO 80클래스 기준 인덱스를 반환할 수 있음.
        # results[0].names 딕셔너리가 {cls_idx: label_str} 실제 매핑을 담고 있으므로
        # 이걸 사용해야 올바른 라벨을 얻을 수 있음.
        out: List[Detection] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls_idx = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            # results[0].names: {int: str} — predict() 시점의 실제 클래스 이름 매핑
            names_map: dict = results[0].names  # e.g. {0: 'cellphone', 3: 'person', ...}

            print(f"[DEBUG] active_labels: {active_labels}")
            print(f"[DEBUG] names_map: {names_map}")
            for i in range(len(xyxy)):
                ci = int(cls_idx[i])
                yolo_label = names_map.get(ci, None)
                print(f"[DEBUG] box[{i}] cls_idx={ci} label={yolo_label} conf={confs[i]:.3f}")

            # active_labels(set_classes에 넘긴 것) 기준으로 필터링
            active_set = set(active_labels)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                ci = int(cls_idx[i])
                yolo_label = names_map.get(ci, None)

                # names_map에 없거나 우리가 요청한 라벨이 아니면 스킵
                if yolo_label is None or yolo_label not in active_set:
                    continue

                # 치환된 라벨 → 사용자가 요청한 원본 라벨명으로 복원
                label = remap_back.get(yolo_label, yolo_label)
                conf_val = float(confs[i])

                out.append(Detection(
                    label=label,
                    x_min=float(x1 / W),
                    y_min=float(y1 / H),
                    x_max=float(x2 / W),
                    y_max=float(y2 / H),
                    confidence=conf_val,
                ))

        return DetectResponse(
            detections=out,
            inference_ms=(time.time() - t0) * 1000.0,
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("YOLO_MODEL", "yolov8l-worldv2.pt"),
        help="YOLO-World 가중치 (s/m/l/x-worldv2.pt). 첫 실행 시 자동 다운로드.",
    )
    parser.add_argument(
        "--conf", type=float,
        default=float(os.environ.get("YOLO_CONF", "0.05")),
        help="confidence threshold. YOLO-World는 0.05~0.15 권장.",
    )
    parser.add_argument(
        "--iou", type=float,
        default=float(os.environ.get("YOLO_IOU", "0.5")),
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--imgsz", type=int,
        default=int(os.environ.get("YOLO_IMGSZ", "640")),
        help="추론 입력 해상도. 320/416/640 권장.",
    )
    args = parser.parse_args()

    app = build_app(
        port=args.port,
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="info", access_log=False)


if __name__ == "__main__":
    main()