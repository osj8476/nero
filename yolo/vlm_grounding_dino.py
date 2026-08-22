#!/usr/bin/env python3
"""
vlm_grounding_dino.py  -  Grounding DINO 기반 개방어휘 검출 서버
=================================================================================
- vlm_boxyolo.py와 API 스키마 100% 호환 (perception_node_sim 코드 변경 없이 드롭인 교체)
- `docs/wiki/vla_vlm_integration_design.md` "1차 검증 실험" 연장선: 순수 챗형 VLM
  (llama-3.2-vision 등)에 프롬프트로 bbox 좌표를 직접 뱉게 하는 방식은 실측
  검증 결과 좌표 정밀도가 전혀 못 쓰는 수준(수십 cm 오차, few-shot 예시를
  그대로 베끼는 앵커링 현상까지 관찰됨)이었음 — 별도 세션 기록 없음, 이
  스크립트 작성 직전 대화에서 실측.
  대신 Grounding DINO(IDEA-Research/grounding-dino-tiny, HuggingFace
  transformers)를 사용 — 이건 "VLM에게 숫자를 말로 시키는" 방식이 아니라
  텍스트 프롬프트 조건부 open-vocab object detector라서 진짜 bbox 회귀를
  한다. 검출이 confidence threshold를 통과했을 때는 박스 중심점 오차가
  이미지 폭 대비 0.2% 수준으로 YOLO와 사실상 동일했다(라이브 프레임 1장
  기준). **다만 아래 "미해결 문제" 때문에 이 정확도가 항상 나오는 게
  아니다 — 결론은 "좌표 정밀도는 증명됐으나 신뢰도(confidence) 보정은
  아직 못 씀".**

[중요] 이 서버는 검증/실험용이다 — **아직 운영 포트 8002에 올리지 않는
것을 권장**(2026-08-18 실측 근거는 아래 "미해결 문제" 참고). 나중에
올리기로 결정하더라도 CLAUDE.md의 "인지 모델 서버 포트 전환 시 주의"
절차를 반드시 따를 것 — 표준 포트 8002 하나에서 기존 프로세스를 멈추고
이 서버를 그 자리에 띄우는 방식으로만 전환(병렬 포트 운용 금지).

[정확도 관련 실측 메모, 2026-08-18]
- 프롬프트는 "box."(Grounding DINO 컨벤션: 소문자, 마침표로 끝) 단순형이
  "cardboard box." 같은 구체적 문구보다 나았다 — 후자는 오검출
  confidence를 더 올리는 역효과가 있었음(아래).
- 콜드스타트(모델 최초 로드+첫 추론)는 CUDA 워밍업 포함 약 3초 걸림 —
  서버 기동 시 헬스체크가 열리기 전에 warmup 추론을 1회 미리 돌려서
  첫 실제 요청이 이 지연을 물지 않게 한다.
- warm 추론 지연은 약 350~400ms/frame (RTX 3080 Ti 기준) — YOLO(~20ms)
  대비 15~20배 느리지만, perception_node_sim의 REQUEST_TIMEOUT(기본 3.0s)
  안에는 충분히 들어온다. 다만 DISPATCH_RATE_HZ(기본 10Hz)만큼 못 따라가고
  ~2.5Hz로 사실상 caps됨 — _inflight 락이 있어 요청이 쌓이지는 않는다.

[미해결 문제 — confidence 범위가 실제 박스와 그리퍼 오검출에서 겹침]
같은 씬에서 실측한 프레임 3장 기준:
  - 실제 박스(prompt="box.") score: 0.756, 0.648 (프레임마다 변동폭 큼 —
    YOLO는 같은 씬에서 항상 0.90~0.91로 훨씬 안정적이었음)
  - 로봇 그리퍼 손끝을 "box."로 오검출한 score: 0.732 (prompt="box."),
    0.800 (prompt="cardboard box.")
이 두 범위가 겹친다(0.648~0.868 vs 0.471~0.800) — **YOLO의 운영
threshold(0.75)를 그대로 가져다 써도 실제 박스를 놓치거나(0.648 프레임)
그리퍼를 박스로 오검출하거나(0.732~0.80) 둘 다 일어날 수 있다.** 단순
threshold 조정만으로는 해결 안 됨 — 시도해볼 만한 다음 단계(미시도):
  1. grounding-dino-tiny 대신 grounding-dino-base로 교체(더 큰 모델,
     느리지만 confidence 분리가 나을 가능성)
  2. 그리퍼가 항상 화면 특정 영역(하단)에 나온다는 걸 이용한 ROI
     마스킹/후처리 필터 추가
  3. negative prompt나 다중 라벨("box. robot gripper.")로 그리퍼를
     별도 클래스로 명시해 억제
**이 중 아무것도 아직 시도 안 함 — 다음 세션에서 이어갈 때는 여기부터.**

요구 패키지: transformers>=4.40, torch, fastapi, uvicorn, pydantic, pillow
    pip install --user "transformers>=4.40,<5.0"   (2026-08-18 기준 이미 설치됨)

실행 (vlm_boxyolo.py와 동일한 방식):
    python3 vlm_grounding_dino.py --port 8002
    python3 vlm_grounding_dino.py --port 8009 --conf 0.75   # 스크래치 포트로 병행 비교 시
"""

import argparse
import base64
import io
import time
from typing import List

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


def build_app(port: int, model_id: str, conf_threshold: float,
              text_threshold: float) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[server :{port}] device={device}")
    if device == "cuda":
        print(f"[server :{port}] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[server :{port}] [WARN] CPU 모드 (매우 느림)")

    print(f"[server :{port}] loading model: {model_id}")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    # 콜드스타트(CUDA 워밍업) 비용을 서버 기동 시점에 미리 지불 —
    # _wait_for_box_server 헬스체크가 열리기 전에 끝내서 첫 실제 요청이
    # 이 지연을 물지 않게 한다 (vlm_boxyolo.py에는 없던 이슈: YOLO는
    # 콜드스타트가 무시할 만큼 짧지만 이 모델은 ~3초 걸림, 실측 확인).
    _dummy = Image.new("RGB", (640, 480))
    _inputs = processor(images=_dummy, text="box.", return_tensors="pt").to(device)
    with torch.no_grad():
        model(**_inputs)
    print(f"[server :{port}] warmup 완료")
    print(f"[server :{port}] conf_threshold: {conf_threshold} text_threshold: {text_threshold}")

    app = FastAPI(title=f"grounding-dino-{port}")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "port": port,
            "device": device,
            "model": model_id,
            "classes": {"open_vocab": True},
            "backend": "grounding-dino",
        }

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        try:
            img_bytes = base64.b64decode(req.image_b64)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"image decode failed: {e}")

        W, H = img_pil.size
        wanted = [l.strip().lower() for l in req.labels if l.strip()]
        if not wanted:
            return DetectResponse(detections=[], inference_ms=0.0)
        # Grounding DINO 컨벤션: 라벨 각각 소문자+마침표, 공백으로 이어붙임
        prompt = " ".join(f"{l}." for l in wanted)

        t0 = time.time()
        try:
            inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                threshold=conf_threshold, text_threshold=text_threshold,
                target_sizes=[(H, W)],
            )[0]
        except Exception as e:
            raise HTTPException(500, f"inference failed: {e}")

        out: List[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            x1, y1, x2, y2 = box.tolist()
            out.append(Detection(
                label=label.strip().lower() or wanted[0],
                x_min=float(x1 / W), y_min=float(y1 / H),
                x_max=float(x2 / W), y_max=float(y2 / H),
                confidence=float(score),
            ))

        return DetectResponse(detections=out, inference_ms=(time.time() - t0) * 1000.0)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--conf", type=float, default=0.75,
                         help="운영 YOLO(box_yolo_v6)와 동일 기본값 0.75 — 낮추면 그리퍼 오검출 재발 가능(실측 확인)")
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    app = build_app(args.port, args.model, args.conf, args.text_threshold)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
