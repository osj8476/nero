#!/usr/bin/env python3
"""
vlm_grip_family_eval.py  -  VLA/VLM 1차 검증 실험: grip_family 판단 정확도 평가
=================================================================================
`docs/wiki/vla_vlm_integration_design.md` "1차 검증 실험 계획"의 구현.

목적: YOLO를 VLM으로 교체했을 때 "이 물체는 top_down/side/pinch 중 어떤
그립으로 잡아야 하는가" 판단이 되는지만 검증한다. **좌표/축 추출 정확도와
의도적으로 분리** — 이 스크립트는 좌표를 전혀 다루지 않는다(라벨 문자열
입출력만). 파인튜닝 없이 프롬프트+few-shot으로 시도한다(설계 문서 결론).

운영 파이프라인(vlm_boxyolo.py, 포트 8002, perception_node_sim)과는
완전히 분리된 1회성 평가 스크립트다 — 실사용 경로에 꽂지 않는다
(CLAUDE.md "인지 모델 서버 포트 전환 시 주의" 참고: 여러 모델 비교는
운영 경로를 안 거치는 별도 스크립트로 하라는 원칙과 동일 이유).

VLM 백엔드: NVIDIA NIM API (`NVIDIA_API_KEY` 환경변수, ~/.bashrc에 이미
있음) 사용, OpenAI 호환 /v1/chat/completions. 기본 모델은
meta/llama-3.2-11b-vision-instruct — 특별히 정해진 모델 없어서 프롬프트
+few-shot이 가능한 것 중 이미 쓸 수 있는 API 키로 바로 시작 가능한 걸
선택함 (다른 후보: nvidia/vila, nvidia/nemotron-nano-12b-v2-vl 등, 필요시
--model로 교체).

라벨 파일 포맷 (JSON, --labels로 지정):
    [
      {"file": "batch1_0000.jpg", "object_label": "box", "expected_grip_family": "top_down"},
      {"file": "rod_003.jpg",     "object_label": "rod", "expected_grip_family": "side"},
      ...
    ]
`file`은 --images-dir 기준 상대경로. expected_grip_family는
top_down/side/pinch 중 하나.

실행:
    export NVIDIA_API_KEY=...   # 이미 ~/.bashrc에 설정돼 있음
    python3 vlm_grip_family_eval.py --images-dir /path/to/images --labels labels.json
    python3 vlm_grip_family_eval.py --images-dir ... --labels ... --out results.json
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

GRIP_FAMILIES = ("top_down", "side", "pinch")

SYSTEM_PROMPT = """You are a robot grasp-planning assistant. Given an image and the name of one target object in it, classify which grip family a parallel-jaw gripper should use to pick it up. Answer with exactly one of these three words: top_down, side, pinch.

Definitions:
- top_down: object has a flat/graspable top face and the gripper approaches straight down from above and closes around its width. Typical: boxes, flat-topped containers, cups sitting upright.
- side: object is long/thin relative to its height, or a flat top-down grasp would be unstable or unreachable, so the gripper approaches horizontally from the side and closes around its body. Typical: rods, bars, bottles lying down, door edges.
- pinch: object has a small protruding/narrow graspable feature (not the object's main body) that must be pinched between the fingertips. Typical: door/drawer handles, knobs, thin tabs, hooks.

Answer with just the single word, nothing else."""

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def classify_grip_family(api_key: str, model: str, image_path: Path,
                          object_label: str, timeout: float = 30.0) -> str:
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    ext = image_path.suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text",
                 "text": f"Target object label: {object_label}. Which grip family should be used?"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}},
            ]},
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return parse_grip_family(raw)


def parse_grip_family(raw: str) -> str:
    cleaned = re.sub(r"[^a-z_]", "", raw.strip().lower())
    for fam in GRIP_FAMILIES:
        if fam in cleaned:
            return fam
    return f"unparseable:{raw.strip()[:40]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True,
                     help="JSON: [{file, object_label, expected_grip_family}, ...]")
    ap.add_argument("--model", type=str, default="meta/llama-3.2-11b-vision-instruct")
    ap.add_argument("--api-key-env", type=str, default="NVIDIA_API_KEY")
    ap.add_argument("--out", type=Path, default=None, help="결과 JSON 저장 경로 (선택)")
    ap.add_argument("--sleep", type=float, default=0.2, help="요청 간 대기(초), 레이트리밋 방지")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"[FATAL] 환경변수 {args.api_key_env} 가 비어있음", file=sys.stderr)
        sys.exit(1)

    items = json.loads(args.labels.read_text())
    if not items:
        print("[FATAL] 라벨 파일이 비어있음", file=sys.stderr)
        sys.exit(1)

    results = []
    correct = 0
    confusion = {}  # (expected, predicted) -> count

    for i, item in enumerate(items):
        img_path = args.images_dir / item["file"]
        expected = item["expected_grip_family"]
        assert expected in GRIP_FAMILIES, f"잘못된 expected_grip_family: {expected} ({item['file']})"

        if not img_path.exists():
            print(f"[SKIP] {img_path} 없음")
            continue

        t0 = time.time()
        try:
            predicted = classify_grip_family(api_key, args.model, img_path, item["object_label"])
        except Exception as e:
            predicted = f"error:{e}"
        dt = time.time() - t0

        is_correct = predicted == expected
        correct += int(is_correct)
        confusion[(expected, predicted)] = confusion.get((expected, predicted), 0) + 1
        results.append({
            "file": item["file"], "object_label": item["object_label"],
            "expected": expected, "predicted": predicted,
            "correct": is_correct, "latency_s": round(dt, 2),
        })
        mark = "OK " if is_correct else "FAIL"
        print(f"[{i+1}/{len(items)}] {mark} {item['file']:30s} "
              f"label={item['object_label']:12s} expected={expected:9s} predicted={predicted}")

        time.sleep(args.sleep)

    n = len(results)
    print("\n=== 요약 ===")
    print(f"정확도: {correct}/{n} = {correct/n*100:.1f}%" if n else "평가된 항목 없음")
    print("\n혼동 행렬 (expected -> predicted : count):")
    for (exp, pred), cnt in sorted(confusion.items()):
        print(f"  {exp:9s} -> {pred:20s} : {cnt}")

    if args.out:
        args.out.write_text(json.dumps({
            "model": args.model, "accuracy": correct / n if n else None,
            "n": n, "results": results,
        }, indent=2, ensure_ascii=False))
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
