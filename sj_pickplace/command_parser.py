#!/usr/bin/env python3
"""
command_parser.py
=================
자연어 명령을 Gemini API로 파싱해서 /arm_command 토픽으로 발행.

사용법:
    GEMINI_API_KEY=your_key ros2 run sj_pickplace command_parser

발행 예시:
    "컵 집어줘"     → {"action":"pick",  "target_label":"cup"}
    "병 옮겨줘"     → {"action":"pick",  "target_label":"bottle"}
    "가위 내려놔"   → {"action":"place", "target_label":"scissors"}
    "원래 자리로"   → {"action":"home"}
"""

import json
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

# ── Gemini SDK ─────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── 상수 ───────────────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-1.5-flash"
VALID_ACTIONS  = {"pick", "place", "home", "stop"}

SYSTEM_PROMPT = """
당신은 로봇팔 픽앤플레이스 시스템의 명령 파서입니다.
사용자의 자연어 명령을 아래 JSON 형식으로만 반환하세요.
다른 설명이나 마크다운은 절대 포함하지 마세요.

JSON 형식:
{
  "action": "pick" | "place" | "home" | "stop",
  "target_label": "물체 이름 (영어, pick/place 일 때만)"
}

물체 이름은 반드시 영어 소문자로 변환:
  컵 → cup, 병 → bottle, 가위 → scissors, 책 → book
  박스 → box, 공 → ball, 리모컨 → remote

action 판단 기준:
  pick  : 집어, 집다, 들어, 가져, 올려, 잡아
  place : 놓아, 내려, 내려놔, 옮겨, 이동
  home  : 원래 자리, 홈, 초기화, 기본 위치
  stop  : 멈춰, 정지, 중단, 취소

예시:
  "컵 집어줘"    → {"action":"pick",  "target_label":"cup"}
  "병 내려놔"    → {"action":"place", "target_label":"bottle"}
  "홈으로 가줘"  → {"action":"home"}
  "멈춰"         → {"action":"stop"}
""".strip()


class CommandParser(Node):
    def __init__(self):
        super().__init__("command_parser")

        # ── Gemini 초기화 ───────────────────────────────────────────────
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not GEMINI_AVAILABLE:
            self.get_logger().error(
                "google-generativeai 미설치. "
                "'pip install google-generativeai' 후 재시작하세요."
            )
            sys.exit(1)
        if not api_key:
            self.get_logger().error(
                "GEMINI_API_KEY 환경변수가 없습니다. "
                "'export GEMINI_API_KEY=your_key' 후 재시작하세요."
            )
            sys.exit(1)

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        self.get_logger().info(f"Gemini 모델 로드 완료: {GEMINI_MODEL}")

        # ── ROS2 Publisher ──────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(String, "/arm_command", qos)

        # ── 입력 구독 (자연어를 /natural_command 토픽으로 받기) ─────────
        self.sub = self.create_subscription(
            String, "/natural_command", self.on_natural_command, qos
        )

        self.get_logger().info(
            "CommandParser 준비 완료.\n"
            "  자연어 입력: ros2 topic pub --once /natural_command "
            "std_msgs/String 'data: \"컵 집어줘\"'"
        )

    # ── 콜백 ────────────────────────────────────────────────────────────────
    def on_natural_command(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        self.get_logger().info(f"[입력] {text}")
        self._parse_and_publish(text)

    # ── 핵심 로직 ────────────────────────────────────────────────────────────
    def _parse_and_publish(self, text: str):
        """Gemini로 자연어 파싱 → JSON 검증 → /arm_command 발행."""
        # 1) Gemini 호출
        try:
            response = self.model.generate_content(text)
            raw = response.text.strip()
        except Exception as e:
            self.get_logger().error(f"Gemini API 오류: {e}")
            return

        self.get_logger().info(f"[Gemini 응답] {raw}")

        # 2) JSON 파싱
        try:
            # 마크다운 코드블록 제거 (```json ... ``` 형태일 경우)
            clean = raw.strip("`").removeprefix("json").strip()
            cmd = json.loads(clean)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON 파싱 실패: {e}\n원문: {raw}")
            return

        # 3) 필드 검증
        action = cmd.get("action", "")
        if action not in VALID_ACTIONS:
            self.get_logger().error(
                f"알 수 없는 action: '{action}'. "
                f"유효한 값: {VALID_ACTIONS}"
            )
            return

        if action in ("pick", "place") and "target_label" not in cmd:
            self.get_logger().error(
                f"'{action}' 명령에 target_label이 없습니다. 원문: {raw}"
            )
            return

        # 4) /arm_command 발행
        out_msg = String()
        out_msg.data = json.dumps(cmd, ensure_ascii=False)
        self.pub.publish(out_msg)

        label = cmd.get("target_label", "")
        self.get_logger().info(
            f"[발행] /arm_command → action={action}"
            + (f", target={label}" if label else "")
        )


def main():
    rclpy.init()
    node = CommandParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
