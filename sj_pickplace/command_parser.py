#!/usr/bin/env python3
"""
command_parser.py
Gemini + MCP Client. /user_command → LLM → MCP Tool → /assistant_reply.
"""

import os
import asyncio
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# .env 로드 (선택적)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="python3",
    args=[os.environ.get(
        "MCP_SERVER_PATH",
        os.path.join(os.path.dirname(__file__), "mcp_robot_server.py"),
    )],
    env=os.environ.copy(),
)

SYSTEM_INSTRUCTION = (
    "너는 AgileX Nero 로봇 팔 픽앤플레이스 시스템의 두뇌다. "
    "사용자의 한국어/영어 자연어 명령을 받아 적절한 도구를 호출해 로봇을 제어한다. "
    "\n"
    "[규칙]\n"
    "1. 물체를 집으라는 요청을 받으면, 먼저 list_detected_objects 로 "
    "   현재 장면을 확인한 뒤, 그 안의 영어 라벨로 pick_object 를 호출하라.\n"
    "2. 한국어 물체명은 영어 라벨로 매핑하라 "
    "   (예: 컵→cup, 병→bottle, 가위→scissors, 폰/전화→phone, "
    "    상자→box, 책→book).\n"
    "3. 장면에 없는 물체를 요청받으면 집으려 시도하지 말고 사용자에게 알려라.\n"
    "4. '저쪽으로 옮겨줘', '상자에 넣어줘' 같은 요청은 pick_object 후 "
    "   place_object 를 연속 호출하라.\n"
    "5. Tool 호출 결과에 status:success 가 아니면 사용자에게 실패 사유를 알려라.\n"
    "6. 모든 응답은 사용자가 사용한 언어로 답하라."
)


class CommandParserNode(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop,
                 session_ready: asyncio.Event):
        super().__init__('command_parser')
        self._loop = loop
        # ────────────────────────────────────────────────────────────────
        # [BUG FIX] asyncio.Event를 모듈 레벨에서 생성하면 asyncio.run()이
        # 만드는 새 이벤트 루프와 다른 루프에 귀속되어 wait()가 영원히 블록됨.
        # main_async() 내부에서 생성한 Event를 인자로 주입받아 사용한다.
        # ────────────────────────────────────────────────────────────────
        self._session_ready = session_ready
        self.sub_cmd = self.create_subscription(
            String, '/user_command', self.on_command, 10)
        self.pub_reply = self.create_publisher(String, '/assistant_reply', 10)
        self.get_logger().info(
            'CommandParser 준비 완료 (Gemini + MCP Tool Call 모드)')

    def on_command(self, msg: String):
        user_text = msg.data
        self.get_logger().info(f'명령 수신: {user_text}')
        asyncio.run_coroutine_threadsafe(
            self._handle(user_text), self._loop)

    async def _handle(self, user_text: str):
        try:
            reply = await run_llm_turn(user_text, self._session_ready)
            self.get_logger().info(f'처리 완료: {reply}')
            out = String()
            out.data = reply
            self.pub_reply.publish(out)
        except Exception as e:
            self.get_logger().error(f'LLM/MCP 처리 오류: {e}')
            out = String()
            out.data = f"오류: {e}"
            self.pub_reply.publish(out)


_session = None
_gemini = None


async def _session_owner(session_ready: asyncio.Event):
    """MCP 세션을 열고 준비 완료를 알린다."""
    global _session
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _session = session
            session_ready.set()          # [BUG FIX] 인자로 받은 Event를 set
            await asyncio.Event().wait() # 세션 유지


async def run_llm_turn(user_text: str,
                       session_ready: asyncio.Event) -> str:
    # ────────────────────────────────────────────────────────────────
    # [BUG FIX] 같은 루프에서 생성된 session_ready를 인자로 받아 대기.
    # ────────────────────────────────────────────────────────────────
    await session_ready.wait()
    response = await _gemini.aio.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=user_text,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[_session],
        ),
    )
    return response.text or "(완료)"


def _ros_spin(node: Node):
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


async def main_async():
    global _gemini

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY 환경변수가 없습니다! "
            ".env 파일 또는 export 로 설정하세요.")
    _gemini = genai.Client(api_key=api_key)

    # ────────────────────────────────────────────────────────────────
    # [BUG FIX] asyncio.Event를 asyncio.run()이 만든 루프 안에서 생성.
    # 모듈 레벨에서 생성하면 이 루프와 다른 루프에 귀속되어
    # run_coroutine_threadsafe로 스케줄된 코루틴에서 wait()가 블록됨.
    # ────────────────────────────────────────────────────────────────
    session_ready = asyncio.Event()

    owner_task = asyncio.create_task(_session_owner(session_ready))
    await session_ready.wait()

    rclpy.init()
    loop = asyncio.get_running_loop()
    node = CommandParserNode(loop, session_ready)
    spin_thread = threading.Thread(
        target=_ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    node.get_logger().info('MCP 세션 연결 완료. /user_command 대기 중...')

    try:
        await asyncio.Event().wait()
    finally:
        owner_task.cancel()
        node.destroy_node()
        rclpy.shutdown()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()