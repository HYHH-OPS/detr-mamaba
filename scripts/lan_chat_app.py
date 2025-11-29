"""局域网聊天助手，附带轻量 AI 回复逻辑。

该脚本提供基于 asyncio 的聊天服务器与客户端，可在局域网运行。内置规则
AI 会响应 ``@电影``、@AI 昵称、``@川小农`` 等提及，提供推荐或提醒。整
体保持零依赖，便于快速练习 AI 驱动的聊天流程。
"""

from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimpleAIResponder:
    """简单规则引擎，针对常见触发词返回固定回复。"""

    ai_name: str = "AI"
    movie_suggestions: Optional[List[str]] = field(
        default_factory=lambda: [
            "流浪地球 2",
            "沙丘 2",
            "银河护卫队 3",
            "入殓师",
            "头号玩家",
        ]
    )
    teammate_tip: str = "@川小农 记得按回车发送消息，大家都能看到哦！"

    def __post_init__(self) -> None:
        if self.movie_suggestions is None:
            self.movie_suggestions = [
                "流浪地球 2",
                "沙丘 2",
                "银河护卫队 3",
                "入殓师",
                "头号玩家",
            ]

    def help_text(self) -> str:
        """返回可用触发词提示。"""

        shortcuts = [
            f"@{self.ai_name} / @ai 呼叫 AI",
            "@电影 获取推荐",
            "@帮助 查看指令",
            "@川小农 提醒队友加入对话",
        ]
        return "指令提示：" + "； ".join(shortcuts)

    def greet(self, nickname: str, room: str) -> str:
        return f"@{nickname} 欢迎来到 {room}！{self.help_text()}"

    def generate_reply(self, message: str, sender: str) -> Optional[str]:
        lower = message.lower()
        if "@电影" in message:
            picks = random.sample(self.movie_suggestions, k=min(3, len(self.movie_suggestions)))
            suggestions = "、".join(picks)
            return f"@{sender} 想看电影吗？可以试试：{suggestions}。"

        if "@" + self.ai_name.lower() in lower or "@ai" in lower:
            return "我在的，需要查资料、提问或推荐都可以喊我。"

        if "@帮助" in message or "help" in lower:
            return self.help_text()

        if "@川小农" in message:
            return self.teammate_tip

        return None


class LanChatServer:
    """支持广播与 AI 自动回复的 asyncio TCP 聊天服务器。"""

    def __init__(
        self, host: str, port: int, ai: Optional[SimpleAIResponder] = None, room: str = "聊天室"
    ):
        self.host = host
        self.port = port
        self.ai = ai or SimpleAIResponder()
        self.room = room
        self.clients: Dict[asyncio.StreamWriter, str] = {}

    async def start(self) -> None:
        server = await asyncio.start_server(self._handle_client, self.host, self.port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(f"服务器已启动，监听: {sockets}，房间：{self.room}")
        async with server:
            await server.serve_forever()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        addr = writer.get_extra_info("peername")
        name = await reader.readline()
        if not name:
            writer.close()
            await writer.wait_closed()
            return

        nickname = name.decode().strip() or f"用户{addr[1]}"
        self.clients[writer] = nickname
        await self._broadcast(f"[{self.room}] {nickname} 加入了聊天。", exclude=None)
        await self._send(writer, f"{self.ai.ai_name}: {self.ai.greet(nickname, self.room)}")
        print(f"{nickname} 已连接: {addr}")

        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                message = raw.decode().rstrip("\n")
                if not message.strip():
                    continue
                formatted = f"{nickname}: {message}"
                await self._broadcast(formatted, exclude=None)
                await self._maybe_reply(message, nickname)
        except asyncio.CancelledError:
            pass
        finally:
            del self.clients[writer]
            await self._broadcast(f"[{self.room}] {nickname} 已离开。", exclude=None)
            writer.close()
            await writer.wait_closed()
            print(f"{nickname} 已断开: {addr}")

    async def _broadcast(self, text: str, exclude: Optional[asyncio.StreamWriter]) -> None:
        dead: List[asyncio.StreamWriter] = []
        for writer in self.clients.keys():
            if writer is exclude:
                continue
            try:
                writer.write((text + "\n").encode())
                await writer.drain()
            except ConnectionError:
                dead.append(writer)
        for writer in dead:
            self.clients.pop(writer, None)

    async def _send(self, writer: asyncio.StreamWriter, text: str) -> None:
        try:
            writer.write((text + "\n").encode())
            await writer.drain()
        except ConnectionError:
            self.clients.pop(writer, None)

    async def _maybe_reply(self, message: str, sender: str) -> None:
        reply = self.ai.generate_reply(message, sender)
        if reply:
            await self._broadcast(f"{self.ai.ai_name}: {reply}", exclude=None)


class LanChatClient:
    """连接至局域网聊天服务器的简单客户端。"""

    def __init__(self, host: str, port: int, nickname: str):
        self.host = host
        self.port = port
        self.nickname = nickname

    async def start(self) -> None:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        writer.write((self.nickname + "\n").encode())
        await writer.drain()
        print("已连接。输入消息并回车发送，Ctrl+C 退出。")

        async def listen() -> None:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                print(line.decode().rstrip("\n"))

        async def send() -> None:
            loop = asyncio.get_running_loop()
            while True:
                try:
                    message = await loop.run_in_executor(None, input, "")
                    if not message.strip():
                        continue
                    writer.write((message + "\n").encode())
                    await writer.drain()
                except (KeyboardInterrupt, EOFError):
                    writer.close()
                    await writer.wait_closed()
                    break

        await asyncio.gather(listen(), send())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAN chat with AI responder")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server_parser = subparsers.add_parser("server", help="启动聊天服务器")
    server_parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    server_parser.add_argument("--port", type=int, default=9009, help="监听端口")
    server_parser.add_argument(
        "--ai-name", default="AI", help="AI 在聊天中的昵称，用于@触发"
    )
    server_parser.add_argument(
        "--movies",
        default=None,
        help="逗号分隔的片单，覆盖默认电影推荐",
    )
    server_parser.add_argument(
        "--room", default="聊天室", help="房间名称，会显示在加入/离开提示中"
    )

    client_parser = subparsers.add_parser("client", help="连接到聊天服务器")
    client_parser.add_argument("--host", default="127.0.0.1", help="服务器地址")
    client_parser.add_argument("--port", type=int, default=9009, help="服务器端口")
    client_parser.add_argument("--name", default="访客", help="你的昵称")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "server":
        movies = None
        if args.movies:
            movies = [m.strip() for m in args.movies.split(",") if m.strip()]
        ai = SimpleAIResponder(ai_name=args.ai_name, movie_suggestions=movies or None)
        server = LanChatServer(args.host, args.port, ai, room=args.room)
        asyncio.run(server.start())
    else:
        client = LanChatClient(args.host, args.port, args.name)
        asyncio.run(client.start())


if __name__ == "__main__":
    main()
