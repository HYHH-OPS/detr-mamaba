"""Async LAN chat helper with a lightweight AI responder.

This module provides a small asyncio-based chat server and client that can run
on a local network. It includes a rule-based AI that reacts to mentions such as
``@电影`` with movie recommendations and can greet specific teammates like
``@川小农``. The goal is to keep the example dependency-free while offering a
practice ground for AI-driven chat flows.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimpleAIResponder:
    """Rule-based helper that returns canned answers for specific triggers."""

    ai_name: str = "AI"
    movie_suggestions: List[str] = field(
        default_factory=lambda: [
            "流浪地球 2",
            "沙丘 2",
            "银河护卫队 3",
            "入殓师",
            "头号玩家",
        ]
    )

    def generate_reply(self, message: str, sender: str) -> Optional[str]:
        lower = message.lower()
        if "@电影" in message:
            suggestions = "、".join(self.movie_suggestions[:3])
            return f"@{sender} 想看电影吗？可以试试：{suggestions}。"

        if "@" + self.ai_name.lower() in lower or "@ai" in lower:
            return "我在的，有什么需要帮忙的？"

        if "@川小农" in message:
            return "@川小农 记得按回车发送消息，大家都能看到哦！"

        return None


class LanChatServer:
    """Asyncio TCP chat server with broadcast and AI hooks."""

    def __init__(self, host: str, port: int, ai: Optional[SimpleAIResponder] = None):
        self.host = host
        self.port = port
        self.ai = ai or SimpleAIResponder()
        self.clients: Dict[asyncio.StreamWriter, str] = {}

    async def start(self) -> None:
        server = await asyncio.start_server(self._handle_client, self.host, self.port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(f"服务器已启动，监听: {sockets}")
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
        await self._broadcast(f"{nickname} 加入了聊天。", exclude=None)
        print(f"{nickname} 已连接: {addr}")

        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                message = raw.decode().rstrip("\n")
                formatted = f"{nickname}: {message}"
                await self._broadcast(formatted, exclude=None)
                await self._maybe_reply(message, nickname)
        except asyncio.CancelledError:
            pass
        finally:
            del self.clients[writer]
            await self._broadcast(f"{nickname} 已离开。", exclude=None)
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

    async def _maybe_reply(self, message: str, sender: str) -> None:
        reply = self.ai.generate_reply(message, sender)
        if reply:
            await self._broadcast(f"{self.ai.ai_name}: {reply}", exclude=None)


class LanChatClient:
    """Simple client that connects to the LAN chat server."""

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

    client_parser = subparsers.add_parser("client", help="连接到聊天服务器")
    client_parser.add_argument("--host", default="127.0.0.1", help="服务器地址")
    client_parser.add_argument("--port", type=int, default=9009, help="服务器端口")
    client_parser.add_argument("--name", default="访客", help="你的昵称")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "server":
        ai = SimpleAIResponder(ai_name=args.ai_name)
        server = LanChatServer(args.host, args.port, ai)
        asyncio.run(server.start())
    else:
        client = LanChatClient(args.host, args.port, args.name)
        asyncio.run(client.start())


if __name__ == "__main__":
    main()
