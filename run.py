"""本地启动入口。

用法::

    python run.py                # 默认 127.0.0.1:8000
    python run.py --port 8080    # 覆盖端口
    python run.py --reload       # 开发热重载
"""

from __future__ import annotations

import argparse
import asyncio
import selectors
import sys

import uvicorn

from app.config import get_settings


class _WindowsSelectorPolicy(asyncio.DefaultEventLoopPolicy):
    """强制 Windows 用 SelectorEventLoop（psycopg3 异步不兼容 Proactor）。"""

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.SelectorEventLoop(selectors.SelectSelector())


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="中学积极心理陪伴智能体")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="开启热重载")
    args = parser.parse_args()

    # Windows 下 psycopg3 异步驱动不兼容默认的 ProactorEventLoop，
    # 连 Postgres 会抛 InterfaceError。uvicorn 在 Windows 上硬编码使用
    # ProactorEventLoop（uvicorn.loops.asyncio.asyncio_loop_factory），
    # 传 loop="none" 可让 uvicorn 不自行建 loop，改走我们设置的 policy。
    loop_setting = "auto"
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(_WindowsSelectorPolicy())
        loop_setting = "none"

    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
        loop=loop_setting,
    )

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
