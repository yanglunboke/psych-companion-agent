"""本地启动入口。

用法::

    python run.py                # 默认 127.0.0.1:8000
    python run.py --port 8080    # 覆盖端口
    python run.py --reload       # 开发热重载
"""

from __future__ import annotations

import argparse

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="中学积极心理陪伴智能体")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="开启热重载")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
