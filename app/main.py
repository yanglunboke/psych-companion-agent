"""FastAPI 应用装配。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import auth_router, chat_router, debug_router, health_router
from app.config import get_settings
from app.container import Container
from app.logging_setup import setup_logging

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "启动 %s v%s | model=%s/%s | memory=%s | risk_threshold=%.2f",
        settings.app_name,
        __version__,
        settings.model_provider,
        settings.model_name,
        settings.memory_backend,
        settings.risk_threshold,
    )
    app.state.container = Container(settings)
    await app.state.container.setup_storage()
    await app.state.container.setup_vector_store()
    # 向量模型在后台预热：否则"重启后第一条真实对话"要独自承担 20–30 秒加载
    app.state.container.start_embedder_warmup()
    logger.info("容器就绪，开始接受请求")
    yield
    await app.state.container.close()
    logger.info("服务已停止")


def create_app() -> FastAPI:
    """构建 FastAPI 实例。"""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "中学积极心理陪伴智能体：4 WORKERS 编排 / 情绪轨迹 / "
            "按需资源建议 / 高危固定上报流程"
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(debug_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/debug/memory", include_in_schema=False)
    async def debug_memory_page() -> FileResponse:
        """短期记忆调试页（与 /api/debug/* 共用同一个开关）。"""
        if not get_settings().debug_api_enabled:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(STATIC_DIR / "debug-memory.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """兼容浏览器自动请求的 favicon 路径。"""
        return Response(status_code=204)

    return app


app = create_app()


__all__ = ["app", "create_app"]
