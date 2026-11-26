"""账号存储装配。

沿用 ``app.memory.factory`` 的口径：连不上 SQL 就降级到进程内，
状态如实记录到 ``/health``，不静默假装成功。
"""

from __future__ import annotations

import logging
from typing import Any

from app.auth.store import AuthStore, InMemoryAuthStore, SqlAuthStore
from app.config import Settings

logger = logging.getLogger(__name__)


def _mask_dsn(dsn: str) -> str:
    """日志里不打印 DSN 里的口令。"""
    if "@" not in dsn:
        return dsn
    head, tail = dsn.split("@", 1)
    if "//" in head:
        scheme, _ = head.split("//", 1)
        return f"{scheme}//***@{tail}"
    return f"***@{tail}"


async def build_auth_store(settings: Settings) -> tuple[AuthStore, dict[str, Any]]:
    """按配置构建账号存储，返回 ``(实现, 状态)``。"""
    if settings.db_backend != "sql":
        return InMemoryAuthStore(), {"backend": "memory", "detail": "进程内实现"}

    dsn = settings.postgres_dsn or settings.sql_dsn
    try:
        store = await SqlAuthStore.connect(dsn)
        return store, {"backend": "sql", "detail": _mask_dsn(dsn)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("账号库不可用（%s），降级到进程内实现", exc)
        if not settings.storage_fallback:
            raise
        return (
            InMemoryAuthStore(),
            {"backend": "memory", "detail": f"SQL 不可用，已降级：{exc}"},
        )


__all__ = ["build_auth_store"]
