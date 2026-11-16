"""存储后端装配与降级。

原则：**服务永远能起来**。Redis / Postgres 连不上时，按配置降级到进程内实现，
并在状态里如实记录降级原因（``/health`` 会暴露），不静默假装成功。

* 短期记忆：``MEMORY_BACKEND=redis`` → Redis；否则进程内。
* 长期存储：``DB_BACKEND=sql`` → SQLAlchemy（Postgres / SQLite）；否则进程内。
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.memory.memory_impl import InMemoryProfileStore, InMemoryShortTermMemory
from app.memory.store import ProfileStore, ShortTermMemory

logger = logging.getLogger(__name__)


async def build_short_term_memory(
    settings: Settings,
) -> tuple[ShortTermMemory, dict[str, Any]]:
    """按配置构建短期记忆，返回 ``(实现, 状态)``。"""
    if settings.memory_backend != "redis":
        return (
            InMemoryShortTermMemory(max_turns=settings.short_term_turns),
            {"backend": "memory", "detail": "进程内实现"},
        )

    try:
        from app.memory.redis_impl import RedisShortTermMemory, create_redis_client

        client = await create_redis_client(settings.redis_url)
        backend = RedisShortTermMemory(
            client,
            max_turns=settings.short_term_turns,
            key_prefix=settings.redis_key_prefix,
        )
        return backend, {"backend": "redis", "detail": settings.redis_url}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 不可用（%s），短期记忆降级到进程内实现", exc)
        if not settings.storage_fallback:
            raise
        return (
            InMemoryShortTermMemory(max_turns=settings.short_term_turns),
            {"backend": "memory", "detail": f"Redis 不可用，已降级：{exc}"},
        )


async def build_profile_store(
    settings: Settings,
) -> tuple[ProfileStore, dict[str, Any]]:
    """按配置构建长期存储，返回 ``(实现, 状态)``。"""
    if settings.db_backend != "sql":
        return InMemoryProfileStore(), {"backend": "memory", "detail": "进程内实现"}

    dsn = settings.postgres_dsn or settings.sql_dsn
    try:
        from app.memory.sql_impl import SqlProfileStore

        store = await SqlProfileStore.create(dsn)
        return store, {"backend": "sql", "detail": dsn}
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQL 存储不可用（%s），长期存储降级到进程内实现", exc)
        if not settings.storage_fallback:
            raise
        return InMemoryProfileStore(), {"backend": "memory", "detail": f"SQL 不可用，已降级：{exc}"}


__all__ = ["build_profile_store", "build_short_term_memory"]
