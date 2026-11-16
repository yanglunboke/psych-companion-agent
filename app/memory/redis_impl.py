"""Redis 短期记忆实现。

文档口径：「redis 放最近 12 轮原话，每次的检索规则一样，是与学生对话的热缓存。」

实现用 Redis 原生命令，**窗口按学生轮次算**（一次学生发言 + 一次助手回复算一轮）：

* ``RPUSH`` 追加一条消息；
* ``LTRIM`` 保留最近 ``max_turns * 2`` 条消息（12 轮 ≈ 24 条）；
* ``LRANGE`` 取回窗口。

连接不可用时由 :mod:`app.memory.factory` 决定是否降级到进程内实现。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.schemas import Turn

logger = logging.getLogger(__name__)


class RedisShortTermMemory:
    """基于 Redis 的短期记忆。"""

    def __init__(
        self,
        client: Any,
        *,
        max_turns: int = 12,
        key_prefix: str = "psych:session",
    ) -> None:
        self.client = client
        self.max_turns = max_turns
        self.key_prefix = key_prefix

    # ------------------------------------------------------------------
    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"

    @staticmethod
    def _encode(turn: Turn) -> str:
        return json.dumps(turn.model_dump(mode="json"), ensure_ascii=False)

    @staticmethod
    def _decode(raw: str) -> Turn | None:
        try:
            return Turn.model_validate(json.loads(raw))
        except Exception:  # noqa: BLE001
            logger.warning("忽略一条无法解析的短期记忆记录")
            return None

    # ------------------------------------------------------------------
    async def get_recent(self, session_id: str, limit: int = 12) -> list[Turn]:
        """取最近 ``limit`` 轮（按学生轮次算）。"""
        raw_items = await self.client.lrange(self._key(session_id), 0, -1)
        turns = [t for t in (self._decode(raw) for raw in raw_items) if t is not None]
        if not turns:
            return []
        student_positions = [i for i, t in enumerate(turns) if t.role == "student"]
        if not student_positions:
            return turns[-limit:]
        start = student_positions[max(0, len(student_positions) - limit)]
        return turns[start:]

    async def append(self, session_id: str, turn: Turn) -> None:
        """追加一条消息并裁剪窗口。"""
        key = self._key(session_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, self._encode(turn))
        pipe.ltrim(key, -(self.max_turns * 2), -1)
        pipe.expire(key, 7 * 24 * 3600)
        await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self.client.delete(self._key(session_id))

    async def close(self) -> None:
        close = getattr(self.client, "aclose", None) or getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


async def create_redis_client(redis_url: str) -> Any:
    """创建 redis.asyncio 客户端并做一次连通性探测。"""
    import redis.asyncio as aioredis  # noqa: PLC0415

    client = aioredis.from_url(redis_url, decode_responses=True)
    await client.ping()
    return client


__all__ = ["RedisShortTermMemory", "create_redis_client"]
