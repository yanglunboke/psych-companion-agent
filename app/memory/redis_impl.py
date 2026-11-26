"""Redis 短期记忆实现。

文档口径：「redis 放最近 12 轮原话，每次的检索规则一样，是与学生对话的热缓存。」

**按学生组织，会话是标签**。键布局（``key_prefix`` 默认 ``psych``）::

    psych:student:<student_id>:session:<session_id>   LIST  该会话的最近若干轮
    psych:student:<student_id>:sessions               ZSET  该学生的会话索引（score=最后活动时间）
    psych:owner:<session_id>                          STR   会话归属，供 history 接口反查

为什么不沿用 ``psych:session:<id>``：会话只是界面的一个壳，人没变。
学生点「新对话」只是清屏，如果窗口挂在会话上，换个会话就什么都不记得了。
所以 ``get_recent`` 默认把该学生**所有会话**合并成一条时间线取最近 N 轮，
前端恢复历史时再按 ``session_id`` 过滤。

窗口按学生轮次算（一次学生发言 + 一次助手回复算一轮）：

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

from app.memory.store import tail_by_turns
from app.schemas import Turn

logger = logging.getLogger(__name__)

#: 跨会话取上下文时最多合并多少个会话（防止一个学生开了几百个会话时扫太久）
MAX_FANOUT = 20

#: 没有 session_id 的消息归到这个桶（旧数据 / 脚本直接写）
UNTAGGED = "_untagged"


class RedisShortTermMemory:
    """基于 Redis 的短期记忆，按学生聚合。"""

    def __init__(
        self,
        client: Any,
        *,
        max_turns: int = 12,
        key_prefix: str = "psych",
        ttl_seconds: int = 7 * 24 * 3600,
    ) -> None:
        self.client = client
        self.max_turns = max_turns
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # 键
    def _student(self, student_id: str) -> str:
        return student_id or "anonymous"

    def _session_key(self, student_id: str, session_id: str) -> str:
        return f"{self.key_prefix}:student:{self._student(student_id)}:session:{session_id}"

    def _sessions_key(self, student_id: str) -> str:
        return f"{self.key_prefix}:student:{self._student(student_id)}:sessions"

    def _owner_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:owner:{session_id}"

    def _bucket_key(self, student_id: str, session_id: str | None) -> str:
        """用于展示的键名（不给会话时展示学生级的桶）。"""
        if session_id is None:
            return f"{self.key_prefix}:student:{self._student(student_id)}"
        return self._session_key(student_id, session_id)

    # ------------------------------------------------------------------
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

    async def _read_session(self, student_id: str, session_id: str) -> list[Turn]:
        raw_items = await self.client.lrange(self._session_key(student_id, session_id), 0, -1)
        return [t for t in (self._decode(raw) for raw in raw_items) if t is not None]

    async def _read_all(self, student_id: str) -> list[Turn]:
        """把该学生最近 ``MAX_FANOUT`` 个会话合并成一条时间线。"""
        index_key = self._sessions_key(student_id)
        # ZREVRANGE：最近活跃的会话排在前面
        sessions = await self.client.zrevrange(index_key, 0, MAX_FANOUT - 1)
        if not sessions:
            return []
        merged: list[Turn] = []
        for session_id in sessions:
            merged.extend(await self._read_session(student_id, session_id))
        # sorted 是稳定排序：同一会话内 ts 相同的两条不会被打乱
        merged.sort(key=lambda t: t.ts)
        return merged

    # ------------------------------------------------------------------
    async def get_recent(
        self, student_id: str, limit: int = 12, *, session_id: str | None = None
    ) -> list[Turn]:
        """取最近 ``limit`` 轮（按学生轮次算）。

        不给 ``session_id`` 时跨会话合并 —— 同一个学生换会话不失忆；
        给了就只返回那个会话的原话，前端恢复历史用。
        """
        if session_id is not None:
            turns = await self._read_session(student_id, session_id)
        else:
            turns = await self._read_all(student_id)
        return tail_by_turns(turns, limit)

    async def append(self, student_id: str, turn: Turn) -> None:
        """追加一条消息、续期、并把会话登记进索引。"""
        session_id = turn.session_id or UNTAGGED
        key = self._session_key(student_id, session_id)
        score = turn.ts or time.time()
        pipe = self.client.pipeline()
        pipe.rpush(key, self._encode(turn))
        pipe.ltrim(key, -(self.max_turns * 2), -1)
        pipe.expire(key, self.ttl_seconds)
        if turn.session_id:
            pipe.zadd(self._sessions_key(student_id), {turn.session_id: score})
            pipe.expire(self._sessions_key(student_id), self.ttl_seconds)
            pipe.set(self._owner_key(turn.session_id), self._student(student_id), ex=self.ttl_seconds)
        await pipe.execute()

    async def clear(self, student_id: str, *, session_id: str | None = None) -> None:
        if session_id is None:
            sessions = await self.client.zrange(self._sessions_key(student_id), 0, -1)
            pipe = self.client.pipeline()
            for sid in sessions or []:
                pipe.delete(self._session_key(student_id, sid))
                pipe.delete(self._owner_key(sid))
            pipe.delete(self._sessions_key(student_id))
            await pipe.execute()
            return

        pipe = self.client.pipeline()
        pipe.delete(self._session_key(student_id, session_id))
        pipe.zrem(self._sessions_key(student_id), session_id)
        pipe.delete(self._owner_key(session_id))
        await pipe.execute()

    # ------------------------------------------------------------------
    # 调试用：暴露"这个窗口在 Redis 里长什么样"，不泄漏客户端本身
    async def describe(self, student_id: str, *, session_id: str | None = None) -> dict[str, Any]:
        if session_id is None:
            turns = await self._read_all(student_id)
            index_size = await self.client.zcard(self._sessions_key(student_id))
            ttl = await self.client.ttl(self._sessions_key(student_id))
            key = self._bucket_key(student_id, None)
            messages = len(turns)
        else:
            raw_items = await self.client.lrange(
                self._session_key(student_id, session_id), 0, -1
            )
            turns = [t for t in (self._decode(raw) for raw in raw_items) if t is not None]
            key = self._session_key(student_id, session_id)
            ttl = await self.client.ttl(key)
            messages = len(raw_items)
            index_size = await self.client.zcard(self._sessions_key(student_id))

        return {
            "backend": "redis",
            "key": key,
            "messages": messages,
            "turns": sum(1 for t in turns if t.role == "student"),
            "sessions": index_size,
            # -1 = 键存在但没有过期时间，-2 = 键不存在
            "ttl_seconds": ttl if ttl >= 0 else None,
        }

    async def list_sessions(self) -> list[dict]:
        """扫描现存的 (学生, 会话) 组合（SCAN，不用 KEYS）。"""
        prefix = f"{self.key_prefix}:student:"
        marker = ":session:"
        rows: list[dict] = []
        async for key in self.client.scan_iter(match=f"{prefix}*{marker}*", count=200):
            rest = key[len(prefix):]
            # 学号不会包含 ":session:"，用 rsplit 把学生和会话切开
            student_id, _, session_id = rest.rpartition(marker)
            raw_items = await self.client.lrange(key, 0, -1)
            turns = [t for t in (self._decode(raw) for raw in raw_items) if t is not None]
            rows.append(
                {
                    "student_id": student_id,
                    "session_id": None if session_id == UNTAGGED else session_id,
                    "messages": len(raw_items),
                    "turns": sum(1 for t in turns if t.role == "student"),
                    "last_ts": max((t.ts for t in turns), default=0.0),
                }
            )
        rows.sort(key=lambda r: r["last_ts"], reverse=True)
        return rows

    async def owner_of(self, session_id: str) -> str | None:
        return await self.client.get(self._owner_key(session_id))

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
