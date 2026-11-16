"""Trace / 工具调用日志。

文档要求："一轮请求进来会先给一个 trace id，这轮调过的工具都记在这个 id 下面。
之后查的话，就查这个 trace id 把顺序拿出来看：先过的安全门，再情绪识别，
再检索，最后生成。"

本模块实现：

* :class:`Trace` —— 单轮追踪器，记录有序的工具调用日志；
* :class:`TraceStore` —— 进程内 trace 仓库（可扩展为 Postgres / ClickHouse）；
* :func:`new_trace_id` —— 生成 trace id。
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from app.schemas import ToolCallLog


def new_trace_id() -> str:
    """生成一轮请求的 trace id。"""
    return f"trace_{uuid.uuid4().hex[:24]}"


class Trace:
    """单轮请求的追踪器。

    用法::

        trace = Trace(new_trace_id())
        trace.record("emotion_recognition", key_args={"text_len": 18}, success=True)
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or new_trace_id()
        self.started_at = time.time()
        self.logs: list[ToolCallLog] = []
        self._seq = 0

    def record(
        self,
        tool_name: str,
        *,
        key_args: dict[str, Any] | None = None,
        success: bool = True,
        duration_ms: float = 0.0,
        detail: str = "",
    ) -> ToolCallLog:
        """记录一条工具调用日志（保持调用顺序）。"""
        self._seq += 1
        entry = ToolCallLog(
            trace_id=self.trace_id,
            seq=self._seq,
            tool_name=tool_name,
            key_args=key_args or {},
            success=success,
            duration_ms=round(duration_ms, 2),
            detail=detail,
            ts=time.time(),
        )
        self.logs.append(entry)
        return entry

    def tool_names(self) -> list[str]:
        return [log.tool_name for log in self.logs]

    def elapsed_ms(self) -> float:
        return round((time.time() - self.started_at) * 1000, 2)


class TraceStore:
    """进程内 trace 仓库。

    保留最近 ``max_traces`` 轮，超出后按先进先出淘汰。生产环境可替换为
    Postgres 实现（``save`` / ``get`` 两个方法签名不变）。
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self._traces: OrderedDict[str, Trace] = OrderedDict()
        self._max_traces = max_traces

    def save(self, trace: Trace) -> None:
        self._traces[trace.trace_id] = trace
        self._traces.move_to_end(trace.trace_id)
        while len(self._traces) > self._max_traces:
            self._traces.popitem(last=False)

    def get(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    def list_ids(self) -> list[str]:
        return list(self._traces.keys())


@runtime_checkable
class TraceSink(Protocol):
    """trace 落库接口。

    有了它，「工具调用可追溯」就不再局限在单进程内存里 —— 换进程、重启服务
    之后仍能按 trace_id 把这一轮的调用顺序查出来。
    """

    async def write(self, logs: list[ToolCallLog]) -> None: ...


class SqlTraceSink:
    """把工具调用日志写进 SQL 存储的 trace sink。"""

    def __init__(self, store: Any) -> None:
        self.store = store

    async def write(self, logs: list[ToolCallLog]) -> None:
        for log in logs:
            await self.store.append_tool_call(log)


__all__ = ["SqlTraceSink", "Trace", "TraceSink", "TraceStore", "new_trace_id"]
