"""请求级上下文（ContextVar）。

把 trace_id 等请求级信息从编排器透传到模型层，供 Prompt 日志等观测组件
读取，避免在调用链里一路显式传参。
"""

from __future__ import annotations

from contextvars import ContextVar

#: 当前请求的 trace_id；编排器在每轮开始前设置，模型层读它落 Prompt 日志。
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


def set_trace_id(trace_id: str | None) -> None:
    """设置当前请求的 trace_id（编排器每轮开始前调用）。"""
    current_trace_id.set(trace_id)


def get_trace_id() -> str | None:
    """读取当前请求的 trace_id，未设置时返回 None。"""
    return current_trace_id.get()


__all__ = ["current_trace_id", "get_trace_id", "set_trace_id"]
