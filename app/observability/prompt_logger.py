"""AI 调用 Prompt 日志。

把每一次调用大模型的**完整提示词**（system prompt + user message）落盘，
供事后排查"这一轮到底给模型看了什么"。

设计要点：

* 落在模型层唯一收口 :class:`~app.models.base.ResilientChatModel` 里，
  真实模型与规则模型、四个 Worker 全部覆盖，无需改动任何 Worker。
* 按天分文件：``data/prompts/YYYY-MM-DD.jsonl``，每行一条 JSON。
* worker 名从 system prompt 开头的 ``[[worker:xxx]]`` 标记解析，
  不需要在调用链里额外透传。
* 写入失败只告警、不影响主流程（Prompt 日志是观测手段，不是业务依赖）。

一行记录的结构::

    {
        "ts": 1710000000.0,
        "trace_id": "trace_xxx",
        "worker": "emotion",          # 从 [[worker:xxx]] 解析，缺省 "guidance"
        "model": "MiniMax-M2.7",
        "backend": "openai",          # openai / mock
        "system_prompt": "...",
        "user_message": "...",
    }
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 系统提示开头的工作标记，如 [[worker:emotion]]
_WORKER_MARKER_RE = re.compile(r"\[\[worker:([a-z_]+)\]\]")


def _extract_worker(system_prompt: str) -> str:
    """从系统提示的工作标记解析 worker 名，缺省 ``guidance``。"""
    match = _WORKER_MARKER_RE.search(system_prompt or "")
    return match.group(1) if match else "guidance"


def _msg_to_text(msg: Any) -> str:
    """把一条 AgentScope ``Msg`` 的内容块拼成纯文本。

    兼容 ``content`` 是文本块列表（TextBlock）或纯字符串两种情况。
    """
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts)


class PromptLogger:
    """把 AI 调用的提示词追加写进按天滚动的 JSONL 文件。"""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self._directory = Path(directory)
        self._enabled = enabled
        # 追加写是原子的前提是单次 write 不被打断；异步并发场景下加锁，
        # 避免两个 Worker 同时写一行导致 JSONL 损坏。
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def directory(self) -> Path:
        return self._directory

    def _today_file(self) -> Path:
        return self._directory / f"{time.strftime('%Y-%m-%d')}.jsonl"

    def record(
        self,
        *,
        trace_id: str,
        model: str,
        backend: str,
        system_prompt: str,
        user_message: str,
    ) -> None:
        """追加一条 Prompt 日志（失败只告警，不抛异常）。"""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            "ts": time.time(),
            "trace_id": trace_id,
            "worker": _extract_worker(system_prompt),
            "model": model,
            "backend": backend,
            "system_prompt": system_prompt,
            "user_message": user_message,
        }
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock:
                with self._today_file().open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.write("\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prompt 日志写入失败：%s", exc)


__all__ = ["PromptLogger", "extract_system_prompt", "extract_user_message"]


def extract_system_prompt(messages: list[Any]) -> str:
    """从 AgentScope 消息列表里取出 system 角色的内容。"""
    for msg in messages or []:
        if getattr(msg, "role", None) == "system":
            return _msg_to_text(msg)
    return ""


def extract_user_message(messages: list[Any]) -> str:
    """从 AgentScope 消息列表里取出 user 角色的内容。"""
    for msg in messages or []:
        if getattr(msg, "role", None) == "user":
            return _msg_to_text(msg)
    return ""
