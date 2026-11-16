"""通知通道适配器。

文档口径：「通知老师 / 家长；停止建议，人工接手」，以及「老师端只收到
『已识别高危』」。

所以通知体是**白名单**构造的：只有 student_id、风险等级、原因码、trace_id、
时间戳，**不含学生原话**。

通道类型：

* ``http(s)://...`` —— 真实 HTTP 投递（webhook / 内部网关）；
* ``outbox`` —— 落地到本地 JSONL（默认，等真实通道接上）；
* 其他名字 —— 只记日志（``console`` 语义）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: 通知体白名单
ALLOWED_NOTIFY_KEYS = ("student_id", "risk_level", "reason_code", "trace_id", "ts")


class NotifyChannel(Protocol):
    """通知通道接口。"""

    name: str

    async def send(self, payload: dict[str, Any]) -> tuple[bool, str]: ...


class WebhookChannel:
    """HTTP webhook 通道。"""

    def __init__(self, name: str, url: str, timeout: float = 15.0) -> None:
        self.name = name
        self.url = url
        self.timeout = timeout

    async def send(self, payload: dict[str, Any]) -> tuple[bool, str]:
        import httpx  # noqa: PLC0415

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.url, json=payload)
            if resp.status_code < 300:
                return True, f"delivered:{resp.status_code}"
            return False, f"failed:http_{resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            logger.error("通知通道 %s 投递失败：%s", self.name, exc)
            return False, f"failed:{type(exc).__name__}"


class OutboxChannel:
    """本地 outbox 通道：把通知落到 JSONL，等待接入真实通道。"""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path

    async def send(self, payload: dict[str, Any]) -> tuple[bool, str]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"kind": "notify", "channel": self.name, "payload": payload},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return True, "outbox"
        except Exception as exc:  # noqa: BLE001
            logger.error("outbox 写入失败：%s", exc)
            return False, f"failed:{type(exc).__name__}"


class ConsoleChannel:
    """只记日志的通道。"""

    def __init__(self, name: str) -> None:
        self.name = name

    async def send(self, payload: dict[str, Any]) -> tuple[bool, str]:
        logger.warning("[通知:%s] %s", self.name, json.dumps(payload, ensure_ascii=False))
        return True, "console"


class Notifier:
    """按通道名分发通知。"""

    def __init__(self, outbox_path: Path, timeout: float = 15.0) -> None:
        self.outbox_path = outbox_path
        self.timeout = timeout

    # ------------------------------------------------------------------
    def build_channel(self, name: str) -> NotifyChannel:
        if name.startswith("http://") or name.startswith("https://"):
            return WebhookChannel(name, name, timeout=self.timeout)
        if name in {"outbox", "file", "local"}:
            return OutboxChannel(name, self.outbox_path)
        return ConsoleChannel(name)

    @staticmethod
    def build_payload(
        *,
        student_id: str,
        risk_level: str,
        reason_code: str,
        trace_id: str,
        ts: float,
    ) -> dict[str, Any]:
        """白名单构造通知体（不含学生原话）。"""
        payload = {
            "student_id": student_id,
            "risk_level": risk_level,
            "reason_code": reason_code,
            "trace_id": trace_id,
            "ts": ts,
        }
        return {k: v for k, v in payload.items() if k in ALLOWED_NOTIFY_KEYS}

    async def notify(
        self, channels: list[str], payload: dict[str, Any]
    ) -> list[str]:
        """逐通道投递，返回成功投递的通道名。"""
        delivered: list[str] = []
        for name in channels:
            if not name:
                continue
            channel = self.build_channel(name)
            ok, status = await channel.send(payload)
            if ok:
                delivered.append(name)
            else:
                logger.error("通知通道 %s 投递失败：%s", name, status)
        return delivered


__all__ = [
    "ALLOWED_NOTIFY_KEYS",
    "ConsoleChannel",
    "Notifier",
    "NotifyChannel",
    "OutboxChannel",
    "WebhookChannel",
]
