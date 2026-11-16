"""高危上报（escalate_risk）。

文档口径极其明确：

    escalate_risk 只传 student_id、风险等级和原因码，不传原话。
    老师端只收到「已识别高危」。

所以本模块的 payload 是**白名单**构造的：学生原话在结构上就进不来，
不是靠"记得别传"。投递记录落在本地 outbox（可查询），若配置了
``ESCALATION_WEBHOOK_URL`` 则改为真实 HTTP 投递；老师 / 家长通知走
:mod:`app.orchestrator.notify` 的通道适配器。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.config import Settings
from app.orchestrator.notify import Notifier
from app.schemas import EscalationRecord

logger = logging.getLogger(__name__)

#: 上报体白名单——只有这几个键可以离开本系统
ALLOWED_PAYLOAD_KEYS = ("student_id", "risk_level", "reason_code", "trace_id", "ts")


class EscalationService:
    """escalate_risk 工具 + 老师/家长通知。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.notifier = Notifier(settings.outbox_file, timeout=settings.model_timeout)

    # ------------------------------------------------------------------
    def build_payload(
        self,
        *,
        student_id: str,
        risk_level: str,
        reason_code: str,
        trace_id: str,
    ) -> dict[str, Any]:
        """构造上报体（白名单，杜绝原话外泄）。"""
        payload = {
            "student_id": student_id,
            "risk_level": risk_level,
            "reason_code": reason_code,
            "trace_id": trace_id,
            "ts": time.time(),
        }
        return {k: v for k, v in payload.items() if k in ALLOWED_PAYLOAD_KEYS}

    # ------------------------------------------------------------------
    async def escalate(
        self,
        *,
        student_id: str,
        risk_level: str,
        reason_code: str,
        trace_id: str,
    ) -> EscalationRecord:
        """执行上报并返回记录（含送达状态）。"""
        payload = self.build_payload(
            student_id=student_id,
            risk_level=risk_level,
            reason_code=reason_code,
            trace_id=trace_id,
        )
        record = EscalationRecord(
            student_id=student_id,
            risk_level=risk_level,
            reason_code=reason_code,
            trace_id=trace_id,
            delivery_status="pending",
            ts=payload["ts"],
        )

        delivered, status = await self._deliver(payload)
        record.delivered = delivered
        record.delivery_status = status

        notified = await self._notify(record)
        record.notified = notified
        return record

    async def _deliver(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """投递到 webhook；未配置时写入本地 outbox。"""
        webhook = self.settings.escalation_webhook_url.strip()
        if webhook:
            from app.orchestrator.notify import WebhookChannel

            return await WebhookChannel(
                "escalate_risk", webhook, timeout=self.settings.model_timeout
            ).send(payload)

        self._write_outbox({"kind": "escalate", "payload": payload})
        return True, "outbox"

    async def _notify(self, record: EscalationRecord) -> list[str]:
        """通知老师与家长。"""
        payload = self.notifier.build_payload(
            student_id=record.student_id,
            risk_level=record.risk_level,
            reason_code=record.reason_code,
            trace_id=record.trace_id,
            ts=record.ts,
        )
        return await self.notifier.notify(
            [
                self.settings.teacher_notify_channel,
                self.settings.parent_notify_channel,
            ],
            payload,
        )

    # ------------------------------------------------------------------
    def _write_outbox(self, item: dict[str, Any]) -> None:
        path: Path = self.settings.outbox_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


__all__ = ["ALLOWED_PAYLOAD_KEYS", "EscalationService"]
