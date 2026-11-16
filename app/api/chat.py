"""对话接口。

* ``POST /api/chat``      —— SSE 流式（学生端用；首字延迟由此观测）
* ``POST /api/chat/sync`` —— 一次性返回完整结果（脚本 / 联调用）
* ``GET  /api/trace/{id}``—— 查这一轮的工具调用顺序（trace）
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse, Route

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _container(request: Request):
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    return container


@router.post("/api/chat")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式对话。"""
    container = _container(request)

    async def event_stream():
        async for event in container.orchestrator.handle(req):
            payload = json.dumps(event.to_dict(), ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/sync", response_model=ChatResponse)
async def chat_sync(req: ChatRequest, request: Request) -> ChatResponse:
    """同步对话（把 SSE 事件收敛成一个结果对象）。"""
    container = _container(request)

    reply_parts: list[str] = []
    outcome = None
    async for event in container.orchestrator.handle(req):
        if event.type == "delta":
            reply_parts.append(str(event.data.get("text", "")))
        elif event.type == "done":
            outcome = event.data

    if outcome is None:
        raise HTTPException(status_code=500, detail="编排器未产生完成事件")

    escalation = outcome.get("escalation")
    return ChatResponse(
        trace_id=str(outcome.get("trace_id", "")),
        session_id=req.session_id,
        student_id=req.student_id,
        reply="".join(reply_parts),
        route=Route(outcome.get("route", "react")),
        risk_level=(escalation or {}).get("risk_level", "low"),
        ttft_ms=float(outcome.get("ttft_ms", 0.0)),
        total_ms=float(outcome.get("total_ms", 0.0)),
        tool_calls=outcome.get("tool_calls", []),
    )


@router.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str, request: Request) -> dict:
    """按 trace id 查这一轮的工具调用顺序。"""
    container = _container(request)
    trace = container.trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace 不存在")
    return {
        "trace_id": trace.trace_id,
        "tools": trace.tool_names(),
        "logs": [log.model_dump(mode="json") for log in trace.logs],
    }


__all__ = ["router"]
