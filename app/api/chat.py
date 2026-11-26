"""对话接口。

* ``POST /api/chat``      —— SSE 流式（学生端用；首字延迟由此观测）
* ``POST /api/chat/sync`` —— 一次性返回完整结果（脚本 / 联调用）
* ``GET  /api/session/{id}/history`` —— 取回会话最近原话（刷新后恢复对话）
* ``GET  /api/trace/{id}``—— 查这一轮的工具调用顺序（trace）
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.auth import resolve_student_id
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
    # 带登录令牌时以令牌里的学生为准，请求体里声明的 student_id 不作数
    student_id = await resolve_student_id(request, req.student_id)
    if student_id != req.student_id:
        req = req.model_copy(update={"student_id": student_id})

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
    student_id = await resolve_student_id(request, req.student_id)
    if student_id != req.student_id:
        req = req.model_copy(update={"student_id": student_id})

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


@router.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, request: Request, limit: int = 24) -> dict:
    """取回某会话的最近若干条原话（学生端刷新页面后恢复对话用）。

    短期记忆按学生存，所以要先确定"这个会话是谁的"：带令牌时以令牌为准，
    没带令牌就反查会话登记过的学生。两者都指向不同的人时拒绝 —— 共享电脑上
    换了账号不该看到上一个人的原话。

    只读短期记忆，不触碰长期画像；会话不存在时返回空列表而不是 404 ——
    对前端来说"新会话"和"记忆已过期"是同一种情况。
    """
    container = _container(request)
    memory = container.short_memory
    capped = max(1, min(limit, 100))

    token_student = await resolve_student_id(request, "")
    owner = await memory.owner_of(session_id) if hasattr(memory, "owner_of") else None
    if token_student and owner and owner != token_student:
        raise HTTPException(status_code=403, detail="这个会话不属于当前登录账号")

    student_id = token_student or owner
    if not student_id:
        return {"session_id": session_id, "student_id": None, "count": 0, "messages": []}

    turns = await memory.get_recent(student_id, limit=capped, session_id=session_id)
    return {
        "session_id": session_id,
        "student_id": student_id,
        "count": len(turns),
        "messages": [
            {"role": turn.role, "content": turn.content, "ts": turn.ts} for turn in turns
        ],
    }


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