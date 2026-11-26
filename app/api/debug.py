"""调试接口：直接查看短期记忆里某个会话的窗口内容。

存在的理由只有一个——**怀疑"上下文到底有没有进窗口"时，不必再猜**。
它能回答三件事：这个会话现在存在哪个后端、窗口里到底几条、取回时是哪几条。

三条硬约束：

* 默认关闭（``DEBUG_API_ENABLED=false``），关闭时返回 404 而不是 403，
  不向外界确认"这里有个调试接口"。
* 只读：只提供 GET，不提供任何写入 / 删除 / 清空能力。
* 没有鉴权，因此只允许在本机可信环境开启；返回的是学生原话，属于敏感数据。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.schemas import Turn

router = APIRouter(tags=["debug"])


def _memory_backend(request: Request) -> Any:
    """取当前生效的短期记忆，未启用调试接口时按"不存在"处理。"""
    settings = get_settings()
    if not settings.debug_api_enabled:
        # 用 404 而不是 403：不暴露调试接口的存在。
        raise HTTPException(status_code=404, detail="Not Found")
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(status_code=503, detail="容器尚未就绪")
    return container.short_memory


@router.get("/api/debug/memory/sessions")
async def list_sessions(request: Request) -> dict:
    """列出当前后端里存在的 (学生, 会话)，便于在页面上点进去看。"""
    memory = _memory_backend(request)
    if not hasattr(memory, "list_sessions"):
        return {"supported": False, "sessions": [], "reason": "该后端不支持枚举会话"}
    sessions = await memory.list_sessions()
    return {"supported": True, "count": len(sessions), "sessions": sessions}


@router.get("/api/debug/memory/{session_id}")
async def inspect_session(request: Request, session_id: str, limit: int = 12) -> dict:
    """查看某会话的窗口：存储概况 + 取回的原话列表。

    短期记忆按学生存，会话 ID 本身不足以定位数据，所以先用 ``owner_of``
    反查归属；查不到就说明这个会话从没写过（或已过期）。
    """
    memory = _memory_backend(request)
    if not hasattr(memory, "describe"):
        raise HTTPException(status_code=501, detail="该后端不支持窗口自描述")

    owner = await memory.owner_of(session_id) if hasattr(memory, "owner_of") else None
    if not owner:
        raise HTTPException(status_code=404, detail="该会话在短期记忆里不存在")

    describe = await memory.describe(owner, session_id=session_id)
    window: list[Turn] = await memory.get_recent(owner, limit, session_id=session_id)

    return {
        "session_id": session_id,
        "student_id": owner,
        **describe,
        "window_limit": limit,
        "window_messages": len(window),
        # get_recent 按"学生轮次"截取：这里显式标出实际生效的条数，
        # 避免把"窗口 12 轮"误读成"窗口 12 条"。
        "window": [
            {
                "role": turn.role,
                "content": turn.content,
                "ts": turn.ts,
                "trace_id": turn.trace_id,
            }
            for turn in window
        ],
    }
