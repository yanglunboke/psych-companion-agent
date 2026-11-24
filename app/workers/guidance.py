"""Worker 2：积极引导（唯一给学生回复的 Worker）+ 高危安全回应。

两条路径：

* :meth:`GuidanceWorker.stream` —— 常规路径，流式吐出学生可见回复，
  首字延迟就是它吐出第一个字的时间；
* :meth:`GuidanceWorker.safety_reply` —— 高危路径，只跑 ``safety_reply``，
  给一段简短的安全回应，不讲道理、不给方法。

Worker 的工具箱里挂了两个真实可调用的工具（``search_strategy`` /
``suggest_exercise``），方便接真模型时让它按需 ReAct 取卡；离线规则模型
不会调用它们，卡片由编排器提前检索好直接注入。

**学生可见内容的边界在这里划定。** AgentScope 的流式事件里除了正文增量
（``TextBlockDeltaEvent``），还有两类绝不能给学生看的东西：

* ``ToolCallDeltaEvent`` —— 工具调用的原始参数 JSON；
* 内联在正文里的推理内容（部分服务不返回 ``reasoning_content``，而是把
  ``<think>...</think>`` 直接塞进 ``content``）。

所以这里只转发正文增量，并在转发前用 :class:`~app.models.sanitize.ThinkFilter`
剥掉推理块。
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from agentscope.event import TextBlockDeltaEvent
from agentscope.message import TextBlock, UserMsg
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit, ToolResponse
from pydantic import BaseModel, Field

from app.models.sanitize import ThinkFilter, strip_think
from app.prompts import GUIDANCE_SYSTEM_PROMPT, SAFETY_SYSTEM_PROMPT, build_user_message
from app.schemas import EmotionCore, NeedResult, RetrievedCard, SessionContext
from app.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class SearchStrategyInput(BaseModel):
    query: str = Field(description="检索用的结构化查询，如 scene=exam_failure distortion=以偏概全")


class SuggestExerciseInput(BaseModel):
    scene: str = Field(default="", description="场景标识，如 exam_anxiety")


class GuidanceWorker(BaseWorker):
    """积极引导 Worker。"""

    name = "positive_guidance"
    max_iters = 4

    def __init__(self, model, *, retriever=None, exercise_worker=None) -> None:
        super().__init__(model)
        self.retriever = retriever
        self.exercise_worker = exercise_worker

    # ------------------------------------------------------------------
    async def build_toolkit(self) -> Toolkit:
        """挂上策略检索与资源建议两个工具。"""
        toolkit = Toolkit()

        async def search_strategy(query: str) -> ToolResponse:
            """按场景检索学生回复策略卡与禁止说法卡。

            Args:
                query: 结构化查询字符串
            """
            if self.retriever is None:
                return ToolResponse(
                    content=[TextBlock(type="text", text="检索器不可用")],
                    state="error",
                )
            from app.schemas import StructuredQuery

            structured = StructuredQuery(
                collection="student_reply",
                scene=query,
                keywords=[query],
            )
            result = self.retriever.search(structured, collections=["student_reply", "ban"])
            lines = [f"- {c.card_id}：{c.text}" for c in result.selected]
            return ToolResponse(
                content=[TextBlock(type="text", text="\n".join(lines) or "没有匹配的卡片")]
            )

        async def suggest_exercise(scene: str) -> ToolResponse:
            """取一张练习卡。

            Args:
                scene: 场景标识
            """
            if self.exercise_worker is None:
                return ToolResponse(
                    content=[TextBlock(type="text", text="资源建议模块不可用")],
                    state="error",
                )
            card = await self.exercise_worker.pick(scene=scene)
            if card is None:
                return ToolResponse(content=[TextBlock(type="text", text="没有匹配的练习卡")])
            text = "\n".join([f"{card.name}（{card.duration}）", *[f"{i}. {s}" for i, s in enumerate(card.steps, 1)]])
            return ToolResponse(content=[TextBlock(type="text", text=text)])

        await toolkit.add_tool(
            FunctionTool(
                func=search_strategy,
                name="search_strategy",
                description="按场景检索学生回复策略卡与禁止说法卡",
                input_schema=SearchStrategyInput,
                # AgentScope 的 FunctionTool 默认要求用户确认（ASK）。本系统
                # 没有交互式确认通道，不显式放行的话工具永远不会被执行——
                # 模型发起调用后会直接结束这一轮，学生拿不到任何回复。
                # 这两个工具都是只读检索，无副作用，可以直接放行。
                permission=PermissionDecision(
                    behavior=PermissionBehavior.ALLOW,
                    message="只读检索，无副作用，自动放行",
                ),
            )
        )
        await toolkit.add_tool(
            FunctionTool(
                func=suggest_exercise,
                name="suggest_exercise",
                description="当学生需要可操作的方法时，取一张练习卡",
                input_schema=SuggestExerciseInput,
                permission=PermissionDecision(
                    behavior=PermissionBehavior.ALLOW,
                    message="只读检索，无副作用，自动放行",
                ),
            )
        )
        return toolkit

    # ------------------------------------------------------------------
    async def stream(
        self,
        ctx: SessionContext,
        emotion: EmotionCore,
        needs: NeedResult,
        cards: Sequence[RetrievedCard] = (),
        *,
        exercise_text: str | None = None,
        reflection_hint: str | None = None,
    ) -> AsyncIterator[str]:
        """流式产出学生可见回复。"""
        content = build_user_message(
            text=ctx.current_text,
            recent_turns=ctx.recent_turns,
            profile=ctx.profile,
            emotion=emotion,
            needs=needs,
            cards=cards,
            exercise_text=exercise_text,
            reflection_hint=reflection_hint,
        )
        agent = self.build_agent(GUIDANCE_SYSTEM_PROMPT, toolkit=await self.build_toolkit())
        think_filter = ThinkFilter()
        started = False
        async for evt in agent.reply_stream(UserMsg(name="student", content=content)):
            # 只转发正文增量：工具调用参数（ToolCallDeltaEvent）与独立推理块
            # （ThinkingBlockDeltaEvent）都不是学生可见内容。
            if not isinstance(evt, TextBlockDeltaEvent):
                continue
            delta = evt.delta
            if not delta:
                continue
            clean = think_filter.feed(delta)
            if not started:
                # 模型在推理块之后通常会留一个空行，剥掉推理后就成了开头的
                # 空行。回复不该以空白开头——在 Worker 这里去掉，保证增量流、
                # 同步接口的拼接结果与落库文本三者完全一致。
                clean = clean.lstrip()
                if not clean:
                    continue
                started = True
            if clean:
                yield clean
        tail = think_filter.flush()
        if not started:
            tail = tail.lstrip()
        if tail:
            yield tail

    async def safety_reply(self, ctx: SessionContext, emotion: EmotionCore | None = None) -> str:
        """高危轮的简短安全回应。"""
        content = build_user_message(
            text=ctx.current_text,
            recent_turns=ctx.recent_turns,
            profile=ctx.profile,
            emotion=emotion,
        )
        agent = self.build_agent(SAFETY_SYSTEM_PROMPT, name="safety_reply")
        msg = await agent.reply(UserMsg(name="student", content=content))
        parts: list[str] = []
        for block in msg.content or []:
            text = getattr(block, "text", None)
            if not text:
                continue
            # 高危轮的草稿会被反思闸门逐字检查，推理内容必须先剥掉——
            # 否则模型英文独白里的问号会被判成"问了不止一个问题"，
            # 整个安全回应（含热线号码）会被拦下不发给学生。
            cleaned = strip_think(str(text)).strip()
            if cleaned:
                parts.append(cleaned)
        return "\n".join(parts).strip()


__all__ = ["GuidanceWorker", "SearchStrategyInput", "SuggestExerciseInput"]
