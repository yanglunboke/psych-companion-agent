"""Worker 2：积极引导（唯一给学生回复的 Worker）+ 高危安全回应。

两条路径：

* :meth:`GuidanceWorker.stream` —— 常规路径，流式吐出学生可见回复，
  首字延迟就是它吐出第一个字的时间；
* :meth:`GuidanceWorker.safety_reply` —— 高危路径，只跑 ``safety_reply``，
  给一段简短的安全回应，不讲道理、不给方法。

Worker 的工具箱里挂了两个真实可调用的工具（``search_strategy`` /
``suggest_exercise``），方便接真模型时让它按需 ReAct 取卡；离线规则模型
不会调用它们，卡片由编排器提前检索好直接注入。
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from agentscope.message import TextBlock, UserMsg
from agentscope.tool import FunctionTool, Toolkit, ToolResponse
from pydantic import BaseModel, Field

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
            )
        )
        await toolkit.add_tool(
            FunctionTool(
                func=suggest_exercise,
                name="suggest_exercise",
                description="当学生需要可操作的方法时，取一张练习卡",
                input_schema=SuggestExerciseInput,
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
        async for evt in agent.reply_stream(UserMsg(name="student", content=content)):
            delta = getattr(evt, "delta", None)
            if isinstance(delta, str) and delta:
                yield delta

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
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()


__all__ = ["GuidanceWorker", "SearchStrategyInput", "SuggestExerciseInput"]
