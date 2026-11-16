"""离线规则模型（AgentScope ``ChatModelBase`` 实现）。

作用：在没有任何 API Key 的环境下，让 Worker / 编排器 / 检索 / 记忆的**全链路
真实跑通**——它不是返回固定假字符串，而是按文档描述的业务语义做真实推断：

* 情绪识别 → 规则引擎产出类型 / 强度 / 诱因 / 认知偏差 / 风险分；
* 需求澄清 → 依据情绪与学生原话判定 ``intervention_fit``；
* 积极引导 → 依据检索到的策略卡与禁止说法卡，按四步结构成稿；
* 风险分 → 只输出 0-1 风险分与依据（对应文档里的"小模型"）。

切换到真实模型只需改 ``MODEL_PROVIDER=openai``，其余代码零改动。
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

from agentscope.credential import CredentialBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolCallState
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from pydantic import BaseModel

from app.models.rule_engine import (
    compose_reply,
    compose_safety_reply,
    compose_teacher_reply,
    infer_emotion,
    infer_needs,
)

#: 系统提示里的角色标记，用于让规则模型知道自己在扮演哪个 Worker
WORKER_MARKER_RE = re.compile(r"\[\[worker:([a-z_]+)\]\]")
#: 上下文 JSON 块的包裹标记
CTX_BLOCK_RE = re.compile(r"<<CTX>>\s*(\{.*?\})\s*<<END>>", re.S)

STRUCTURED_TOOL_NAME = "GenerateStructuredOutput"


class _LocalCredential(CredentialBase):
    """本地模型无需真实凭据。"""

    name: str = "local-rule-model"


def _msg_text(msg: Msg) -> str:
    """把一条消息的内容块拼成纯文本。"""
    parts: list[str] = []
    for block in msg.content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _system_prompt(messages: list[Msg]) -> str:
    for msg in messages:
        if msg.role == "system":
            return _msg_text(msg)
    return ""


def _last_user_text(messages: list[Msg]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return _msg_text(msg)
    return ""


def _extract_context(messages: list[Msg]) -> dict[str, Any]:
    """从提示里解析 <<CTX>>{...}<<END>> 上下文块。"""
    for msg in reversed(messages):
        match = CTX_BLOCK_RE.search(_msg_text(msg))
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return {}
    return {}


class RuleBasedChatModel(ChatModelBase):
    """用本地规则引擎顶替大模型的 ``ChatModelBase`` 实现。"""

    class Parameters(BaseModel):
        temperature: float = 0.0
        max_tokens: int | None = None
        purpose: str = "main"

    def __init__(
        self,
        *,
        purpose: str = "main",
        temperature: float = 0.0,
        stream: bool = True,
    ) -> None:
        super().__init__(
            credential=_LocalCredential(),
            model=f"local-rule-{purpose}",
            parameters=self.Parameters(temperature=temperature, purpose=purpose),
            stream=stream,
            max_retries=0,
        )
        self.formatter = OpenAIChatFormatter()
        self.purpose = purpose

    # ------------------------------------------------------------------
    # ChatModelBase 契约
    # ------------------------------------------------------------------
    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        tool_names = [t.get("function", {}).get("name") for t in (tools or [])]

        if STRUCTURED_TOOL_NAME in tool_names:
            payload = self._structured_payload(messages, tools or [])
            async def gen() -> AsyncGenerator[ChatResponse, None]:
                yield ChatResponse(
                    content=[
                        ToolCallBlock(
                            type="tool_call",
                            id=f"tc-{self.purpose}",
                            name=STRUCTURED_TOOL_NAME,
                            input=json.dumps(payload, ensure_ascii=False),
                            state=ToolCallState.PENDING,
                        ),
                    ],
                    is_last=True,
                    finished_reason=FinishedReason.COMPLETED,
                )

            return gen()

        text = self._plain_text(messages)
        async def gen() -> AsyncGenerator[ChatResponse, None]:
            for chunk in _chunk_text(text):
                yield ChatResponse(
                    content=[TextBlock(type="text", text=chunk)],
                    is_last=False,
                    finished_reason=FinishedReason.COMPLETED,
                )
            # AgentScope 以 is_last=True 的分片作为"完整回复"构造最终 Msg，
            # 所以最后一片必须带上完整文本（流式时它不会被重复吐成 delta）。
            yield ChatResponse(
                content=[TextBlock(type="text", text=text)] if text else [],
                is_last=True,
                finished_reason=FinishedReason.COMPLETED,
            )

        return gen()

    # ------------------------------------------------------------------
    # 规则实现
    # ------------------------------------------------------------------
    def _worker_role(self, messages: list[Msg]) -> str:
        marker = WORKER_MARKER_RE.search(_system_prompt(messages))
        return marker.group(1) if marker else "guidance"

    def _structured_payload(
        self,
        messages: list[Msg],
        tools: list[dict],
    ) -> dict[str, Any]:
        """按目标 schema 计算结构化输出。"""
        schema: dict[str, Any] = {}
        for tool in tools:
            if tool.get("function", {}).get("name") == STRUCTURED_TOOL_NAME:
                schema = tool["function"].get("parameters", {}) or {}
                break
        props = set(schema.get("properties", {}).keys())

        ctx = _extract_context(messages)
        text = str(ctx.get("text") or _last_user_text(messages))
        recent = ctx.get("recent_texts") or []

        # 风险分小模型：只输出 risk_score / risk_reason
        if props and props <= {"risk_score", "risk_reason", "confidence"}:
            emotion = infer_emotion(text, recent)
            payload: dict[str, Any] = {
                "risk_score": emotion.risk_score,
                "risk_reason": emotion.risk_reason,
            }
            if "confidence" in props:
                payload["confidence"] = emotion.confidence
            return payload

        # 情绪识别
        if {"emotion_type", "intensity"} & props:
            emotion = infer_emotion(text, recent)
            payload = {
                "emotion_type": emotion.emotion_type,
                "intensity": emotion.intensity,
                "cause_domain": emotion.cause_domain.value,
                "cause_detail": emotion.cause_detail,
                "candidate_distortions": emotion.candidate_distortions,
                "risk_score": emotion.risk_score,
                "risk_reason": emotion.risk_reason,
                "confidence": emotion.confidence,
            }
            return {k: v for k, v in payload.items() if k in props}

        # 需求澄清
        if {"needs", "intervention_fit"} & props:
            emotion = infer_emotion(text, recent)
            needs = infer_needs(text, emotion, ctx.get("profile"))
            payload = {
                "needs": needs.needs,
                "intervention_fit": needs.intervention_fit.value,
                "reasoning": needs.reasoning,
            }
            return {k: v for k, v in payload.items() if k in props}

        # 兜底：返回空对象
        return {}

    def _plain_text(self, messages: list[Msg]) -> str:
        role = self._worker_role(messages)
        ctx = _extract_context(messages)
        text = str(ctx.get("text") or _last_user_text(messages))

        if role == "safety":
            return compose_safety_reply()
        if role == "needs":
            emotion = infer_emotion(text, ctx.get("recent_texts") or [])
            needs = infer_needs(text, emotion, ctx.get("profile"))
            return json.dumps(
                {
                    "needs": needs.needs,
                    "intervention_fit": needs.intervention_fit.value,
                    "reasoning": needs.reasoning,
                },
                ensure_ascii=False,
            )

        emotion = infer_emotion(text, ctx.get("recent_texts") or [])
        needs = infer_needs(text, emotion, ctx.get("profile"))
        cards = ctx.get("cards") or []
        if str(ctx.get("speaker") or "student") == "teacher":
            return compose_teacher_reply(text, emotion, needs)
        reply = compose_reply(text, emotion, needs, cards, ctx.get("profile"))
        if ctx.get("exercise_text"):
            reply = f"{reply}\n\n{ctx['exercise_text']}"
        return reply


def _chunk_text(text: str, size: int = 12) -> list[str]:
    """把整段文本切成便于流式观测的小块。"""
    if not text:
        return [""]
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    return chunks


__all__ = ["RuleBasedChatModel"]
