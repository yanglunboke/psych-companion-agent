"""模型层测试：规则模型能否在 AgentScope Agent 上产出结构化输出与流式文本。"""

from __future__ import annotations

import pytest
from agentscope.agent import Agent
from agentscope.message import UserMsg
from pydantic import BaseModel, Field

from app.models.mock_model import RuleBasedChatModel


class EmotionOut(BaseModel):
    emotion_type: str
    intensity: int
    cause_domain: str
    candidate_distortions: list[str] = Field(default_factory=list)
    risk_score: float


class NeedsOut(BaseModel):
    needs: list[str] = Field(default_factory=list)
    intervention_fit: str
    reasoning: str = ""


@pytest.mark.asyncio
async def test_structured_emotion_output() -> None:
    agent = Agent(
        name="emotion",
        system_prompt="[[worker:emotion]] 你是情绪识别助手，只输出 JSON。",
        model=RuleBasedChatModel(),
    )
    msg = await agent.reply(
        UserMsg(name="student", content="这次月考又考砸了，我真的太笨了"),
        structured_schema=EmotionOut,
    )
    assert msg.structured_output is not None
    assert msg.structured_output["emotion_type"] == "沮丧"
    assert msg.structured_output["intensity"] == 4
    assert msg.structured_output["risk_score"] < 0.72


@pytest.mark.asyncio
async def test_structured_needs_output() -> None:
    agent = Agent(
        name="needs",
        system_prompt="[[worker:needs]] 你是需求澄清助手。",
        model=RuleBasedChatModel(),
    )
    msg = await agent.reply(
        UserMsg(name="student", content="我只想说说，没什么大事"),
        structured_schema=NeedsOut,
    )
    assert msg.structured_output is not None
    assert msg.structured_output["intervention_fit"] == "只想说说"


@pytest.mark.asyncio
async def test_streaming_emits_text_deltas() -> None:
    agent = Agent(
        name="guidance",
        system_prompt="[[worker:guidance]] 你是积极引导助手。",
        model=RuleBasedChatModel(),
    )
    chunks: list[str] = []
    async for evt in agent.reply_stream(UserMsg(name="student", content="这次月考又考砸了")):
        delta = getattr(evt, "delta", None)
        if isinstance(delta, str):
            chunks.append(delta)
    assert "".join(chunks).strip() != ""
