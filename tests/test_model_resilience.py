"""模型容错测试：真实模型不可用时必须自动降级，且如实暴露状态。"""

from __future__ import annotations

import pytest
from agentscope.message import UserMsg
from agentscope.model import ChatModelBase, ChatResponse

from app.config import Settings
from app.models.base import ModelFactory, ResilientChatModel
from app.models.mock_model import RuleBasedChatModel


class BrokenModel(ChatModelBase):
    """总是失败的模型（模拟鉴权失败 / 网络不通）。"""

    class Parameters(ChatModelBase.Parameters):
        temperature: float = 0.0
        max_tokens: int | None = None

    def __init__(self) -> None:
        from app.models.mock_model import _LocalCredential

        super().__init__(
            credential=_LocalCredential(),
            model="broken-model",
            parameters=self.Parameters(),
            stream=True,
            max_retries=0,
        )
        self.formatter = RuleBasedChatModel().formatter

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        raise RuntimeError("401 Unauthorized: invalid api key")


@pytest.mark.asyncio
async def test_resilient_model_falls_back_and_records_reason() -> None:
    resilient = ResilientChatModel(
        BrokenModel(),
        RuleBasedChatModel(),
        purpose="main",
        model_name="broken-model",
    )
    result = await resilient(
        [UserMsg(name="student", content="我这次月考考砸了")],
    )
    text = ""
    async for chunk in result:
        for block in chunk.content:
            text += getattr(block, "text", "") or ""
    assert text.strip() != ""
    assert resilient.status["backend"] == "mock"
    assert "401" in (resilient.status["fallback_reason"] or "")


@pytest.mark.asyncio
async def test_fallback_keeps_working_for_structured_output() -> None:
    from agentscope.agent import Agent

    resilient = ResilientChatModel(
        BrokenModel(),
        RuleBasedChatModel(),
        purpose="main",
        model_name="broken-model",
    )
    agent = Agent(name="emotion", system_prompt="[[worker:emotion]]", model=resilient)
    msg = await agent.reply(
        UserMsg(name="student", content="这次月考又考砸了"),
        structured_schema=None,
    )
    assert msg is not None
    assert resilient.status["backend"] == "mock"


def test_factory_status_reports_configured_provider() -> None:
    settings = Settings(model_provider="mock")
    factory = ModelFactory(settings)
    factory.create("main")
    status = factory.status()
    assert {s["purpose"] for s in status} == {"main", "risk"}
    assert all(s["backend"] == "mock" for s in status)


def test_factory_without_fallback_raises_on_bad_config() -> None:
    """构造阶段就失败（例如没有 API Key）且不允许降级时，必须抛出来。"""
    settings = Settings(
        model_provider="openai",
        model_api_key="",
        model_fallback_to_mock=False,
    )
    factory = ModelFactory(settings)
    with pytest.raises(Exception):
        factory.create("main")


def test_factory_falls_back_when_key_missing() -> None:
    settings = Settings(
        model_provider="openai",
        model_api_key="",
        model_fallback_to_mock=True,
    )
    factory = ModelFactory(settings)
    instance = factory.create("main")
    assert isinstance(instance, RuleBasedChatModel)


@pytest.mark.asyncio
async def test_circuit_breaker_stops_retrying_the_real_model() -> None:
    """真实模型失败后熔断：不再每轮白跑一次注定失败的请求。"""
    calls = {"n": 0}

    class CountingBrokenModel(BrokenModel):
        async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
            calls["n"] += 1
            raise RuntimeError("401 Unauthorized")

    resilient = ResilientChatModel(
        CountingBrokenModel(),
        RuleBasedChatModel(),
        purpose="main",
        model_name="broken-model",
        circuit_breaker=True,
    )
    for _ in range(3):
        result = await resilient([UserMsg(name="student", content="我这次月考考砸了")])
        async for _chunk in result:
            pass
    assert calls["n"] == 1, "熔断后不应再调用真实模型"
    assert resilient.status["circuit_breaker_open"] is True


@pytest.mark.asyncio
async def test_circuit_breaker_can_be_disabled() -> None:
    calls = {"n": 0}

    class CountingBrokenModel(BrokenModel):
        async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
            calls["n"] += 1
            raise RuntimeError("401 Unauthorized")

    resilient = ResilientChatModel(
        CountingBrokenModel(),
        RuleBasedChatModel(),
        purpose="main",
        model_name="broken-model",
        circuit_breaker=False,
    )
    for _ in range(3):
        result = await resilient([UserMsg(name="student", content="我这次月考考砸了")])
        async for _chunk in result:
            pass
    assert calls["n"] == 3
    assert resilient.status["circuit_breaker_open"] is False
