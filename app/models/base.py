"""模型工厂 + 真实模型的容错包装。

设计要点：**服务永远能起来**。

真实模型可能因为鉴权失败、额度耗尽、网络不通而不可用。这时如果直接抛异常，
整个陪伴服务就挂了 —— 但对一个中学生心理陪伴场景来说，"模型不可用"必须有
兜底，不能让学生那边看到 500。

所以这里做两件事：

1. :class:`ResilientChatModel` 包装真实模型，第一次调用失败时**自动切换**到
   本地规则模型，并把失败原因记进 ``status``，由 ``/health`` 如实暴露；
2. :class:`ModelFactory` 按用途（主回复模型 / 风险分小模型）产出实例。
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, AsyncGenerator, Literal

from agentscope.model import ChatModelBase, ChatResponse

from app.config import Settings
from app.models.mock_model import RuleBasedChatModel
from app.models.openai_model import build_openai_model
from app.observability.context import get_trace_id
from app.observability.prompt_logger import (
    PromptLogger,
    extract_system_prompt,
    extract_user_message,
)

logger = logging.getLogger(__name__)

Purpose = Literal["main", "risk"]


class ResilientChatModel(ChatModelBase):
    """真实模型 + 本地规则模型兜底。"""

    class Parameters(ChatModelBase.Parameters):
        temperature: float = 0.0
        max_tokens: int | None = None

    def __init__(
        self,
        primary: ChatModelBase,
        fallback: RuleBasedChatModel,
        *,
        purpose: str = "main",
        model_name: str = "",
        circuit_breaker: bool = True,
        prompt_logger: PromptLogger | None = None,
    ) -> None:
        super().__init__(
            credential=primary.credential,
            model=model_name or primary.model,
            parameters=self.Parameters(),
            stream=True,
            max_retries=0,
        )
        self.primary = primary
        self.fallback = fallback
        self.purpose = purpose
        self.model_name = model_name or primary.model
        self.fallback_reason: str | None = None
        self._use_fallback = False
        self._last_messages: list = []
        #: 熔断：真实模型一旦失败就不再重试（鉴权失败/额度耗尽不会自己好，
        #: 每轮再白跑一次注定失败的请求纯属浪费首字延迟）
        self._circuit_breaker = circuit_breaker
        self._breaker_open = False
        self._prompt_logger = prompt_logger

    # ------------------------------------------------------------------
    @property
    def formatter(self):  # type: ignore[override]
        return self.primary.formatter

    @property
    def status(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "model": self.model_name,
            "backend": "mock" if self._use_fallback else "openai",
            "fallback_reason": self.fallback_reason,
            "circuit_breaker_open": self._breaker_open,
        }

    def _mark_fallback(self, exc: Exception) -> None:
        """记一次降级。

        ``_use_fallback`` 表示"最近一次走了兜底"（用于状态展示）；
        ``_breaker_open`` 表示"永久不再试真实模型"（熔断）。
        """
        self._use_fallback = True
        self.fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
        if self._circuit_breaker:
            self._breaker_open = True
        logger.error(
            "模型 %s 调用失败，已降级到本地规则模型%s：%s",
            self.model_name,
            "（并熔断，后续不再重试真实模型）" if self._circuit_breaker else "",
            self.fallback_reason,
        )

    def _log_prompt(self, messages: list, *, backend: str) -> None:
        """把本轮发给模型的提示词落盘（观测用途，失败不影响主流程）。"""
        if self._prompt_logger is None or not self._prompt_logger.enabled:
            return
        try:
            self._prompt_logger.record(
                trace_id=get_trace_id() or "",
                model=self.model_name,
                backend=backend,
                system_prompt=extract_system_prompt(messages),
                user_message=extract_user_message(messages),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prompt 日志埋点失败：%s", exc)

    # ------------------------------------------------------------------
    async def _call_api(
        self,
        model_name: str,
        messages: list,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        if self._breaker_open:
            # 已熔断：直接走兜底，不再发起注定失败的请求
            return await self.fallback._call_api(
                model_name, messages, tools=tools, tool_choice=tool_choice, **kwargs
            )
        self._last_messages = messages
        self._log_prompt(messages, backend="openai")
        try:
            result = await self.primary._call_api(
                model_name, messages, tools=tools, tool_choice=tool_choice, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            self._mark_fallback(exc)
            return await self.fallback._call_api(
                model_name, messages, tools=tools, tool_choice=tool_choice, **kwargs
            )

        if inspect.isasyncgen(result):
            return self._guard_stream(result)
        return result

    async def _guard_stream(
        self, generator: AsyncGenerator[ChatResponse, None]
    ) -> AsyncGenerator[ChatResponse, None]:
        """流式过程中失败也要能降级（鉴权失败往往发生在第一次迭代时）。"""
        try:
            async for chunk in generator:
                yield chunk
        except Exception as exc:  # noqa: BLE001
            self._mark_fallback(exc)
            fallback = await self.fallback._call_api(
                self.model_name, self._last_messages, tools=None, tool_choice=None
            )
            if inspect.isasyncgen(fallback):
                async for chunk in fallback:
                    yield chunk
            elif fallback is not None:
                yield fallback


class ModelFactory:
    """按配置创建模型实例。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._instances: dict[Purpose, ChatModelBase] = {}
        #: Prompt 日志器：把每次 AI 调用的提示词落盘（供事后排查）。
        self.prompt_logger = PromptLogger(
            settings.prompt_log_directory,
            enabled=settings.prompt_log_enabled,
        )

    # ------------------------------------------------------------------
    def create(self, purpose: Purpose = "main") -> ChatModelBase:
        """返回模型实例（同用途复用同一个，便于查看状态）。"""
        if purpose in self._instances:
            return self._instances[purpose]
        instance = self._build(purpose)
        self._instances[purpose] = instance
        return instance

    def _build(self, purpose: Purpose) -> ChatModelBase:
        settings = self.settings
        fallback = RuleBasedChatModel(
            purpose=purpose,
            temperature=(
                settings.risk_model_temperature
                if purpose == "risk"
                else settings.model_temperature
            ),
        )
        if settings.model_provider == "mock":
            return fallback

        try:
            if purpose == "risk":
                primary = build_openai_model(
                    model=settings.risk_model_name or settings.model_name,
                    base_url=settings.risk_model_base_url or settings.model_base_url,
                    api_key=settings.risk_model_api_key or settings.model_api_key,
                    temperature=settings.risk_model_temperature,
                    stream=False,
                    extra_headers=dict(settings.risk_model_extra_headers),
                    timeout=settings.model_timeout,
                )
                name = settings.risk_model_name
            else:
                primary = build_openai_model(
                    model=settings.model_name,
                    base_url=settings.model_base_url,
                    api_key=settings.model_api_key,
                    temperature=settings.model_temperature,
                    max_tokens=settings.model_max_tokens,
                    extra_headers=dict(settings.model_extra_headers),
                    timeout=settings.model_timeout,
                )
                name = settings.model_name
        except Exception as exc:  # noqa: BLE001
            if not settings.model_fallback_to_mock:
                raise
            logger.error("构造真实模型失败（%s），直接使用本地规则模型", exc)
            return fallback

        if not settings.model_fallback_to_mock:
            return primary
        return ResilientChatModel(
            primary,
            fallback,
            purpose=purpose,
            model_name=name,
            circuit_breaker=settings.model_circuit_breaker,
            prompt_logger=self.prompt_logger,
        )

    # ------------------------------------------------------------------
    def status(self) -> list[dict[str, Any]]:
        """各用途模型当前状态（供 /health 如实暴露）。"""
        out: list[dict[str, Any]] = []
        for purpose in ("main", "risk"):
            instance = self._instances.get(purpose)
            if instance is None:
                out.append(
                    {
                        "purpose": purpose,
                        "model": self.settings.model_name
                        if purpose == "main"
                        else self.settings.risk_model_name,
                        "backend": self.settings.model_provider,
                        "fallback_reason": None,
                    }
                )
            elif isinstance(instance, ResilientChatModel):
                out.append(instance.status)
            else:
                out.append(
                    {
                        "purpose": purpose,
                        "model": instance.model,
                        "backend": "mock" if isinstance(instance, RuleBasedChatModel) else "openai",
                        "fallback_reason": None,
                    }
                )
        return out


__all__ = ["ModelFactory", "Purpose", "ResilientChatModel"]
