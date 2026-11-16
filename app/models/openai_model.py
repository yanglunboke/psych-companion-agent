"""OpenAI 兼容模型后端。

支持任何暴露 ``/chat/completions`` 的服务：OpenAI、DeepSeek、Moonshot、
通义千问兼容模式、智谱、OpenRouter、火山方舟、Ollama、vLLM、LM Studio ……
统一通过 ``MODEL_BASE_URL`` + ``MODEL_API_KEY`` 配置。

额外 HTTP 头（例如 OpenRouter 的 ``HTTP-Referer`` / ``X-Title``）通过
``MODEL_EXTRA_HEADERS`` 传入。
"""

from __future__ import annotations

from typing import Any

from agentscope.credential import OpenAICredential
from agentscope.model import ChatModelBase, OpenAIChatModel


def build_openai_model(
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int | None = None,
    stream: bool = True,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> ChatModelBase:
    """构造一个 OpenAI 兼容的聊天模型。"""
    if not api_key:
        raise ValueError(
            "MODEL_PROVIDER=openai 时必须提供 MODEL_API_KEY；"
            "没有密钥请把 MODEL_PROVIDER 改回 mock。"
        )
    credential = OpenAICredential(
        name="openai-compatible",
        api_key=api_key,
        base_url=base_url.rstrip("/") if base_url else None,
    )
    parameters = OpenAIChatModel.Parameters(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if extra_headers:
        # openai SDK 的参数名是 default_headers，不是 headers
        client_kwargs["default_headers"] = dict(extra_headers)
    return OpenAIChatModel(
        credential=credential,
        model=model,
        parameters=parameters,
        stream=stream,
        client_kwargs=client_kwargs,
    )


__all__ = ["build_openai_model"]
