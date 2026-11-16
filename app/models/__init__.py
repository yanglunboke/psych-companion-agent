"""LLM 适配层。

对外只暴露 :func:`get_model_factory`，其余是内部实现细节。
"""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.models.base import ModelFactory, Purpose


@lru_cache(maxsize=1)
def get_model_factory(settings: Settings | None = None) -> ModelFactory:
    """返回进程内单例模型工厂。"""
    return ModelFactory(settings or get_settings())


__all__ = ["ModelFactory", "Purpose", "get_model_factory"]
