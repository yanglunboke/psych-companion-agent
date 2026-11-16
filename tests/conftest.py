"""pytest 公共夹具。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 测试必须离线且确定：强制走本地规则模型。
# 环境变量优先级高于 .env，所以这一行会覆盖 .env 里的 MODEL_PROVIDER。
os.environ["MODEL_PROVIDER"] = "mock"

from app.config import Settings, reload_settings  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    """测试用配置：强制使用离线规则模型与内存存储。"""
    return reload_settings()


@pytest.fixture()
def offline_settings() -> Settings:
    """单测级别的配置副本。"""
    return Settings(
        model_provider="mock",
        memory_backend="memory",
        risk_threshold=0.72,
    )
