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
# 测试同样强制离线：向量化走本地哈希（避免加载 bge-m3 / 依赖 PGVector），
# 保证无网络、无模型权重、无 Docker 也能秒级通过。向量存储固定内存。
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["VECTOR_BACKEND"] = "memory"
# 短期记忆固定进程内：测试不应依赖"本机恰好起了 Redis"，
# 否则同样的代码在有/无 Redis 的机器上跑出两种结果。
os.environ["MEMORY_BACKEND"] = "memory"
# 调试接口默认关闭：.env 里可能为本机开发开了，测试要按"默认关"的契约跑。
os.environ["DEBUG_API_ENABLED"] = "false"

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
