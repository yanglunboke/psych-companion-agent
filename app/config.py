"""全局配置。

所有阈值、检索参数、存储后端都集中在这里，避免散落在业务代码里。
取值来源优先级：环境变量 > .env > 默认值。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置。"""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------- 服务 ----------------
    app_name: str = "中学积极心理陪伴智能体"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # ---------------- 模型后端 ----------------
    model_provider: Literal["mock", "openai"] = "mock"
    model_name: str = "psych-companion-main"
    model_base_url: str = "https://api.openai.com/v1"
    model_api_key: str = ""
    model_temperature: float = 0.3
    model_max_tokens: int = 1024
    model_timeout: float = 60.0
    #: 额外的 HTTP 头（OpenRouter 需要 HTTP-Referer / X-Title 等）
    model_extra_headers: dict[str, str] = Field(default_factory=dict)
    #: 真实模型不可用（鉴权失败/网络不通）时，是否透明降级到本地规则模型
    model_fallback_to_mock: bool = True
    #: 真实模型失败后是否熔断（不再重试），避免每轮白跑一个注定失败的请求
    model_circuit_breaker: bool = True

    # 风险分专用小模型：更小更快，temperature=0
    risk_model_name: str = "psych-companion-risk"
    risk_model_base_url: str = ""
    risk_model_api_key: str = ""
    risk_model_temperature: float = 0.0
    risk_model_extra_headers: dict[str, str] = Field(default_factory=dict)

    # ---------------- 安全门 ----------------
    risk_threshold: float = 0.72
    lexicon_path: str = "app/data/lexicon/high_risk_lexicon.json"

    # ---------------- 记忆 ----------------
    #: 短期记忆后端：memory（进程内）/ redis（热缓存，最近 12 轮）
    memory_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_key_prefix: str = "psych:session"
    #: 长期存储后端：memory（进程内）/ sql（Postgres 生产 / SQLite 本地）
    db_backend: Literal["memory", "sql"] = "memory"
    #: SQLAlchemy 异步 DSN。生产：postgresql+psycopg://user:pw@host:5432/psych
    #: 本地：sqlite+aiosqlite:///./data/psych_companion.db
    sql_dsn: str = "sqlite+aiosqlite:///./data/psych_companion.db"
    postgres_dsn: str = ""
    #: 后端连不上时是否自动降级到进程内实现（保证服务永远能起）
    storage_fallback: bool = True
    short_term_turns: int = 12
    intensity_ema_alpha: float = 0.3
    profile_evidence_threshold: int = 3

    # ---------------- RAG 检索 ----------------
    #: 向量化后端：hashing（本地哈希，无语义，离线默认）/ bge（真实语义模型）
    embedding_backend: Literal["hashing", "bge"] = "hashing"
    #: 语义向量模型（embedding_backend=bge 时生效）
    embedding_model: str = "BAAI/bge-m3"
    #: 向量持久化：memory（进程内，默认）/ pgvector（Postgres + pgvector）
    vector_backend: Literal["memory", "pgvector"] = "memory"
    #: PGVector 连接串（vector_backend=pgvector 时生效）
    #: 形如 postgresql+psycopg://user:pw@host:5432/psych
    pg_dsn: str = ""
    #: 向量维度（bge-m3 = 1024）
    embedding_dim: int = 1024
    recall_top_k: int = 20
    vector_min_score: float = 0.35
    rrf_k: int = 60
    rerank_min_score: float = 0.45
    rerank_top_k: int = 20
    dynamic_k_by_fit: dict[str, int] = Field(
        default_factory=lambda: {
            "只想说说": 2,
            "情绪疏导": 3,
            "轻量引导": 2,
            "需要练习": 4,
        },
    )
    score_gap_threshold: float = 0.15
    #: 本轮 Prompt 里策略卡正文的字符预算（超出挤占上下文、拖慢首字）
    card_char_budget: int = 1200

    # ---------------- 高危上报 ----------------
    escalation_webhook_url: str = ""
    escalation_outbox_path: str = "data/outbox/escalations.jsonl"
    teacher_notify_channel: str = "teacher_console"
    parent_notify_channel: str = "parent_console"

    # ---------------- 评测 ----------------
    eval_seed: int = 2026

    # ---------------- 派生路径 ----------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "app" / "data"

    @property
    def lexicon_file(self) -> Path:
        path = Path(self.lexicon_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def outbox_file(self) -> Path:
        path = Path(self.escalation_outbox_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @field_validator("dynamic_k_by_fit", mode="before")
    @classmethod
    def _parse_dynamic_k(cls, value: object) -> object:
        """支持 ``DYNAMIC_K_BY_FIT=只想说说:2,情绪疏导:3`` 形式的环境变量。"""
        if isinstance(value, str):
            parsed: dict[str, int] = {}
            for chunk in value.split(","):
                if not chunk.strip():
                    continue
                key, _, val = chunk.partition(":")
                parsed[key.strip()] = int(val.strip())
            return parsed
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内单例配置。"""
    return Settings()


def reload_settings() -> Settings:
    """清缓存重新加载配置（测试用）。"""
    get_settings.cache_clear()
    return get_settings()


__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
    "reload_settings",
]
