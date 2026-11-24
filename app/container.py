"""依赖装配。

把配置变成一整套可用的组件（模型、索引、检索器、四个 Worker、编排器、
记忆、trace、反思），上层（API / 脚本 / 测试）只跟容器打交道。

存储后端（Redis / SQL）是**异步装配**的：:meth:`Container.setup_storage`
在应用启动时调用；连不上会按配置降级到进程内实现并记录原因。
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings
from app.memory.factory import build_profile_store, build_short_term_memory
from app.memory.memory_impl import InMemoryProfileStore, InMemoryShortTermMemory
from app.models.base import ModelFactory
from app.observability.reflection import ReflectionService
from app.observability.trace import SqlTraceSink, TraceStore
from app.orchestrator.escalate import EscalationService
from app.orchestrator.orchestrator import Orchestrator
from app.orchestrator.router import ToolRouter
from app.orchestrator.safety import Lexicon, SafetyGate
from app.rag.index import build_default_index
from app.rag.embedding import build_embedder
from app.rag.retriever import Retriever, RetrieverConfig
from app.workers.emotion import EmotionWorker, RiskScorer
from app.workers.exercise import ExerciseWorker
from app.workers.guidance import GuidanceWorker
from app.workers.needs import NeedsWorker

logger = logging.getLogger(__name__)


class Container:
    """组件容器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        s = self.settings

        # ---- 模型 ----
        self.model_factory = ModelFactory(s)
        main_model = self.model_factory.create("main")
        risk_model = self.model_factory.create("risk")

        # ---- 知识库与检索 ----
        self.embedder = build_embedder(s.embedding_backend, s.embedding_model)
        self.card_index = build_default_index(s.data_dir, embedder=self.embedder)
        self.vector_store: Any = None
        self.vector_status: dict[str, Any] = {
            "embedder": self.embedder.name,
            "store": "memory",
            "detail": "进程内向量索引",
        }
        self.retriever = Retriever(
            self.card_index,
            RetrieverConfig(
                recall_top_k=s.recall_top_k,
                vector_min_score=s.vector_min_score,
                rrf_k=s.rrf_k,
                rerank_min_score=s.rerank_min_score,
                rerank_top_k=s.rerank_top_k,
                dynamic_k_by_fit=dict(s.dynamic_k_by_fit),
                score_gap_threshold=s.score_gap_threshold,
                card_char_budget=s.card_char_budget,
            ),
        )

        # ---- 安全门 ----
        self.safety_gate = SafetyGate(
            Lexicon.from_file(s.lexicon_file),
            threshold=s.risk_threshold,
        )

        # ---- 记忆（默认进程内；setup_storage 会按配置升级）----
        self.short_memory: Any = InMemoryShortTermMemory(max_turns=s.short_term_turns)
        self.profile_store: Any = InMemoryProfileStore()
        self.storage_status: dict[str, Any] = {
            "short_term": {"backend": "memory", "detail": "进程内实现"},
            "profile": {"backend": "memory", "detail": "进程内实现"},
        }

        # ---- 支撑 ----
        self.trace_store = TraceStore()
        self.reflection = ReflectionService()
        self.escalation = EscalationService(s)

        # ---- 四个 Worker ----
        self.emotion_worker = EmotionWorker(main_model)
        self.risk_scorer = RiskScorer(risk_model)
        self.needs_worker = NeedsWorker(main_model)
        self.exercise_worker = ExerciseWorker(main_model, self.retriever)
        self.guidance_worker = GuidanceWorker(
            main_model,
            retriever=self.retriever,
            exercise_worker=self.exercise_worker,
        )

        # ---- 编排器 ----
        self.orchestrator = Orchestrator(
            s,
            emotion_worker=self.emotion_worker,
            risk_scorer=self.risk_scorer,
            needs_worker=self.needs_worker,
            guidance_worker=self.guidance_worker,
            exercise_worker=self.exercise_worker,
            retriever=self.retriever,
            safety_gate=self.safety_gate,
            escalation=self.escalation,
            reflection=self.reflection,
            short_memory=self.short_memory,
            profile_store=self.profile_store,
            trace_store=self.trace_store,
            router=ToolRouter(),
        )
        logger.info(
            "容器就绪：卡片 %d 张，集合 %s",
            len(self.card_index.cards),
            ",".join(self.card_index.collections()),
        )

    # ------------------------------------------------------------------
    async def setup_storage(self) -> None:
        """按配置装配 Redis / SQL 存储（连不上则降级）。"""
        self.short_memory, short_status = await build_short_term_memory(self.settings)
        self.storage_status["short_term"] = short_status

        self.profile_store, profile_status = await build_profile_store(self.settings)
        self.storage_status["profile"] = profile_status

        # 编排器换成新的存储实现
        self.orchestrator.short_memory = self.short_memory
        self.orchestrator.profile_store = self.profile_store

        # 长期存储可用时，把工具调用日志也落库（跨进程可追溯）
        if profile_status.get("backend") == "sql":
            self.orchestrator.trace_sink = SqlTraceSink(self.profile_store)

        logger.info(
            "存储装配完成：短期记忆=%s，长期存储=%s",
            short_status.get("backend"),
            profile_status.get("backend"),
        )

    async def setup_vector_store(self) -> None:
        """按配置装配 PGVector（连不上则保持内存索引）。"""
        s = self.settings
        if s.vector_backend != "pgvector" or not s.pg_dsn:
            return
        try:
            from app.rag.pgvector_store import PgVectorStore

            store = PgVectorStore(s.pg_dsn, dim=s.embedding_dim)
            await store.init_schema()
            # 把当前内存索引里的卡片向量灌进 PGVector（幂等 upsert）
            items = [
                (c.card_id, c.collection, c.vector)
                for c in self.card_index.cards
                if c.vector is not None
            ]
            await store.upsert_many(items)
            self.vector_store = store
            self.vector_status = {
                "embedder": self.embedder.name,
                "store": "pgvector",
                "detail": s.pg_dsn,
                "dim": s.embedding_dim,
            }
            logger.info("PGVector 就绪：写入 %d 条卡片向量", len(items))
        except Exception as exc:  # noqa: BLE001
            logger.warning("PGVector 不可用（%s），向量检索保持进程内索引", exc)
            self.vector_status = {
                "embedder": self.embedder.name,
                "store": "memory",
                "detail": f"PGVector 不可用，已降级：{exc}",
            }

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """运行状态（供 /health 如实暴露）。"""
        return {
            "model": self.model_factory.status(),
            "storage": self.storage_status,
            "vector": self.vector_status,
            "cards": len(self.card_index.cards),
            "collections": self.card_index.collections(),
        }


__all__ = ["Container"]
