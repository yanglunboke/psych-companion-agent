"""依赖装配。

把配置变成一整套可用的组件（模型、索引、检索器、四个 Worker、编排器、
记忆、trace、反思），上层（API / 脚本 / 测试）只跟容器打交道。

存储后端（Redis / SQL）是**异步装配**的：:meth:`Container.setup_storage`
在应用启动时调用；连不上会按配置降级到进程内实现并记录原因。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.auth.factory import build_auth_store
from app.auth.store import InMemoryAuthStore
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


def content_hash_of(card: Any) -> str:
    """卡片**检索文本**的内容指纹。

    卡片正文或检索锚点改了 → 指纹变 → 库里的向量视为过期，重新编码。
    只比对 card_id 是不够的：同一张卡换了检索文本却复用旧向量，
    会让库里存的是"旧文本的向量"，与当前语义完全错位。
    """
    import hashlib

    return hashlib.sha256(card.search_text.encode("utf-8")).hexdigest()


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
        # bge-m3 编码几十张卡要几十秒。启用语义模型时先不编码，
        # 启动装配阶段优先从 PGVector 复用已持久化的向量，只对缺失的补编。
        self.card_index = build_default_index(
            s.data_dir,
            embedder=self.embedder,
            encode=s.embedding_backend != "bge",
        )
        self.vector_store: Any = None
        self.vector_status: dict[str, Any] = {
            "embedder": self.embedder.name,
            "store": "memory",
            "detail": "进程内向量索引",
        }
        #: 后台预热任务（bge 等真实向量模型才需要）
        self._warmup_task: asyncio.Task | None = None
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
                theory_rerank_min_score=s.theory_rerank_min_score,
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
        # ---- 账号（登录给出稳定 student_id）----
        self.auth: Any = InMemoryAuthStore()
        self.storage_status: dict[str, Any] = {
            "short_term": {"backend": "memory", "detail": "进程内实现"},
            "profile": {"backend": "memory", "detail": "进程内实现"},
            "auth": {"backend": "memory", "detail": "进程内实现"},
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

        self.auth, auth_status = await build_auth_store(self.settings)
        self.storage_status["auth"] = auth_status

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
        """装配向量存储。

        无论是否连上 PGVector，都要保证内存索引里**有**向量，否则语义召回会空转。
        装配顺序：

        1. 连 PGVector（连不上则降级为进程内索引）；
        2. 从库里装载已持久化的向量（省掉重复编码——bge-m3 编码几十张卡要几十秒）；
        3. 只对**缺失**的卡编码；
        4. 增量写回，并清理库里已不存在的卡（prune），保持库与卡片源一致。
        """
        s = self.settings
        store = None

        if s.vector_backend == "pgvector" and s.pg_dsn:
            try:
                from app.rag.pgvector_store import PgVectorStore

                store = PgVectorStore(s.pg_dsn, dim=s.embedding_dim)
                await store.init_schema()
            except Exception as exc:  # noqa: BLE001
                logger.warning("PGVector 不可用（%s），向量检索保持进程内索引", exc)
                store = None

        loaded = 0
        pruned = 0
        encoded = 0

        if store is not None:
            try:
                existing = await store.fetch_all()
            except Exception as exc:  # noqa: BLE001
                logger.warning("PGVector 读取向量失败（%s），改用进程内索引", exc)
                existing = []
                store = None

        if store is not None:
            by_id = {c.card_id: c for c in self.card_index.cards}
            loaded_ids: set[str] = set()
            stale: set[str] = set()
            for card_id, collection, vec, content_hash in existing:
                card = by_id.get(card_id)
                # 卡片已删除或换了集合 → 这条向量该被清理，不装载
                if card is None or card.collection != collection:
                    continue
                # 维度与当前模型不符（换过 embedding 模型）→ 丢弃，重新编码
                if len(vec) != s.embedding_dim:
                    continue
                # 内容指纹不符（卡片正文/检索文本改过）→ 向量过期，重新编码
                if content_hash != content_hash_of(card):
                    stale.add(card_id)
                    continue
                if card.vector is None:
                    self.card_index.set_vector(card_id, vec)
                    loaded_ids.add(card_id)
            loaded = len(loaded_ids)

            # 只编码库里没有或已过期的卡
            encoded = self.card_index.encode_missing()
            if encoded:
                pending = [
                    (c.card_id, c.collection, c.vector, content_hash_of(c))
                    for c in self.card_index.cards
                    if c.vector is not None and c.card_id not in loaded_ids
                ]
                await store.upsert_many(pending)

            # 清理库里已不存在（或换了集合）的卡
            pruned = await store.delete_missing([c.card_id for c in self.card_index.cards])

        if store is not None:
            self.vector_store = store
            self.vector_status = {
                "embedder": self.embedder.name,
                "store": "pgvector",
                "detail": s.pg_dsn,
                "dim": s.embedding_dim,
                "loaded_from_pgvector": loaded,
                "encoded": encoded,
                "stale": len(stale),
                "pruned": pruned,
                "cards": len(self.card_index.cards),
            }
            logger.info(
                "PGVector 就绪：装载 %d 条、补编 %d 条、过期 %d 条、清理 %d 条、卡片 %d 张",
                loaded,
                encoded,
                len(stale),
                pruned,
                len(self.card_index.cards),
            )
        else:
            # 没连上 PG：也要保证向量存在（首次要全量编码）
            encoded = self.card_index.encode_missing()
            self.vector_status = {
                "embedder": self.embedder.name,
                "store": "memory",
                "detail": "进程内向量索引",
                "encoded": encoded,
                "cards": len(self.card_index.cards),
            }
            logger.info("向量索引就绪：进程内，补编 %d 条", encoded)

    # ------------------------------------------------------------------
    def start_embedder_warmup(self) -> None:
        """后台预热向量模型（不阻塞启动，也不占首个用户请求）。

        为什么必须有这一步：bge-m3 的权重是**懒加载**的，而启动时 68 张卡的
        向量已在 PGVector 里（``encoded=0``），根本不会触发加载。于是这份
        20–30 秒的加载成本就落在"重启后的第一条真实对话"上——实测出现过
        首轮 39 秒、第二轮 1.6 秒的落差。

        预热放在后台线程里：服务立即可用（``/health`` 返回 ok），
        正在预热的这几十秒内如果有请求进来，两者会**共用同一把加载锁**，
        不会各加载一份模型。
        """
        if getattr(self, "_warmup_task", None) is not None:
            return
        embedder = self.embedder
        if not getattr(embedder, "needs_warmup", False):
            self.vector_status["warmup"] = "skipped"
            return
        self.vector_status["warmup"] = "warming"
        self._warmup_task = asyncio.create_task(self._run_warmup())

    async def _run_warmup(self) -> None:
        started = time.perf_counter()
        try:
            elapsed = await asyncio.to_thread(self.embedder.warmup)
            self.vector_status["warmup"] = f"ready ({elapsed:.1f}s)"
            logger.info("向量模型预热完成：%.1f 秒（首轮检索不再承担加载成本）", elapsed)
        except Exception as exc:  # noqa: BLE001
            self.vector_status["warmup"] = f"failed: {exc}"
            logger.warning("向量模型预热失败（首轮仍会按需加载）：%s", exc)
        finally:
            logger.debug("预热流程耗时 %.1f 秒", time.perf_counter() - started)

    # ------------------------------------------------------------------
    async def close(self) -> None:
        """释放外部连接（Redis 客户端 / PostgreSQL 连接池）。

        任一项失败只告警，不影响进程退出。
        """
        task = getattr(self, "_warmup_task", None)
        if task is not None and not task.done():
            task.cancel()
        for name, target, method in (
            ("短期记忆", self.short_memory, "close"),
            ("长期存储", self.profile_store, "dispose"),
            ("账号存储", self.auth, "close"),
            ("向量存储", self.vector_store, "dispose"),
        ):
            closer = getattr(target, method, None)
            if closer is None:
                continue
            try:
                result = closer()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.warning("释放%s连接失败：%s", name, exc)

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
