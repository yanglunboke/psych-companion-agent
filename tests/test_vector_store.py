"""语义向量与 PGVector 存储层测试（不依赖 Docker / 真实模型权重）。

* ``build_embedder`` 按配置返回正确的向量器；
* ``SentenceTransformerEmbedder`` 接口可单测（monkeypatch 掉模型加载，避免
  真实下载 bge-m3）；
* ``PgVectorStore`` 在无 Postgres 环境下不 import 即崩，构建失败被上层降级。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.embedding import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)


def test_build_embedder_default_is_hashing() -> None:
    embedder = build_embedder("hashing")
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.name == "hashing"


def test_build_embedder_bge_returns_semantic_embedder() -> None:
    embedder = build_embedder("bge", "BAAI/bge-m3")
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.name == "bge-m3"
    # 语义向量器不提供标定值，检索层回落到文档口径 0.35
    assert embedder.calibrated_min_score is None


def test_hashing_embedder_normalized_cosine() -> None:
    embedder = HashingEmbedder(dim=256)
    a = embedder.embed("考试失利 我很笨")
    b = embedder.embed("考试失利 学业自我否定")
    c = embedder.embed("同伴冲突 被孤立")
    # L2 归一化后点积即余弦相似度
    assert embedder.similarity(a, a) == pytest.approx(1.0, abs=1e-3)
    # 相近文本比不相近文本更相似
    assert embedder.similarity(a, b) > embedder.similarity(a, c)


class _FakeModel:
    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        if isinstance(texts, str):
            return np.ones(self._dim, dtype=np.float32)
        return np.ones((len(texts), self._dim), dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return self._dim


def test_sentence_transformer_embedder_lazy_load(monkeypatch) -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    # 未加载前不触发模型下载
    assert embedder._model is None
    # 注入假模型，验证 embed 走语义向量、返回 float32、维度正确
    embedder._model = _FakeModel(dim=16)
    vec = embedder.embed("考砸了")
    assert vec.shape == (16,)
    assert vec.dtype == np.float32
    assert embedder.dim == 16


def test_sentence_transformer_embed_many(monkeypatch) -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    embedder._model = _FakeModel(dim=8)
    vecs = embedder.embed_many(["a", "b", "c"])
    assert vecs.shape == (3, 8)


def test_pgvector_store_does_not_import_on_construct() -> None:
    """构造 PgVectorStore 不应触发 sqlalchemy/pgvector import（可离线单测）。"""
    from app.rag.pgvector_store import PgVectorStore

    store = PgVectorStore("postgresql+psycopg://x:y@localhost:5432/x")
    assert store.dim == 1024
    assert store._engine is None  # 尚未建 engine


# ----------------------------------------------------------------------
# 延迟编码 + 装载（PGVector 作为向量权威源）
# ----------------------------------------------------------------------


def test_card_index_encode_false_leaves_vectors_empty() -> None:
    """encode=False 时不编码，卡片向量全为空。"""
    from app.rag.index import IndexedCard, CardIndex

    cards = [
        IndexedCard(card_id="a", collection="student_reply", scene="s", title="t", text="x"),
        IndexedCard(card_id="b", collection="student_reply", scene="s", title="t", text="y"),
    ]
    index = CardIndex(cards, embedder=HashingEmbedder(dim=32), encode=False)
    assert len(index.missing_vector_ids()) == 2
    assert all(c.vector is None for c in index.cards)


def test_card_index_encode_missing_only_encodes_pending() -> None:
    """encode_missing 只补编没有向量的卡，已有向量不重复编码。"""
    from app.rag.index import IndexedCard, CardIndex

    cards = [
        IndexedCard(card_id="a", collection="student_reply", scene="s", title="t", text="x"),
        IndexedCard(card_id="b", collection="student_reply", scene="s", title="t", text="y"),
    ]
    index = CardIndex(cards, embedder=HashingEmbedder(dim=32), encode=False)

    # 模拟从 PGVector 装载：给 a 灌一个向量
    existing = np.ones(32, dtype=np.float32)
    assert index.set_vector("a", existing) is True
    assert index.set_vector("missing-id", existing) is False

    assert index.missing_vector_ids() == ["b"]
    assert index.encode_missing() == 1
    assert index.missing_vector_ids() == []
    # a 的向量没被覆盖
    assert np.allclose(index.get("a").vector, existing)
    # b 被编码成了 32 维向量
    assert index.get("b").vector is not None
    assert index.get("b").vector.shape == (32,)


def test_card_index_encode_missing_is_idempotent() -> None:
    """重复调用 encode_missing 不会重复编码（第二次返回 0）。"""
    from app.rag.index import IndexedCard, CardIndex

    cards = [
        IndexedCard(card_id="a", collection="student_reply", scene="s", title="t", text="x"),
    ]
    index = CardIndex(cards, embedder=HashingEmbedder(dim=16), encode=False)
    assert index.encode_missing() == 1
    assert index.encode_missing() == 0


def test_pgvector_store_exposes_sync_api() -> None:
    """PGVector 提供装载（fetch_all）与清理（delete_missing）接口。"""
    from app.rag.pgvector_store import PgVectorStore

    store = PgVectorStore("postgresql+psycopg://x:y@localhost:5432/x")
    assert hasattr(store, "fetch_all")
    assert hasattr(store, "delete_missing")
    assert hasattr(store, "upsert_many")


# ----------------------------------------------------------------------
# 检索锚点（keywords）+ 内容指纹失效
# ----------------------------------------------------------------------


def test_card_index_keywords_override_search_text() -> None:
    """有 keywords 时检索文本改用「标题 + 关键词」，否则用「标题+场景+正文」。"""
    from app.rag.index import IndexedCard

    anchored = IndexedCard(
        card_id="a",
        collection="theory",
        scene="*",
        title="幸福的两个根本困惑",
        text="很长很长的正文……",
        keywords=["幸福指数", "和别人比"],
    )
    plain = IndexedCard(
        card_id="b",
        collection="student_reply",
        scene="exam_failure",
        title="考试失利",
        text="正文内容",
    )
    assert anchored.search_text == "幸福的两个根本困惑 幸福指数 和别人比"
    assert plain.search_text == "考试失利 exam_failure 正文内容"


def test_content_hash_changes_with_search_text() -> None:
    """检索文本变了 → 内容指纹变 → 库里的向量应被判为过期。"""
    from app.container import content_hash_of
    from app.rag.index import IndexedCard

    card = IndexedCard(
        card_id="a",
        collection="theory",
        scene="*",
        title="T",
        text="正文",
        keywords=["旧锚点"],
    )
    before = content_hash_of(card)
    card.keywords = ["新锚点"]
    after = content_hash_of(card)
    assert before != after


def test_retriever_min_score_override() -> None:
    """min_score 覆盖重排阈值（理论卡用更低的独立阈值）。"""
    from app.rag.index import CardIndex, IndexedCard
    from app.rag.retriever import Retriever, RetrieverConfig
    from app.schemas import StructuredQuery

    cards = [
        IndexedCard(
            card_id="t1",
            collection="theory",
            scene="*",
            title="理论卡",
            text="关于幸福与优势的论述内容",
            keywords=["幸福", "优势"],
        ),
    ]
    index = CardIndex(cards, embedder=HashingEmbedder(dim=64))
    retriever = Retriever(
        index, RetrieverConfig(rerank_min_score=0.45, theory_rerank_min_score=0.35)
    )
    query = StructuredQuery(collection="theory", scene="", keywords=["幸福优势"])

    strict = retriever.search(query, collections=["theory"])
    loose = retriever.search(query, collections=["theory"], min_score=0.0)
    # 放宽阈值后候选只会更多或相等
    assert len(loose.reranked) >= len(strict.reranked)


def test_retriever_max_selected_override() -> None:
    """max_selected 覆盖档位 K，理论卡不挤占策略卡名额。"""
    from app.rag.index import CardIndex, IndexedCard
    from app.rag.retriever import Retriever, RetrieverConfig
    from app.schemas import StructuredQuery

    cards = [
        IndexedCard(
            card_id=f"t{i}",
            collection="theory",
            scene="*",
            title=f"理论卡{i}",
            text="幸福与优势的理论论述内容",
            keywords=["幸福", "优势"],
        )
        for i in range(6)
    ]
    index = CardIndex(cards, embedder=HashingEmbedder(dim=64))
    retriever = Retriever(index, RetrieverConfig(dynamic_k_by_fit={"情绪疏导": 3}))
    query = StructuredQuery(collection="theory", scene="", keywords=["幸福优势"], fit="情绪疏导")

    limited = retriever.search(query, collections=["theory"], max_selected=2)
    assert len(limited.selected) <= 2
    # dynamic_k 记录的是**实际生效**的上限（已被 max_selected 覆盖）
    assert limited.dynamic_k == 2
