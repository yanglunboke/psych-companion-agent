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
