"""策略知识库检索层。

    tokenizer  →  分词（向量与 BM25 共用）
    embedding  →  向量化（默认本地哈希向量器，可换真实 embedding 模型）
    bm25       →  BM25 索引
    index      →  卡片内存索引（按 collection 隔离）
    query      →  结构化查询构造
    retriever  →  四步检索：召回 → RRF → 重排 → 截断
"""

from app.rag.bm25 import BM25Index
from app.rag.embedding import Embedder, HashingEmbedder
from app.rag.index import CardIndex, IndexedCard, build_default_index
from app.rag.query import build_exercise_query, build_query, infer_scene, infer_steps
from app.rag.retriever import (
    CrossEncoderReranker,
    LocalCrossEncoder,
    Retriever,
    RetrieverConfig,
)

__all__ = [
    "BM25Index",
    "CardIndex",
    "CrossEncoderReranker",
    "Embedder",
    "HashingEmbedder",
    "IndexedCard",
    "LocalCrossEncoder",
    "Retriever",
    "RetrieverConfig",
    "build_default_index",
    "build_exercise_query",
    "build_query",
    "infer_scene",
    "infer_steps",
]
