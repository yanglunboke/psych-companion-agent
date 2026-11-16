"""四步检索：召回 → RRF 融合 → 重排 → 截断。

文档口径：

1. **召回**：向量召回与 BM25 并行，各取 top-20；向量相似度低于 0.35 的丢掉。
2. **融合**：RRF，k 固定 60，只看最终总名次，不看两路各自的原始分。
   ``score = 1/(60+向量排名) + 1/(60+BM25排名)``；只出现在一路里的卡，另一路不贡献。
3. **重排**：读卡片正文，用 Cross-Encoder 打分，低于 0.45 的丢掉，其余按新分重排。
4. **截断**：按本轮对话动态决定留几张 —— 断崖截断 + token 预算 + 档位 K。

默认的重排器 :class:`LocalCrossEncoder` 是**可插拔的本地实现**（字段匹配 +
IDF 词项覆盖 + 余弦），生产可替换为真实 cross-encoder 模型（如 bge-reranker），
只需实现 :meth:`CrossEncoderReranker.score`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.rag.index import CardIndex, IndexedCard
from app.schemas import RetrievedCard, RetrievalResult, StructuredQuery


class CrossEncoderReranker:
    """重排器接口。"""

    name: str = "reranker"

    def score(self, query: StructuredQuery, card: IndexedCard) -> float:  # pragma: no cover
        raise NotImplementedError


class LocalCrossEncoder(CrossEncoderReranker):
    """本地重排器：字段匹配 + IDF 词项覆盖 + 余弦相似度。

    之所以要"读正文"：召回阶段只看标签，场景标签相近但规则正文不同的卡会混进来；
    这一步把查询与卡片全文放在一起比较，才知道是不是"在说同一件事"。
    """

    name = "local-cross-encoder"

    def __init__(self, index: CardIndex) -> None:
        self.index = index

    def score(self, query: StructuredQuery, card: IndexedCard) -> float:
        payload = card.payload
        score = 0.0

        # 1) 场景匹配（主信号：同一场景的回应卡与禁止说法卡必须同档）
        if query.scene and card.scene == query.scene:
            score += 0.45
        elif query.scene and card.scene == "*":
            score += 0.27

        # 2) 干预档位匹配
        applicable = [str(x) for x in payload.get("applicable_fit", [])]
        if query.fit and query.fit in applicable:
            score += 0.20

        # 3) 认知偏差匹配
        distortions = {str(x) for x in payload.get("applicable_distortions", [])}
        if query.distortion and distortions:
            hit = distortions & set(query.distortion)
            score += 0.15 * (len(hit) / len(set(query.distortion)))

        # 4) 策略步骤匹配
        card_steps = {str(x) for x in payload.get("steps", [])}
        if query.steps and card_steps:
            hit = card_steps & set(query.steps)
            score += 0.10 * (len(hit) / len(set(query.steps)))

        # 5) 词项覆盖（IDF 加权）
        score += 0.10 * self._coverage(query, card)

        return round(min(score, 1.0), 4)

    def _coverage(self, query: StructuredQuery, card: IndexedCard) -> float:
        """查询词项在卡片正文中的 IDF 加权覆盖率。"""
        from app.rag.tokenizer import tokenize

        bm25 = self.index.bm25()
        query_tokens = tokenize(query.to_query_string())
        if not query_tokens:
            return 0.0
        card_tokens = set(tokenize(card.search_text))
        weights = 0.0
        covered = 0.0
        for token in set(query_tokens):
            idf = bm25.idf(token)
            if idf <= 0:
                continue
            weights += idf
            if token in card_tokens:
                covered += idf
        return (covered / weights) if weights > 0 else 0.0


@dataclass
class RetrieverConfig:
    """检索参数（来自 Settings，此处解耦便于单测）。"""

    recall_top_k: int = 20
    vector_min_score: float = 0.35
    rrf_k: int = 60
    rerank_min_score: float = 0.45
    rerank_top_k: int = 20
    dynamic_k_by_fit: dict[str, int] | None = None
    score_gap_threshold: float = 0.15
    card_char_budget: int = 1200


class Retriever:
    """四步检索的执行器。"""

    def __init__(
        self,
        index: CardIndex,
        config: RetrieverConfig | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.index = index
        self.config = config or RetrieverConfig()
        self.reranker = reranker or LocalCrossEncoder(index)

    # ------------------------------------------------------------------
    def effective_vector_min_score(self) -> float:
        """实际使用的向量相似度下限。

        文档口径是 0.35（按稠密 embedding 模型标定）。本地哈希向量器稀疏得多，
        余弦分布整体偏低，所以它自带一个标定值；两者取更宽松的那个。
        换成真实 embedding 模型（``calibrated_min_score is None``）时，
        这里就是文档口径的 0.35。
        """
        configured = self.config.vector_min_score
        calibrated = getattr(self.index.embedder, "calibrated_min_score", None)
        if calibrated is None:
            return configured
        return min(configured, calibrated)

    def search(
        self,
        query: StructuredQuery,
        *,
        collections: list[str] | None = None,
    ) -> RetrievalResult:
        """执行完整四步检索。"""
        collections = collections or [query.collection]
        candidates = [c for col in collections for c in self.index.by_collection(col)]
        if not candidates:
            return RetrievalResult(query=query, dynamic_k=0)

        # ---------- 步骤 1：召回（向量 ∥ BM25） ----------
        min_score = self.effective_vector_min_score()
        qv = self.index.embedder.embed(query.to_query_string())
        vector_scored: list[tuple[IndexedCard, float]] = []
        for card in candidates:
            if card.vector is None:
                continue
            sim = float(np.dot(qv, card.vector))
            if sim >= min_score:
                vector_scored.append((card, sim))
        vector_scored.sort(key=lambda item: item[1], reverse=True)
        vector_scored = vector_scored[: self.config.recall_top_k]

        bm25 = self.index.bm25()
        bm25_hits = bm25.search(
            query.to_query_string(),
            top_k=self.config.recall_top_k,
            allowed={c.card_id for c in candidates},
        )

        # ---------- 步骤 2：RRF 融合 ----------
        vector_rank = {card.card_id: i + 1 for i, (card, _) in enumerate(vector_scored)}
        bm25_rank = {doc_id: i + 1 for i, (doc_id, _) in enumerate(bm25_hits)}
        card_by_id = {c.card_id: c for c in candidates}

        fused: list[RetrievedCard] = []
        for card_id in set(vector_rank) | set(bm25_rank):
            score = 0.0
            if card_id in vector_rank:
                score += 1.0 / (self.config.rrf_k + vector_rank[card_id])
            if card_id in bm25_rank:
                score += 1.0 / (self.config.rrf_k + bm25_rank[card_id])
            card = card_by_id[card_id]
            fused.append(
                RetrievedCard(
                    card_id=card.card_id,
                    collection=card.collection,
                    scene=card.scene,
                    title=card.title,
                    text=card.text,
                    vector_rank=vector_rank.get(card_id),
                    bm25_rank=bm25_rank.get(card_id),
                    rrf_score=round(score, 6),
                    payload=card.payload,
                )
            )
        fused.sort(key=lambda c: c.rrf_score, reverse=True)
        fused = fused[: self.config.rerank_top_k]

        # ---------- 步骤 3：重排（读正文） ----------
        reranked: list[RetrievedCard] = []
        for item in fused:
            card = card_by_id[item.card_id]
            rerank_score = self.reranker.score(query, card)
            if rerank_score < self.config.rerank_min_score:
                continue
            item.rerank_score = rerank_score
            item.final_score = rerank_score
            reranked.append(item)
        reranked.sort(key=lambda c: c.final_score, reverse=True)
        reranked = reranked[: self.config.rerank_top_k]

        # ---------- 步骤 4：截断 ----------
        selected, dynamic_k, token_used = self._truncate(query, reranked)

        return RetrievalResult(
            query=query,
            recalled_vector=[
                RetrievedCard(
                    card_id=card.card_id,
                    collection=card.collection,
                    scene=card.scene,
                    title=card.title,
                    text=card.text,
                    final_score=round(sim, 6),
                    payload=card.payload,
                )
                for card, sim in vector_scored
            ],
            recalled_bm25=[
                RetrievedCard(
                    card_id=doc_id,
                    collection=card_by_id[doc_id].collection,
                    scene=card_by_id[doc_id].scene,
                    title=card_by_id[doc_id].title,
                    text=card_by_id[doc_id].text,
                    final_score=round(score, 6),
                    payload=card_by_id[doc_id].payload,
                )
                for doc_id, score in bm25_hits
            ],
            fused=fused,
            reranked=reranked,
            selected=selected,
            dynamic_k=dynamic_k,
            token_budget_used=token_used,
        )

    # ------------------------------------------------------------------
    def _truncate(
        self,
        query: StructuredQuery,
        reranked: list[RetrievedCard],
    ) -> tuple[list[RetrievedCard], int, int]:
        """断崖截断 + 档位 K + token 预算。"""
        if not reranked:
            return [], 0, 0

        # 断崖：与上一个保留项分差过大即舍弃后面的
        kept: list[RetrievedCard] = []
        prev_score: float | None = None
        for card in reranked:
            if prev_score is not None and (prev_score - card.final_score) > self.config.score_gap_threshold:
                break
            kept.append(card)
            prev_score = card.final_score

        # 档位 K
        dynamic_k = self.config.dynamic_k_by_fit or {}
        limit = dynamic_k.get(query.fit, len(kept))

        # token 预算
        budget = self.config.card_char_budget
        used = 0
        selected: list[RetrievedCard] = []
        for card in kept:
            if len(selected) >= limit:
                break
            cost = len(card.text) + len(card.title)
            if used + cost > budget and selected:
                break
            used += cost
            selected.append(card)

        return selected, limit, used


__all__ = [
    "CrossEncoderReranker",
    "LocalCrossEncoder",
    "Retriever",
    "RetrieverConfig",
]
