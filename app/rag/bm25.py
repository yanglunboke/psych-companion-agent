"""BM25（Okapi）实现。

纯 Python，无外部依赖。文档指出 BM25 这一路能精确命中"以偏概全"这类
**逐字出现**的词，而向量召回做不到这么准——所以两路要并行。
"""

from __future__ import annotations

import math
from collections import Counter

from app.rag.tokenizer import tokenize


class BM25Index:
    """Okapi BM25 索引。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_freq: Counter[str] = Counter()
        self._doc_len: dict[str, int] = {}
        self._avg_len = 0.0
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._doc_tokens[doc_id] = tokens
        self._doc_len[doc_id] = len(tokens)
        for token in set(tokens):
            self._doc_freq[token] += 1

    def finalize(self) -> None:
        """所有文档加入后计算 IDF 与平均长度。"""
        n_docs = len(self._doc_tokens)
        self._avg_len = (
            sum(self._doc_len.values()) / n_docs if n_docs else 0.0
        )
        self._idf = {}
        for token, freq in self._doc_freq.items():
            # 平滑 IDF，保证非负
            self._idf[token] = max(
                0.0,
                math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)),
            )

    # ------------------------------------------------------------------
    def idf(self, token: str) -> float:
        return self._idf.get(token, 0.0)

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        doc_tokens = self._doc_tokens.get(doc_id)
        if not doc_tokens:
            return 0.0
        freq = Counter(doc_tokens)
        doc_len = self._doc_len[doc_id]
        total = 0.0
        for token in query_tokens:
            tf = freq.get(token, 0)
            if tf == 0:
                continue
            idf = self._idf.get(token, 0.0)
            denom = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / self._avg_len if self._avg_len else 1.0)
            )
            total += idf * (tf * (self.k1 + 1)) / denom
        return total

    def search(
        self,
        query: str,
        top_k: int = 20,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """返回 ``[(doc_id, score), ...]``，按分数降序。

        ``allowed`` 用于把召回限制在本次可见的 collection 内（学生共情轮只看
        得到回应卡和禁止卡，练习卡在 intervention_fit = 需要练习 之前不可见）。
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self._doc_tokens:
            return []
        doc_ids = self._doc_tokens.keys()
        if allowed is not None:
            doc_ids = [d for d in doc_ids if d in allowed]
        scored = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in doc_ids]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


__all__ = ["BM25Index"]
