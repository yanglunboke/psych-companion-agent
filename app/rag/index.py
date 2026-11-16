"""策略知识库索引。

把 ``app/data/cards/*.json`` 里的卡片读进内存，建好三样东西：

* 每张卡的向量（用于向量召回）；
* 每张卡的倒排统计（用于 BM25）；
* 按 ``collection`` 分组的索引（学生共情轮只看得到回应卡和禁止卡，
  练习要等 ``intervention_fit = 需要练习`` 才可见 —— 靠 collection 做隔离）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.rag.bm25 import BM25Index
from app.rag.embedding import Embedder, HashingEmbedder
from app.schemas import Collection


@dataclass
class IndexedCard:
    """索引中的一张卡。"""

    card_id: str
    collection: str
    scene: str
    title: str
    text: str
    payload: dict = field(default_factory=dict)
    vector: np.ndarray | None = None

    @property
    def search_text(self) -> str:
        return " ".join([self.title, self.scene, self.text])


class CardIndex:
    """全部卡片的内存索引。"""

    def __init__(self, cards: list[IndexedCard], embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._cards: dict[str, IndexedCard] = {c.card_id: c for c in cards}
        self._by_collection: dict[str, list[IndexedCard]] = {}
        for card in cards:
            self._by_collection.setdefault(card.collection, []).append(card)
        self._bm25 = BM25Index()
        for card in cards:
            card.vector = self.embedder.embed(card.search_text)
            self._bm25.add(card.card_id, card.search_text)
        self._bm25.finalize()

    # ------------------------------------------------------------------
    @property
    def cards(self) -> list[IndexedCard]:
        return list(self._cards.values())

    def get(self, card_id: str) -> IndexedCard | None:
        return self._cards.get(card_id)

    def by_collection(self, collection: str) -> list[IndexedCard]:
        return list(self._by_collection.get(collection, []))

    def collections(self) -> list[str]:
        return sorted(self._by_collection.keys())

    def bm25(self) -> BM25Index:
        return self._bm25

    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        files: list[Path],
        embedder: Embedder | None = None,
    ) -> "CardIndex":
        """从 JSON 文件加载卡片。"""
        cards: list[IndexedCard] = []
        for path in files:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            collection = raw.get("collection", "")
            for item in raw.get("cards", []):
                payload = {k: v for k, v in item.items() if k not in {"id", "text"}}
                cards.append(
                    IndexedCard(
                        card_id=item["id"],
                        collection=collection,
                        scene=item.get("scene", ""),
                        title=item.get("title", ""),
                        text=item.get("text", ""),
                        payload=payload,
                    ),
                )
        return cls(cards, embedder=embedder)


def default_card_files(data_dir: Path) -> list[Path]:
    """知识库默认文件清单。"""
    return [
        data_dir / "cards" / "student_reply_cards.json",
        data_dir / "cards" / "ban_cards.json",
        data_dir / "cards" / "exercise_cards.json",
    ]


def build_default_index(data_dir: Path, embedder: Embedder | None = None) -> CardIndex:
    """构建默认索引（只加载存在的文件）。"""
    files = [p for p in default_card_files(data_dir) if p.exists()]
    return CardIndex.from_files(files, embedder=embedder)


__all__ = [
    "CardIndex",
    "IndexedCard",
    "Collection",
    "build_default_index",
    "default_card_files",
]
