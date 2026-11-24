"""向量化。

默认实现是**本地哈希向量器**（hash trick + 字符/词特征 + L2 归一化）：
零模型下载、确定性、可离线运行，用于把"向量召回"这一路真实跑通。

生产环境把 :class:`Embedder` 换成真实 embedding 模型（bge-m3 / text-embedding-v3 /
SentenceTransformer）即可，检索层其余代码无需改动。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np

from app.rag.tokenizer import tokenize


class Embedder:
    """向量化器接口。"""

    name: str = "embedder"
    #: 该实现**自己标定**的相似度下限（None = 用配置里的文档口径阈值）。
    #:
    #: 文档给的"相似度低于 0.35 丢弃"是按**稠密 embedding 模型**的相似度
    #: 标定的。稀疏哈希向量的余弦分布完全不同，直接套 0.35 会把召回砍掉
    #: 近九成。所以本地占位实现自带一个标定值；换成真实 embedding 模型时
    #: 该值为 None，就回到文档口径的 0.35。
    calibrated_min_score: float | None = None

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


class HashingEmbedder(Embedder):
    """哈希向量器。

    token → 稳定哈希 → 固定维度桶，权重 ``1 + log(tf)``，最后 L2 归一化，
    因此点积即余弦相似度。

    标定：在本仓库的合成评测集上，金标卡相似度中位数 0.285、p5 0.157，
    非金标卡中位数 0.111。取 0.15 可保留 95% 的金标卡、滤掉 65% 的非金标卡。
    """

    name = "hashing"
    calibrated_min_score = 0.15

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = Counter(tokenize(text))
        for token, tf in counts.items():
            idx = self._bucket(token)
            vec[idx] += 1.0 + math.log(tf)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


class SentenceTransformerEmbedder(Embedder):
    """真实语义向量器（默认 bge-m3）。

    用 ``sentence-transformers`` 加载 bge-m3 产出稠密语义向量，让"向量召回"
    这一路真正按**语义**命中（"考砸了"≈"没考好"），而不是字符级重合。

    * ``calibrated_min_score`` 保持 ``None``：检索层会回落到文档口径的
      0.35（该阈值正是按稠密 embedding 模型的余弦相似度标定的）。
    * 首次调用会触发模型下载（约 2.3GB），建议设 ``HF_ENDPOINT`` 指向镜像。
    * 模型在第一次 ``embed`` 时**延迟加载**，避免拖慢进程启动、也避免测试
      在无模型环境下 import 即失败。
    """

    name = "bge-m3"
    calibrated_min_score = None

    def __init__(self, model_name: str = "BAAI/bge-m3", *, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._load()
        # bge 系列建议对查询加指令前缀，提升检索质量；文档侧不加。
        vec = model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter() or 0)


def build_embedder(backend: str = "hashing", model_name: str = "BAAI/bge-m3") -> Embedder:
    """按配置构造向量化器。

    ``hashing`` → 本地哈希向量器（离线默认，无语义）；
    ``bge`` → 真实语义向量器（需 sentence-transformers 与模型权重）。
    """
    if backend == "bge":
        return SentenceTransformerEmbedder(model_name=model_name)
    return HashingEmbedder()


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
]
