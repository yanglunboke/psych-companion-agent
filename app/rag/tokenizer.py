"""中英文混合分词。

检索层（向量占位器与 BM25）共用同一套分词，保证两条召回路的 token 口径一致。

* 英文/数字：按 ``[a-z_][a-z0-9_]*`` 切出完整词（``exam_failure`` 不会被切碎）；
* 中文：jieba 词 + 相邻字的二元组（二元组让"以偏概全"这类固定搭配更稳）。
"""

from __future__ import annotations

import re

_SEG_RE = re.compile(r"([a-z_][a-z0-9_]*|\d+)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = set("，。！？、；：,.!?;:'\"“”‘’（）()《》【】…—·「」『』【】 \t\n\r")

_jieba = None


def _get_jieba():
    """延迟加载 jieba（首次调用约 0.5s，避免拖慢进程启动）。"""
    global _jieba
    if _jieba is None:
        import jieba  # noqa: PLC0415

        jieba.setLogLevel(60)
        _jieba = jieba
    return _jieba


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def tokenize(text: str) -> list[str]:
    """把一段文本切成检索用的 token 列表。"""
    if not text:
        return []
    text = text.lower()
    tokens: list[str] = []
    for segment in _SEG_RE.split(text):
        if not segment or not segment.strip():
            continue
        if _SEG_RE.fullmatch(segment):
            tokens.append(segment)
            continue
        # 中文/标点混合：jieba 切词 + 二元组
        for word in _get_jieba().cut(segment):
            word = word.strip()
            if not word or word in _PUNCT:
                continue
            if not _is_cjk(word):
                tokens.append(word.lower())
                continue
            tokens.append(word)
            if len(word) > 1:
                for i in range(len(word) - 1):
                    tokens.append(word[i : i + 2])
    return tokens


__all__ = ["tokenize"]
