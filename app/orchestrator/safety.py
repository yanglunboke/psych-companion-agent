"""安全门：关键词词表 + 风险分阈值。

两路判定的口径来自文档：

* **第一路 关键词触发**：对学生这句话做词表和模式匹配，覆盖文档里的保密例外
  —— 明确的自伤或自杀、伤人、未成年人受侵害。命中直接返回
  ``lexicon_hit = true``，同时给出原因码。
* **第二路 风险分**：情绪识别 Worker 里那个更小更快、温度 0 的小模型给出的
  ``risk_score``，达到阈值（0.72）就按高危风险处理。

两路结果交回同一个函数，入参是 ``lexicon_hit`` 和 ``risk_score``，
由函数判断启用哪个流程。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.schemas import LexiconHit, RiskSource, RiskVerdict


@dataclass
class LexiconCategory:
    """一个高危类别。"""

    code: str
    category: str
    severity: str
    terms: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    note: str = ""
    confidentiality_exception: bool = False
    _compiled: list[re.Pattern[str]] = field(default_factory=list, repr=False)

    def compile(self) -> None:
        self._compiled = [re.compile(p) for p in self.patterns]


@dataclass
class SuppressRule:
    """抑制规则：命中这些表达时不算该类别的高危。"""

    pattern: str
    reason: str
    category: str
    _compiled: re.Pattern[str] | None = field(default=None, repr=False)

    def compile(self) -> None:
        self._compiled = re.compile(self.pattern)


class Lexicon:
    """高危关键词词表。"""

    def __init__(self, categories: list[LexiconCategory], suppress: list[SuppressRule]) -> None:
        self.categories = categories
        self.suppress = suppress

    @classmethod
    def from_file(cls, path: Path) -> "Lexicon":
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        categories = [
            LexiconCategory(
                code=item["code"],
                category=item.get("category", ""),
                severity=item.get("severity", "critical"),
                terms=list(item.get("terms", [])),
                patterns=list(item.get("patterns", [])),
                note=item.get("note", ""),
                confidentiality_exception=bool(item.get("confidentiality_exception", False)),
            )
            for item in raw.get("categories", [])
        ]
        for category in categories:
            category.compile()
        suppress = [
            SuppressRule(
                pattern=item["pattern"],
                reason=item.get("reason", ""),
                category=item.get("category", ""),
            )
            for item in raw.get("suppress", [])
        ]
        for rule in suppress:
            rule.compile()
        return cls(categories, suppress)

    def _suppressed(self, text: str, category_code: str) -> SuppressRule | None:
        for rule in self.suppress:
            if rule.category and rule.category != category_code:
                continue
            if rule._compiled and rule._compiled.search(text):
                return rule
        return None

    def scan(self, text: str) -> LexiconHit:
        """对一句话做词表与模式匹配。"""
        if not text:
            return LexiconHit()
        for category in self.categories:
            matched = [term for term in category.terms if term in text]
            if not matched:
                matched = []
                for pattern in category._compiled:
                    found = pattern.search(text)
                    if found:
                        matched.append(found.group(0))
            if not matched:
                continue
            suppressed = self._suppressed(text, category.code)
            if suppressed is not None:
                continue
            return LexiconHit(
                hit=True,
                reason_code=category.code,
                matched_terms=matched[:5],
                category=category.category,
            )
        return LexiconHit()


class SafetyGate:
    """安全门：把两路判定合成一个结论。"""

    def __init__(self, lexicon: Lexicon, threshold: float = 0.72) -> None:
        self.lexicon = lexicon
        self.threshold = threshold

    def scan(self, text: str) -> LexiconHit:
        return self.lexicon.scan(text)

    def verdict(self, hit: LexiconHit, risk_score: float) -> RiskVerdict:
        """入参 lexicon_hit 与 risk_score，由函数决定启用哪个流程。"""
        score_hit = risk_score >= self.threshold
        is_high = bool(hit.hit) or score_hit

        if hit.hit and score_hit:
            source = RiskSource.BOTH
        elif hit.hit:
            source = RiskSource.LEXICON
        elif score_hit:
            source = RiskSource.MODEL
        else:
            source = RiskSource.NONE

        evidence = (
            f"关键词命中 {hit.matched_terms}（{hit.category}）"
            if hit.hit
            else f"风险分 {risk_score:.2f} 未达阈值 {self.threshold:.2f}"
        )
        if hit.hit and score_hit:
            evidence += f"；风险分 {risk_score:.2f} 也已越线"

        return RiskVerdict(
            lexicon_hit=hit.hit,
            reason_code=hit.reason_code or (None if not score_hit else "risk_score_threshold"),
            matched_terms=list(hit.matched_terms),
            risk_score=risk_score,
            threshold=self.threshold,
            is_high_risk=is_high,
            source=source,
            evidence=evidence,
        )


__all__ = ["Lexicon", "LexiconCategory", "SafetyGate", "SuppressRule"]
