"""反思模块（复盘）。

文档口径：

* 常规轮：**异步**跑，不让流程卡在这里；主要看两件事 —— 是不是一上来就讲道理，
  以及学生明明只想说说，回复里却给了方法。有问题就记在这轮日志上，
  下一轮带上一句提醒。
* 高危轮：**同步**查。上报如果没有成功，或者这轮草稿里带了诊断词，
  这轮回复就不发给学生。

因此本模块提供两个入口：

* :meth:`ReflectionService.review` —— 常规轮的质量复盘（异步调用）；
* :meth:`ReflectionService.gate_high_risk` —— 高危轮的放行闸门（同步调用）。

除文档点名的两条之外，还额外检查"结尾问了不止一个问题"——那是积极引导
自己的硬性规则（只问一个问题），一并纳入复盘口径。
"""

from __future__ import annotations

import logging
import re

from app.schemas import InterventionFit, ReflectionResult

logger = logging.getLogger(__name__)

#: 诊断词——高危草稿里出现就不该发给学生
DIAGNOSTIC_TERMS = (
    "抑郁症",
    "焦虑症",
    "双相",
    "确诊",
    "心理疾病",
    "精神疾病",
    "你这是",
    "你有病",
    "应该去看医生",
    "需要看心理医生",
    "典型的",
)

#: 讲道理/说教的开场特征
LECTURE_TERMS = (
    "你应该",
    "你必须",
    "你要学会",
    "你应该懂",
    "首先你",
    "第一，",
    "第二，",
    "道理",
    "换位思考",
    "想开点",
    "别想太多",
)

#: 给方法的特征
METHOD_TERMS = (
    "方法",
    "步骤",
    "练习",
    "可以试试",
    "教你",
    "深呼吸",
    "建议你",
    "计划表",
    "去做",
    "第一，你可以",
    "建议你试试",
)

#: 句末标点（中英文）
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n；;]+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


class ReflectionService:
    """复盘服务。"""

    def __init__(self) -> None:
        self._hints: dict[str, str] = {}

    # ------------------------------------------------------------------
    def set_hint(self, student_id: str, hint: str | None) -> None:
        """保存给下一轮的提醒。"""
        if hint:
            self._hints[student_id] = hint
        else:
            self._hints.pop(student_id, None)

    def get_hint(self, student_id: str) -> str | None:
        return self._hints.get(student_id)

    # ------------------------------------------------------------------
    def review(
        self,
        draft: str,
        fit: InterventionFit,
        *,
        trace_id: str = "",
    ) -> ReflectionResult:
        """常规轮的质量复盘。"""
        issues: list[str] = []
        sentences = _sentences(draft)

        # 1) 一上来就讲道理：首句就带说教特征
        if sentences and any(term in sentences[0] for term in LECTURE_TERMS):
            issues.append("一上来就讲道理")
        # 后面任何一句突然开始说教，同样算问题
        elif len(sentences) > 1 and any(
            term in s for s in sentences[1:] for term in ("你应该", "你必须", "第一，")
        ):
            issues.append("中途开始讲道理")

        # 2) 学生只想说说，回复里却给了方法
        if fit == InterventionFit.只想说说 and any(term in draft for term in METHOD_TERMS):
            issues.append("学生只想说说，回复里却给了方法")

        # 3) 结尾问了不止一个问题（积极引导的硬性规则）
        if draft.count("？") + draft.count("?") > 1:
            issues.append("问了不止一个问题")

        hint = self._hint_for(issues)
        return ReflectionResult(
            trace_id=trace_id,
            issues=issues,
            passed=not issues,
            blocking=False,
            hint_for_next_turn=hint,
        )

    @staticmethod
    def _hint_for(issues: list[str]) -> str | None:
        if "一上来就讲道理" in issues or "中途开始讲道理" in issues:
            return "上一轮你一上来就讲道理了，这一轮先接住情绪，只问一个问题。"
        if "学生只想说说，回复里却给了方法" in issues:
            return "上一轮学生只想说说，你却给了方法。这一轮只听，不给方法。"
        if "问了不止一个问题" in issues:
            return "上一轮你问了不止一个问题。这一轮结尾只留一个问题。"
        return None

    # ------------------------------------------------------------------
    def gate_high_risk(
        self,
        draft: str,
        *,
        escalation_ok: bool,
        trace_id: str = "",
    ) -> ReflectionResult:
        """高危轮的同步闸门：不过检就不发给学生。"""
        issues: list[str] = []
        if not escalation_ok:
            issues.append("高危上报未成功")
        if any(term in draft for term in DIAGNOSTIC_TERMS):
            issues.append("草稿里带了诊断词")
        if draft.count("？") + draft.count("?") > 1:
            issues.append("安全回应里问了不止一个问题")
        return ReflectionResult(
            trace_id=trace_id,
            issues=issues,
            passed=not issues,
            blocking=bool(issues),
            hint_for_next_turn=None,
        )


__all__ = [
    "DIAGNOSTIC_TERMS",
    "LECTURE_TERMS",
    "METHOD_TERMS",
    "ReflectionService",
]
