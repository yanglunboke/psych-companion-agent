"""Tool Router。

文档口径："编排器里的 Tool Router 看三样东西：风险结论、intervention_fit、
说话的是学生还是老师。"

路由结果只有两条路：

* ``REACT`` —— 常规对话，按需调用（情绪识别 → 需求澄清 → 检索 → 积极引导
  → 资源建议〔仅需要练习时〕）；
* ``PLAN_EXECUTE`` —— 高危处理，固定计划三步（安全回应 → escalate_risk →
  通知老师家长），不再自由挑选工具。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import (
    InterventionFit,
    OrchestratorDecision,
    RiskVerdict,
    Route,
    Speaker,
)


@dataclass
class RoutePlan:
    """一轮的执行计划。"""

    route: Route
    steps: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reason: str = ""

    def to_decision(
        self,
        verdict: RiskVerdict,
        fit: InterventionFit | None,
        speaker: Speaker,
    ) -> OrchestratorDecision:
        return OrchestratorDecision(
            route=self.route,
            reason=self.reason,
            risk_verdict=verdict,
            intervention_fit=fit,
            speaker=speaker,
            skipped_tools=list(self.skipped),
        )


class ToolRouter:
    """按风险结论 + 干预档位 + 说话人决定本轮走哪条路、调哪些工具。"""

    def is_high_risk(self, verdict: RiskVerdict) -> bool:
        return verdict.is_high_risk

    def plan_high_risk(self, verdict: RiskVerdict) -> RoutePlan:
        """高危固定计划：不再自由挑选工具，停止一切建议。"""
        return RoutePlan(
            route=Route.PLAN_EXECUTE,
            steps=["safety_reply", "escalate_risk", "notify_teacher_parent"],
            skipped=[
                "need_clarification",
                "strategy_retrieval",
                "resource_suggestion",
                "positive_guidance_normal",
            ],
            reason=(
                f"高危固定上报流程（来源 {verdict.source.value}）："
                f"{verdict.evidence or '命中风险阈值'}"
            ),
        )

    def plan_react(
        self,
        fit: InterventionFit,
        speaker: Speaker,
    ) -> RoutePlan:
        """常规对话：按需调用。"""
        steps = ["need_clarification", "strategy_retrieval", "positive_guidance"]
        skipped: list[str] = []

        if fit == InterventionFit.需要练习:
            steps.append("resource_suggestion")
        else:
            skipped.append("resource_suggestion")

        if speaker == Speaker.老师:
            # 老师端不推学生练习卡（文档未定义老师侧行为，这里取最小惊讶策略）
            if "resource_suggestion" in steps:
                steps.remove("resource_suggestion")
                skipped.append("resource_suggestion")
            skipped.append("student_support_reply")

        return RoutePlan(
            route=Route.REACT,
            steps=steps,
            skipped=skipped,
            reason=(
                f"常规 ReAct 调用；intervention_fit={fit.value}；"
                f"说话人={speaker.value}"
            ),
        )


__all__ = ["RoutePlan", "ToolRouter"]
