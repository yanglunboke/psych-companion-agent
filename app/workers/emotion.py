"""Worker 0：情绪识别（每轮都跑，只出 JSON）。

输出两段：

* :class:`EmotionCore` —— 情绪类型 / 强度 / 诱因域 / 候选认知偏差；
* :class:`RiskScore` —— 风险分。文档强调风险分由**更小更快的模型**、温度 0
  单独产出，所以这里用一个独立的小模型实例，而不是让主模型顺手给一个分。
"""

from __future__ import annotations

import logging

from agentscope.message import UserMsg

from app.models.mock_model import mark_rule_usage
from app.models.rule_engine import infer_emotion
from app.prompts import EMOTION_SYSTEM_PROMPT, RISK_SYSTEM_PROMPT, build_user_message
from app.schemas import EmotionCore, RiskScore, SessionContext
from app.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class EmotionWorker(BaseWorker):
    """情绪识别 Worker。"""

    name = "emotion_recognition"

    async def run(self, ctx: SessionContext) -> EmotionCore:
        """识别情绪结构化结果（不含风险分）。"""
        content = build_user_message(
            text=ctx.current_text,
            recent_turns=ctx.recent_turns,
            profile=ctx.profile,
        )
        agent = self.build_agent(EMOTION_SYSTEM_PROMPT)
        try:
            msg = await agent.reply(
                UserMsg(name="student", content=content),
                structured_schema=EmotionCore,
            )
            payload = msg.structured_output
            if not payload:
                raise ValueError("情绪识别没有返回结构化结果")
            return EmotionCore.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("情绪识别调用失败，回退到规则引擎：%s", exc)
            return self._fallback(ctx)

    def _fallback(self, ctx: SessionContext) -> EmotionCore:
        mark_rule_usage()
        result = infer_emotion(
            ctx.current_text,
            [t.content for t in ctx.recent_turns if t.role == "student"],
        )
        return EmotionCore(
            emotion_type=result.emotion_type,
            intensity=result.intensity,
            cause_domain=result.cause_domain,
            cause_detail=result.cause_detail,
            candidate_distortions=result.candidate_distortions,
            confidence=result.confidence,
        )


class RiskScorer(BaseWorker):
    """风险分小模型（temperature=0）。"""

    name = "risk_scorer"
    max_iters = 4

    async def score(self, ctx: SessionContext) -> RiskScore:
        """输出 0-1 的风险分与依据。"""
        content = build_user_message(
            text=ctx.current_text,
            recent_turns=ctx.recent_turns,
            profile=ctx.profile,
        )
        agent = self.build_agent(RISK_SYSTEM_PROMPT, name="risk_scorer")
        try:
            msg = await agent.reply(
                UserMsg(name="student", content=content),
                structured_schema=RiskScore,
            )
            payload = msg.structured_output
            if not payload:
                raise ValueError("风险分没有返回结构化结果")
            return RiskScore.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("风险分调用失败，回退到规则引擎：%s", exc)
            mark_rule_usage()
            result = infer_emotion(
                ctx.current_text,
                [t.content for t in ctx.recent_turns if t.role == "student"],
            )
            return RiskScore(risk_score=result.risk_score, risk_reason=result.risk_reason)


__all__ = ["EmotionWorker", "RiskScorer"]
