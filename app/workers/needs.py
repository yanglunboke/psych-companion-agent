"""Worker 1：需求澄清。

输入：情绪识别结果 + 心理侧写摘要（**不再读数据库**）。
输出：学生的具体需求 + ``intervention_fit``。
"""

from __future__ import annotations

import logging

from agentscope.message import UserMsg

from app.models.mock_model import mark_rule_usage
from app.models.rule_engine import infer_needs
from app.prompts import NEEDS_SYSTEM_PROMPT, build_user_message
from app.schemas import EmotionCore, NeedResult, SessionContext
from app.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class NeedsWorker(BaseWorker):
    """需求澄清 Worker。"""

    name = "need_clarification"

    async def run(self, ctx: SessionContext, emotion: EmotionCore) -> NeedResult:
        content = build_user_message(
            text=ctx.current_text,
            recent_turns=ctx.recent_turns,
            profile=ctx.profile,
            emotion=emotion,
        )
        agent = self.build_agent(NEEDS_SYSTEM_PROMPT)
        try:
            msg = await agent.reply(
                UserMsg(name="student", content=content),
                structured_schema=NeedResult,
            )
            payload = msg.structured_output
            if not payload:
                raise ValueError("需求澄清没有返回结构化结果")
            return NeedResult.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("需求澄清调用失败，回退到规则引擎：%s", exc)
            mark_rule_usage()
            profile = ctx.profile
            profile_obj = None
            if profile is not None:
                from app.schemas import StudentProfile

                profile_obj = StudentProfile(
                    student_id=profile.student_id,
                    grade=profile.grade,
                    aspiration=profile.aspiration,
                    strengths=list(profile.strengths),
                    stable_distortions=list(profile.stable_distortions),
                    unmet_needs=list(profile.unmet_needs),
                    risk_level=profile.risk_level,
                    last_cause=profile.last_cause,
                )
            from app.schemas import EmotionResult

            emotion_result = EmotionResult(
                emotion_type=emotion.emotion_type,
                intensity=emotion.intensity,
                cause_domain=emotion.cause_domain,
                cause_detail=emotion.cause_detail,
                candidate_distortions=list(emotion.candidate_distortions),
                risk_score=0.0,
                confidence=emotion.confidence,
            )
            return infer_needs(ctx.current_text, emotion_result, profile_obj)


__all__ = ["NeedsWorker"]
