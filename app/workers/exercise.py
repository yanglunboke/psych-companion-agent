"""资源建议 Worker（练习条目）。

文档口径：**只有 ``intervention_fit = 需要练习`` 时才运行**，
从工具库返回一张卡片：名称、时长、步骤（例如考前焦虑对应 4-7-8 呼吸）。

卡片自带 ``do_not_use_when``（高危预警 / 学生说只想说说），这里做最后一道闸：
命中就不给。
"""

from __future__ import annotations

import logging

from app.models.rule_engine import compose_exercise_text
from app.rag.query import build_exercise_query
from app.rag.retriever import Retriever
from app.schemas import EmotionCore, ExerciseCard, InterventionFit, NeedResult, SessionContext
from app.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class ExerciseWorker(BaseWorker):
    """资源建议 Worker。"""

    name = "resource_suggestion"

    def __init__(self, model, retriever: Retriever) -> None:
        super().__init__(model)
        self.retriever = retriever

    # ------------------------------------------------------------------
    async def run(
        self,
        ctx: SessionContext,
        emotion: EmotionCore,
        needs: NeedResult,
    ) -> ExerciseCard | None:
        """按场景检索练习卡并返回最合适的一张。"""
        if needs.intervention_fit != InterventionFit.需要练习:
            logger.debug("intervention_fit=%s，跳过资源建议", needs.intervention_fit)
            return None

        query = build_exercise_query(emotion, needs)
        result = await self.retriever.asearch(query, collections=["exercise"])
        for card in result.selected:
            payload = card.payload
            if not payload.get("steps"):
                continue
            return ExerciseCard(
                id=card.card_id,
                scene=card.scene,
                name=str(payload.get("name", card.title)),
                duration=str(payload.get("duration", "")),
                steps=[str(s) for s in payload.get("steps", [])],
                do_not_use_when=[str(s) for s in payload.get("do_not_use_when", [])],
                text=card.text,
            )
        return None

    async def pick(self, scene: str = "") -> ExerciseCard | None:
        """按场景直接取卡（供工具调用使用）。"""
        from app.schemas import StructuredQuery

        query = StructuredQuery(collection="exercise", scene=scene, keywords=[scene])
        result = await self.retriever.asearch(query, collections=["exercise"])
        for card in result.selected:
            payload = card.payload
            if payload.get("steps"):
                return ExerciseCard(
                    id=card.card_id,
                    scene=card.scene,
                    name=str(payload.get("name", card.title)),
                    duration=str(payload.get("duration", "")),
                    steps=[str(s) for s in payload.get("steps", [])],
                    do_not_use_when=[str(s) for s in payload.get("do_not_use_when", [])],
                    text=card.text,
                )
        return None

    @staticmethod
    def render(card: ExerciseCard) -> str:
        """练习卡转成给学生的文本。"""
        return compose_exercise_text(card)


__all__ = ["ExerciseWorker"]
