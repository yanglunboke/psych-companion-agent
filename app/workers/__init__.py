"""四个职责固定的 Worker。

    EmotionWorker  情绪识别（每轮必跑，只出 JSON）
    NeedsWorker    需求澄清（读情绪 JSON + 侧写，不再读数据库）
    GuidanceWorker 积极引导（唯一给学生回复的 Worker，流式）
    ExerciseWorker 资源建议（仅 intervention_fit = 需要练习时运行）
"""

from app.workers.base import BaseWorker
from app.workers.emotion import EmotionWorker, RiskScorer
from app.workers.exercise import ExerciseWorker
from app.workers.guidance import GuidanceWorker
from app.workers.needs import NeedsWorker

__all__ = [
    "BaseWorker",
    "EmotionWorker",
    "ExerciseWorker",
    "GuidanceWorker",
    "NeedsWorker",
    "RiskScorer",
]
