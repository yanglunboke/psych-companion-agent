"""结构化查询构造。

文档明确：**不用学生原句当查询**。情绪识别出来后，用结构化字段编一条查询：

    scene=exam_failure
    distortion=以偏概全
    fit=轻量引导
    steps=共情接纳、认知解离、优势强化

好处有两个：向量侧能匹配到语义相近的卡；BM25 侧能靠"以偏概全"这种
**逐字出现**的词精确命中。
"""

from __future__ import annotations

from app.schemas import EmotionCore, InterventionFit, NeedResult, StructuredQuery

#: 认知偏差 → 策略步骤
DISTORTION_STEPS: dict[str, list[str]] = {
    "以偏概全": ["认知解离", "优势强化"],
    "灾难化": ["认知解离"],
    "非黑即白": ["认知解离"],
    "过度自责": ["优势强化"],
    "读心术": ["事实澄清"],
}

#: 诱因域 + 情绪类型 → 场景标识
_SCENE_TABLE: tuple[tuple[str, str, str], ...] = (
    ("学业", "沮丧", "exam_failure"),
    ("学业", "焦虑", "exam_anxiety"),
    ("学业", "迷茫", "exam_anxiety"),
    ("人际", "委屈", "peer_conflict"),
    ("人际", "愤怒", "peer_conflict"),
    ("家庭", "委屈", "family_conflict"),
    ("家庭", "愤怒", "family_conflict"),
    ("自我", "自责", "self_doubt"),
    ("自我", "迷茫", "self_doubt"),
    ("自我", "绝望", "self_doubt"),
)

#: 诱因域兜底场景
_DOMAIN_FALLBACK = {
    "学业": "exam_failure",
    "人际": "peer_conflict",
    "家庭": "family_conflict",
    "自我": "self_doubt",
    "未知": "self_doubt",
}

#: 场景 → 补充关键词（让向量/BM25 更容易命中对应卡片）
SCENE_KEYWORDS: dict[str, list[str]] = {
    "exam_failure": ["考试失利", "学业自我否定", "考砸"],
    "exam_anxiety": ["考前焦虑", "紧张", "睡不着"],
    "exam_progress": ["考试进步", "被表扬", "考得比上次好"],
    "peer_conflict": ["同伴冲突", "被孤立", "吵架"],
    "family_conflict": ["家庭冲突", "父母", "争执"],
    "self_doubt": ["自我否定", "没用", "我很笨"],
    "loneliness": ["孤独", "没人可以说"],
    "bullying": ["校园欺凌", "起绰号", "被排挤"],
    "subject_pressure": ["选科压力", "文理分科", "选科"],
    "sports_failure": ["体育挫败", "比赛失利", "体测"],
    "appearance": ["外貌焦虑", "身材焦虑", "照镜子"],
    "friend_leaving": ["好友离别", "转学", "分开"],
    "club_competition": ["社团竞争", "名额落选", "没选上"],
    "teacher_misunderstanding": ["被老师误解", "被冤枉", "老师批评"],
    "dormitory": ["宿舍关系", "室友", "作息"],
    "parents_control": ["父母管控", "手机被收", "限制外出"],
    "procrastination": ["拖延", "时间管理", "做不完"],
    "positive_progress": ["进步", "做到了"],
}


def infer_scene(emotion: EmotionCore) -> str:
    """由情绪结果推断场景标识。"""
    cause = emotion.cause_domain.value
    for domain, emo, scene in _SCENE_TABLE:
        if cause == domain and emotion.emotion_type == emo:
            return scene
    # 诱因描述里带明确信号时优先按关键词判断
    detail = emotion.cause_detail or ""
    if "霸凌" in detail or "欺负" in detail:
        return "bullying"
    if "孤独" in detail:
        return "loneliness"
    return _DOMAIN_FALLBACK.get(cause, "self_doubt")


def infer_steps(emotion: EmotionCore) -> list[str]:
    """由认知偏差推断策略步骤。"""
    steps = ["共情接纳"]
    for distortion in emotion.candidate_distortions:
        for step in DISTORTION_STEPS.get(distortion, []):
            if step not in steps:
                steps.append(step)
    if len(steps) == 1:
        steps.append("优势强化")
    return steps


def build_query(
    emotion: EmotionCore,
    needs: NeedResult,
    *,
    collection: str = "student_reply",
) -> StructuredQuery:
    """构造结构化检索查询。"""
    scene = infer_scene(emotion)
    keywords = list(SCENE_KEYWORDS.get(scene, []))
    if emotion.cause_detail:
        keywords.append(emotion.cause_detail.split("（")[0])
    return StructuredQuery(
        collection=collection,
        scene=scene,
        distortion=list(emotion.candidate_distortions),
        fit=needs.intervention_fit.value,
        steps=infer_steps(emotion),
        keywords=keywords,
    )


def build_exercise_query(
    emotion: EmotionCore,
    needs: NeedResult,
) -> StructuredQuery:
    """资源建议阶段的查询（collection = exercise）。"""
    query = build_query(emotion, needs, collection="exercise")
    return query


__all__ = [
    "DISTORTION_STEPS",
    "SCENE_KEYWORDS",
    "build_exercise_query",
    "build_query",
    "infer_scene",
    "infer_steps",
]
