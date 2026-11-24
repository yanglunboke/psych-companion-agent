"""心理侧写聚合逻辑。

文档口径：

* **情绪轨迹**是按时间排的流水，每轮情绪识别落地后追加一条（时间、情绪类型、
  强度、诱因域）。强度用 ``0.7 * 旧值 + 0.3 * 本轮``，保证一句气话不会把主导情绪改掉。
* **心理侧写**是从很多轮里收出来的稳定结论，字段固定：年级和升学目标、优势 Top3、
  已经三次验证过的认知偏差、没被满足的需求、风险等级、上一次诱因。
  认知偏差每次出现都记候选，``evidence_count`` 到 3 才写进稳定项。
"""

from __future__ import annotations

import re
import time

from app.schemas import (
    CauseDomain,
    EmotionResult,
    InterventionFit,
    NeedResult,
    ProfileSummary,
    StudentProfile,
)

#: 优势提取线索
_STRENGTH_PATTERNS = (
    re.compile(r"我(?:比较|特别|最|挺)?擅长([^，。！？,.!?\n]{1,12})"),
    re.compile(r"我(?:的)?优点(?:是|就是)?([^，。！？,.!?\n]{1,12})"),
    re.compile(r"我(?:最)?拿手(?:的)?(?:是)?([^，。！？,.!?\n]{1,12})"),
    # 侧写摘要里的「优势：xxx」也要能解析回去（摘要会被原样塞进 Prompt）
    re.compile(r"优势[:：]\s*([^，。、\n]{1,12})"),
)

#: 年级线索
_GRADE_PATTERNS = (
    re.compile(r"(高[一二三]|初[一二三]|中[一二三])(年级)?"),
)

# ---------------------------------------------------------------------------
# 三维度：积极认知资源 + VIA 品格优势 + 情绪趋势
# ---------------------------------------------------------------------------

#: 积极认知资源线索 → 归一化标签（成长型思维 / 积极归因 / 自我接纳）
_POSITIVE_COGNITION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("成长型思维", re.compile(r"(我会?努力|还能进步|慢慢来|只要.*就能|坚持.*就|练习.*就能|多练)")),
    ("积极归因", re.compile(r"(这次没考好|这次没做好).{0,6}(没关系|正常|不代表|还能|下次)|不是我(不行|笨).*(而是|是)")),
    ("自我接纳", re.compile(r"(接受?自己|接纳自己|我就是我|也挺好|没关系|别太苛责|放过自己)")),
)

#: VIA 品格优势关键词 → 优势名
_VIA_STRENGTH_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("韧性", ("坚持", "不放弃", "挺过来", "扛住", "熬过去", "没被打倒", "爬起来")),
    ("创造力", ("创意", "点子", "想出新", "原创", "想象力", "灵感")),
    ("同理心", ("理解别人", "体谅", "替别人想", "感同身受", "能理解")),
    ("勇敢", ("鼓起勇气", "敢", "不怕", "站出来", "豁出去")),
    ("感恩", ("感谢", "谢谢", "感激", "感恩", "珍惜")),
    ("好奇心", ("好奇", "想知道", "探索", "钻研", "感兴趣")),
    ("自律", ("坚持早起", "按计划", "管住自己", "自律", "打卡")),
    ("善良", ("帮助", "帮别人", "照顾", "安慰", "陪")),
)


def extract_positive_cognitions(text: str) -> list[str]:
    """从学生原话里提取积极认知资源（成长型思维 / 积极归因 / 自我接纳）。"""
    found: list[str] = []
    for label, pattern in _POSITIVE_COGNITION_PATTERNS:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


def extract_via_strengths(text: str) -> list[str]:
    """按 VIA 品格优势关键词识别优势（比"我擅长"正则覆盖面更广）。"""
    found: list[str] = []
    for name, keywords in _VIA_STRENGTH_KEYWORDS:
        if any(kw in text for kw in keywords) and name not in found:
            found.append(name)
    return found


def detect_emotion_trend(previous_ema: float, current_ema: float, delta: float = 0.15) -> str:
    """由前后两次强度平滑值判断情绪趋势。

    当前值显著高于前值 → 上升；显著低于 → 下降；否则平稳。
    """
    if current_ema - previous_ema > delta:
        return "上升"
    if previous_ema - current_ema > delta:
        return "下降"
    return "平稳"


def infer_attribution_style(positive_counts: dict[str, int], stable_distortions: list[str]) -> str:
    """由积极归因证据与稳定认知偏差推断归因风格。

    积极归因证据占优 → 积极归因；稳定偏差占优 → 消极归因；否则尚未明确。
    """
    positive = positive_counts.get("积极归因", 0)
    negative = len(stable_distortions)
    if positive >= 3 and positive > negative:
        return "积极归因"
    if negative >= 3 and negative >= positive:
        return "消极归因"
    return "尚未明确"


def extract_strengths(text: str) -> list[str]:
    """从学生原话里抽取"我擅长/优点是"这类自述优势。"""
    found: list[str] = []
    for pattern in _STRENGTH_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).strip(" 的了，,")
            if value and value not in found:
                found.append(value)
    return found


def extract_grade(text: str) -> str | None:
    """从学生原话里抽取年级。"""
    for pattern in _GRADE_PATTERNS:
        match = pattern.search(text)
        if match:
            grade = match.group(1)
            suffix = match.group(2) or ""
            return grade + suffix
    return None


def smooth_intensity(previous: float, current: int, alpha: float = 0.3) -> float:
    """情绪强度的指数平滑：``0.7 * 旧值 + 0.3 * 本轮``。"""
    return round((1 - alpha) * previous + alpha * current, 4)


def update_profile(
    profile: StudentProfile,
    emotion: EmotionResult,
    needs: NeedResult | None = None,
    *,
    alpha: float = 0.3,
    evidence_threshold: int = 3,
    student_text: str = "",
) -> StudentProfile:
    """用本轮的情绪与需求更新侧写（原地返回新对象）。"""
    updated = profile.model_copy(deep=True)

    # 1) 情绪强度指数平滑
    updated.intensity_ema = smooth_intensity(profile.intensity_ema, emotion.intensity, alpha)

    # 2) 认知偏差：先记候选，证据满 3 次才进稳定项
    for distortion in emotion.candidate_distortions:
        count = updated.evidence_counts.get(distortion, 0) + 1
        updated.evidence_counts[distortion] = count
        if count >= evidence_threshold and distortion not in updated.stable_distortions:
            updated.stable_distortions.append(distortion)

    # 2b) 积极认知资源：同样走 3 次交叉验证（认知维度的积极面）
    for cognition in extract_positive_cognitions(student_text):
        count = updated.positive_cognition_counts.get(cognition, 0) + 1
        updated.positive_cognition_counts[cognition] = count
        if count >= evidence_threshold and cognition not in updated.positive_cognitions:
            updated.positive_cognitions.append(cognition)

    # 3) 未被满足的需求
    if needs:
        for need in needs.needs:
            if need not in updated.unmet_needs:
                updated.unmet_needs.append(need)
        updated.unmet_needs = updated.unmet_needs[:5]

    # 4) 上一次诱因与情绪
    #    只有"有实质情绪"的一轮才更新——否则一句平淡的话会把上一次诱因冲成"未知"，
    #    侧写就跟着抖了。
    meaningful = emotion.cause_domain != CauseDomain.未知 and emotion.emotion_type != "平静"
    if meaningful:
        updated.last_cause = emotion.cause_detail or emotion.cause_domain.value
        updated.last_emotion = emotion.emotion_type

    # 5) 优势 Top3 + VIA 品格优势（优势聚焦原则：每轮都找优势）
    for strength in extract_strengths(student_text):
        if strength not in updated.strengths:
            updated.strengths.append(strength)
    for via in extract_via_strengths(student_text):
        updated.via_strengths[via] = updated.via_strengths.get(via, 0) + 1
    # VIA 优势按证据次数补进 strengths，保持 Top3
    ranked_via = sorted(updated.via_strengths.items(), key=lambda kv: kv[1], reverse=True)
    for via, _count in ranked_via:
        if via not in updated.strengths:
            updated.strengths.append(via)
    updated.strengths = updated.strengths[:3]

    # 5b) 情绪趋势：比较平滑前后的强度
    updated.emotion_trend = detect_emotion_trend(
        profile.intensity_ema, updated.intensity_ema
    )

    # 5c) 归因风格 / 认知模式
    updated.attribution_style = infer_attribution_style(
        updated.positive_cognition_counts, updated.stable_distortions
    )

    # 6) 年级
    grade = extract_grade(student_text)
    if grade:
        updated.grade = grade

    # 7) 风险等级
    if emotion.risk_score >= 0.72:
        updated.risk_level = "high"
    elif emotion.risk_score >= 0.45:
        updated.risk_level = "medium"
    elif updated.risk_level != "high":
        updated.risk_level = "low"

    updated.updated_at = time.time()
    return updated


def to_summary(profile: StudentProfile) -> ProfileSummary:
    """把侧写压成放进 Prompt 的摘要。"""
    return ProfileSummary(
        student_id=profile.student_id,
        grade=profile.grade,
        aspiration=profile.aspiration,
        strengths=list(profile.strengths),
        stable_distortions=list(profile.stable_distortions),
        positive_cognitions=list(profile.positive_cognitions),
        attribution_style=profile.attribution_style,
        unmet_needs=list(profile.unmet_needs),
        risk_level=profile.risk_level,
        last_cause=profile.last_cause,
        emotion_trend=profile.emotion_trend,
    )


__all__ = [
    "detect_emotion_trend",
    "extract_grade",
    "extract_positive_cognitions",
    "extract_strengths",
    "extract_via_strengths",
    "infer_attribution_style",
    "smooth_intensity",
    "to_summary",
    "update_profile",
]
