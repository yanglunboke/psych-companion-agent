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

    # 5) 优势 Top3
    for strength in extract_strengths(student_text):
        if strength not in updated.strengths:
            updated.strengths.append(strength)
    updated.strengths = updated.strengths[:3]

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
        unmet_needs=list(profile.unmet_needs),
        risk_level=profile.risk_level,
        last_cause=profile.last_cause,
    )


__all__ = [
    "extract_grade",
    "extract_strengths",
    "smooth_intensity",
    "to_summary",
    "update_profile",
]
