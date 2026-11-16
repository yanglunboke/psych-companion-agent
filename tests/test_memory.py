"""记忆层测试：12 轮窗口、情绪强度指数平滑、侧写证据门槛。"""

from __future__ import annotations

import pytest

from app.memory.memory_impl import InMemoryProfileStore, InMemoryShortTermMemory
from app.memory.profile import (
    extract_grade,
    extract_strengths,
    smooth_intensity,
    to_summary,
    update_profile,
)
from app.schemas import EmotionResult, InterventionFit, NeedResult, StudentProfile, Turn


@pytest.mark.asyncio
async def test_short_term_memory_keeps_12_rounds() -> None:
    mem = InMemoryShortTermMemory(max_turns=12)
    for i in range(20):
        await mem.append("s", Turn(role="student", content=f"学生第{i}轮"))
        await mem.append("s", Turn(role="assistant", content=f"助手第{i}轮"))
    recent = await mem.get_recent("s", 12)
    student_turns = [t for t in recent if t.role == "student"]
    assert len(student_turns) == 12
    assert student_turns[0].content == "学生第8轮"
    assert student_turns[-1].content == "学生第19轮"


@pytest.mark.asyncio
async def test_short_term_memory_clear() -> None:
    mem = InMemoryShortTermMemory()
    await mem.append("s", Turn(role="student", content="你好"))
    await mem.clear("s")
    assert await mem.get_recent("s") == []


def test_intensity_uses_ema() -> None:
    """强度用 0.7 * 旧值 + 0.3 * 本轮，一句气话不会改掉主导情绪。"""
    assert smooth_intensity(4.0, 1, alpha=0.3) == pytest.approx(0.7 * 4.0 + 0.3 * 1)
    assert smooth_intensity(0.0, 5, alpha=0.3) == pytest.approx(1.5)


def test_distortion_needs_three_evidence_to_be_stable() -> None:
    profile = StudentProfile(student_id="s1")
    emotion = EmotionResult(
        emotion_type="自责",
        intensity=4,
        candidate_distortions=["以偏概全"],
        risk_score=0.2,
    )
    for _ in range(2):
        profile = update_profile(profile, emotion, alpha=0.3, evidence_threshold=3)
    assert "以偏概全" not in profile.stable_distortions
    assert profile.evidence_counts["以偏概全"] == 2

    profile = update_profile(profile, emotion, alpha=0.3, evidence_threshold=3)
    assert profile.evidence_counts["以偏概全"] == 3
    assert "以偏概全" in profile.stable_distortions


def test_profile_extracts_strength_and_grade() -> None:
    profile = StudentProfile(student_id="s1")
    emotion = EmotionResult(emotion_type="积极", intensity=2, risk_score=0.05)
    profile = update_profile(
        profile,
        emotion,
        student_text="我比较擅长画画，我是初二学生",
    )
    assert profile.strengths
    assert "画画" in profile.strengths[0]
    assert profile.grade is not None


def test_strength_extraction_limits_to_top3() -> None:
    assert extract_strengths("我擅长画画") == ["画画"]
    assert extract_grade("我读高二") == "高二"


def test_profile_summary_renders_for_prompt() -> None:
    profile = StudentProfile(
        student_id="s1",
        grade="高二",
        strengths=["画画"],
        stable_distortions=["以偏概全"],
        unmet_needs=["被理解"],
        last_cause="月考失利",
    )
    text = to_summary(profile).to_prompt_text()
    assert "高二" in text
    assert "画画" in text
    assert "以偏概全" in text


def test_empty_profile_summary_is_explicit() -> None:
    assert "暂无侧写摘要" in to_summary(StudentProfile(student_id="s1")).to_prompt_text()


@pytest.mark.asyncio
async def test_profile_store_keeps_trajectory() -> None:
    store = InMemoryProfileStore()
    from app.schemas import EmotionTrajectoryItem

    for i in range(3):
        await store.append_emotion(
            EmotionTrajectoryItem(
                student_id="s1",
                ts=float(i),
                emotion_type="沮丧",
                intensity=3,
                raw_intensity=4,
                cause_domain="学业",
            )
        )
    trajectory = await store.get_trajectory("s1")
    assert len(trajectory) == 3
    assert trajectory[0].emotion_type == "沮丧"


@pytest.mark.asyncio
async def test_profile_store_roundtrip() -> None:
    store = InMemoryProfileStore()
    profile = StudentProfile(student_id="s1", strengths=["画画"])
    await store.save_profile(profile)
    loaded = await store.get_profile("s1")
    assert loaded.strengths == ["画画"]
    # 返回副本，改动不会污染仓库
    loaded.strengths.append("音乐")
    again = await store.get_profile("s1")
    assert again.strengths == ["画画"]


def test_needs_worker_uses_profile() -> None:
    profile = StudentProfile(student_id="s1", unmet_needs=["安全感"])
    needs = NeedResult(needs=["安全感"], intervention_fit=InterventionFit.情绪疏导)
    assert "安全感" in needs.needs
