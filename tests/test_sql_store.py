"""SQL 长期存储测试（SQLite，生产同一套代码走 Postgres）。"""

from __future__ import annotations

import pytest

from app.memory.sql_impl import SqlProfileStore
from app.schemas import EmotionTrajectoryItem, StudentProfile, ToolCallLog


@pytest.fixture()
async def store(tmp_path):
    dsn = f"sqlite+aiosqlite:///{(tmp_path / 'psych.db').as_posix()}"
    sql_store = await SqlProfileStore.create(dsn)
    yield sql_store
    await sql_store.dispose()


@pytest.mark.asyncio
async def test_profile_roundtrip(store) -> None:
    profile = StudentProfile(
        student_id="stu1",
        grade="高二",
        strengths=["画画"],
        stable_distortions=["以偏概全"],
        unmet_needs=["被理解"],
        risk_level="medium",
        last_cause="月考失利",
        intensity_ema=3.4,
        evidence_counts={"以偏概全": 3},
    )
    await store.save_profile(profile)
    loaded = await store.get_profile("stu1")
    assert loaded.grade == "高二"
    assert loaded.strengths == ["画画"]
    assert loaded.stable_distortions == ["以偏概全"]
    assert loaded.evidence_counts == {"以偏概全": 3}
    assert loaded.intensity_ema == pytest.approx(3.4)


@pytest.mark.asyncio
async def test_profile_update_is_upsert(store) -> None:
    await store.save_profile(StudentProfile(student_id="stu1", grade="高一"))
    await store.save_profile(StudentProfile(student_id="stu1", grade="高二"))
    loaded = await store.get_profile("stu1")
    assert loaded.grade == "高二"


@pytest.mark.asyncio
async def test_missing_profile_returns_empty(store) -> None:
    loaded = await store.get_profile("nobody")
    assert loaded.student_id == "nobody"
    assert loaded.strengths == []


@pytest.mark.asyncio
async def test_emotion_trajectory_appends_in_order(store) -> None:
    for i in range(3):
        await store.append_emotion(
            EmotionTrajectoryItem(
                student_id="stu1",
                ts=float(i),
                emotion_type="沮丧",
                intensity=3.0,
                raw_intensity=4,
                cause_domain="学业",
                trace_id=f"t{i}",
            )
        )
    trajectory = await store.get_trajectory("stu1")
    assert len(trajectory) == 3
    assert [t.ts for t in trajectory] == [0.0, 1.0, 2.0]


@pytest.mark.asyncio
async def test_utterances_are_stored(store) -> None:
    await store.append_utterance("stu1", "student", "我很难过", "t1")
    await store.append_utterance("stu1", "assistant", "我在听你说。", "t1")
    rows = await store.get_utterances("stu1")
    assert len(rows) == 2
    assert rows[0]["role"] == "student"


@pytest.mark.asyncio
async def test_tool_call_log_is_traceable(store) -> None:
    for seq, name in enumerate(["safety_lexicon", "emotion_recognition", "positive_guidance"]):
        await store.append_tool_call(
            ToolCallLog(
                trace_id="trace-1",
                seq=seq,
                tool_name=name,
                key_args={"n": seq},
                success=True,
                duration_ms=1.5,
            )
        )
    logs = await store.get_tool_calls("trace-1")
    assert [log.tool_name for log in logs] == [
        "safety_lexicon",
        "emotion_recognition",
        "positive_guidance",
    ]
    assert logs[0].seq == 0


@pytest.mark.asyncio
async def test_delete_student_clears_everything(store) -> None:
    await store.save_profile(StudentProfile(student_id="stu1", grade="高一"))
    await store.append_utterance("stu1", "student", "你好")
    await store.append_emotion(
        EmotionTrajectoryItem(
            student_id="stu1", ts=1.0, emotion_type="平静", intensity=1.0
        )
    )
    await store.delete_student("stu1")
    assert (await store.get_profile("stu1")).grade is None
    assert await store.get_utterances("stu1") == []
    assert await store.get_trajectory("stu1") == []


@pytest.mark.asyncio
async def test_sql_backend_selection_and_fallback(tmp_path) -> None:
    from app.config import Settings
    from app.memory.factory import build_profile_store

    dsn = f"sqlite+aiosqlite:///{(tmp_path / 'ok.db').as_posix()}"
    ok_store, status = await build_profile_store(Settings(db_backend="sql", sql_dsn=dsn))
    assert status["backend"] == "sql"
    assert isinstance(ok_store, SqlProfileStore)

    bad_store, bad_status = await build_profile_store(
        Settings(db_backend="sql", sql_dsn="postgresql+psycopg://nobody@127.0.0.1:1/x")
    )
    assert bad_status["backend"] == "memory"
    assert "降级" in bad_status["detail"]
