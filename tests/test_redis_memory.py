"""Redis 短期记忆测试（用 fakeredis 跑真实 Redis 命令路径）。

重点覆盖"按学生组织"这件事：跨会话合并、按会话过滤、学生之间隔离、
会话归属可反查。
"""

from __future__ import annotations

import pytest

from app.memory.redis_impl import RedisShortTermMemory
from app.schemas import Turn


@pytest.fixture()
def redis_client():
    import fakeredis.aioredis as fake

    return fake.FakeRedis(decode_responses=True)


@pytest.fixture()
def memory(redis_client):
    return RedisShortTermMemory(redis_client, max_turns=12, key_prefix="test")


@pytest.mark.asyncio
async def test_append_and_read(memory) -> None:
    await memory.append(
        "u1", Turn(role="student", content="我这次月考考砸了", ts=1.0, session_id="s1")
    )
    await memory.append(
        "u1", Turn(role="assistant", content="我在听你说。", ts=1.0, session_id="s1")
    )
    turns = await memory.get_recent("u1", session_id="s1")
    assert len(turns) == 2
    assert turns[0].content == "我这次月考考砸了"
    assert turns[1].role == "assistant"


@pytest.mark.asyncio
async def test_keys_are_namespaced_by_student(memory, redis_client) -> None:
    """键必须落在学生下面，而不是平铺在 psych:session:*。"""
    await memory.append("u1", Turn(role="student", content="你好", ts=1.0, session_id="s1"))
    keys = {k async for k in redis_client.scan_iter(match="*")}
    assert "test:student:u1:session:s1" in keys
    assert "test:student:u1:sessions" in keys
    assert "test:owner:s1" in keys
    assert not any(k.startswith("test:session:") for k in keys)


@pytest.mark.asyncio
async def test_window_is_12_rounds(memory) -> None:
    for i in range(20):
        await memory.append(
            "u1", Turn(role="student", content=f"学生第{i}轮", ts=float(i), session_id="s")
        )
        await memory.append(
            "u1", Turn(role="assistant", content=f"助手第{i}轮", ts=float(i), session_id="s")
        )
    turns = await memory.get_recent("u1", 12, session_id="s")
    student_turns = [t for t in turns if t.role == "student"]
    assert len(student_turns) == 12
    assert student_turns[0].content == "学生第8轮"
    assert student_turns[-1].content == "学生第19轮"


@pytest.mark.asyncio
async def test_clear(memory) -> None:
    await memory.append("u1", Turn(role="student", content="你好", ts=1.0, session_id="s"))
    await memory.clear("u1", session_id="s")
    assert await memory.get_recent("u1", session_id="s") == []


@pytest.mark.asyncio
async def test_cross_session_merge_for_same_student(memory) -> None:
    """同一学生换会话不失忆：不带 session_id 时把多个会话合并成时间线。"""
    await memory.append("u1", Turn(role="student", content="A 里说的事", ts=1.0, session_id="A"))
    await memory.append("u1", Turn(role="assistant", content="A 的回复", ts=1.0, session_id="A"))
    await memory.append("u1", Turn(role="student", content="B 里说的事", ts=2.0, session_id="B"))

    merged = [t.content for t in await memory.get_recent("u1")]
    assert merged == ["A 里说的事", "A 的回复", "B 里说的事"]

    only_a = [t.content for t in await memory.get_recent("u1", session_id="A")]
    assert only_a == ["A 里说的事", "A 的回复"]
    assert await memory.owner_of("B") == "u1"


@pytest.mark.asyncio
async def test_sessions_are_isolated(memory) -> None:
    await memory.append("u1", Turn(role="student", content="A", ts=1.0, session_id="a"))
    await memory.append("u2", Turn(role="student", content="B", ts=2.0, session_id="b"))
    assert (await memory.get_recent("u1", session_id="a"))[0].content == "A"
    assert (await memory.get_recent("u2", session_id="b"))[0].content == "B"
    # 学生之间绝不串味
    assert await memory.get_recent("u1", session_id="b") == []
    assert await memory.get_recent("u2", session_id="a") == []


@pytest.mark.asyncio
async def test_list_sessions_reports_student_and_session(memory) -> None:
    await memory.append("u1", Turn(role="student", content="一", ts=1.0, session_id="s1"))
    await memory.append("u2", Turn(role="student", content="二", ts=2.0, session_id="s2"))
    rows = await memory.list_sessions()
    pairs = {(r["student_id"], r["session_id"]) for r in rows}
    assert pairs == {("u1", "s1"), ("u2", "s2")}
    # 最近活跃的排在前面
    assert rows[0]["session_id"] == "s2"


@pytest.mark.asyncio
async def test_redis_backend_is_selected_when_configured() -> None:
    """配置成 redis 且连不上时，降级到进程内并说明原因。"""
    from app.config import Settings
    from app.memory.factory import build_short_term_memory

    settings = Settings(
        memory_backend="redis",
        redis_url="redis://127.0.0.1:63999/0",  # 故意指向没人监听的端口
        storage_fallback=True,
    )
    backend, status = await build_short_term_memory(settings)
    assert status["backend"] == "memory"
    assert "降级" in status["detail"]


@pytest.mark.asyncio
async def test_no_fallback_raises() -> None:
    from app.config import Settings
    from app.memory.factory import build_short_term_memory

    settings = Settings(
        memory_backend="redis",
        redis_url="redis://127.0.0.1:63999/0",
        storage_fallback=False,
    )
    with pytest.raises(Exception):
        await build_short_term_memory(settings)
