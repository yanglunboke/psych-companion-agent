"""Redis 短期记忆测试（用 fakeredis 跑真实 Redis 命令路径）。"""

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
    return RedisShortTermMemory(redis_client, max_turns=12, key_prefix="test:session")


@pytest.mark.asyncio
async def test_append_and_read(memory) -> None:
    await memory.append("s1", Turn(role="student", content="我这次月考考砸了"))
    await memory.append("s1", Turn(role="assistant", content="我在听你说。"))
    turns = await memory.get_recent("s1")
    assert len(turns) == 2
    assert turns[0].content == "我这次月考考砸了"
    assert turns[1].role == "assistant"


@pytest.mark.asyncio
async def test_window_is_12_rounds(memory) -> None:
    for i in range(20):
        await memory.append("s", Turn(role="student", content=f"学生第{i}轮"))
        await memory.append("s", Turn(role="assistant", content=f"助手第{i}轮"))
    turns = await memory.get_recent("s", 12)
    student_turns = [t for t in turns if t.role == "student"]
    assert len(student_turns) == 12
    assert student_turns[0].content == "学生第8轮"
    assert student_turns[-1].content == "学生第19轮"


@pytest.mark.asyncio
async def test_clear(memory) -> None:
    await memory.append("s", Turn(role="student", content="你好"))
    await memory.clear("s")
    assert await memory.get_recent("s") == []


@pytest.mark.asyncio
async def test_sessions_are_isolated(memory) -> None:
    await memory.append("a", Turn(role="student", content="A"))
    await memory.append("b", Turn(role="student", content="B"))
    assert (await memory.get_recent("a"))[0].content == "A"
    assert (await memory.get_recent("b"))[0].content == "B"


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
