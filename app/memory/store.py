"""记忆存储接口。

* **短期记忆**（Redis）：最近 12 轮原话，检索规则固定，是与学生对话的热缓存。
* **长期存储**（Postgres）：情绪轨迹、心理侧写、学生对话原话。

阶段二提供进程内实现（开发/测试可直接跑）；阶段三接入真实 Redis / Postgres。
接口方法签名保持一致，切换后端不影响上层。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas import EmotionTrajectoryItem, StudentProfile, Turn


def tail_by_turns(turns: list[Turn], limit: int) -> list[Turn]:
    """按「学生轮次」截取尾部，两个后端共用同一套口径。

    一轮 = 一次学生发言 + 它的回复。窗口装的是 12 轮而不是 12 条，
    所以截取点必须落在学生发言上，否则会出现「有回答没提问」的半截对话。
    """
    if limit <= 0 or not turns:
        return []
    student_positions = [i for i, t in enumerate(turns) if t.role == "student"]
    if not student_positions:
        return list(turns[-limit:])
    start = student_positions[max(0, len(student_positions) - limit)]
    return list(turns[start:])


@runtime_checkable
class ShortTermMemory(Protocol):
    """短期记忆：**按学生**存最近 N 轮原话，会话只是其中一层标签。

    为什么按学生而不是按会话：同一个学生点「新对话」只是界面上清屏，
    人没变。若窗口挂在会话上，换个会话就什么都不记得了 —— 那不像陪伴，
    像每次都在跟陌生人说话。

    ``get_recent`` 不给 ``session_id`` 时返回该学生**跨会话**的最近 N 轮；
    给了就只返回那个会话的原话（前端恢复历史用）。
    """

    async def get_recent(
        self, student_id: str, limit: int = 12, *, session_id: str | None = None
    ) -> list[Turn]: ...

    async def append(self, student_id: str, turn: Turn) -> None: ...

    async def clear(self, student_id: str, *, session_id: str | None = None) -> None: ...

    # 以下三个方法只服务于调试接口（``/api/debug/memory``）。调用方必须先
    # ``hasattr`` 探测，缺失时如实说明"该后端不支持"，而不是假装成功。
    async def describe(
        self, student_id: str, *, session_id: str | None = None
    ) -> dict: ...

    async def list_sessions(self) -> list[dict]: ...

    async def owner_of(self, session_id: str) -> str | None: ...


@runtime_checkable
class ProfileStore(Protocol):
    """长期存储：情绪轨迹 + 心理侧写 + 对话原话。"""

    async def get_profile(self, student_id: str) -> StudentProfile: ...

    async def save_profile(self, profile: StudentProfile) -> None: ...

    async def append_emotion(self, item: EmotionTrajectoryItem) -> None: ...

    async def get_trajectory(self, student_id: str, limit: int = 50) -> list[EmotionTrajectoryItem]: ...

    async def append_utterance(self, student_id: str, role: str, content: str, trace_id: str | None = None) -> None: ...


__all__ = ["ProfileStore", "ShortTermMemory", "tail_by_turns"]
