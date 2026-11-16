"""记忆存储接口。

* **短期记忆**（Redis）：最近 12 轮原话，检索规则固定，是与学生对话的热缓存。
* **长期存储**（Postgres）：情绪轨迹、心理侧写、学生对话原话。

阶段二提供进程内实现（开发/测试可直接跑）；阶段三接入真实 Redis / Postgres。
接口方法签名保持一致，切换后端不影响上层。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas import EmotionTrajectoryItem, StudentProfile, Turn


@runtime_checkable
class ShortTermMemory(Protocol):
    """短期记忆：按 session 存最近 N 轮原话。"""

    async def get_recent(self, session_id: str, limit: int = 12) -> list[Turn]: ...

    async def append(self, session_id: str, turn: Turn) -> None: ...

    async def clear(self, session_id: str) -> None: ...


@runtime_checkable
class ProfileStore(Protocol):
    """长期存储：情绪轨迹 + 心理侧写 + 对话原话。"""

    async def get_profile(self, student_id: str) -> StudentProfile: ...

    async def save_profile(self, profile: StudentProfile) -> None: ...

    async def append_emotion(self, item: EmotionTrajectoryItem) -> None: ...

    async def get_trajectory(self, student_id: str, limit: int = 50) -> list[EmotionTrajectoryItem]: ...

    async def append_utterance(self, student_id: str, role: str, content: str, trace_id: str | None = None) -> None: ...


__all__ = ["ProfileStore", "ShortTermMemory"]
