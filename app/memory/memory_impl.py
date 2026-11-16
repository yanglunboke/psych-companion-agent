"""进程内记忆实现（开发 / 测试默认后端）。"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from app.schemas import EmotionTrajectoryItem, StudentProfile, Turn


class InMemoryShortTermMemory:
    """最近 N 轮原话（默认 12 轮）。

    "一轮" = 一次学生发言 + 一次陪伴助手回复。裁剪时以**学生轮次**为准，
    保证窗口里始终是完整的 12 轮对话，而不是 12 条消息。
    """

    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, list[Turn]] = {}

    async def get_recent(self, session_id: str, limit: int = 12) -> list[Turn]:
        items = self._sessions.get(session_id) or []
        if not items:
            return []
        student_positions = [i for i, t in enumerate(items) if t.role == "student"]
        if not student_positions:
            return items[-limit:]
        start = student_positions[max(0, len(student_positions) - limit)]
        return list(items[start:])

    async def append(self, session_id: str, turn: Turn) -> None:
        items = self._sessions.setdefault(session_id, [])
        items.append(turn)
        student_positions = [i for i, t in enumerate(items) if t.role == "student"]
        if len(student_positions) > self.max_turns:
            cut = student_positions[len(student_positions) - self.max_turns]
            del items[:cut]

    async def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class InMemoryProfileStore:
    """情绪轨迹 + 心理侧写 + 对话原话。"""

    def __init__(self, max_trajectory: int = 500) -> None:
        self._profiles: dict[str, StudentProfile] = {}
        self._trajectories: OrderedDict[str, list[EmotionTrajectoryItem]] = OrderedDict()
        self._utterances: dict[str, list[tuple[str, str, str | None]]] = {}
        self._max_trajectory = max_trajectory
        self._lock = asyncio.Lock()

    async def get_profile(self, student_id: str) -> StudentProfile:
        async with self._lock:
            return self._profiles.get(student_id, StudentProfile(student_id=student_id)).model_copy(
                deep=True
            )

    async def save_profile(self, profile: StudentProfile) -> None:
        async with self._lock:
            self._profiles[profile.student_id] = profile.model_copy(deep=True)

    async def append_emotion(self, item: EmotionTrajectoryItem) -> None:
        async with self._lock:
            bucket = self._trajectories.setdefault(item.student_id, [])
            bucket.append(item)
            if len(bucket) > self._max_trajectory:
                del bucket[: len(bucket) - self._max_trajectory]

    async def get_trajectory(
        self, student_id: str, limit: int = 50
    ) -> list[EmotionTrajectoryItem]:
        async with self._lock:
            return list(self._trajectories.get(student_id, []))[-limit:]

    async def append_utterance(
        self, student_id: str, role: str, content: str, trace_id: str | None = None
    ) -> None:
        async with self._lock:
            self._utterances.setdefault(student_id, []).append((role, content, trace_id))


__all__ = ["InMemoryProfileStore", "InMemoryShortTermMemory"]
