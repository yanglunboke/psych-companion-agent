"""进程内记忆实现（开发 / 测试默认后端）。"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from app.memory.store import tail_by_turns
from app.schemas import EmotionTrajectoryItem, StudentProfile, Turn


class InMemoryShortTermMemory:
    """按学生保存最近 N 轮原话（默认 12 轮），会话是其中一层标签。

    结构是 ``{student_id: {session_id: [Turn, ...]}}``：
    ``get_recent`` 不给会话时把该学生所有会话合起来按时间排序，所以换会话
    不会失忆；给了会话就只看那个会话，供前端恢复历史。

    "一轮" = 一次学生发言 + 一次陪伴助手回复。裁剪时以**学生轮次**为准，
    保证窗口里始终是完整的 12 轮对话，而不是 12 条消息。
    """

    #: 没有 ``session_id`` 的消息（旧数据 / 脚本直接写）归到这个桶
    UNTAGGED = "_untagged"

    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns
        self._students: dict[str, dict[str, list[Turn]]] = {}
        self._owners: dict[str, str] = {}

    # ------------------------------------------------------------------
    def _bucket(self, student_id: str, session_id: str | None) -> list[Turn]:
        per_student = self._students.setdefault(student_id or "", {})
        return per_student.setdefault(session_id or self.UNTAGGED, [])

    def _merged(self, student_id: str) -> list[Turn]:
        """该学生所有会话的消息按时间合并（同会话内保持原顺序）。"""
        buckets = self._students.get(student_id or "") or {}
        if len(buckets) == 1:
            return list(next(iter(buckets.values())))
        merged: list[Turn] = []
        for turns in buckets.values():
            merged.extend(turns)
        # sorted 是稳定排序：同一会话内 ts 相同的两条不会被打乱
        merged.sort(key=lambda t: t.ts)
        return merged

    # ------------------------------------------------------------------
    async def get_recent(
        self, student_id: str, limit: int = 12, *, session_id: str | None = None
    ) -> list[Turn]:
        if session_id is not None:
            turns = list(self._students.get(student_id or "", {}).get(session_id) or [])
        else:
            turns = self._merged(student_id)
        return tail_by_turns(turns, limit)

    async def append(self, student_id: str, turn: Turn) -> None:
        items = self._bucket(student_id, turn.session_id)
        items.append(turn)
        if turn.session_id:
            self._owners[turn.session_id] = student_id or ""
        # 单会话窗口不超 max_turns 轮，避免一个话痨会话把内存撑爆
        kept = tail_by_turns(items, self.max_turns)
        if len(kept) != len(items):
            items[:] = kept

    async def clear(self, student_id: str, *, session_id: str | None = None) -> None:
        if session_id is None:
            self._students.pop(student_id or "", None)
            return
        per_student = self._students.get(student_id or "")
        if per_student:
            per_student.pop(session_id, None)
        if self._owners.get(session_id) == (student_id or ""):
            self._owners.pop(session_id, None)

    # ------------------------------------------------------------------
    # 调试用：让上层不必知道具体后端就能展示窗口内容
    async def describe(self, student_id: str, *, session_id: str | None = None) -> dict:
        """当前窗口在这个后端里的存储概况。"""
        if session_id is not None:
            items = list(self._students.get(student_id or "", {}).get(session_id) or [])
            key = f"memory:student:{student_id}:session:{session_id}"
        else:
            items = self._merged(student_id)
            key = f"memory:student:{student_id}"
        return {
            "backend": "memory",
            "key": key,
            "messages": len(items),
            "turns": sum(1 for t in items if t.role == "student"),
            "sessions": len(self._students.get(student_id or "") or {}),
            "ttl_seconds": None,
        }

    async def list_sessions(self) -> list[dict]:
        """当前进程内已缓存的 (学生, 会话) 组合。"""
        rows: list[dict] = []
        for student_id, buckets in self._students.items():
            for session_id, turns in buckets.items():
                rows.append(
                    {
                        "student_id": student_id,
                        "session_id": None if session_id == self.UNTAGGED else session_id,
                        "messages": len(turns),
                        "turns": sum(1 for t in turns if t.role == "student"),
                        "last_ts": max((t.ts for t in turns), default=0.0),
                    }
                )
        rows.sort(key=lambda r: r["last_ts"], reverse=True)
        return rows

    async def owner_of(self, session_id: str) -> str | None:
        return self._owners.get(session_id)


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
