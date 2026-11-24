"""SQL 长期存储实现（Postgres 生产 / SQLite 本地与测试）。

文档口径：「Postgres：情绪轨迹（类型、强度、诱因、时间）、心理侧写（年级、
优势、稳定偏差、需求、风险等级）、学生对话原话」，另外还落一张
``tool_call_log`` 表，让「工具调用可追溯」跨进程成立。

用 SQLAlchemy 2.x 异步 ORM：同一份代码，生产指向
``postgresql+psycopg://``，本地/测试指向 ``sqlite+aiosqlite://``，
**只换 DSN 不换代码**。所有写入都用绑定参数，不做字符串拼接。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import JSON, Float, Integer, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.schemas import (
    EmotionTrajectoryItem,
    StudentProfile,
    ToolCallLog,
    Turn,
)

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class EmotionTrajectoryRow(Base):
    """情绪轨迹（按时间排的流水）。"""

    __tablename__ = "emotion_trajectory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    emotion_type: Mapped[str] = mapped_column(String(32))
    intensity: Mapped[float] = mapped_column(Float)
    raw_intensity: Mapped[int] = mapped_column(Integer, default=0)
    cause_domain: Mapped[str] = mapped_column(String(32), default="")
    cause_detail: Mapped[str] = mapped_column(Text, default="")
    trace_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)


class StudentProfileRow(Base):
    """心理侧写（跨轮稳定结论）。"""

    __tablename__ = "student_profile"

    student_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    grade: Mapped[str | None] = mapped_column(String(32), default=None)
    aspiration: Mapped[str | None] = mapped_column(String(128), default=None)
    strengths: Mapped[list] = mapped_column(JSON, default=list)
    via_strengths: Mapped[dict] = mapped_column(JSON, default=dict)
    stable_distortions: Mapped[list] = mapped_column(JSON, default=list)
    positive_cognitions: Mapped[list] = mapped_column(JSON, default=list)
    attribution_style: Mapped[str] = mapped_column(String(32), default="")
    unmet_needs: Mapped[list] = mapped_column(JSON, default=list)
    risk_level: Mapped[str] = mapped_column(String(16), default="low")
    last_cause: Mapped[str | None] = mapped_column(String(255), default=None)
    last_emotion: Mapped[str | None] = mapped_column(String(32), default=None)
    intensity_ema: Mapped[float] = mapped_column(Float, default=0.0)
    emotion_trend: Mapped[str] = mapped_column(String(16), default="平稳")
    evidence_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    positive_cognition_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[float] = mapped_column(Float, default=0.0)


class StudentUtteranceRow(Base):
    """学生对话原话。"""

    __tablename__ = "student_utterance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    ts: Mapped[float] = mapped_column(Float, default=0.0)


class ToolCallLogRow(Base):
    """工具调用日志（trace 可追溯）。"""

    __tablename__ = "tool_call_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    key_args: Mapped[dict] = mapped_column(JSON, default=dict)
    success: Mapped[bool] = mapped_column(default=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str] = mapped_column(Text, default="")
    ts: Mapped[float] = mapped_column(Float, default=0.0)


class SqlProfileStore:
    """基于 SQLAlchemy 异步引擎的长期存储。"""

    def __init__(self, dsn: str, *, echo: bool = False) -> None:
        self.dsn = dsn
        self.engine: AsyncEngine = create_async_engine(dsn, echo=echo, future=True)
        self.sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    # ------------------------------------------------------------------
    @classmethod
    async def create(cls, dsn: str, *, echo: bool = False) -> "SqlProfileStore":
        """创建实例、建表并做一次连通性探测。"""
        store = cls(dsn, echo=echo)
        await store.init_schema()
        async with store.sessionmaker() as session:
            await session.execute(select(1))
        return store

    async def init_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # SQLite 的 create_all 不会给已存在的表加新列；这里对心理侧写表
            # 做一次轻量迁移，把后加的列补上（Postgres 生产库建议用 Alembic，
            # 此处只为本地/测试的 SQLite 提供兜底，幂等，重复执行无副作用）。
            await conn.run_sync(self._migrate_sqlite_columns)

    @staticmethod
    def _migrate_sqlite_columns(sync_conn) -> None:
        """为 SQLite 的 student_profile 表补齐新增列（幂等）。"""
        if sync_conn.dialect.name != "sqlite":
            return
        additions = [
            ("via_strengths", "JSON", "'{}'"),
            ("positive_cognitions", "JSON", "'[]'"),
            ("attribution_style", "VARCHAR(32)", "''"),
            ("emotion_trend", "VARCHAR(16)", "'平稳'"),
            ("positive_cognition_counts", "JSON", "'{}'"),
        ]
        existing = {
            row[1]
            for row in sync_conn.exec_driver_sql("PRAGMA table_info(student_profile)")
        }
        for col, ctype, default in additions:
            if col in existing:
                continue
            sync_conn.exec_driver_sql(
                f'ALTER TABLE student_profile ADD COLUMN {col} {ctype} DEFAULT {default}'
            )

    async def dispose(self) -> None:
        await self.engine.dispose()

    # ------------------------------------------------------------------
    async def get_profile(self, student_id: str) -> StudentProfile:
        async with self.sessionmaker() as session:
            row = await session.get(StudentProfileRow, student_id)
            if row is None:
                return StudentProfile(student_id=student_id)
            return StudentProfile(
                student_id=row.student_id,
                grade=row.grade,
                aspiration=row.aspiration,
                strengths=list(row.strengths or []),
                via_strengths=dict(row.via_strengths or {}),
                stable_distortions=list(row.stable_distortions or []),
                positive_cognitions=list(row.positive_cognitions or []),
                attribution_style=row.attribution_style or "",
                unmet_needs=list(row.unmet_needs or []),
                risk_level=row.risk_level,
                last_cause=row.last_cause,
                last_emotion=row.last_emotion,
                intensity_ema=row.intensity_ema,
                emotion_trend=row.emotion_trend or "平稳",
                evidence_counts=dict(row.evidence_counts or {}),
                positive_cognition_counts=dict(row.positive_cognition_counts or {}),
                updated_at=row.updated_at,
            )

    async def save_profile(self, profile: StudentProfile) -> None:
        async with self.sessionmaker() as session:
            row = await session.get(StudentProfileRow, profile.student_id)
            if row is None:
                row = StudentProfileRow(student_id=profile.student_id)
                session.add(row)
            row.grade = profile.grade
            row.aspiration = profile.aspiration
            row.strengths = list(profile.strengths)
            row.via_strengths = dict(profile.via_strengths)
            row.stable_distortions = list(profile.stable_distortions)
            row.positive_cognitions = list(profile.positive_cognitions)
            row.attribution_style = profile.attribution_style
            row.unmet_needs = list(profile.unmet_needs)
            row.risk_level = profile.risk_level
            row.last_cause = profile.last_cause
            row.last_emotion = profile.last_emotion
            row.intensity_ema = profile.intensity_ema
            row.emotion_trend = profile.emotion_trend
            row.evidence_counts = dict(profile.evidence_counts)
            row.positive_cognition_counts = dict(profile.positive_cognition_counts)
            row.updated_at = profile.updated_at
            await session.commit()

    async def append_emotion(self, item: EmotionTrajectoryItem) -> None:
        async with self.sessionmaker() as session:
            session.add(
                EmotionTrajectoryRow(
                    student_id=item.student_id,
                    ts=item.ts,
                    emotion_type=item.emotion_type,
                    intensity=item.intensity,
                    raw_intensity=item.raw_intensity,
                    cause_domain=item.cause_domain,
                    cause_detail=item.cause_detail,
                    trace_id=item.trace_id,
                )
            )
            await session.commit()

    async def get_trajectory(
        self, student_id: str, limit: int = 50
    ) -> list[EmotionTrajectoryItem]:
        stmt = (
            select(EmotionTrajectoryRow)
            .where(EmotionTrajectoryRow.student_id == student_id)
            .order_by(EmotionTrajectoryRow.ts.desc())
            .limit(limit)
        )
        async with self.sessionmaker() as session:
            rows = list((await session.execute(stmt)).scalars())
        rows.reverse()
        return [
            EmotionTrajectoryItem(
                student_id=r.student_id,
                ts=r.ts,
                emotion_type=r.emotion_type,
                intensity=r.intensity,
                raw_intensity=r.raw_intensity,
                cause_domain=r.cause_domain,
                cause_detail=r.cause_detail,
                trace_id=r.trace_id,
            )
            for r in rows
        ]

    async def append_utterance(
        self, student_id: str, role: str, content: str, trace_id: str | None = None
    ) -> None:
        import time as _time

        async with self.sessionmaker() as session:
            session.add(
                StudentUtteranceRow(
                    student_id=student_id,
                    role=role,
                    content=content,
                    trace_id=trace_id,
                    ts=_time.time(),
                )
            )
            await session.commit()

    async def get_utterances(self, student_id: str, limit: int = 50) -> list[dict]:
        stmt = (
            select(StudentUtteranceRow)
            .where(StudentUtteranceRow.student_id == student_id)
            .order_by(StudentUtteranceRow.id.desc())
            .limit(limit)
        )
        async with self.sessionmaker() as session:
            rows = list((await session.execute(stmt)).scalars())
        rows.reverse()
        return [
            {"role": r.role, "content": r.content, "trace_id": r.trace_id, "ts": r.ts}
            for r in rows
        ]

    # ------------------------------------------------------------------
    async def append_tool_call(self, log: ToolCallLog) -> None:
        async with self.sessionmaker() as session:
            session.add(
                ToolCallLogRow(
                    trace_id=log.trace_id,
                    seq=log.seq,
                    tool_name=log.tool_name,
                    key_args=json.loads(json.dumps(log.key_args, default=str)),
                    success=log.success,
                    duration_ms=log.duration_ms,
                    detail=log.detail[:2000],
                    ts=log.ts,
                )
            )
            await session.commit()

    async def get_tool_calls(self, trace_id: str) -> list[ToolCallLog]:
        stmt = (
            select(ToolCallLogRow)
            .where(ToolCallLogRow.trace_id == trace_id)
            .order_by(ToolCallLogRow.seq.asc())
        )
        async with self.sessionmaker() as session:
            rows = list((await session.execute(stmt)).scalars())
        return [
            ToolCallLog(
                trace_id=r.trace_id,
                seq=r.seq,
                tool_name=r.tool_name,
                key_args=r.key_args or {},
                success=r.success,
                duration_ms=r.duration_ms,
                detail=r.detail,
                ts=r.ts,
            )
            for r in rows
        ]

    async def delete_student(self, student_id: str) -> None:
        """删除某个学生的全部数据（合规要求：可遗忘）。"""
        async with self.sessionmaker() as session:
            await session.execute(
                delete(EmotionTrajectoryRow).where(
                    EmotionTrajectoryRow.student_id == student_id
                )
            )
            await session.execute(
                delete(StudentUtteranceRow).where(
                    StudentUtteranceRow.student_id == student_id
                )
            )
            await session.execute(
                delete(StudentProfileRow).where(
                    StudentProfileRow.student_id == student_id
                )
            )
            await session.commit()


__all__ = [
    "Base",
    "EmotionTrajectoryRow",
    "SqlProfileStore",
    "StudentProfileRow",
    "StudentUtteranceRow",
    "ToolCallLogRow",
]
