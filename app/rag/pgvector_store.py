"""PGVector 持久化向量存储。

把卡片向量落进带 ``pgvector`` 扩展的 Postgres，让"向量召回"这一路不再依赖
进程内哈希向量，而是真正的语义向量 + 可持久化 / 可扩展的最近邻检索。

设计原则沿用整个记忆层的口径：

* **可降级**：连不上 Postgres、或 ``pgvector`` 扩展未启用时，上层回退到
  进程内 ``CardIndex``（内存哈希/内存向量），服务照常能跑；
* **幂等建表**：``create_all`` + 保证扩展已启用；重复调用无副作用；
* **接口最小**：只暴露 ``upsert`` / ``search`` / ``delete``，与
  :class:`app.rag.retriever.Retriever` 解耦，检索四步管线无需改动。

生产建议索引：卡片量级小（几十张）用 IVFFlat 即可；量大再换 HNSW。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: 向量维度（bge-m3 默认 1024）。切换模型时同步改这里或改为运行时探测。
DEFAULT_DIM = 1024

#: 扩展启用与建索引的 SQL 片段（Postgres 方言专用）。
_ENABLE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"
_IVFFLAT_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_card_vectors_ivfflat "
    "ON card_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 16)"
)


class PgVectorStore:
    """基于 SQLAlchemy + pgvector 的卡片向量存储。"""

    def __init__(self, dsn: str, *, dim: int = DEFAULT_DIM, echo: bool = False) -> None:
        self.dsn = dsn
        self.dim = dim
        self._echo = echo
        self._engine = None
        self._sessionmaker = None

    # ------------------------------------------------------------------
    def _build_engine(self):
        """延迟建 engine，避免无 pgvector 环境下 import 即失败。"""
        if self._engine is not None:
            return self._engine, self._sessionmaker
        from sqlalchemy import (  # noqa: PLC0415
            JSON,
            Column,
            Float,
            Integer,
            String,
        )
        from sqlalchemy.ext.asyncio import (  # noqa: PLC0415
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlalchemy.orm import DeclarativeBase  # noqa: PLC0415
        from pgvector.sqlalchemy import Vector  # noqa: PLC0415

        class Base(DeclarativeBase):
            pass

        class CardVectorRow(Base):  # noqa: WPS431
            __tablename__ = "card_vectors"

            card_id: Any = Column(String(128), primary_key=True)
            collection: Any = Column(String(32), index=True)
            embedding: Any = Column(Vector(self.dim))

        self._Base = Base
        self._Row = CardVectorRow
        self._engine = create_async_engine(self.dsn, echo=self._echo, future=True)
        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        return self._engine, self._sessionmaker

    # ------------------------------------------------------------------
    async def init_schema(self) -> None:
        """建表 + 启用扩展 + 建索引（幂等）。"""
        engine, _ = self._build_engine()
        async with engine.begin() as conn:
            # 必须先启用扩展，再建表——表的 embedding 列用了 vector 类型
            if conn.dialect.name == "postgresql":
                await conn.exec_driver_sql(_ENABLE_EXTENSION)
            await conn.run_sync(self._Base.metadata.create_all)
            if conn.dialect.name == "postgresql":
                await conn.exec_driver_sql(_IVFFLAT_INDEX)

    async def upsert_many(self, items: list[tuple[str, str, np.ndarray]]) -> None:
        """批量写入 ``(card_id, collection, embedding)``。

        用 Postgres 的 ``INSERT ... ON CONFLICT DO UPDATE`` 做 upsert，
        重复写入同一 card_id 时覆盖向量。
        """
        if not items:
            return
        _, sessionmaker = self._build_engine()
        from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: PLC0415

        rows = [
            {
                "card_id": card_id,
                "collection": collection,
                "embedding": np.asarray(vec, dtype=np.float32).tolist(),
            }
            for card_id, collection, vec in items
        ]
        async with sessionmaker() as session:
            stmt = pg_insert(self._Row).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[self._Row.card_id],
                set_={
                    "collection": stmt.excluded.collection,
                    "embedding": stmt.excluded.embedding,
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 20,
        collections: list[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """余弦相似度检索，返回 ``[(card_id, collection, similarity), ...]``。

        similarity 由 ``1 - cosine_distance`` 得出，与检索层其余代码用的
        余弦相似度口径一致。
        """
        _, sessionmaker = self._build_engine()
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select  # noqa: PLC0415

        qvec = np.asarray(query_vec, dtype=np.float32).tolist()
        async with sessionmaker() as session:
            stmt = select(
                self._Row.card_id,
                self._Row.collection,
                (1.0 - self._Row.embedding.cosine_distance(qvec)).label("sim"),
            )
            if collections:
                stmt = stmt.where(self._Row.collection.in_(collections))
            stmt = stmt.order_by(
                self._Row.embedding.cosine_distance(qvec)
            ).limit(top_k)
            result = await session.execute(stmt)
            return [
                (row.card_id, row.collection, float(row.sim)) for row in result.all()
            ]

    async def fetch_all(self) -> list[tuple[str, str, np.ndarray]]:
        """读出库里全部向量。

        供启动时装载：把已持久化的向量灌回内存索引，省掉重新编码
        （bge-m3 编码几十张卡要几十秒）。
        """
        _, sessionmaker = self._build_engine()
        from sqlalchemy import select  # noqa: PLC0415

        async with sessionmaker() as session:
            stmt = select(
                self._Row.card_id,
                self._Row.collection,
                self._Row.embedding,
            )
            result = await session.execute(stmt)
            return [
                (row.card_id, row.collection, np.asarray(row.embedding, dtype=np.float32))
                for row in result.all()
            ]

    async def delete_missing(self, keep_ids: list[str]) -> int:
        """删除不在 ``keep_ids`` 里的向量（清理卡片删除/改 id 后的残留）。

        返回删除条数。
        """
        _, sessionmaker = self._build_engine()
        from sqlalchemy import delete  # noqa: PLC0415

        async with sessionmaker() as session:
            if keep_ids:
                stmt = delete(self._Row).where(self._Row.card_id.notin_(keep_ids))
            else:
                stmt = delete(self._Row)
            result = await session.execute(stmt)
            await session.commit()
            return int(result.rowcount or 0)

    async def delete(self, card_id: str) -> None:
        _, sessionmaker = self._build_engine()
        async with sessionmaker() as session:
            row = await session.get(self._Row, card_id)
            if row is not None:
                await session.delete(row)
                await session.commit()

    async def dispose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()


__all__ = ["PgVectorStore", "DEFAULT_DIM"]
