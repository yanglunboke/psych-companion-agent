"""账号与令牌存储。

两个实现，接口一致：

* :class:`InMemoryAuthStore` —— 进程内（测试 / ``DB_BACKEND=memory``）；
* :class:`SqlAuthStore`      —— SQLAlchemy 异步（本地 SQLite、生产 Postgres）。

密码只存 PBKDF2 摘要 + 盐，令牌只存 SHA-256 指纹，两者都不落明文。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy import JSON, Float, String, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.auth.security import (
    hash_password,
    new_salt,
    new_token,
    token_fingerprint,
    verify_password,
)


@dataclass(slots=True)
class Account:
    """一个学生账号。"""

    student_id: str
    name: str
    salt: str
    password_hash: str
    created_at: float

    def check(self, password: str) -> bool:
        return verify_password(password, self.salt, self.password_hash)


class AuthError(Exception):
    """认证失败（口令错误 / 令牌失效）。"""


@runtime_checkable
class AuthStore(Protocol):
    """账号存储接口。"""

    async def get(self, student_id: str) -> Account | None: ...

    async def create(self, student_id: str, name: str, password: str) -> Account: ...

    async def issue_token(self, student_id: str, ttl_seconds: int) -> tuple[str, float]: ...

    async def resolve(self, token: str) -> Account | None: ...

    async def revoke(self, token: str) -> bool: ...

    async def close(self) -> None: ...


class InMemoryAuthStore:
    """进程内实现（重启即清空，只适合测试与演示）。"""

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        #: token 指纹 -> (student_id, 过期时间)
        self._tokens: dict[str, tuple[str, float]] = {}

    async def get(self, student_id: str) -> Account | None:
        return self._accounts.get(student_id)

    async def create(self, student_id: str, name: str, password: str) -> Account:
        salt = new_salt()
        account = Account(
            student_id=student_id,
            name=name,
            salt=salt,
            password_hash=hash_password(password, salt),
            created_at=time.time(),
        )
        self._accounts[student_id] = account
        return account

    async def issue_token(self, student_id: str, ttl_seconds: int) -> tuple[str, float]:
        token = new_token()
        expires_at = time.time() + ttl_seconds
        self._tokens[token_fingerprint(token)] = (student_id, expires_at)
        return token, expires_at

    async def resolve(self, token: str) -> Account | None:
        record = self._tokens.get(token_fingerprint(token))
        if record is None:
            return None
        student_id, expires_at = record
        if expires_at < time.time():
            self._tokens.pop(token_fingerprint(token), None)
            return None
        return self._accounts.get(student_id)

    async def revoke(self, token: str) -> bool:
        return self._tokens.pop(token_fingerprint(token), None) is not None

    async def close(self) -> None:  # noqa: D401 - 接口对齐
        self._accounts.clear()
        self._tokens.clear()


# ----------------------------------------------------------------------
# SQL 实现
# ----------------------------------------------------------------------
class AuthBase(DeclarativeBase):
    """认证表独立 Base：不污染记忆库的表结构。"""


class AccountRow(AuthBase):
    """学生账号。

    后续要加年级、班级、班主任等字段时：
    * 加一列 ``Mapped[...] = mapped_column(...)``；
    * 在 :meth:`SqlAuthStore._migrate_sqlite_columns` 的 ``ADDED_COLUMNS`` 里补一条。

    ``extra`` 是给"还没想清楚要不要单独成列"的信息准备的 JSON 扩展位，
    加它不必改表结构。
    """

    __tablename__ = "student_account"

    student_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    salt: Mapped[str] = mapped_column(String(64))
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[float] = mapped_column(Float, default=0.0)
    last_login_at: Mapped[float] = mapped_column(Float, default=0.0)
    #: 预留扩展位（年级 / 班级 / 班主任 / 校方工号……）
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class AuthTokenRow(AuthBase):
    """登录令牌（只存指纹）。"""

    __tablename__ = "auth_token"

    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(String(64), index=True)
    issued_at: Mapped[float] = mapped_column(Float, default=0.0)
    expires_at: Mapped[float] = mapped_column(Float, index=True)


class SqlAuthStore:
    """SQLAlchemy 异步实现。"""

    #: 后加的列（SQLite 的 create_all 不会给已存在的表补列）。
    #: 格式：表名 -> [(列名, 类型, 默认值)]
    ADDED_COLUMNS: dict[str, list[tuple[str, str, str]]] = {
        "student_account": [("extra", "JSON", "'{}'")],
    }

    def __init__(self, dsn: str, *, echo: bool = False) -> None:
        self.dsn = dsn
        self.engine: AsyncEngine = create_async_engine(dsn, echo=echo, future=True)
        self.sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    @classmethod
    async def connect(cls, dsn: str, *, echo: bool = False) -> "SqlAuthStore":
        """建立连接并建表。

        注意：这里不能叫 ``create`` —— 类里已有一个 ``create``（建账号），
        同名会互相覆盖，工厂方法会退化成实例方法。
        """
        store = cls(dsn, echo=echo)
        async with store.engine.begin() as conn:
            await conn.run_sync(AuthBase.metadata.create_all)
            await conn.run_sync(store._migrate_sqlite_columns)
        return store

    @classmethod
    def _migrate_sqlite_columns(cls, sync_conn) -> None:
        """给已有的 SQLite 表补新列（幂等）。Postgres 生产库请用 Alembic。"""
        if sync_conn.dialect.name != "sqlite":
            return
        for table, columns in cls.ADDED_COLUMNS.items():
            existing = {
                row[1] for row in sync_conn.exec_driver_sql(f"PRAGMA table_info({table})")
            }
            for name, ctype, default in columns:
                if name in existing:
                    continue
                sync_conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {name} {ctype} DEFAULT {default}"
                )

    @staticmethod
    def _to_account(row: AccountRow) -> Account:
        return Account(
            student_id=row.student_id,
            name=row.name or "",
            salt=row.salt,
            password_hash=row.password_hash,
            created_at=row.created_at,
        )

    async def get(self, student_id: str) -> Account | None:
        async with self.sessionmaker() as session:
            row = await session.get(AccountRow, student_id)
            return self._to_account(row) if row else None

    async def create(self, student_id: str, name: str, password: str) -> Account:
        salt = new_salt()
        now = time.time()
        row = AccountRow(
            student_id=student_id,
            name=name,
            salt=salt,
            password_hash=hash_password(password, salt),
            created_at=now,
            last_login_at=now,
        )
        try:
            async with self.sessionmaker() as session:
                session.add(row)
                await session.commit()
        except IntegrityError:
            # 并发注册：以先落库的那条为准，调用方随后重新校验口令
            existing = await self.get(student_id)
            if existing is None:
                raise
            return existing
        return self._to_account(row)

    async def touch_login(self, student_id: str) -> None:
        async with self.sessionmaker() as session:
            row = await session.get(AccountRow, student_id)
            if row is not None:
                row.last_login_at = time.time()
                await session.commit()

    async def issue_token(self, student_id: str, ttl_seconds: int) -> tuple[str, float]:
        token = new_token()
        now = time.time()
        expires_at = now + ttl_seconds
        async with self.sessionmaker() as session:
            session.add(
                AuthTokenRow(
                    fingerprint=token_fingerprint(token),
                    student_id=student_id,
                    issued_at=now,
                    expires_at=expires_at,
                )
            )
            await session.commit()
        return token, expires_at

    async def resolve(self, token: str) -> Account | None:
        if not token:
            return None
        fingerprint = token_fingerprint(token)
        async with self.sessionmaker() as session:
            row = await session.get(AuthTokenRow, fingerprint)
            if row is None:
                return None
            if row.expires_at < time.time():
                await session.execute(
                    delete(AuthTokenRow).where(AuthTokenRow.fingerprint == fingerprint)
                )
                await session.commit()
                return None
            account = await session.get(AccountRow, row.student_id)
            return self._to_account(account) if account else None

    async def revoke(self, token: str) -> bool:
        fingerprint = token_fingerprint(token)
        async with self.sessionmaker() as session:
            row = await session.get(AuthTokenRow, fingerprint)
            if row is None:
                return False
            await session.execute(
                delete(AuthTokenRow).where(AuthTokenRow.fingerprint == fingerprint)
            )
            await session.commit()
            return True

    async def purge_expired(self) -> int:
        """清掉过期令牌（登录时顺手做，避免表无限增长）。"""
        async with self.sessionmaker() as session:
            result = await session.execute(
                delete(AuthTokenRow).where(AuthTokenRow.expires_at < time.time())
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def close(self) -> None:
        await self.engine.dispose()


__all__ = [
    "Account",
    "AccountRow",
    "AuthBase",
    "AuthError",
    "AuthStore",
    "AuthTokenRow",
    "InMemoryAuthStore",
    "SqlAuthStore",
]
