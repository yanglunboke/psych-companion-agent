"""账号与登录。

学生身份必须是**稳定**的：长期画像按 ``student_id`` 存、短期上下文按
``session_id`` 存，任何一个每次刷新都变，表现得就像"它不记得我了"。
所以身份由登录态给出，不再由前端随机生成。
"""

from app.auth.store import Account, AuthStore, InMemoryAuthStore, SqlAuthStore

__all__ = ["Account", "AuthStore", "InMemoryAuthStore", "SqlAuthStore"]
