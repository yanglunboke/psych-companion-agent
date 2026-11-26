"""口令哈希与登录令牌。

只用标准库：``hashlib.pbkdf2_hmac`` 做慢哈希，``secrets`` 出令牌，
``hmac.compare_digest`` 做恒定时间比较。**不引入 passlib / pyjwt** ——
这个项目不需要 JWT 的无状态性，反而需要"能立即吊销"，所以令牌是
服务端存一条记录、客户端只拿随机串。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

#: PBKDF2 迭代次数。2026 年的手机/笔记本上单次约 100ms，
#: 登录一次感觉不到，暴力破解代价却很高。
PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16


def new_salt() -> str:
    """生成随机盐（十六进制）。"""
    return secrets.token_hex(SALT_BYTES)


def hash_password(password: str, salt: str) -> str:
    """把明文口令揉成十六进制摘要。"""
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return digest.hex()


def verify_password(password: str, salt: str, expected: str) -> bool:
    """恒定时间比对口令；任何异常都当作不匹配。"""
    if not password or not salt or not expected:
        return False
    actual = hash_password(password, salt)
    return hmac.compare_digest(actual, expected)


def new_token() -> str:
    """生成登录令牌（URL 安全的随机串）。"""
    return secrets.token_urlsafe(32)


def token_fingerprint(token: str) -> str:
    """令牌入库前先做摘要。

    库里存的是指纹而不是令牌本身：万一数据库泄漏，拿到的只是摘要，
    没法直接当令牌去调接口。
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def mask_student_id(student_id: str) -> str:
    """日志里用的脱敏学号。"""
    if len(student_id) <= 3:
        return student_id[0] + "**"
    return f"{student_id[:2]}***{student_id[-1]}"


__all__ = [
    "PBKDF2_ITERATIONS",
    "hash_password",
    "mask_student_id",
    "new_salt",
    "new_token",
    "token_fingerprint",
    "verify_password",
]
