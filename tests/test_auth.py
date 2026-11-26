"""登录接口测试。

重点不是"能登录"，而是三件事：

1. 同一学号第二次登录拿到的是**同一个 student_id**（记忆才有落点）；
2. 密码错必须 401，且不能泄漏"这个学号存在/不存在"；
3. 带令牌的对话以令牌里的学生为准，请求体里声明的 student_id 不作数。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.security import hash_password, new_salt, token_fingerprint, verify_password
from app.auth.store import InMemoryAuthStore
from app.config import reload_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _reset_settings():
    yield
    reload_settings()


@pytest.fixture()
def client(monkeypatch):
    """离线客户端：内存存储 + 规则模型。"""
    monkeypatch.setenv("DB_BACKEND", "memory")
    monkeypatch.setenv("AUTH_AUTO_REGISTER", "true")
    reload_settings()
    with TestClient(create_app()) as test_client:
        yield test_client


def _login(client: TestClient, student_id: str = "20260312", password: str = "hunter2x"):
    return client.post(
        "/api/auth/login",
        json={"student_id": student_id, "password": password, "name": "小宇"},
    )


# ----------------------------------------------------------------------
# 口令哈希
# ----------------------------------------------------------------------
def test_password_hash_is_salted_and_verified() -> None:
    salt_a, salt_b = new_salt(), new_salt()
    assert salt_a != salt_b
    digest = hash_password("hunter2x", salt_a)
    assert digest != "hunter2x"
    assert verify_password("hunter2x", salt_a, digest)
    assert not verify_password("hunter2y", salt_a, digest)
    # 同一口令换盐，摘要必须不同（防彩虹表）
    assert hash_password("hunter2x", salt_b) != digest


def test_token_fingerprint_is_stable_and_not_the_token() -> None:
    token = "abc123"
    assert token_fingerprint(token) == token_fingerprint(token)
    assert token_fingerprint(token) != token


# ----------------------------------------------------------------------
# 登录
# ----------------------------------------------------------------------
def test_first_login_creates_account_then_reuses_student_id(client: TestClient) -> None:
    first = _login(client)
    assert first.status_code == 200
    body = first.json()
    assert body["created"] is True
    assert body["student_id"] == "20260312"
    assert body["token"]

    second = _login(client)
    assert second.status_code == 200
    again = second.json()
    assert again["created"] is False
    assert again["student_id"] == body["student_id"]
    assert again["token"] != body["token"]  # 每次登录发新令牌，旧的不动


def test_wrong_password_is_401_and_does_not_echo_reason(client: TestClient) -> None:
    assert _login(client).status_code == 200
    bad = _login(client, password="wrongpass")
    assert bad.status_code == 401
    assert bad.json()["detail"] == "学号或密码不正确"


def test_short_password_is_rejected_by_schema(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login", json={"student_id": "20260312", "password": "123"}
    )
    assert response.status_code == 422


def test_me_requires_valid_token(client: TestClient) -> None:
    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.get("/api/auth/me", headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )

    token = _login(client).json()["token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["student_id"] == "20260312"


def test_logout_revokes_token(client: TestClient) -> None:
    token = _login(client).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/auth/logout", headers=headers).json()["revoked"] is True
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_auto_register_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "memory")
    monkeypatch.setenv("AUTH_AUTO_REGISTER", "false")
    reload_settings()
    with TestClient(create_app()) as client:
        assert _login(client).status_code == 401


# ----------------------------------------------------------------------
# 令牌决定这一轮属于谁
# ----------------------------------------------------------------------
def test_chat_uses_token_identity_over_body_student_id(client: TestClient) -> None:
    """请求体里冒用别人的学号，带令牌时无效。"""
    token = _login(client, "20260312").json()["token"]
    response = client.post(
        "/api/chat/sync",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "session_id": "s-token-1",
            "student_id": "别人的学号",
            "text": "今天有点累",
            "speaker": "student",
        },
    )
    assert response.status_code == 200
    # 归属令牌里的学生，而不是请求体声称的那个
    assert response.json()["student_id"] == "20260312"


def test_chat_without_token_still_accepts_body_student_id(client: TestClient) -> None:
    """脚本 / 联调不需要令牌，沿用请求体里的 student_id。"""
    response = client.post(
        "/api/chat/sync",
        json={"session_id": "s-script-1", "student_id": "script-user", "text": "你好"},
    )
    assert response.status_code == 200
    assert response.json()["student_id"] == "script-user"


# ----------------------------------------------------------------------
# 存储实现
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_in_memory_store_expires_tokens() -> None:
    store = InMemoryAuthStore()
    account = await store.create("20260312", "小宇", "hunter2x")
    assert account.check("hunter2x")

    token, _ = await store.issue_token("20260312", ttl_seconds=-1)
    assert await store.resolve(token) is None


@pytest.mark.asyncio
async def test_in_memory_store_revokes_token() -> None:
    store = InMemoryAuthStore()
    await store.create("20260312", "小宇", "hunter2x")
    token, _ = await store.issue_token("20260312", ttl_seconds=600)
    assert await store.resolve(token) is not None
    assert await store.revoke(token) is True
    assert await store.resolve(token) is None
