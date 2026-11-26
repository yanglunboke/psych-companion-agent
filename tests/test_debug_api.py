"""调试接口测试：默认必须关闭，开启后才能读到窗口内容。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import reload_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _reset_settings():
    """每个用例结束后把配置缓存恢复到 conftest 的默认值。"""
    yield
    reload_settings()


def _client(monkeypatch, enabled: bool) -> TestClient:
    # 显式设置环境变量：.env 里为本地开发开了这个开关，
    # 单测必须自己决定开/关，不能跟着本地 .env 走。
    monkeypatch.setenv("DEBUG_API_ENABLED", "true" if enabled else "false")
    reload_settings()
    return TestClient(create_app())


def test_debug_api_disabled_by_default(monkeypatch) -> None:
    """未开启时按"不存在"处理：404，而不是 403。"""
    with _client(monkeypatch, enabled=False) as client:
        assert client.get("/api/debug/memory/sessions").status_code == 404
        assert client.get("/api/debug/memory/any-session").status_code == 404
        assert client.get("/debug/memory").status_code == 404


def test_debug_api_shows_window(monkeypatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        sid = "debug-session"
        for text in ("最近睡不好", "还是睡不着"):
            resp = client.post(
                "/api/chat/sync",
                json={"session_id": sid, "student_id": "debug-stu", "text": text},
            )
            assert resp.status_code == 200

        # 会话列表里能看到它（按学生聚合，所以两项都要对得上）
        listing = client.get("/api/debug/memory/sessions").json()
        assert listing["supported"] is True
        assert any(
            row["student_id"] == "debug-stu" and row["session_id"] == sid
            for row in listing["sessions"]
        )

        # 窗口内容：两轮 = 4 条
        detail = client.get(f"/api/debug/memory/{sid}").json()
        assert detail["backend"] == "memory"
        assert detail["student_id"] == "debug-stu"
        assert detail["messages"] == 4
        assert detail["turns"] == 2
        assert detail["sessions"] == 1
        assert detail["window_messages"] == 4
        assert [t["role"] for t in detail["window"]] == [
            "student",
            "assistant",
            "student",
            "assistant",
        ]
        assert detail["window"][0]["content"] == "最近睡不好"
        assert all(t["trace_id"] for t in detail["window"])

        # 页面本身可访问
        page = client.get("/debug/memory")
        assert page.status_code == 200
        assert "短期记忆调试台" in page.text


def test_debug_api_unknown_session(monkeypatch) -> None:
    """查一个不存在的会话：短期记忆按学生存，没有归属就找不到。"""
    with _client(monkeypatch, enabled=True) as client:
        response = client.get("/api/debug/memory/not-exist-session")
        assert response.status_code == 404
