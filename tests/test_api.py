"""HTTP 接口测试。"""

from __future__ import annotations

import json
import sqlite3

import pytest
from agentscope.message import TextBlock, ToolCallBlock, ToolCallState
from agentscope.model import ChatResponse, FinishedReason
from fastapi.testclient import TestClient

from app.config import reload_settings
from app.models.mock_model import RuleBasedChatModel
from app.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_health_still_works(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_frontend_served_by_same_app(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "心语" in page.text
    assert '/static/app.js' in page.text
    assert 'rel="icon"' in page.text
    assert client.get("/favicon.ico").status_code == 204
    for path, content_type in [
        ("/static/style.css", "text/css"),
        ("/static/app.js", "javascript"),
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert content_type in response.headers["content-type"]


def test_mock_chat_does_not_write_sql(tmp_path, monkeypatch) -> None:
    """规则引擎轮次保留聊天功能，但 SQL 的四张表均无新数据。"""
    db = tmp_path / "mock_chat.db"
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("DB_BACKEND", "sql")
    monkeypatch.setenv("SQL_DSN", f"sqlite+aiosqlite:///{db.as_posix()}")
    monkeypatch.setenv("ESCALATION_OUTBOX_PATH", str(tmp_path / "escalations.jsonl"))
    reload_settings()
    try:
        with TestClient(create_app()) as local_client:
            assert local_client.app.state.container.storage_status["profile"]["backend"] == "sql"
            for text in ("考试考砸了，我很难过", "我不想活了"):
                response = local_client.post(
                    "/api/chat/sync",
                    json={"session_id": "no-sql-session", "student_id": "no-sql-student", "text": text},
                )
                assert response.status_code == 200
                assert response.json()["reply"]
                if text == "我不想活了":
                    assert response.json()["risk_level"] == "high"
            assert (tmp_path / "escalations.jsonl").exists()
            with sqlite3.connect(db) as connection:
                for table in ("student_utterance", "emotion_trajectory", "student_profile", "tool_call_log"):
                    assert connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0
            assert len(local_client.app.state.container.short_memory._sessions["no-sql-session"]) == 4
    finally:
        monkeypatch.delenv("DB_BACKEND")
        monkeypatch.delenv("SQL_DSN")
        reload_settings()


@pytest.mark.parametrize("fail_on", [None, 2, 3])
def test_real_model_sql_persistence_and_midturn_fallback(
    tmp_path, monkeypatch, fail_on: int | None
) -> None:
    """真实模型成功才落库；任一 Worker 中途降级则整轮不落库。"""
    from app.models import base as model_base

    db = tmp_path / "chat.db"
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("DB_BACKEND", "sql")
    monkeypatch.setenv("SQL_DSN", f"sqlite+aiosqlite:///{db.as_posix()}")
    monkeypatch.setenv("MODEL_CIRCUIT_BREAKER", "false")
    calls = {"n": 0}

    def build_fake_model(**kwargs):
        fake = RuleBasedChatModel(stream=kwargs.get("stream", True))

        async def call_api(model_name, messages, tools=None, tool_choice=None, **extra):
            calls["n"] += 1
            if calls["n"] == fail_on:
                raise RuntimeError("test model failure")
            if tools:
                payload = fake._structured_payload(messages, tools)
                return ChatResponse(
                    content=[ToolCallBlock(
                        type="tool_call", id="tc-test", name="GenerateStructuredOutput",
                        input=json.dumps(payload, ensure_ascii=False), state=ToolCallState.PENDING,
                    )],
                    is_last=True, finished_reason=FinishedReason.COMPLETED,
                )
            text = fake._plain_text(messages)
            return ChatResponse(
                content=[TextBlock(type="text", text=text)],
                is_last=True, finished_reason=FinishedReason.COMPLETED,
            )

        fake._call_api = call_api
        return fake

    monkeypatch.setattr(model_base, "build_openai_model", build_fake_model)
    reload_settings()
    try:
        with TestClient(create_app()) as local_client:
            assert local_client.app.state.container.storage_status["profile"]["backend"] == "sql"
            response = local_client.post(
                "/api/chat/sync",
                json={"session_id": "test-session", "student_id": "test-student", "text": "考试考砸了，很难过"},
            )
            assert response.status_code == 200
            assert response.json()["reply"]
            assert calls["n"] >= (fail_on or 1)
            with sqlite3.connect(db) as connection:
                counts = {
                    table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                    for table in ("student_utterance", "emotion_trajectory", "student_profile", "tool_call_log")
                }
            if fail_on is None:
                assert counts["student_utterance"] == 2
                assert counts["emotion_trajectory"] == 1
                assert counts["student_profile"] == 1
                assert counts["tool_call_log"] > 0
            else:
                assert all(count == 0 for count in counts.values()), counts
    finally:
        # 清除缓存中的 openai 配置；pytest 其余用例固定使用离线模型。
        monkeypatch.setenv("MODEL_PROVIDER", "mock")
        reload_settings()


def test_sync_chat_normal_turn(client) -> None:
    resp = client.post(
        "/api/chat/sync",
        json={
            "session_id": "api-1",
            "student_id": "api-stu-1",
            "text": "这次月考又考砸了，我真的太笨了",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "react"
    assert body["reply"]
    assert body["trace_id"]
    tools = [t["tool_name"] for t in body["tool_calls"]]
    assert "emotion_recognition" in tools
    assert "positive_guidance" in tools


def test_sync_chat_high_risk_triggers_escalation(client) -> None:
    resp = client.post(
        "/api/chat/sync",
        json={
            "session_id": "api-2",
            "student_id": "api-stu-2",
            "text": "我不想活了",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "plan_execute"
    assert body["risk_level"] == "high"
    assert "老师" in body["reply"]


def test_sse_stream_emits_events(client) -> None:
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "session_id": "api-3",
            "student_id": "api-stu-3",
            "text": "这次月考又考砸了，我真的太笨了",
        },
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in resp.iter_lines() if line]

    payloads = [json.loads(line[len("data: "):]) for line in lines if line.startswith("data: ")]
    types = [p["type"] for p in payloads]
    assert types[0] == "meta"
    assert "delta" in types
    assert types[-1] == "done"
    text = "".join(p["text"] for p in payloads if p["type"] == "delta")
    assert text.strip()


def test_trace_endpoint(client) -> None:
    resp = client.post(
        "/api/chat/sync",
        json={
            "session_id": "api-4",
            "student_id": "api-stu-4",
            "text": "我这次月考考砸了",
        },
    )
    trace_id = resp.json()["trace_id"]
    trace_resp = client.get(f"/api/trace/{trace_id}")
    assert trace_resp.status_code == 200
    body = trace_resp.json()
    assert body["trace_id"] == trace_id
    assert "emotion_recognition" in body["tools"]


def test_trace_endpoint_404(client) -> None:
    assert client.get("/api/trace/not-a-trace").status_code == 404
