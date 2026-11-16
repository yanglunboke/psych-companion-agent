"""HTTP 接口测试。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_health_still_works(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


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
