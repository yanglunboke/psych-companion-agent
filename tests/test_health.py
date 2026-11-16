"""健康检查与元信息接口测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["risk_threshold"] == 0.72
    # 测试套件强制离线（conftest 里 MODEL_PROVIDER=mock）
    assert body["model_provider"] == "mock"


def test_meta_exposes_retrieval_params() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval"]["rrf_k"] == 60
    assert body["retrieval"]["dynamic_k_by_fit"]["轻量引导"] == 2
    assert body["memory"]["short_term_turns"] == 12
