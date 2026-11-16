"""健康检查与运行元信息。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.config import get_settings
from app.schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """存活探针 + 当前生效的关键配置。"""
    settings = get_settings()
    container = getattr(request.app.state, "container", None)
    model_status = container.model_factory.status() if container else []
    provider = settings.model_provider
    name = settings.model_name
    if model_status:
        main = next((m for m in model_status if m["purpose"] == "main"), None)
        if main:
            provider = main["backend"]
            name = main["model"]
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        model_provider=provider,
        model_name=name,
        memory_backend=settings.memory_backend,
        risk_threshold=settings.risk_threshold,
    )


@router.get("/api/meta")
async def meta(request: Request) -> dict:
    """返回本轮生效的关键参数，便于确认部署配置是否正确。"""
    settings = get_settings()
    container = getattr(request.app.state, "container", None)
    payload = {
        "app": settings.app_name,
        "version": __version__,
        "model": {
            "provider": settings.model_provider,
            "name": settings.model_name,
            "risk_model_name": settings.risk_model_name,
            "temperature": settings.model_temperature,
            "risk_temperature": settings.risk_model_temperature,
        },
        "safety": {
            "risk_threshold": settings.risk_threshold,
            "lexicon_path": str(settings.lexicon_file),
        },
        "memory": {
            "backend": settings.memory_backend,
            "short_term_turns": settings.short_term_turns,
            "intensity_ema_alpha": settings.intensity_ema_alpha,
            "profile_evidence_threshold": settings.profile_evidence_threshold,
        },
        "retrieval": {
            "recall_top_k": settings.recall_top_k,
            "vector_min_score": settings.vector_min_score,
            "rrf_k": settings.rrf_k,
            "rerank_min_score": settings.rerank_min_score,
            "rerank_top_k": settings.rerank_top_k,
            "dynamic_k_by_fit": settings.dynamic_k_by_fit,
            "score_gap_threshold": settings.score_gap_threshold,
        },
    }
    if container is not None:
        payload["runtime"] = container.status()
    return payload
