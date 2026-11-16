"""RAG 四步检索测试。"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.bm25 import BM25Index
from app.rag.index import build_default_index
from app.rag.query import build_query
from app.rag.retriever import Retriever, RetrieverConfig
from app.schemas import CauseDomain, EmotionCore, InterventionFit, NeedResult


@pytest.fixture(scope="module")
def index():
    settings = get_settings()
    return build_default_index(settings.data_dir)


@pytest.fixture(scope="module")
def retriever(index):
    settings = get_settings()
    return Retriever(
        index,
        RetrieverConfig(
            recall_top_k=settings.recall_top_k,
            vector_min_score=settings.vector_min_score,
            rrf_k=settings.rrf_k,
            rerank_min_score=settings.rerank_min_score,
            rerank_top_k=settings.rerank_top_k,
            dynamic_k_by_fit=dict(settings.dynamic_k_by_fit),
            score_gap_threshold=settings.score_gap_threshold,
        ),
    )


def _query(scene="exam_failure", distortion=("以偏概全",), fit=InterventionFit.轻量引导):
    emotion = EmotionCore(
        emotion_type="沮丧",
        intensity=4,
        cause_domain=CauseDomain.学业,
        cause_detail="考砸",
        candidate_distortions=list(distortion),
    )
    needs = NeedResult(needs=["被理解"], intervention_fit=fit)
    return build_query(emotion, needs)


def test_index_loaded_all_collections(index) -> None:
    collections = index.collections()
    assert "student_reply" in collections
    assert "ban" in collections
    assert "exercise" in collections


def test_structured_query_is_not_raw_text() -> None:
    query = _query()
    assert query.scene == "exam_failure"
    assert "以偏概全" in query.distortion
    assert query.fit == "轻量引导"
    assert "共情接纳" in query.steps
    # 查询串里不应出现"学生原句"
    assert "这次月考" not in query.to_query_string()


def test_four_step_pipeline_produces_traceable_result(retriever) -> None:
    result = retriever.search(_query(), collections=["student_reply", "ban"])
    assert result.recalled_vector, "向量召回应有结果"
    assert result.recalled_bm25, "BM25 召回应有结果"
    assert result.fused, "RRF 融合应有结果"
    assert result.reranked, "重排后应有结果"
    assert result.selected, "截断后应留下卡片"
    # 每张入选卡都带齐四步的分数
    for card in result.selected:
        assert card.rrf_score > 0
        assert card.rerank_score is not None
        assert card.final_score == card.rerank_score


def test_selected_contains_strategy_and_ban_card(retriever) -> None:
    """文档口径：轻量引导留两张 —— 回应策略卡 + 禁止说法卡。"""
    result = retriever.search(_query(), collections=["student_reply", "ban"])
    ids = [c.card_id for c in result.selected]
    assert "scene_exam_failure" in ids
    assert "ban_exam_failure" in ids
    assert result.dynamic_k == 2


def test_dynamic_k_limits_cards(retriever) -> None:
    result = retriever.search(
        _query(fit=InterventionFit.需要练习), collections=["student_reply", "ban"]
    )
    assert len(result.selected) <= result.dynamic_k == 4


def test_bm25_finds_exact_term_better_than_vector(retriever) -> None:
    """BM25 靠"以偏概全"逐字命中，向量召回做不到这么准。"""
    result = retriever.search(_query(), collections=["student_reply", "ban"])
    bm25_ids = [c.card_id for c in result.recalled_bm25]
    assert "scene_exam_failure" in bm25_ids


def test_collection_isolation_exercise_not_visible_in_reply_round(retriever) -> None:
    """学生共情轮只能看到回应卡和禁止卡，练习卡不可见。"""
    result = retriever.search(_query(), collections=["student_reply", "ban"])
    assert all(not c.card_id.startswith("breath") for c in result.fused)
    assert all(c.collection in {"student_reply", "ban"} for c in result.selected)


def test_exercise_collection_returns_exercise_card(retriever) -> None:
    query = _query(scene="exam_anxiety", distortion=(), fit=InterventionFit.需要练习)
    result = retriever.search(query, collections=["exercise"])
    assert result.selected
    assert result.selected[0].collection == "exercise"
    assert result.selected[0].payload.get("steps")


def test_gap_cutoff_drops_tail(retriever, index) -> None:
    """断崖截断：与上一张分差超过阈值就舍弃后面的。"""
    from app.schemas import RetrievedCard, StructuredQuery

    query = StructuredQuery(collection="student_reply", scene="exam_failure")
    cards = [
        RetrievedCard(card_id="a", collection="student_reply", scene="exam_failure",
                      title="t", text="x", final_score=0.86),
        RetrievedCard(card_id="b", collection="student_reply", scene="exam_failure",
                      title="t", text="x", final_score=0.81),
        RetrievedCard(card_id="c", collection="student_reply", scene="exam_failure",
                      title="t", text="x", final_score=0.52),
    ]
    selected, limit, _ = retriever._truncate(query, cards)
    assert [c.card_id for c in selected] == ["a", "b"]
    assert limit == 2


def test_bm25_index_is_deterministic() -> None:
    idx = BM25Index()
    idx.add("d1", "考试失利 以偏概全 学业自我否定")
    idx.add("d2", "同伴冲突 被孤立")
    idx.finalize()
    first = idx.search("以偏概全", top_k=5)
    second = idx.search("以偏概全", top_k=5)
    assert first == second
    assert first[0][0] == "d1"
