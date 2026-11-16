"""评测脚手架测试：保证评测集有效、评测脚本能跑、结论可复现。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def eval_data():
    result = _run("gen_eval_data.py", "--queries", "60", "--dialogs", "10")
    assert result.returncode == 0, result.stderr
    retrieval = [
        json.loads(line)
        for line in (EVAL_DIR / "retrieval_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    profile = [
        json.loads(line)
        for line in (EVAL_DIR / "profile_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return retrieval, profile


def test_eval_set_files_exist(eval_data) -> None:
    retrieval, profile = eval_data
    assert len(retrieval) == 60
    assert len(profile) == 10


def test_gold_cards_exist_in_knowledge_base(eval_data) -> None:
    from app.config import get_settings
    from app.rag.index import build_default_index

    retrieval, _ = eval_data
    index = build_default_index(get_settings().data_dir)
    known = {c.card_id for c in index.cards}
    for item in retrieval:
        assert item["gold_card_id"] in known, item["gold_card_id"]


def test_eval_queries_have_structured_fields(eval_data) -> None:
    retrieval, _ = eval_data
    for item in retrieval:
        assert item["scene"]
        assert item["collection"] == "student_reply"
        assert item["fit"]
        assert isinstance(item["distortion"], list)
        assert item["variant"] in {"exact", "paraphrase"}


def test_dialogs_have_eight_rounds_and_gold_facts(eval_data) -> None:
    _, profile = eval_data
    for dialog in profile:
        assert len(dialog["turns"]) == 16  # 8 轮 × 2 条消息
        assert dialog["gold"]["grade"]
        assert dialog["gold"]["strength"]
        assert dialog["gold"]["cause"]


def test_retrieval_eval_script_runs(eval_data) -> None:
    result = _run("eval_retrieval.py")
    assert result.returncode == 0, result.stderr
    assert "只走向量" in result.stdout
    assert "完整管线" in result.stdout or "+断崖" in result.stdout


def test_profile_eval_script_runs(eval_data) -> None:
    result = _run("eval_profile_recall.py")
    assert result.returncode == 0, result.stderr
    assert "侧写事实召回评测" in result.stdout


def test_profile_summary_recovers_early_advantage() -> None:
    """核心结论：早期优势掉出原话窗口后，靠侧写摘要仍能捞回来。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_profile_recall import _answer_grade, _answer_strength, _build_profile

    turns = [
        {"role": "student", "content": "我比较擅长画画，画了三年了"},
        {"role": "assistant", "content": "我在听你说。"},
        {"role": "student", "content": "对了，我高二"},
        {"role": "assistant", "content": "我在听你说。"},
        {"role": "student", "content": "这次月考失利我挺难受的"},
        {"role": "assistant", "content": "我在听你说。"},
    ]
    profile = _build_profile(turns, "stu_x")
    from app.memory.profile import to_summary

    summary = to_summary(profile).to_prompt_text()

    # 只留后面两条原话（优势已不在窗口里）→ 答不出来
    assert _answer_strength(["对了，我高二", "这次月考失利我挺难受的"]) == ""
    # 加上侧写摘要 → 捞得回来
    assert "画画" in _answer_strength([summary])
    # 年级与诱因本来就在窗口里
    assert _answer_grade(["对了，我高二"]) == "高二"
    assert _answer_strength([summary]) == "画画"


def test_vector_threshold_is_calibrated_per_embedder() -> None:
    """文档的 0.35 是稠密模型口径；本地占位实现自带标定值。"""
    from app.config import get_settings
    from app.rag.index import build_default_index
    from app.rag.retriever import Retriever, RetrieverConfig

    settings = get_settings()
    index = build_default_index(settings.data_dir)
    retriever = Retriever(index, RetrieverConfig())
    effective = retriever.effective_vector_min_score()
    assert effective < settings.vector_min_score
    assert effective == index.embedder.calibrated_min_score
