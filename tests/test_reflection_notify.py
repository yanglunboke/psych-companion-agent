"""反思模块与通知通道测试。"""

from __future__ import annotations

import json

import pytest

from app.observability.reflection import ReflectionService
from app.orchestrator.notify import ALLOWED_NOTIFY_KEYS, Notifier
from app.schemas import InterventionFit


@pytest.fixture()
def reflection() -> ReflectionService:
    return ReflectionService()


# --------------------------------------------------------------------------
# 反思
# --------------------------------------------------------------------------
def test_lecture_first_is_detected(reflection) -> None:
    result = reflection.review(
        "你应该先冷静下来，道理其实你都知道。\n我听着。",
        InterventionFit.情绪疏导,
    )
    assert "一上来就讲道理" in result.issues
    assert result.passed is False
    assert result.hint_for_next_turn


def test_empathy_first_is_not_flagged(reflection) -> None:
    result = reflection.review(
        "我听到了，这次确实很难受。\n你现在最想说的是哪一件？",
        InterventionFit.情绪疏导,
    )
    assert result.passed is True
    assert result.issues == []


def test_method_for_just_talk_is_detected(reflection) -> None:
    result = reflection.review(
        "我听到了。\n你可以试试深呼吸这个方法。\n现在感觉怎么样？",
        InterventionFit.只想说说,
    )
    assert "学生只想说说，回复里却给了方法" in result.issues
    assert "只听" in (result.hint_for_next_turn or "")


def test_method_for_practice_fit_is_allowed(reflection) -> None:
    result = reflection.review(
        "我听到了。\n我们做一个练习：吸气 4 秒。\n现在感觉怎么样？",
        InterventionFit.需要练习,
    )
    assert "学生只想说说，回复里却给了方法" not in result.issues


def test_more_than_one_question_is_detected(reflection) -> None:
    result = reflection.review(
        "我听到了。\n你最近睡得怎么样？\n考试压力大吗？",
        InterventionFit.情绪疏导,
    )
    assert "问了不止一个问题" in result.issues


def test_high_risk_gate_blocks_on_diagnosis(reflection) -> None:
    gate = reflection.gate_high_risk(
        "你这是抑郁症，需要看心理医生。", escalation_ok=True, trace_id="t1"
    )
    assert gate.blocking is True
    assert "草稿里带了诊断词" in gate.issues


def test_high_risk_gate_blocks_on_failed_escalation(reflection) -> None:
    gate = reflection.gate_high_risk("我在这儿。", escalation_ok=False, trace_id="t1")
    assert gate.blocking is True
    assert "高危上报未成功" in gate.issues


def test_high_risk_gate_passes_clean_draft(reflection) -> None:
    gate = reflection.gate_high_risk("我在这儿，先别一个人扛。", escalation_ok=True)
    assert gate.blocking is False
    assert gate.passed is True


def test_hint_is_carried_to_next_turn(reflection) -> None:
    reflection.set_hint("stu1", "上一轮你一上来就讲道理了。")
    assert reflection.get_hint("stu1") == "上一轮你一上来就讲道理了。"
    reflection.set_hint("stu1", None)
    assert reflection.get_hint("stu1") is None


# --------------------------------------------------------------------------
# 通知
# --------------------------------------------------------------------------
def test_notify_payload_is_whitelisted() -> None:
    payload = Notifier.build_payload(
        student_id="stu1",
        risk_level="high",
        reason_code="self_harm_statement",
        trace_id="t1",
        ts=1.0,
    )
    assert set(payload) <= set(ALLOWED_NOTIFY_KEYS)
    assert "我不想活" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_outbox_channel_writes_file(tmp_path) -> None:
    notifier = Notifier(tmp_path / "outbox.jsonl")
    delivered = await notifier.notify(
        ["outbox"],
        {"student_id": "stu1", "risk_level": "high", "reason_code": "x", "trace_id": "t", "ts": 1.0},
    )
    assert delivered == ["outbox"]
    content = (tmp_path / "outbox.jsonl").read_text(encoding="utf-8")
    assert "stu1" in content


@pytest.mark.asyncio
async def test_console_channel_counts_as_delivered(tmp_path) -> None:
    notifier = Notifier(tmp_path / "outbox.jsonl")
    delivered = await notifier.notify(["teacher_console"], {"student_id": "s"})
    assert delivered == ["teacher_console"]


@pytest.mark.asyncio
async def test_failed_webhook_is_not_counted(tmp_path) -> None:
    notifier = Notifier(tmp_path / "outbox.jsonl")
    delivered = await notifier.notify(
        ["http://127.0.0.1:63998/hook"], {"student_id": "s"}
    )
    assert delivered == []
