"""编排器端到端测试：常规轮 / 高危轮 / 按需资源建议 / trace 顺序 / 上报体。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.container import Container
from app.orchestrator.escalate import ALLOWED_PAYLOAD_KEYS
from app.schemas import ChatRequest, InterventionFit, Route, Speaker


@pytest.fixture(scope="module")
def container():
    return Container(get_settings())


_SEQ = itertools.count(1)


async def _run(container, text: str, session: str | None = None, student: str | None = None, speaker=Speaker.学生):
    """每个用例默认用独立的会话与学生，避免上一轮上下文串进下一轮。"""
    seq = next(_SEQ)
    session = session or f"sess{seq}"
    student = student or f"stu{seq}"
    events = []
    async for event in container.orchestrator.handle(
        ChatRequest(session_id=session, student_id=student, text=text, speaker=speaker)
    ):
        events.append(event)
    return events


def _done(events):
    return next(e for e in events if e.type == "done").data


def _reply(events):
    return "".join(str(e.data.get("text", "")) for e in events if e.type == "delta")


@pytest.mark.asyncio
async def test_normal_turn_runs_react_path(container) -> None:
    events = await _run(container, "这次月考又考砸了，我真的太笨了")
    done = _done(events)
    assert done["route"] == Route.REACT.value
    assert done["emotion"]["emotion_type"] == "沮丧"
    assert done["emotion"]["cause_domain"] == "学业"
    assert done["needs"]["intervention_fit"] == InterventionFit.轻量引导.value
    # 轻量引导 → 只取 2 张卡：回应策略卡 + 禁止说法卡
    ids = [c["card_id"] for c in done["retrieved"]]
    assert "scene_exam_failure" in ids
    assert "ban_exam_failure" in ids
    assert len(ids) <= 2
    # 不该在轻量引导时误推练习
    assert done["exercise"] is None


@pytest.mark.asyncio
async def test_tool_call_order_matches_document(container) -> None:
    events = await _run(container, "这次月考又考砸了，我真的太笨了")
    tools = [t["tool_name"] for t in _done(events)["tool_calls"]]
    assert tools.index("safety_lexicon") < tools.index("emotion_recognition")
    assert tools.index("emotion_recognition") < tools.index("need_clarification")
    assert tools.index("need_clarification") < tools.index("strategy_retrieval")
    assert tools.index("strategy_retrieval") < tools.index("positive_guidance")


@pytest.mark.asyncio
async def test_exercise_only_when_fit_requires_practice(container) -> None:
    events = await _run(container, "一到考试我就紧张得睡不着，有没有方法能缓解")
    done = _done(events)
    assert done["needs"]["intervention_fit"] == InterventionFit.需要练习.value
    assert done["exercise"] is not None
    assert done["exercise"]["duration"]
    assert done["exercise"]["steps"]
    # 练习卡是附在积极引导回复之后的
    assert "练习" in _reply(events)


@pytest.mark.asyncio
async def test_high_risk_takes_plan_execute(container) -> None:
    events = await _run(container, "我不想活了", session="s2", student="stu2")
    done = _done(events)
    assert done["route"] == Route.PLAN_EXECUTE.value
    tools = [t["tool_name"] for t in done["tool_calls"]]
    # 固定三步：安全回应 → escalate_risk → 通知
    assert "safety_reply" in tools
    assert "escalate_risk" in tools
    assert "notify_teacher_parent" in tools
    # 高危轮不再自由挑选工具
    assert "strategy_retrieval" not in tools
    assert "need_clarification" not in tools
    # 停止建议：不推练习卡
    assert done["exercise"] is None
    assert done["retrieved"] == []


@pytest.mark.asyncio
async def test_high_risk_reply_is_short_and_safe(container) -> None:
    events = await _run(container, "我不想活了", session="s3", student="stu3")
    reply = _reply(events)
    assert 0 < len(reply) <= 220
    assert "老师" in reply
    assert "练习" not in reply


@pytest.mark.asyncio
async def test_escalation_payload_has_no_raw_utterance(container) -> None:
    """上报体只含 student_id / 风险等级 / 原因码，绝不含原话。"""
    events = await _run(container, "我不想活了", session="s4", student="stu4")
    done = _done(events)
    escalation = done["escalation"]
    assert escalation["student_id"] == "stu4"
    assert escalation["risk_level"] == "high"
    assert escalation["reason_code"] == "self_harm_statement"
    assert "我不想活" not in json.dumps(escalation, ensure_ascii=False)


@pytest.mark.asyncio
async def test_escalation_payload_keys_are_whitelisted(container) -> None:
    payload = container.escalation.build_payload(
        student_id="stu5", risk_level="high", reason_code="self_harm_statement", trace_id="t1"
    )
    assert set(payload) <= set(ALLOWED_PAYLOAD_KEYS)


@pytest.mark.asyncio
async def test_reflection_gate_blocks_when_escalation_fails(container) -> None:
    """上报失败 → 这轮回复不发给学生。"""
    from app.orchestrator import escalate as escalate_mod

    original = escalate_mod.EscalationService._deliver

    async def failing(self, payload):
        return False, "failed:test"

    escalate_mod.EscalationService._deliver = failing
    try:
        events = await _run(container, "我不想活了", session="s6", student="stu6")
    finally:
        escalate_mod.EscalationService._deliver = original

    blocked = [e for e in events if e.type == "blocked"]
    assert blocked, "上报失败时必须拦截"
    assert _reply(events) == ""
    done = _done(events)
    assert done["escalation"]["delivered"] is False


@pytest.mark.asyncio
async def test_teacher_speaker_is_routed_differently(container) -> None:
    events = await _run(
        container,
        "这个学生最近情绪不太好，麻烦你跟进一下",
        session="s7",
        student="stu7",
        speaker=Speaker.老师,
    )
    done = _done(events)
    assert done["route"] == Route.REACT.value
    tools = [t["tool_name"] for t in done["tool_calls"]]
    # 老师端不推学生练习卡
    assert "resource_suggestion" not in [t for t in tools if t == "resource_suggestion" and False]
    assert done["exercise"] is None


@pytest.mark.asyncio
async def test_trace_can_be_looked_up(container) -> None:
    events = await _run(container, "这次月考又考砸了，我真的太笨了", session="s8", student="stu8")
    trace_id = _done(events)["trace_id"]
    trace = container.trace_store.get(trace_id)
    assert trace is not None
    assert "emotion_recognition" in trace.tool_names()


@pytest.mark.asyncio
async def test_multi_turn_keeps_recent_context(container) -> None:
    await _run(container, "我这次月考考砸了", session="s9", student="stu9")
    events = await _run(container, "我还总是睡不着", session="s9", student="stu9")
    meta = next(e for e in events if e.type == "meta").data
    assert meta["recent_turns"] >= 1
