"""回归测试：学生可见的回复里不能出现模型的推理内容或工具调用参数。

背景（线上真实缺陷）：MiniMax / DeepSeek-R1 / Qwen3 这类 OpenAI 兼容服务
**不返回**独立的 ``reasoning_content`` 字段，而是把 ``<think>...</think>``
内联进 ``content``。AgentScope 的 OpenAI 适配器拿不到独立通道，这段文字就
以普通正文增量事件（``TextBlockDeltaEvent``）到达 Worker；工具调用参数则以
``ToolCallDeltaEvent`` 到达。修复前两者都被原样转发，学生先读到模型的英文
内心独白，才读到真正的回复。

这些用例不依赖真实服务：用假模型复刻上述两个特征，保证以后再改 Worker
也不会把推理内容放回去。
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from agentscope.credential import CredentialBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import TextBlock, ToolCallBlock, ToolCallState
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from pydantic import BaseModel

from app.models.sanitize import ThinkFilter, strip_think
from app.schemas import (
    CauseDomain,
    EmotionCore,
    InterventionFit,
    NeedResult,
    SessionContext,
)
from app.workers.guidance import GuidanceWorker

THINK_MARKERS = ("<think>", "</think>")


# --------------------------------------------------------------------------
# 纯函数层：流式过滤器
# --------------------------------------------------------------------------
def test_plain_text_passes_through() -> None:
    filt = ThinkFilter()
    assert filt.feed("我听到了，") == "我听到了，"
    assert filt.feed("这份难受是真的。") == "这份难受是真的。"
    assert filt.flush() == ""


def test_think_block_is_removed() -> None:
    filt = ThinkFilter()
    assert filt.feed("<think>模型在自言自语</think>学生看到的部分") == "学生看到的部分"
    assert filt.flush() == ""


def test_think_tag_split_across_chunks_is_removed() -> None:
    """标签跨增量块断开时，逐块正则替换一定会漏——必须按状态机处理。"""
    filt = ThinkFilter()
    assert filt.feed("<thi") == ""
    assert filt.feed("nk>这是推理") == ""
    assert filt.feed("，还在推理</think>真正") == "真正"
    assert filt.feed("的回复") == "的回复"
    assert filt.flush() == ""


def test_close_tag_split_across_chunks_is_removed() -> None:
    filt = ThinkFilter()
    assert filt.feed("<think>推理") == ""
    assert filt.feed("</thin") == ""
    assert filt.feed("k>正文") == "正文"


def test_unterminated_think_discards_the_rest() -> None:
    """推理块没闭合：宁可少给，也不能把推理当成回复发出去。"""
    filt = ThinkFilter()
    assert filt.feed("<think>推理还没说完") == ""
    assert filt.flush() == ""


def test_multiple_think_blocks_are_removed() -> None:
    filt = ThinkFilter()
    out = filt.feed("<think>一</think>甲<think>二</think>乙")
    assert out == "甲乙"


def test_stray_close_tag_is_removed() -> None:
    filt = ThinkFilter()
    assert filt.feed("正文</think>还是正文") == "正文还是正文"


def test_flush_returns_pending_tail() -> None:
    """暂扣的尾巴在流结束时必须吐出去。"""
    filt = ThinkFilter()
    assert filt.feed("正文结尾是 <thi") == "正文结尾是 "
    assert filt.flush() == "<thi"


def test_angle_brackets_in_plain_text_are_kept() -> None:
    """只认推理标签，正文里普通的尖括号不能被误伤。"""
    filt = ThinkFilter()
    text = "他说 <b>这样</b> 和 <notathink> 都算正文"
    assert filt.feed(text) == text
    assert filt.flush() == ""


# --------------------------------------------------------------------------
# 纯函数层：一次性过滤
# --------------------------------------------------------------------------
def test_strip_think_removes_block() -> None:
    assert strip_think("<think>推理</think>回复") == "回复"


def test_strip_think_keeps_plain_text() -> None:
    assert strip_think("没有任何推理标签") == "没有任何推理标签"


def test_strip_think_removes_multiline_block() -> None:
    text = "<think>第一行\n第二行\n第三行</think>\n真正回复"
    assert strip_think(text) == "\n真正回复"


def test_strip_think_discards_unterminated_tail() -> None:
    assert strip_think("回复前面<think>没闭合的推理") == "回复前面"


def test_strip_think_handles_empty_input() -> None:
    assert strip_think("") == ""


# --------------------------------------------------------------------------
# 假模型：复刻真实服务的两个特征
# --------------------------------------------------------------------------
class _FakeCredential(CredentialBase):
    name: str = "fake"


class _LeakyStreamingModel(ChatModelBase):
    """模拟把推理内联进 ``content`` 的 OpenAI 兼容服务。

    刻意贴近真实：

    * 推理内容与正文在**同一个** ``content`` 里，没有独立推理通道；
    * 推理标签**跨块断开**（``<thi`` + ``nk>...``）；
    * 工具调用参数走 ``ToolCallBlock``，会变成 ``ToolCallDeltaEvent``；
    * AgentScope 只把 ``is_last=True`` 分片的内容存进上下文，所以工具调用
      必须同时出现在 ``is_last`` 分片里才会真的被执行。
    """

    class Parameters(BaseModel):
        temperature: float = 0.0
        max_tokens: int | None = None

    def __init__(
        self,
        *,
        think_parts: list[str] | None = None,
        answer_parts: list[str] | None = None,
        tool_call: tuple[str, dict] | None = None,
        risk_score: float = 0.05,
    ) -> None:
        super().__init__(
            credential=_FakeCredential(),
            model="fake-leaky-model",
            parameters=self.Parameters(),
            stream=True,
            max_retries=0,
        )
        self.formatter = OpenAIChatFormatter()
        self.think_parts = list(think_parts or [])
        self.answer_parts = list(answer_parts or [])
        self.tool_call = tool_call
        self.risk_score = risk_score
        self.text_calls = 0
        self.structured_calls = 0

    # -- 结构化输出（情绪识别 / 需求澄清 / 风险分）------------------------
    def _structured_payload(self, tools: list[dict]) -> dict:
        props: set[str] = set()
        for tool in tools:
            fn = tool.get("function", {})
            if fn.get("name") == "GenerateStructuredOutput":
                props = set((fn.get("parameters") or {}).get("properties", {}).keys())
                break
        if props and props <= {"risk_score", "risk_reason", "confidence"}:
            return {"risk_score": self.risk_score, "risk_reason": "测试用风险分"}
        if {"emotion_type", "intensity"} & props:
            return {
                "emotion_type": "沮丧",
                "intensity": 4,
                "cause_domain": CauseDomain.学业.value,
                "cause_detail": "月考失利",
                "candidate_distortions": ["以偏概全"],
                "confidence": 0.8,
            }
        if {"needs", "intervention_fit"} & props:
            return {
                "needs": ["被理解"],
                "intervention_fit": InterventionFit.情绪疏导.value,
                "reasoning": "测试用判定",
            }
        return {}

    # -- 正文流 -----------------------------------------------------------
    async def _call_api(
        self,
        model_name: str,
        messages: list,
        tools: list[dict] | None = None,
        tool_choice=None,
        **kwargs,
    ):
        tool_names = {t.get("function", {}).get("name") for t in (tools or [])}
        if "GenerateStructuredOutput" in tool_names:
            self.structured_calls += 1
            payload = self._structured_payload(tools or [])
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id="tc-structured",
                        name="GenerateStructuredOutput",
                        input=json.dumps(payload, ensure_ascii=False),
                        state=ToolCallState.PENDING,
                    )
                ],
                is_last=True,
                finished_reason=FinishedReason.COMPLETED,
            )

        self.text_calls += 1
        first_turn = self.text_calls == 1
        # 真实服务把**累积后的完整 content** 放在最后一片（is_last=True），
        # AgentScope 用它构造最终 Msg —— 所以推理内容也会出现在最终消息里。
        accumulated = "".join(self.think_parts)

        async def gen():
            if first_turn:
                for part in self.think_parts:
                    yield ChatResponse(
                        content=[TextBlock(type="text", text=part)],
                        is_last=False,
                        finished_reason=FinishedReason.COMPLETED,
                    )
                if self.tool_call is not None:
                    name, args = self.tool_call
                    payload = json.dumps(args, ensure_ascii=False)
                    block = ToolCallBlock(
                        type="tool_call",
                        id="tc-fake",
                        name=name,
                        input=payload,
                        state=ToolCallState.PENDING,
                    )
                    # 非 is_last 分片：产生 ToolCallDeltaEvent（学生可见性风险点）
                    yield ChatResponse(
                        content=[block],
                        is_last=False,
                        finished_reason=FinishedReason.COMPLETED,
                    )
                    # is_last 分片：AgentScope 只把这里的内容存进上下文，
                    # 工具调用才会真的被执行；累积文本也一并带上
                    yield ChatResponse(
                        content=[TextBlock(type="text", text=accumulated), block],
                        is_last=True,
                        finished_reason=FinishedReason.COMPLETED,
                    )
                    return

            for part in self.answer_parts:
                yield ChatResponse(
                    content=[TextBlock(type="text", text=part)],
                    is_last=False,
                    finished_reason=FinishedReason.COMPLETED,
                )
            yield ChatResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=accumulated + "".join(self.answer_parts),
                    )
                ],
                is_last=True,
                finished_reason=FinishedReason.COMPLETED,
            )

        return gen()


def _ctx(text: str = "这次月考又考砸了，我真的太笨了") -> SessionContext:
    return SessionContext(session_id="s1", student_id="stu1", current_text=text)


def _emotion() -> EmotionCore:
    return EmotionCore(
        emotion_type="沮丧",
        intensity=4,
        cause_domain=CauseDomain.学业,
        cause_detail="月考失利",
        candidate_distortions=["以偏概全"],
    )


def _needs() -> NeedResult:
    return NeedResult(
        needs=["被理解"],
        intervention_fit=InterventionFit.情绪疏导,
        reasoning="测试用",
    )


# --------------------------------------------------------------------------
# Worker 层
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_guidance_stream_does_not_leak_think_content() -> None:
    """推理内容（标签跨块断开）不得出现在学生可见的流里。"""
    model = _LeakyStreamingModel(
        think_parts=["<thi", "nk>We need to reply ", "in Chinese.</think>"],
        answer_parts=["我听到了，", "这份难受是真的。"],
    )
    worker = GuidanceWorker(model)
    chunks = [chunk async for chunk in worker.stream(_ctx(), _emotion(), _needs())]
    visible = "".join(chunks)

    assert visible == "我听到了，这份难受是真的。"
    assert "think" not in visible.lower()
    for marker in THINK_MARKERS:
        assert marker not in visible
    assert "We need to reply" not in visible


@pytest.mark.asyncio
async def test_guidance_stream_does_not_leak_tool_arguments() -> None:
    """工具调用参数不得出现在学生可见的流里。"""
    tool_args = {"query": "scene=exam_failure distortion=以偏概全"}
    model = _LeakyStreamingModel(
        think_parts=["<think>先检索一张卡</think>"],
        answer_parts=["我听到了。"],
        tool_call=("search_strategy", tool_args),
    )
    worker = GuidanceWorker(model)
    chunks = [chunk async for chunk in worker.stream(_ctx(), _emotion(), _needs())]
    visible = "".join(chunks)

    assert "search_strategy" not in visible
    assert "scene=exam_failure" not in visible
    assert "{'query'" not in visible and '{"query"' not in visible
    # 工具确实被执行了（模型拿到了工具结果才给出最终回复）
    assert model.text_calls >= 2


@pytest.mark.asyncio
async def test_safety_reply_does_not_leak_think_content() -> None:
    """高危安全回应是最关键的一段话，推理内容必须剥干净。"""
    model = _LeakyStreamingModel(
        think_parts=[
            "<think>We need <=4 lines. ",
            "Could mention 120/110? Ensure hotline.</think>",
        ],
        answer_parts=[
            "我在这里，没有走开，我会马上找一位老师来陪你。",
            "现在请离开危险的地方，或拨打心理援助热线 12356。",
        ],
    )
    worker = GuidanceWorker(model)
    draft = await worker.safety_reply(_ctx("我不想活了"), _emotion())

    assert "我在这里" in draft
    assert "12356" in draft
    assert "think" not in draft.lower()
    for marker in THINK_MARKERS:
        assert marker not in draft
    assert "120/110" not in draft


@pytest.mark.asyncio
async def test_safety_reply_passes_reflection_gate() -> None:
    """带推理的草稿曾经因为英文问号被判成"问了不止一个问题"而被拦下。"""
    from app.observability.reflection import ReflectionService

    model = _LeakyStreamingModel(
        think_parts=["<think>Should I ask two questions?</think>"],
        answer_parts=["我在这里，没有走开。", "请拨打 12356。"],
    )
    worker = GuidanceWorker(model)
    draft = await worker.safety_reply(_ctx("我不想活了"), _emotion())

    gate = ReflectionService().gate_high_risk(draft, escalation_ok=True, trace_id="t")
    assert gate.passed, gate.issues


# --------------------------------------------------------------------------
# 端到端：编排器 + 落库
# --------------------------------------------------------------------------
def test_chat_reply_and_persisted_utterance_have_no_think_content(
    tmp_path, monkeypatch
) -> None:
    """整轮跑完：返回给学生的文本与落库的"助手原话"都不能带推理内容。"""
    from fastapi.testclient import TestClient

    from app.api.chat import router  # noqa: F401  确保路由模块已加载
    from app.config import reload_settings
    from app.main import create_app
    from app.models import base as model_base

    db = tmp_path / "leak.db"
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("DB_BACKEND", "sql")
    monkeypatch.setenv("SQL_DSN", f"sqlite+aiosqlite:///{db.as_posix()}")
    monkeypatch.setenv("MODEL_CIRCUIT_BREAKER", "false")

    def build_fake_model(**kwargs):
        return _LeakyStreamingModel(
            think_parts=["<thi", "nk>内部推理，不该给学生看</think>"],
            answer_parts=["我听到了，这份难受是真的。", "你现在最想被回应的是什么？"],
        )

    monkeypatch.setattr(model_base, "build_openai_model", build_fake_model)
    reload_settings()
    try:
        with TestClient(create_app()) as client:
            response = client.post(
                "/api/chat/sync",
                json={
                    "session_id": "leak-session",
                    "student_id": "leak-student",
                    "text": "这次月考又考砸了，我真的太笨了",
                },
            )
            assert response.status_code == 200
            reply = response.json()["reply"]

            assert reply, "回复不应为空"
            for marker in THINK_MARKERS:
                assert marker not in reply
            assert "内部推理" not in reply
            assert "我听到了" in reply

        # 落库的"助手原话"同样不能带推理内容
        with sqlite3.connect(db) as connection:
            rows = connection.execute(
                "SELECT role, content FROM student_utterance"
            ).fetchall()
        assert rows, "真实模型轮应当落库"
        for role, content in rows:
            for marker in THINK_MARKERS:
                assert marker not in content
            assert "内部推理" not in content
    finally:
        monkeypatch.setenv("MODEL_PROVIDER", "mock")
        monkeypatch.delenv("DB_BACKEND")
        monkeypatch.delenv("SQL_DSN")
        reload_settings()


def test_sse_stream_does_not_leak_think_content(tmp_path, monkeypatch) -> None:
    """SSE 增量事件里同样不能出现推理内容。"""
    from fastapi.testclient import TestClient

    from app.config import reload_settings
    from app.main import create_app
    from app.models import base as model_base

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("MODEL_CIRCUIT_BREAKER", "false")

    def build_fake_model(**kwargs):
        return _LeakyStreamingModel(
            think_parts=["<think>内部推理，不该给学生看</think>"],
            answer_parts=["我听到了。"],
        )

    monkeypatch.setattr(model_base, "build_openai_model", build_fake_model)
    reload_settings()
    try:
        with TestClient(create_app()) as client:
            with client.stream(
                "POST",
                "/api/chat",
                json={
                    "session_id": "sse-session",
                    "student_id": "sse-student",
                    "text": "这次月考又考砸了",
                },
            ) as response:
                assert response.status_code == 200
                body = response.read().decode("utf-8")

        deltas = [
            json.loads(line[5:].strip())
            for line in body.splitlines()
            if line.startswith("data:") and '"delta"' in line
        ]
        assert deltas, "应当有正文增量"
        visible = "".join(d["text"] for d in deltas)
        assert visible, "学生可见文本不应为空"
        for marker in THINK_MARKERS:
            assert marker not in body
        assert "内部推理" not in body
    finally:
        monkeypatch.setenv("MODEL_PROVIDER", "mock")
        reload_settings()
