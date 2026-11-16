"""编排器。

文档口径（第 4 / 8 问）：

* **思考**：学生的一句话先做高危判断（关键词 + 风险分），再进编排器；
  编排器里的 Tool Router 看三样东西 —— 风险结论、intervention_fit、
  说话的是学生还是老师。
* **行动**：不高危走 react；高危走 Plan-execute，固定三步：
  短的安全回应 → escalate_risk（高位上报，一个工具调用）→ 停止建议。
* **观察反思**：每步的工具名、入参和是否成功写入 tool_call_log；
  反思模块事后核对（该跳过资源建议时有没有误推练习、高危轮有没有上报成功）。

数据流（第 8 问）：网关收到 session_id / student_id / 原话，同时做三件事 ——
Redis 取最近 12 轮、Postgres 取侧写摘要、词表匹配；命中自伤伤人则直接
触发危机上报。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.config import Settings
from app.memory.profile import to_summary, update_profile
from app.models.mock_model import RuleUsage, rule_usage
from app.observability.reflection import ReflectionService
from app.observability.trace import Trace, TraceStore
from app.orchestrator.escalate import EscalationService
from app.orchestrator.router import RoutePlan, ToolRouter
from app.orchestrator.safety import SafetyGate
from app.rag.query import build_query
from app.rag.retriever import Retriever
from app.schemas import (
    ChatOutcome,
    ChatRequest,
    EmotionCore,
    EmotionResult,
    EmotionTrajectoryItem,
    EscalationRecord,
    ExerciseCard,
    InterventionFit,
    NeedResult,
    OrchestratorDecision,
    RetrievedCard,
    RiskVerdict,
    Route,
    SessionContext,
    Speaker,
    Turn,
)
from app.workers.emotion import EmotionWorker, RiskScorer
from app.workers.exercise import ExerciseWorker
from app.workers.guidance import GuidanceWorker
from app.workers.needs import NeedsWorker

logger = logging.getLogger(__name__)


@dataclass
class ChatEvent:
    """一轮对话向外抛出的事件（SSE data）。"""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


class Orchestrator:
    """对话编排器。"""

    def __init__(
        self,
        settings: Settings,
        *,
        emotion_worker: EmotionWorker,
        risk_scorer: RiskScorer,
        needs_worker: NeedsWorker,
        guidance_worker: GuidanceWorker,
        exercise_worker: ExerciseWorker,
        retriever: Retriever,
        safety_gate: SafetyGate,
        escalation: EscalationService,
        reflection: ReflectionService,
        short_memory,
        profile_store,
        trace_store: TraceStore,
        router: ToolRouter | None = None,
        trace_sink=None,
    ) -> None:
        self.settings = settings
        self.emotion_worker = emotion_worker
        self.risk_scorer = risk_scorer
        self.needs_worker = needs_worker
        self.guidance_worker = guidance_worker
        self.exercise_worker = exercise_worker
        self.retriever = retriever
        self.safety_gate = safety_gate
        self.escalation = escalation
        self.reflection = reflection
        self.short_memory = short_memory
        self.profile_store = profile_store
        self.trace_store = trace_store
        self.router = router or ToolRouter()
        self.trace_sink = trace_sink
        self._background: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    async def handle(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
        """按请求隔离规则引擎使用状态，流式迭代结束时清理。"""
        token = rule_usage.set(RuleUsage())
        try:
            async for event in self._handle(req):
                yield event
        finally:
            rule_usage.reset(token)

    async def _handle(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
        """处理一轮对话，按事件流式返回。"""
        trace = Trace()
        self.trace_store.save(trace)
        started = time.perf_counter()

        # ---------- 0. 网关：同时做三件事 ----------
        recent = await self.short_memory.get_recent(
            req.session_id, self.settings.short_term_turns
        )
        profile = await self.profile_store.get_profile(req.student_id)
        summary = to_summary(profile)
        ctx = SessionContext(
            session_id=req.session_id,
            student_id=req.student_id,
            current_text=req.text,
            recent_turns=recent,
            profile=summary,
            reflection_hint=self.reflection.get_hint(req.student_id),
        )

        yield ChatEvent(
            "meta",
            {
                "trace_id": trace.trace_id,
                "session_id": req.session_id,
                "student_id": req.student_id,
                "recent_turns": len(recent),
                "profile_known": bool(summary.strengths or summary.stable_distortions),
            },
        )

        # ---------- 1. 安全门第一路：关键词触发 ----------
        t0 = time.perf_counter()
        hit = self.safety_gate.scan(req.text)
        trace.record(
            "safety_lexicon",
            key_args={
                "hit": hit.hit,
                "reason_code": hit.reason_code,
                "matched": hit.matched_terms,
            },
            success=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            detail=hit.category or "",
        )

        # ---------- 2. 情绪识别（每轮都跑） ----------
        t0 = time.perf_counter()
        emotion_core: EmotionCore = await self.emotion_worker.run(ctx)
        trace.record(
            "emotion_recognition",
            key_args={
                "emotion_type": emotion_core.emotion_type,
                "intensity": emotion_core.intensity,
                "cause_domain": emotion_core.cause_domain.value,
                "distortions": emotion_core.candidate_distortions,
            },
            success=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        # ---------- 3. 风险分（小模型，temperature=0） ----------
        t0 = time.perf_counter()
        risk = await self.risk_scorer.score(ctx)
        trace.record(
            "risk_scorer",
            key_args={"risk_score": round(risk.risk_score, 3)},
            success=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            detail=risk.risk_reason,
        )

        emotion = EmotionResult(
            emotion_type=emotion_core.emotion_type,
            intensity=emotion_core.intensity,
            cause_domain=emotion_core.cause_domain,
            cause_detail=emotion_core.cause_detail,
            candidate_distortions=list(emotion_core.candidate_distortions),
            risk_score=risk.risk_score,
            risk_reason=risk.risk_reason,
            confidence=emotion_core.confidence,
        )

        # ---------- 4. 安全门裁决 ----------
        verdict: RiskVerdict = self.safety_gate.verdict(hit, risk.risk_score)
        trace.record(
            "safety_gate",
            key_args={
                "is_high_risk": verdict.is_high_risk,
                "source": verdict.source.value,
                "threshold": verdict.threshold,
            },
            success=True,
            detail=verdict.evidence,
        )

        outcome = ChatOutcome(
            trace_id=trace.trace_id,
            session_id=req.session_id,
            student_id=req.student_id,
            route=Route.PLAN_EXECUTE if verdict.is_high_risk else Route.REACT,
            emotion=emotion,
            risk_verdict=verdict,
        )

        # ---------- 5. 路由 ----------
        if verdict.is_high_risk:
            async for event in self._plan_execute(req, ctx, trace, outcome):
                yield event
        else:
            async for event in self._react(req, ctx, trace, outcome):
                yield event

        # ---------- 6. 落库 ----------
        # 任一 Worker 在本轮使用规则引擎（含真实模型中途降级）时，不写长期数据库。
        # 短期上下文仍保留；高危上报在 _plan_execute 中已独立执行。
        used_rule_engine = bool(rule_usage.get() and rule_usage.get().used)
        await self._persist(req, ctx, outcome, persist_long_term=not used_rule_engine)

        outcome.tool_calls = list(trace.logs)
        outcome.total_ms = (time.perf_counter() - started) * 1000

        # trace 落库：规则引擎轮次也不写工具调用日志到 SQL。
        if self.trace_sink is not None and not used_rule_engine:
            try:
                await self.trace_sink.write(outcome.tool_calls)
            except Exception as exc:  # noqa: BLE001
                logger.warning("trace 落库失败：%s", exc)

        yield ChatEvent(
            "done",
            {
                "trace_id": trace.trace_id,
                "route": outcome.route.value,
                "ttft_ms": round(outcome.ttft_ms, 2),
                "total_ms": round(outcome.total_ms, 2),
                "risk_verdict": outcome.risk_verdict.model_dump(mode="json") if outcome.risk_verdict else None,
                "emotion": outcome.emotion.model_dump(mode="json") if outcome.emotion else None,
                "needs": outcome.needs.model_dump(mode="json") if outcome.needs else None,
                "retrieved": [c.model_dump(mode="json") for c in outcome.retrieved],
                "exercise": outcome.exercise.model_dump(mode="json") if outcome.exercise else None,
                "escalation": outcome.escalation.model_dump(mode="json") if outcome.escalation else None,
                "tool_calls": [log.model_dump(mode="json") for log in outcome.tool_calls],
            },
        )

    # ------------------------------------------------------------------
    # 常规对话：ReAct 按需调用
    # ------------------------------------------------------------------
    async def _react(
        self,
        req: ChatRequest,
        ctx: SessionContext,
        trace: Trace,
        outcome: ChatOutcome,
    ) -> AsyncIterator[ChatEvent]:
        # 需求澄清
        t0 = time.perf_counter()
        needs: NeedResult = await self.needs_worker.run(ctx, outcome.emotion)
        trace.record(
            "need_clarification",
            key_args={
                "needs": needs.needs,
                "intervention_fit": needs.intervention_fit.value,
            },
            success=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            detail=needs.reasoning,
        )
        outcome.needs = needs

        plan: RoutePlan = self.router.plan_react(needs.intervention_fit, req.speaker)
        outcome.route = plan.route
        decision = plan.to_decision(outcome.risk_verdict, needs.intervention_fit, req.speaker)
        trace.record(
            "tool_router",
            key_args={
                "route": plan.route.value,
                "steps": plan.steps,
                "skipped": plan.skipped,
            },
            success=True,
            detail=plan.reason,
        )
        outcome_decision: OrchestratorDecision = decision

        # 策略检索（回应卡 + 禁止说法卡同时取）
        t0 = time.perf_counter()
        query = build_query(outcome.emotion, needs)
        retrieval = self.retriever.search(query, collections=["student_reply", "ban"])
        outcome.retrieved = list(retrieval.selected)
        trace.record(
            "strategy_retrieval",
            key_args={
                "collection": query.collection,
                "scene": query.scene,
                "distortion": query.distortion,
                "fit": query.fit,
                "recalled_vector": len(retrieval.recalled_vector),
                "recalled_bm25": len(retrieval.recalled_bm25),
                "fused": len(retrieval.fused),
                "reranked": len(retrieval.reranked),
                "selected": [c.card_id for c in retrieval.selected],
                "dynamic_k": retrieval.dynamic_k,
            },
            success=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        # 积极引导（流式）
        t0 = time.perf_counter()
        ttft = 0.0
        chunks: list[str] = []
        async for delta in self.guidance_worker.stream(
            ctx,
            outcome.emotion,
            needs,
            retrieval.selected,
            reflection_hint=ctx.reflection_hint,
        ):
            if not chunks:
                ttft = (time.perf_counter() - t0) * 1000
            chunks.append(delta)
            yield ChatEvent("delta", {"text": delta})
        reply = "".join(chunks)
        trace.record(
            "positive_guidance",
            key_args={"chars": len(reply), "cards": len(retrieval.selected)},
            success=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            detail=f"ttft={ttft:.0f}ms",
        )
        outcome.ttft_ms = ttft

        # 资源建议：仅 intervention_fit = 需要练习
        exercise: ExerciseCard | None = None
        if "resource_suggestion" in plan.steps:
            t0 = time.perf_counter()
            exercise = await self.exercise_worker.run(ctx, outcome.emotion, needs)
            outcome.exercise = exercise
            trace.record(
                "resource_suggestion",
                key_args={"exercise_id": exercise.id if exercise else None},
                success=exercise is not None,
                duration_ms=(time.perf_counter() - t0) * 1000,
                detail="" if exercise else "没有匹配的练习卡",
            )
            if exercise is not None:
                card_text = "\n\n" + ExerciseWorker.render(exercise)
                reply += card_text
                yield ChatEvent("card", {"exercise": exercise.model_dump(mode="json")})
                yield ChatEvent("delta", {"text": card_text})
        else:
            trace.record(
                "resource_suggestion",
                key_args={"skipped": True},
                success=True,
                detail=f"按干预档位跳过：{needs.intervention_fit.value}",
            )

        outcome.reply = reply

        # 反思：常规轮异步跑，不让流程卡在这里
        self._spawn_reflection(
            draft=reply,
            needs=needs,
            trace_id=trace.trace_id,
            student_id=req.student_id,
        )

        yield ChatEvent(
            "decision",
            {
                "route": outcome_decision.route.value,
                "reason": outcome_decision.reason,
                "intervention_fit": needs.intervention_fit.value,
                "speaker": req.speaker.value,
                "skipped_tools": outcome_decision.skipped_tools,
            },
        )

    # ------------------------------------------------------------------
    # 高危处理：Plan-execute 固定计划
    # ------------------------------------------------------------------
    async def _plan_execute(
        self,
        req: ChatRequest,
        ctx: SessionContext,
        trace: Trace,
        outcome: ChatOutcome,
    ) -> AsyncIterator[ChatEvent]:
        plan = self.router.plan_high_risk(outcome.risk_verdict)
        trace.record(
            "tool_router",
            key_args={"route": plan.route.value, "steps": plan.steps},
            success=True,
            detail=plan.reason,
        )

        # 第一步：简短的安全回应
        t0 = time.perf_counter()
        draft = await self.guidance_worker.safety_reply(ctx, outcome.emotion)
        trace.record(
            "safety_reply",
            key_args={"chars": len(draft)},
            success=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        # 第二步：escalate_risk（高位上报，一个工具调用）
        t0 = time.perf_counter()
        record: EscalationRecord = await self.escalation.escalate(
            student_id=req.student_id,
            risk_level=self._risk_level(outcome.risk_verdict),
            reason_code=outcome.risk_verdict.reason_code or "high_risk",
            trace_id=trace.trace_id,
        )
        outcome.escalation = record
        trace.record(
            "escalate_risk",
            key_args={
                "student_id": record.student_id,
                "risk_level": record.risk_level,
                "reason_code": record.reason_code,
            },
            success=record.delivered,
            duration_ms=(time.perf_counter() - t0) * 1000,
            detail=f"delivery_status={record.delivery_status}; notified={record.notified}",
        )

        # 第三步：通知老师/家长（已在 escalate 内部完成），停止建议
        trace.record(
            "notify_teacher_parent",
            key_args={"channels": record.notified},
            success=bool(record.notified),
            detail="停止建议，人工接手",
        )

        # 同步闸门：上报没成功，或草稿带诊断词 → 这轮回复不发给学生
        gate = self.reflection.gate_high_risk(
            draft,
            escalation_ok=record.delivered,
            trace_id=trace.trace_id,
        )
        outcome.reflection = gate
        trace.record(
            "reflection_gate",
            key_args={"passed": gate.passed, "issues": gate.issues},
            success=gate.passed,
        )

        if gate.blocking:
            outcome.reply = ""
            yield ChatEvent(
                "blocked",
                {
                    "trace_id": trace.trace_id,
                    "issues": gate.issues,
                    "detail": "高危轮未通过放行闸门，回复不发给学生",
                },
            )
        else:
            outcome.reply = draft
            yield ChatEvent("delta", {"text": draft})
            outcome.ttft_ms = (time.perf_counter() - t0) * 1000

        yield ChatEvent(
            "escalate",
            {
                "escalation": record.model_dump(mode="json"),
                "detail": "已识别高危，老师端只收到风险结论，不含学生原话",
            },
        )

    # ------------------------------------------------------------------
    def _risk_level(self, verdict: RiskVerdict | None) -> str:
        if verdict is None:
            return "high"
        if verdict.lexicon_hit:
            return "high"
        if verdict.risk_score >= 0.72:
            return "high"
        if verdict.risk_score >= 0.45:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    def _spawn_reflection(
        self,
        *,
        draft: str,
        needs: NeedResult,
        trace_id: str,
        student_id: str,
    ) -> None:
        """常规轮异步复盘：不阻塞主流程。"""
        try:
            task = asyncio.create_task(self._run_reflection(draft, needs, trace_id, student_id))
        except RuntimeError:  # 没有运行中的事件循环（同步调用场景）
            self.reflection.set_hint(student_id, None)
            return
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _run_reflection(
        self,
        draft: str,
        needs: NeedResult,
        trace_id: str,
        student_id: str,
    ) -> None:
        try:
            result = self.reflection.review(draft, needs.intervention_fit, trace_id=trace_id)
            self.reflection.set_hint(student_id, result.hint_for_next_turn)
            if result.issues:
                logger.info("复盘发现问题 %s（trace=%s）", result.issues, trace_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("复盘失败：%s", exc)

    # ------------------------------------------------------------------
    async def _persist(
        self,
        req: ChatRequest,
        ctx: SessionContext,
        outcome: ChatOutcome,
        *,
        persist_long_term: bool = True,
    ) -> None:
        """保留短期上下文；仅真实模型轮次写长期存储。"""
        now = time.time()
        await self.short_memory.append(
            req.session_id,
            Turn(role="student", content=req.text, ts=now, trace_id=outcome.trace_id),
        )
        if outcome.reply:
            await self.short_memory.append(
                req.session_id,
                Turn(role="assistant", content=outcome.reply, ts=now, trace_id=outcome.trace_id),
            )

        if not persist_long_term:
            return

        if outcome.emotion is not None:
            await self.profile_store.append_emotion(
                EmotionTrajectoryItem(
                    student_id=req.student_id,
                    ts=now,
                    emotion_type=outcome.emotion.emotion_type,
                    intensity=outcome.emotion.intensity,
                    raw_intensity=outcome.emotion.intensity,
                    cause_domain=outcome.emotion.cause_domain.value,
                    cause_detail=outcome.emotion.cause_detail,
                    trace_id=outcome.trace_id,
                )
            )

        profile = await self.profile_store.get_profile(req.student_id)
        if outcome.emotion is not None:
            profile = update_profile(
                profile,
                outcome.emotion,
                outcome.needs,
                alpha=self.settings.intensity_ema_alpha,
                evidence_threshold=self.settings.profile_evidence_threshold,
                student_text=req.text,
            )
        await self.profile_store.save_profile(profile)

        await self.profile_store.append_utterance(
            req.student_id, "student", req.text, outcome.trace_id
        )
        if outcome.reply:
            await self.profile_store.append_utterance(
                req.student_id, "assistant", outcome.reply, outcome.trace_id
            )


__all__ = ["ChatEvent", "Orchestrator"]
