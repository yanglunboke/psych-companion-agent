"""领域模型与 API 契约。

这里的类型是全项目唯一的"口径来源"：Worker 的结构化输出、检索结果、
记忆快照、编排器决策、上报记录都以本文件为准。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# 枚举 / 常量
# --------------------------------------------------------------------------
class CauseDomain(str, Enum):
    """情绪诱因域（文档：学业、人际、家庭、自我）。"""

    学业 = "学业"
    人际 = "人际"
    家庭 = "家庭"
    自我 = "自我"
    未知 = "未知"


class InterventionFit(str, Enum):
    """需求澄清 Worker 的干预适配档位。"""

    只想说说 = "只想说说"
    情绪疏导 = "情绪疏导"
    轻量引导 = "轻量引导"
    需要练习 = "需要练习"


class Speaker(str, Enum):
    """说话人。Tool Router 需要区分学生与老师。"""

    学生 = "student"
    老师 = "teacher"


class Collection(str, Enum):
    """策略知识库集合（文档：靠 collection 做隔离）。"""

    STUDENT_REPLY = "student_reply"
    BAN = "ban"
    EXERCISE = "exercise"
    #: 理论卡（积极心理学理论、研究证据），给模型提供可引用的依据
    THEORY = "theory"


class RiskSource(str, Enum):
    LEXICON = "lexicon"
    MODEL = "model"
    BOTH = "both"
    NONE = "none"


class Route(str, Enum):
    """编排器本轮走哪条路径。"""

    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class WorkerName(str, Enum):
    EMOTION = "emotion_recognition"
    NEEDS = "need_clarification"
    GUIDANCE = "positive_guidance"
    EXERCISE = "resource_suggestion"


# --------------------------------------------------------------------------
# Worker 0 —— 情绪识别（每轮都跑，只出 JSON，不写回复）
# --------------------------------------------------------------------------
class EmotionResult(BaseModel):
    """情绪识别 Worker 的结构化输出。"""

    emotion_type: str = Field(description="情绪类型，如 沮丧 / 焦虑 / 愤怒")
    intensity: int = Field(ge=1, le=5, description="情绪强度 1-5")
    cause_domain: CauseDomain = Field(default=CauseDomain.未知, description="诱因域")
    cause_detail: str = Field(default="", description="具体诱因描述")
    candidate_distortions: list[str] = Field(
        default_factory=list,
        description="候选认知偏差，如 以偏概全 / 灾难化 / 非黑即白",
    )
    risk_score: float = Field(ge=0.0, le=1.0, description="风险分 0-1")
    risk_reason: str = Field(default="", description="风险分依据")
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class EmotionCore(BaseModel):
    """情绪识别的"非风险"部分。

    风险分由**独立的更小更快的模型**（temperature=0）单独产出，见 :class:`RiskScore`，
    两部分合并后才是完整的 :class:`EmotionResult`。
    """

    emotion_type: str = Field(description="情绪类型，如 沮丧 / 焦虑 / 愤怒")
    intensity: int = Field(ge=1, le=5, description="情绪强度 1-5")
    cause_domain: CauseDomain = Field(default=CauseDomain.未知, description="诱因域")
    cause_detail: str = Field(default="", description="具体诱因描述")
    candidate_distortions: list[str] = Field(
        default_factory=list,
        description="候选认知偏差，如 以偏概全 / 灾难化 / 非黑即白",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class RiskScore(BaseModel):
    """风险分小模型的输出（temperature=0，只出一个 0-1 分与简短依据）。"""

    risk_score: float = Field(ge=0.0, le=1.0, description="风险分 0-1")
    risk_reason: str = Field(default="", description="简短依据")


# --------------------------------------------------------------------------
# Worker 1 —— 需求澄清
# --------------------------------------------------------------------------
class NeedResult(BaseModel):
    """需求澄清 Worker 的结构化输出。"""

    needs: list[str] = Field(default_factory=list, description="学生的具体需求")
    intervention_fit: InterventionFit = Field(
        default=InterventionFit.情绪疏导,
        description="干预适配档位",
    )
    reasoning: str = Field(default="", description="判断依据")


# --------------------------------------------------------------------------
# 策略知识库
# --------------------------------------------------------------------------
class StrategyCard(BaseModel):
    """学生场景回复策略卡 / 禁止说法卡。"""

    id: str
    collection: Collection
    scene: str = Field(description="场景标识，如 exam_failure")
    title: str = ""
    must: list[str] = Field(default_factory=list, description="必须遵循的步骤")
    bans: list[str] = Field(default_factory=list, description="禁止出现的说法")
    applicable_fit: list[InterventionFit] = Field(default_factory=list)
    applicable_distortions: list[str] = Field(default_factory=list)
    forbidden_when: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list, description="策略卡展开成的步骤")
    text: str = Field(default="", description="卡片全文，参与重排打分")


class ExerciseCard(BaseModel):
    """练习条目（资源建议 Worker 的返回物）。"""

    id: str
    collection: Collection = Collection.EXERCISE
    scene: str = ""
    name: str = Field(description="练习名称")
    duration: str = Field(default="", description="时长，如 5 分钟")
    steps: list[str] = Field(default_factory=list, description="步骤")
    do_not_use_when: list[str] = Field(default_factory=list)
    text: str = ""


class RetrievedCard(BaseModel):
    """带检索分数的卡片。"""

    card_id: str
    collection: str
    scene: str
    title: str
    text: str
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


class StructuredQuery(BaseModel):
    """结构化检索查询（不用学生原句当查询）。"""

    collection: str = Collection.STUDENT_REPLY.value
    scene: str = ""
    distortion: list[str] = Field(default_factory=list)
    fit: str = ""
    steps: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    def to_query_string(self) -> str:
        """拼成给向量/BM25 用的查询串。"""
        parts: list[str] = [self.scene]
        parts.extend(self.distortion)
        parts.extend(self.steps)
        parts.extend(self.keywords)
        if self.fit:
            parts.append(self.fit)
        return " ".join(p for p in parts if p)


class RetrievalResult(BaseModel):
    """四步检索的完整产物（可追溯）。"""

    query: StructuredQuery
    recalled_vector: list[RetrievedCard] = Field(default_factory=list)
    recalled_bm25: list[RetrievedCard] = Field(default_factory=list)
    fused: list[RetrievedCard] = Field(default_factory=list)
    reranked: list[RetrievedCard] = Field(default_factory=list)
    selected: list[RetrievedCard] = Field(default_factory=list)
    dynamic_k: int = 0
    token_budget_used: int = 0


# --------------------------------------------------------------------------
# 安全门
# --------------------------------------------------------------------------
class LexiconHit(BaseModel):
    """关键词词表命中结果。"""

    hit: bool = False
    reason_code: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    category: str | None = None


class RiskVerdict(BaseModel):
    """安全门结论：关键词命中 ∨ 风险分达阈值。"""

    lexicon_hit: bool = False
    reason_code: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    risk_score: float = 0.0
    threshold: float = 0.72
    is_high_risk: bool = False
    source: RiskSource = RiskSource.NONE
    evidence: str = ""


# --------------------------------------------------------------------------
# 记忆
# --------------------------------------------------------------------------
class Turn(BaseModel):
    """一轮对话。"""

    role: str = Field(description="student / assistant / teacher")
    content: str
    ts: float = 0.0
    trace_id: str | None = None


class EmotionTrajectoryItem(BaseModel):
    """情绪轨迹流水（Postgres）。"""

    student_id: str
    ts: float
    emotion_type: str
    intensity: float = Field(description="指数平滑后的强度")
    raw_intensity: int = 0
    cause_domain: str = ""
    cause_detail: str = ""
    trace_id: str | None = None


class StudentProfile(BaseModel):
    """心理侧写（Postgres，跨轮稳定结论）。

    三个维度（情绪 / 认知 / 需求与优势）：

    * **情绪**：``intensity_ema``（强度平滑）、``emotion_trend``（趋势）、
      ``last_cause`` / ``last_emotion``（上一次诱因与情绪）。
    * **认知**：``stable_distortions``（不合理认知，3 次验证）、
      ``positive_cognitions``（积极认知资源，3 次验证）、
      ``attribution_style``（归因风格 / 认知模式）。
    * **需求与优势**：``unmet_needs``、``strengths``（VIA 品格优势）。
    """

    student_id: str
    grade: str | None = None
    aspiration: str | None = None
    strengths: list[str] = Field(default_factory=list, description="VIA 品格优势 Top3")
    via_strengths: dict[str, int] = Field(
        default_factory=dict,
        description="VIA 品格优势 → 证据计数（从学生原话提取）",
    )
    stable_distortions: list[str] = Field(
        default_factory=list,
        description="已三次验证过的认知偏差",
    )
    positive_cognitions: list[str] = Field(
        default_factory=list,
        description="已三次验证过的积极认知资源（成长型思维/积极归因/自我接纳）",
    )
    attribution_style: str = Field(
        default="",
        description="归因风格 / 认知模式总结（积极归因 / 消极归因 / 尚未明确）",
    )
    unmet_needs: list[str] = Field(default_factory=list, description="没被满足的需求")
    risk_level: str = "low"
    last_cause: str | None = None
    last_emotion: str | None = None
    intensity_ema: float = Field(
        default=0.0,
        description="情绪强度的指数平滑值：0.7*旧值 + 0.3*本轮",
    )
    emotion_trend: str = Field(
        default="平稳",
        description="情绪变化趋势：上升 / 下降 / 平稳（连续对话跟踪）",
    )
    evidence_counts: dict[str, int] = Field(default_factory=dict)
    positive_cognition_counts: dict[str, int] = Field(
        default_factory=dict,
        description="积极认知资源 → 证据计数",
    )
    updated_at: float = 0.0


class ProfileSummary(BaseModel):
    """侧写摘要：固定放进 Prompt 的那段文本。"""

    student_id: str
    grade: str | None = None
    aspiration: str | None = None
    strengths: list[str] = Field(default_factory=list)
    stable_distortions: list[str] = Field(default_factory=list)
    positive_cognitions: list[str] = Field(default_factory=list)
    attribution_style: str = ""
    unmet_needs: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    last_cause: str | None = None
    emotion_trend: str = "平稳"

    def to_prompt_text(self) -> str:
        if not any(
            [
                self.grade,
                self.aspiration,
                self.strengths,
                self.stable_distortions,
                self.positive_cognitions,
                self.unmet_needs,
                self.last_cause,
            ],
        ):
            return "（暂无侧写摘要）"
        lines = ["【心理侧写摘要】"]
        if self.grade:
            lines.append(f"- 年级/升学目标：{self.grade}")
        if self.aspiration:
            lines.append(f"- 升学目标：{self.aspiration}")
        if self.strengths:
            lines.append(f"- 优势：{'、'.join(self.strengths)}")
        if self.stable_distortions:
            lines.append(f"- 稳定认知偏差：{'、'.join(self.stable_distortions)}")
        if self.positive_cognitions:
            lines.append(f"- 积极认知资源：{'、'.join(self.positive_cognitions)}")
        if self.attribution_style:
            lines.append(f"- 归因风格：{self.attribution_style}")
        if self.unmet_needs:
            lines.append(f"- 未被满足的需求：{'、'.join(self.unmet_needs)}")
        if self.last_cause:
            lines.append(f"- 上一次情绪诱因：{self.last_cause}")
        if self.emotion_trend and self.emotion_trend != "平稳":
            lines.append(f"- 情绪趋势：{self.emotion_trend}")
        lines.append(f"- 风险等级：{self.risk_level}")
        return "\n".join(lines)


class SessionContext(BaseModel):
    """本轮交给 Worker 的会话上下文。"""

    session_id: str
    student_id: str
    current_text: str
    recent_turns: list[Turn] = Field(default_factory=list)
    profile: ProfileSummary | None = None
    reflection_hint: str | None = Field(default=None, description="上一轮反思提醒")


# --------------------------------------------------------------------------
# 编排器
# --------------------------------------------------------------------------
class ToolCallLog(BaseModel):
    """一次工具调用记录（trace）。"""

    trace_id: str
    seq: int
    tool_name: str
    key_args: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    duration_ms: float = 0.0
    detail: str = ""
    ts: float = 0.0


class OrchestratorDecision(BaseModel):
    """Tool Router 的决策记录。"""

    route: Route
    reason: str
    risk_verdict: RiskVerdict
    intervention_fit: InterventionFit | None = None
    speaker: Speaker = Speaker.学生
    skipped_tools: list[str] = Field(default_factory=list)


class EscalationRecord(BaseModel):
    """escalate_risk 上报记录（只传 student_id / 风险等级 / 原因码，不传原话）。"""

    student_id: str
    risk_level: str
    reason_code: str
    trace_id: str
    delivered: bool = False
    delivery_status: str = "pending"
    channel: str = ""
    notified: list[str] = Field(default_factory=list)
    ts: float = 0.0


class ReflectionResult(BaseModel):
    """反思模块产出。"""

    trace_id: str
    issues: list[str] = Field(default_factory=list)
    passed: bool = True
    blocking: bool = False
    hint_for_next_turn: str | None = None


class ChatOutcome(BaseModel):
    """一轮完整对话的落库/返回产物。"""

    trace_id: str
    session_id: str
    student_id: str
    reply: str = ""
    route: Route = Route.REACT
    emotion: EmotionResult | None = None
    needs: NeedResult | None = None
    risk_verdict: RiskVerdict | None = None
    retrieved: list[RetrievedCard] = Field(default_factory=list)
    #: 理论卡（独立通道检索，不占策略卡的档位 K）
    theory: list[RetrievedCard] = Field(default_factory=list)
    exercise: ExerciseCard | None = None
    escalation: EscalationRecord | None = None
    reflection: ReflectionResult | None = None
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    ttft_ms: float = 0.0
    total_ms: float = 0.0


# --------------------------------------------------------------------------
# API 契约
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """``POST /api/chat`` 请求体。"""

    session_id: str = Field(min_length=1, max_length=128)
    student_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=4000)
    speaker: Speaker = Speaker.学生
    seed_profile: StudentProfile | None = None


class ChatResponse(BaseModel):
    """``POST /api/chat`` 同步响应体。"""

    trace_id: str
    session_id: str
    student_id: str
    reply: str
    route: Route
    risk_level: str = "low"
    emotion: EmotionResult | None = None
    needs: NeedResult | None = None
    exercise: ExerciseCard | None = None
    escalation: EscalationRecord | None = None
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    ttft_ms: float = 0.0
    total_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    model_provider: str
    model_name: str
    memory_backend: str
    risk_threshold: float


__all__ = [
    "CauseDomain",
    "ChatOutcome",
    "ChatRequest",
    "ChatResponse",
    "Collection",
    "EmotionCore",
    "EmotionResult",
    "EmotionTrajectoryItem",
    "EscalationRecord",
    "ExerciseCard",
    "HealthResponse",
    "InterventionFit",
    "LexiconHit",
    "NeedResult",
    "OrchestratorDecision",
    "ProfileSummary",
    "ReflectionResult",
    "RetrievedCard",
    "RetrievalResult",
    "RiskScore",
    "RiskSource",
    "RiskVerdict",
    "Route",
    "SessionContext",
    "Speaker",
    "StrategyCard",
    "StructuredQuery",
    "StudentProfile",
    "ToolCallLog",
    "Turn",
    "WorkerName",
]
