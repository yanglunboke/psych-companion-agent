"""本地规则引擎。

这是一套**不依赖大模型**的领域推理实现，用两件事：

1. 离线默认模式下充当可解释的"模型"，让全链路（安全门 → Worker → 编排 →
   检索 → 记忆 → 上报）在没有任何 API Key 的情况下真实跑通，而不是返回假数据；
2. 给接真模型时提供**基线**：评测脚本用同一份规则跑对照组。

它刻意做成纯函数 + 无外部依赖，方便单测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from app.schemas import (
    CauseDomain,
    EmotionResult,
    ExerciseCard,
    InterventionFit,
    NeedResult,
    RetrievedCard,
    StudentProfile,
)


# --------------------------------------------------------------------------
# 词表
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """一条情绪规则。"""

    terms: tuple[str, ...]
    emotion_type: str
    cause: CauseDomain
    intensity: int
    weight: float = 1.0


#: 高危信号（与 app/data/lexicon/high_risk_lexicon.json 保持同源语义）
SELF_HARM_TERMS = (
    "自杀",
    "想死",
    "不想活",
    "活不下去",
    "结束生命",
    "跳楼",
    "割腕",
    "服毒",
    "轻生",
    "消失算了",
)
VIOLENCE_TERMS = (
    "想打他",
    "想打她",
    "想揍",
    "想杀",
    "弄死",
    "报复",
    "砍他",
    "拿刀",
)
ABUSE_TERMS = (
    "被欺负",
    "被霸凌",
    "被虐待",
    "被打",
    "打我",
    "被骂",
    "骂我",
    "侵犯",
    "猥亵",
    "强迫",
    "被威胁",
)

EMOTION_RULES: tuple[Rule, ...] = (
    Rule(SELF_HARM_TERMS, "绝望", CauseDomain.自我, 5, weight=3.0),
    Rule(VIOLENCE_TERMS, "愤怒", CauseDomain.人际, 5, weight=2.8),
    Rule(ABUSE_TERMS, "恐惧", CauseDomain.家庭, 4, weight=2.6),
    Rule(
        ("考砸", "考试失利", "月考", "期末", "成绩", "分数", "排名", "没考好", "挂了", "复读"),
        "沮丧",
        CauseDomain.学业,
        4,
        weight=1.2,
    ),
    Rule(
        ("作业", "补习", "上课", "老师", "考试", "学习", "背不下来", "写不完"),
        "焦虑",
        CauseDomain.学业,
        3,
    ),
    Rule(
        ("同学", "朋友", "没人理", "孤立", "被排斥", "吵架", "分手", "合不来", "被说"),
        "委屈",
        CauseDomain.人际,
        4,
        weight=1.1,
    ),
    Rule(
        ("爸妈", "父母", "家里", "父亲", "母亲", "爸爸", "妈妈", "爷爷奶奶"),
        "委屈",
        CauseDomain.家庭,
        4,
    ),
    Rule(
        ("我很笨", "我好笨", "我不行", "我真没用", "我什么都不会", "我是不是很差",
         "废物", "失败者", "一无是处"),
        "自责",
        CauseDomain.自我,
        4,
        weight=1.2,
    ),
    Rule(
        ("睡不着", "失眠", "紧张", "心跳快", "手抖", "压力", "喘不过气"),
        "焦虑",
        CauseDomain.自我,
        4,
    ),
    Rule(
        ("迷茫", "不知道该", "没方向", "未来", "怎么办", "该不该"),
        "迷茫",
        CauseDomain.自我,
        3,
    ),
    Rule(
        ("开心", "高兴", "考好了", "进步了", "我做到了"),
        "积极",
        CauseDomain.自我,
        2,
    ),
)

#: 认知偏差线索
DISTORTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("以偏概全", ("我不行", "我什么都不会", "我永远都", "每次都", "总是")),
    ("灾难化", ("完了", "彻底", "再也", "全毁了", "没救了", "人生完了")),
    ("非黑即白", ("要么成功要么失败", "不是……就是", "只有……才", "否则就")),
    ("过度自责", ("都怪我", "是我的错", "我害了", "我害得")),
    ("读心术", ("他们肯定觉得", "他们一定认为", "大家都讨厌我")),
)

#: 自我否定标签
SELF_LABEL_PATTERN = re.compile(
    r"我(?:真的|真|好|太|就是|就)?(?:是)?(?:很|太)?(笨|差|没用|不行|傻|废物|垃圾|失败者)",
)
#: 事件片段（用于"把一次事件和自我评价拆开"）
EVENT_PATTERN = re.compile(
    r"([^，。！？,.!?]{2,18}(?:考砸|失利|考差|没考好|考试|被骂|吵架|分开|受伤|生病))",
)

#: 需求信号
JUST_TALK_TERMS = ("只想说", "想聊聊", "说说而已", "没什么事", "就是想说说", "随便聊聊")
METHOD_TERMS = ("怎么办", "有什么方法", "教我", "技巧", "练习", "怎么缓解", "怎么放松", "帮我改")


# --------------------------------------------------------------------------
# 情绪识别
# --------------------------------------------------------------------------
def _match(text: str, terms: Iterable[str]) -> list[str]:
    return [t for t in terms if t in text]


def infer_emotion(
    text: str,
    recent_texts: Sequence[str] = (),
    *,
    confidence: float = 0.75,
) -> EmotionResult:
    """从学生这句话（可选结合最近几轮）推断情绪结构化结果。

    规则命中越多、权重越高，情绪越强；风险分由高危信号 + 情绪强度 +
    认知偏差共同决定，高危信号直接顶到阈值以上。
    """
    corpus = " ".join([text, *recent_texts[-3:]])

    hits: list[tuple[Rule, list[str]]] = []
    for rule in EMOTION_RULES:
        matched = _match(corpus, rule.terms)
        if matched:
            hits.append((rule, matched))

    if not hits:
        return EmotionResult(
            emotion_type="平静",
            intensity=1,
            cause_domain=CauseDomain.未知,
            cause_detail="未命中明显情绪线索",
            candidate_distortions=[],
            risk_score=0.03,
            risk_reason="无高危信号、无明显情绪扰动",
            confidence=0.5,
        )

    # 取权重最高的一条作为主情绪，同权重时取强度更高的
    best_rule, matched = max(hits, key=lambda item: (item[0].weight, item[0].intensity))

    distortions = [
        name for name, terms in DISTORTION_RULES if _match(corpus, terms)
    ]
    # 出现"我很笨/我太笨"这类自我否定标签时，本身就是以偏概全的信号
    if _extract_self_label(corpus) and "以偏概全" not in distortions:
        distortions.insert(0, "以偏概全")

    high_risk_hit = _match(text, SELF_HARM_TERMS) or _match(text, VIOLENCE_TERMS) or _match(
        text, ABUSE_TERMS
    )
    if high_risk_hit:
        risk_score = 0.95 if _match(text, SELF_HARM_TERMS) else 0.90
        risk_reason = f"命中高危信号：{'、'.join(high_risk_hit)}"
    else:
        risk_score = round(
            min(0.60, 0.04 * best_rule.intensity + 0.05 * len(distortions) + 0.02 * len(hits)),
            3,
        )
        risk_reason = (
            f"情绪强度 {best_rule.intensity}/5，认知偏差 {len(distortions)} 项，无高危信号"
        )

    return EmotionResult(
        emotion_type=best_rule.emotion_type,
        intensity=best_rule.intensity,
        cause_domain=best_rule.cause,
        cause_detail=f"{best_rule.terms[0]}（命中：{'、'.join(matched[:3])}）",
        candidate_distortions=distortions,
        risk_score=risk_score,
        risk_reason=risk_reason,
        confidence=confidence,
    )


# --------------------------------------------------------------------------
# 需求澄清
# --------------------------------------------------------------------------
def _profile_field(profile: object, name: str, default: Any = None) -> Any:
    """兼容 pydantic 对象与 dict 两种侧写形态。"""
    if profile is None:
        return default
    if isinstance(profile, dict):
        value = profile.get(name, default)
    else:
        value = getattr(profile, name, default)
    return default if value is None else value


def infer_needs(
    text: str,
    emotion: EmotionResult,
    profile: StudentProfile | None = None,
) -> NeedResult:
    """判断学生到底需要什么，以及该走哪一档干预。"""
    needs: list[str] = []

    if emotion.emotion_type in ("自责", "沮丧") or _match(text, ("我很笨", "我不行")):
        needs.append("被理解")
        needs.append("成就感")
    if emotion.emotion_type in ("焦虑", "迷茫", "恐惧"):
        needs.append("安全感")
        needs.append("方法")
    if emotion.emotion_type in ("委屈",):
        needs.append("被理解")
        needs.append("归属感")
    if emotion.emotion_type in ("愤怒",):
        needs.append("被公平对待")
    if not needs:
        needs.append("被倾听")

    if _match(text, JUST_TALK_TERMS):
        fit = InterventionFit.只想说说
        reasoning = "学生明确表示只想说说，不应主动给方法"
    elif _match(text, METHOD_TERMS) or (
        emotion.emotion_type == "焦虑" and emotion.intensity >= 4
    ):
        fit = InterventionFit.需要练习
        reasoning = "学生明确在找可操作的方法，适合给结构化练习"
    elif emotion.intensity >= 4:
        fit = InterventionFit.轻量引导
        reasoning = "情绪强度较高但仍在倾诉语境，给轻量引导而非完整方案"
    else:
        fit = InterventionFit.情绪疏导
        reasoning = "中等强度情绪，先做情绪疏导"

    # 侧写里累计未满足的需求也带进来
    for need in _profile_field(profile, "unmet_needs", []) or []:
        if need not in needs:
            needs.append(need)

    return NeedResult(needs=needs, intervention_fit=fit, reasoning=reasoning)


# --------------------------------------------------------------------------
# 积极引导回复
# --------------------------------------------------------------------------
def _extract_self_label(text: str) -> str | None:
    match = SELF_LABEL_PATTERN.search(text)
    return match.group(0) if match else None


def _extract_event(text: str) -> str | None:
    match = EVENT_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _acknowledge(text: str, emotion: EmotionResult) -> str:
    """第一步：共情接纳。"""
    core = text.strip().split("，")[0].split("。")[0].strip()
    if not core:
        core = "你说的这件事"
    return f"我听到了——{core}，这份{'难受' if emotion.intensity >= 3 else '在意'}是真的，不用急着把它压下去。"


def _reframe(text: str) -> str:
    """第二步：把「一次事件」和「自我评价」拆开。"""
    label = _extract_self_label(text)
    event = _extract_event(text)
    if label and event:
        return f"有一件事我想跟你拆开看：「{event}」是一次具体发生的事，「{label}」是你对自己的评价，它们不是同一件事。"
    if label:
        return f"有一件事我想跟你拆开看：「{label}」是此刻的自我感受，不等于你的真实水平。"
    return "我想先帮你把「发生了什么」和「你对自己的评价」分开看，这样你会更清楚一点。"


def _resilience(profile: StudentProfile | None) -> str:
    """第三步：点出韧性。"""
    strengths = _profile_field(profile, "strengths", []) or []
    if strengths:
        return f"另外，你之前说过自己擅长{strengths[0]}——能扛住这段时间的你，不是一个只会否定自己的人。"
    return "另外，能把这句话说出口，说明你还在跟这件事较劲，这份较劲本身就是韧性。"


def _one_question(text: str, emotion: EmotionResult, needs: NeedResult) -> str:
    """第四步：只问一个问题。"""
    if "被理解" in needs.needs:
        return "你更希望我先听你把这件事说完，还是我们一起拆一下刚才那句最扎心的话？"
    if "安全感" in needs.needs or "方法" in needs.needs:
        return "你现在最难受的是身体上的紧绷，还是脑子里的停不下来的想法？"
    return "此刻最想被回应的，是那件事本身，还是它带给你的那种感觉？"


def _apply_bans(reply: str, bans: Sequence[str]) -> str:
    """把禁止说法从草稿里剔除（命中则替换为中性表达）。"""
    for ban in bans:
        if ban and ban in reply:
            reply = reply.replace(ban, "（换成更贴近你的说法）")
    return reply


def _card_field(card: object, name: str, default: Any = None) -> Any:
    """兼容 RetrievedCard 对象与 dict 两种卡片形态。"""
    if isinstance(card, dict):
        value = card.get(name, default)
    else:
        value = getattr(card, name, default)
    return default if value is None else value


def compose_reply(
    text: str,
    emotion: EmotionResult,
    needs: NeedResult,
    cards: Sequence[RetrievedCard] = (),
    profile: StudentProfile | None = None,
) -> str:
    """按"共情接纳 → 拆解 → 韧性 → 只问一个问题"生成回复。

    若检索到了策略卡，其 ``must`` 步骤会被显式落到回复里；禁止说法卡在
    最后做一次剔除，保证"策略卡 + 禁止说法卡"确实作用于生成结果。
    """
    strategy_card = None
    ban_card = None
    for card in cards:
        collection = str(_card_field(card, "collection", ""))
        payload = _card_field(card, "payload", {}) or {}
        if collection == "student_reply" and payload.get("must") and strategy_card is None:
            strategy_card = card
        if collection == "ban" and payload.get("bans") and ban_card is None:
            ban_card = card

    parts: list[str] = [_acknowledge(text, emotion), _reframe(text), _resilience(profile)]

    if strategy_card is not None:
        must_steps = [
            str(s) for s in (_card_field(strategy_card, "payload", {}) or {}).get("must", [])
        ]
        if must_steps:
            parts.append("我打算这样陪你走一遍：" + "；".join(must_steps) + "。")

    parts.append(_one_question(text, emotion, needs))

    reply = "\n".join(parts)
    if ban_card is not None:
        bans = (_card_field(ban_card, "payload", {}) or {}).get("bans", [])
        reply = _apply_bans(reply, [str(b) for b in bans])
    return reply


def compose_safety_reply(emotion: EmotionResult | None = None) -> str:
    """高危轮的"简短安全回应"。

    要求：短、不讲道理、不做诊断、明确表示会有人帮忙，且不给练习建议。
    """
    return (
        "我在这儿，先别一个人扛。\n"
        "你刚才说的这些让我很担心，我需要现在就找一位老师一起帮你。\n"
        "这不是你做错了什么，我只是需要确认你是安全的。\n"
        "如果你此刻正有伤害自己的念头，请先离开窗边、远离可能伤到自己的东西，"
        "马上找身边可信的人陪着你，或者直接拨打 24 小时心理援助热线 12356。"
    )


def compose_teacher_reply(
    text: str,
    emotion: EmotionResult,
    needs: NeedResult,
) -> str:
    """老师发来的消息：回给老师，而不是回给学生。

    文档只说明 Tool Router 要看"说话的是学生还是老师"，没有定义老师侧口径。
    这里只陈述本轮**系统实际记录到的事实**（情绪、诱因、干预档位），
    不对老师做评判，也不替学生承诺任何事。
    """
    return (
        f"收到，已记录这名学生本轮的情况：情绪是{emotion.emotion_type}（强度 {emotion.intensity}/5），"
        f"诱因指向{emotion.cause_domain.value}，当前判断的干预档位是{needs.intervention_fit.value}。\n"
        "本轮的原文与侧写已写入轨迹，需要我把这名学生转介给心理老师时告诉我。"
    )


def compose_exercise_text(card: ExerciseCard) -> str:
    """练习卡转成给学生的文本。"""
    lines = [f"给你一个小练习：{card.name}"]
    if card.duration:
        lines.append(f"用时：{card.duration}")
    for idx, step in enumerate(card.steps, start=1):
        lines.append(f"{idx}. {step}")
    return "\n".join(lines)


__all__ = [
    "ABUSE_TERMS",
    "DISTORTION_RULES",
    "EMOTION_RULES",
    "SELF_HARM_TERMS",
    "VIOLENCE_TERMS",
    "compose_exercise_text",
    "compose_reply",
    "compose_safety_reply",
    "compose_teacher_reply",
    "infer_emotion",
    "infer_needs",
]
