"""提示词与提示构造。

``build_user_message`` 是唯一的上下文拼装入口：把「12 轮原话 + 侧写摘要 +
情绪 JSON + 检索到的卡」四样东西同时塞进用户消息 —— 既给人看的可读部分，
也给规则模型用的 ``<<CTX>>`` JSON 块。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from app.prompts.templates import (
    EMOTION_SYSTEM_PROMPT,
    GUIDANCE_SYSTEM_PROMPT,
    NEEDS_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
    SAFETY_SYSTEM_PROMPT,
)
from app.schemas import EmotionResult, NeedResult, ProfileSummary, RetrievedCard, Turn


def _dump(value: Any) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_user_message(
    *,
    text: str,
    recent_turns: Sequence[Turn] = (),
    profile: ProfileSummary | None = None,
    emotion: EmotionResult | None = None,
    needs: NeedResult | None = None,
    cards: Sequence[RetrievedCard] = (),
    theory_cards: Sequence[RetrievedCard] = (),
    exercise_text: str | None = None,
    reflection_hint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """把本轮上下文拼成一条用户消息。"""
    ctx: dict[str, Any] = {
        "text": text,
        "recent_texts": [t.content for t in recent_turns if t.role == "student"],
        "recent_turns": [
            {"role": t.role, "content": t.content, "ts": t.ts} for t in recent_turns
        ],
        "profile": profile.model_dump(mode="json") if profile else None,
        "emotion": emotion.model_dump(mode="json") if emotion else None,
        "needs": needs.model_dump(mode="json") if needs else None,
        "cards": [
            {
                "card_id": c.card_id,
                "collection": c.collection,
                "scene": c.scene,
                "title": c.title,
                "text": c.text,
                "payload": c.payload,
                "final_score": c.final_score,
            }
            for c in cards
        ],
        "theory_cards": [
            {
                "card_id": c.card_id,
                "collection": c.collection,
                "scene": c.scene,
                "title": c.title,
                "text": c.text,
                "final_score": c.final_score,
            }
            for c in theory_cards
        ],
        "exercise_text": exercise_text,
        "reflection_hint": reflection_hint,
    }
    if extra:
        ctx.update(extra)

    blocks: list[str] = []
    blocks.append("<<CTX>>")
    blocks.append(_dump(ctx))
    blocks.append("<<END>>")
    blocks.append("")
    blocks.append("【学生这一轮的原话】")
    blocks.append(text)
    if recent_turns:
        blocks.append("")
        blocks.append("【最近 12 轮原话】")
        for turn in recent_turns:
            role = {"student": "学生", "assistant": "陪伴助手", "teacher": "老师"}.get(
                turn.role, turn.role
            )
            blocks.append(f"{role}：{turn.content}")
    if profile is not None:
        blocks.append("")
        blocks.append(profile.to_prompt_text())
    if emotion is not None:
        blocks.append("")
        blocks.append("【情绪识别结果】")
        blocks.append(_dump(emotion))
    if needs is not None:
        blocks.append("")
        blocks.append("【需求澄清结果】")
        blocks.append(_dump(needs))
    if cards:
        blocks.append("")
        blocks.append("【本轮检索到的策略卡】")
        for card in cards:
            blocks.append(f"- {card.card_id}（{card.collection} / {card.scene}）：{card.text}")
    if theory_cards:
        blocks.append("")
        blocks.append("【可引用的心理学理论依据】")
        blocks.append(
            "以下是与本轮情绪/需求相关的积极心理学理论与研究证据。"
            "仅在与当前回复真正相关时才引用（一句话带出即可），"
            "不要堆砌、不要为了显示专业而强行引用；不相关就完全不要提。"
        )
        for card in theory_cards:
            blocks.append(f"- {card.card_id}（{card.title}）：{card.text}")
    if exercise_text:
        blocks.append("")
        blocks.append("【练习卡】")
        blocks.append(exercise_text)
    if reflection_hint:
        blocks.append("")
        blocks.append("【上一轮的复盘提醒】")
        blocks.append(reflection_hint)

    return "\n".join(blocks)


__all__ = [
    "EMOTION_SYSTEM_PROMPT",
    "GUIDANCE_SYSTEM_PROMPT",
    "NEEDS_SYSTEM_PROMPT",
    "RISK_SYSTEM_PROMPT",
    "SAFETY_SYSTEM_PROMPT",
    "build_user_message",
]
