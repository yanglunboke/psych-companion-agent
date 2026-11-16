"""规则引擎测试：情绪识别 / 需求澄清 / 回复成稿 / 安全回应。"""

from __future__ import annotations

from app.models.rule_engine import (
    compose_reply,
    compose_safety_reply,
    infer_emotion,
    infer_needs,
)
from app.schemas import CauseDomain, InterventionFit


def test_exam_failure_is_depressed_academic() -> None:
    result = infer_emotion("这次月考又考砸了，我真的太笨了")
    assert result.emotion_type == "沮丧"
    assert result.cause_domain == CauseDomain.学业
    assert result.intensity == 4
    assert "以偏概全" in result.candidate_distortions
    assert result.risk_score < 0.72


def test_self_harm_statement_pushes_risk_over_threshold() -> None:
    result = infer_emotion("我不想活了，真的撑不下去了")
    assert result.risk_score >= 0.72
    assert result.emotion_type == "绝望"


def test_violence_statement_pushes_risk_over_threshold() -> None:
    result = infer_emotion("我真的想打他一顿，想弄死他")
    assert result.risk_score >= 0.72


def test_minor_abuse_statement_pushes_risk_over_threshold() -> None:
    result = infer_emotion("他们总是打我，我好害怕")
    assert result.risk_score >= 0.72


def test_only_talk_request_is_detected() -> None:
    emotion = infer_emotion("我只想说说，没什么大事")
    needs = infer_needs("我只想说说，没什么大事", emotion)
    assert needs.intervention_fit == InterventionFit.只想说说


def test_looking_for_method_maps_to_need_practice() -> None:
    emotion = infer_emotion("一到考试我就紧张得睡不着，有没有方法能缓解")
    needs = infer_needs("一到考试我就紧张得睡不着，有没有方法能缓解", emotion)
    assert needs.intervention_fit == InterventionFit.需要练习


def test_reply_follows_four_step_structure() -> None:
    emotion = infer_emotion("这次月考又考砸了，我真的太笨了")
    needs = infer_needs("这次月考又考砸了，我真的太笨了", emotion)
    reply = compose_reply("这次月考又考砸了，我真的太笨了", emotion, needs)
    # 共情接纳
    assert "我听到了" in reply
    # 拆解：把事件与自我评价分开
    assert "拆开看" in reply
    # 韧性
    assert "韧性" in reply or "较劲" in reply
    # 只问一个问题
    question_marks = reply.count("？")
    assert question_marks == 1


def test_banned_phrases_are_removed() -> None:
    from app.schemas import RetrievedCard

    emotion = infer_emotion("这次月考又考砸了，我真的太笨了")
    needs = infer_needs("这次月考又考砸了，我真的太笨了", emotion)
    ban = RetrievedCard(
        card_id="ban_exam_failure",
        collection="ban",
        scene="exam_failure",
        title="禁止说法",
        text="一次考试有什么大不了",
        payload={"bans": ["一次考试有什么大不了", "你就是不够努力"]},
    )
    reply = compose_reply("这次月考又考砸了，我真的太笨了", emotion, needs, cards=[ban])
    assert "一次考试有什么大不了" not in reply


def test_safety_reply_is_short_and_has_no_method() -> None:
    reply = compose_safety_reply()
    assert len(reply) < 220
    assert "老师" in reply
    assert "练习" not in reply
    assert "深呼吸" not in reply
