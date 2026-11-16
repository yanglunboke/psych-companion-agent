"""安全门测试：词表命中、抑制规则、双路裁决。"""

from __future__ import annotations

from app.config import get_settings
from app.orchestrator.safety import Lexicon, SafetyGate
from app.schemas import RiskSource


def _gate() -> SafetyGate:
    settings = get_settings()
    return SafetyGate(Lexicon.from_file(settings.lexicon_file), threshold=settings.risk_threshold)


def test_self_harm_phrase_triggers_lexicon() -> None:
    hit = _gate().scan("我不想活了，真的撑不下去了")
    assert hit.hit is True
    assert hit.reason_code == "self_harm_statement"


def test_harm_others_triggers_lexicon() -> None:
    hit = _gate().scan("我真的想打他一顿，想弄死他")
    assert hit.hit is True
    assert hit.reason_code == "harm_others"


def test_minor_safety_triggers_lexicon() -> None:
    hit = _gate().scan("他们总是打我，我好害怕")
    assert hit.hit is True
    assert hit.reason_code == "minor_safety"


def test_normal_sentence_does_not_trigger() -> None:
    hit = _gate().scan("这次月考又考砸了，我真的太笨了")
    assert hit.hit is False
    assert hit.reason_code is None


def test_protective_expression_is_suppressed() -> None:
    """保护性表达不应被当成自杀意图。"""
    hit = _gate().scan("我不想死，我只是想说我最近很痛苦")
    assert hit.hit is False


def test_verdict_uses_lexicon_path() -> None:
    gate = _gate()
    hit = gate.scan("我想自杀")
    verdict = gate.verdict(hit, risk_score=0.1)
    assert verdict.is_high_risk is True
    assert verdict.source == RiskSource.LEXICON


def test_verdict_uses_risk_score_path() -> None:
    gate = _gate()
    verdict = gate.verdict(gate.scan("我很难过"), risk_score=0.72)
    assert verdict.is_high_risk is True
    assert verdict.source == RiskSource.MODEL
    assert verdict.reason_code == "risk_score_threshold"


def test_verdict_both_paths() -> None:
    gate = _gate()
    verdict = gate.verdict(gate.scan("我想自杀"), risk_score=0.9)
    assert verdict.is_high_risk is True
    assert verdict.source == RiskSource.BOTH


def test_verdict_low_risk() -> None:
    gate = _gate()
    verdict = gate.verdict(gate.scan("我今天有点烦"), risk_score=0.11)
    assert verdict.is_high_risk is False
    assert verdict.source == RiskSource.NONE


def test_threshold_boundary_is_inclusive() -> None:
    gate = _gate()
    assert gate.verdict(gate.scan("普通句子"), 0.72).is_high_risk is True
    assert gate.verdict(gate.scan("普通句子"), 0.719).is_high_risk is False
