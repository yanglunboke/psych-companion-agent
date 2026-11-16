"""侧写事实召回评测。

文档口径：

    35%的提升评的是 50 段对话、每段 8 轮，最后问三件早期事实：
    年级或升学目标、上一轮情绪诱因、学生说过的一个优势。
    只留 12 轮原话时，早期优势经常已经不在窗口里，答对大约 60%。
    把侧写摘要固定放进 Prompt 后，答对大约 81%。

本脚本在**合成评测集**上跑四个配置：

    A1  最近 12 条原话
    A2  最近 12 轮原话（= 24 条，本项目的 Redis 窗口默认）
    B1  最近 12 条原话 + 侧写摘要
    B2  最近 12 轮原话 + 侧写摘要

「原话」这里按**消息条数**理解：文档说"最近 12 轮原话"，而"原话"本身指
学生/助手说过的一句话；8 轮对话共 16 条，超过 12 条窗口后第 1 轮掉出窗口，
早期事实随之消失——这正是文档描述的现象。两种口径都跑，结论一目了然。

⚠️ 本评测衡量的是**"该信息在上下文里还在不在"**（信息可得性），不是大模型的
理解能力。答题用规则抽取完成，便于复现。

用法::

    ./.venv/Scripts/python.exe scripts/gen_eval_data.py
    ./.venv/Scripts/python.exe scripts/eval_profile_recall.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.memory.profile import extract_grade, extract_strengths, to_summary, update_profile  # noqa: E402
from app.models.rule_engine import infer_emotion  # noqa: E402
from app.schemas import EmotionResult, StudentProfile  # noqa: E402

EVAL_FILE = ROOT / "eval" / "profile_eval.jsonl"

#: 诱因关键词（按特异度排序）
CAUSE_KEYWORDS = ("月考失利", "考试失利", "考砸", "月考", "考试", "成绩")


def _answer_grade(texts: list[str]) -> str:
    for text in texts:
        grade = extract_grade(text)
        if grade:
            return grade
    return ""


def _answer_strength(texts: list[str]) -> str:
    for text in texts:
        strengths = extract_strengths(text)
        if strengths:
            return strengths[0]
    return ""


def _answer_cause(texts: list[str]) -> str:
    joined = " ".join(texts)
    for keyword in CAUSE_KEYWORDS:
        if keyword in joined:
            return keyword
    return ""


def _build_profile(turns: list[dict], student_id: str) -> StudentProfile:
    """把整段对话跑一遍情绪识别与侧写聚合（端到端，不手工填）。"""
    profile = StudentProfile(student_id=student_id)
    for turn in turns:
        if turn["role"] != "student":
            continue
        result = infer_emotion(turn["content"])
        emotion = EmotionResult(
            emotion_type=result.emotion_type,
            intensity=result.intensity,
            cause_domain=result.cause_domain,
            cause_detail=result.cause_detail,
            candidate_distortions=list(result.candidate_distortions),
            risk_score=result.risk_score,
            risk_reason=result.risk_reason,
            confidence=result.confidence,
        )
        profile = update_profile(profile, emotion, student_text=turn["content"])
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="侧写事实召回评测")
    parser.add_argument("--json", default="", help="同时把报告写到该 JSON 文件")
    args = parser.parse_args(argv)

    if not EVAL_FILE.exists():
        print(f"找不到评测集 {EVAL_FILE}，请先运行 scripts/gen_eval_data.py")
        return 1
    dialogs = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not dialogs:
        print("评测集为空")
        return 1

    configs = {
        "A1 只留最近 12 条原话": {"messages": 12, "use_profile": False},
        "A2 只留最近 12 轮原话": {"messages": 24, "use_profile": False},
        "B1 12 条原话 + 侧写摘要": {"messages": 12, "use_profile": True},
        "B2 12 轮原话 + 侧写摘要": {"messages": 24, "use_profile": True},
    }
    questions = ("grade", "strength", "cause")
    counters = {name: {q: 0 for q in questions} for name in configs}

    for dialog in dialogs:
        turns = dialog["turns"]
        gold = dialog["gold"]
        profile = _build_profile(turns, dialog["id"])
        summary_text = to_summary(profile).to_prompt_text()

        for name, cfg in configs.items():
            window = turns[-cfg["messages"] :]
            texts = [t["content"] for t in window]
            if cfg["use_profile"]:
                texts = texts + [summary_text]
            answers = {
                "grade": _answer_grade(texts),
                "strength": _answer_strength(texts),
                "cause": _answer_cause(texts),
            }
            for q in questions:
                if gold[q] and gold[q] in answers[q]:
                    counters[name][q] += 1

    total = len(dialogs)
    print("=" * 72)
    print(f"侧写事实召回评测（合成评测集，{total} 段对话 × 8 轮，每段问 3 件早期事实）")
    print("=" * 72)
    header = f"  {'配置':26s} {'年级/升学目标':>14s} {'优势':>10s} {'上一次诱因':>12s} {'合计':>10s}"
    print(header)
    print("-" * 72)
    for name in configs:
        c = counters[name]
        rates = {q: c[q] / total * 100 for q in questions}
        overall = sum(rates.values()) / 3
        print(
            f"  {name:26s} {rates['grade']:13.1f}% {rates['strength']:9.1f}%"
            f" {rates['cause']:11.1f}% {overall:9.1f}%"
        )
    print("-" * 72)
    a1 = sum(counters["A1 只留最近 12 条原话"].values()) / (total * 3) * 100
    b1 = sum(counters["B1 12 条原话 + 侧写摘要"].values()) / (total * 3) * 100
    print(f"  12 条原话：{a1:.1f}%  →  加侧写摘要：{b1:.1f}%   （{b1 - a1:+.1f} 个百分点）")
    print("-" * 72)
    print("注：本评测衡量的是\"信息在上下文里还在不在\"（信息可得性），不是模型的")
    print("    理解能力；答题用规则抽取完成，便于复现。绝对值不等价于原系统")
    print("    （60% → 81%），可参考的是\"加侧写摘要\"带来的相对提升。")

    if args.json:
        report = {
            "total": total,
            "configs": {
                name: {
                    **{
                        q: counters[name][q] / total for q in questions
                    },
                    "overall": sum(counters[name].values()) / (total * 3),
                }
                for name in configs
            },
            "improvement_12_messages_pp": b1 - a1,
        }
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
