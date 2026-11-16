"""合成评测集生成器。

⚠️ 重要说明
-----------
原文档给了评测口径与指标（检索 200 条查询的 Top5 命中率 62% → 82%；
侧写事实召回 60% → 81%），但**没有给原始数据**。本脚本生成的是**合成评测集**，
用来在本仓库里复现"同样的评测方法"与"同样的趋势"。

因此：

* 绝对数值**不等价于**原系统的数字（向量器、重排器、卡片库都不同）；
* 相对关系（加入 BM25 + RRF + 重排后命中率的提升；把侧写摘要固定进 Prompt
  后事实召回的提升）是本脚本真正要验证的东西。

用法::

    ./.venv/Scripts/python.exe scripts/gen_eval_data.py            # 默认 200 + 50
    ./.venv/Scripts/python.exe scripts/gen_eval_data.py --queries 200 --dialogs 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.index import build_default_index  # noqa: E402
from app.rag.query import SCENE_KEYWORDS  # noqa: E402

EVAL_DIR = ROOT / "eval"

#: 通用认知偏差（当某张卡没有专属偏差时用）
GENERAL_DISTORTIONS = ["以偏概全", "灾难化", "非黑即白", "过度自责", "读心术"]

#: 每个场景的"同义改写"——不直接包含卡片正文里的原词，用来制造向量召回的困难样本
PARAPHRASES: dict[str, list[str]] = {
    "exam_failure": ["月考没考好", "分数不理想", "这次测验砸了"],
    "exam_anxiety": ["快考试了心慌", "面对考试很紧张", "考前来不及了"],
    "exam_progress": ["这次考得比上次好", "成绩往上走了", "分数有进步"],
    "peer_conflict": ["和同学闹别扭", "跟朋友处不来", "被同班的人排挤"],
    "family_conflict": ["和家里人吵起来", "父母不理解我", "在家待着难受"],
    "self_doubt": ["觉得自己很没用", "怀疑自己能力", "总觉得自己不行"],
    "loneliness": ["身边没个说话的人", "没人可倾诉", "一个人很没意思"],
    "bullying": ["被同学欺负", "在班上被针对", "被人欺负过"],
    "subject_pressure": ["选科的事很纠结", "分科方向定不下来", "文科理科怎么选"],
    "sports_failure": ["体测没过", "比赛输了", "跑步成绩不行"],
    "appearance": ["嫌自己不好看", "身材很在意", "照镜子越看越烦"],
    "friend_leaving": ["好朋友要转走", "同窗要分开", "认识的人要走了"],
    "club_competition": ["社团没选上", "名额被别人拿了", "没进想进的社团"],
    "teacher_misunderstanding": ["被老师冤枉了", "被老师误解", "老师批评我不对"],
    "dormitory": ["和室友处不来", "宿舍里老有矛盾", "室友作息跟我冲突"],
    "parents_control": ["爸妈管我很严", "手机被收了", "出门被限制"],
    "procrastination": ["作业一直拖着", "总是拖到 deadline", "计划做不完"],
    "positive_progress": ["做成了一件难事", "这次表现不错", "完成了一个挑战"],
}

#: 练习卡场景（评测集不用，但保持场景词表完整）
EXERCISE_SCENES = {
    "exam_anxiety",
    "exam_failure",
    "self_doubt",
    "peer_conflict",
    "family_conflict",
    "loneliness",
    "appearance",
    "subject_pressure",
    "procrastination",
}


# --------------------------------------------------------------------------
# 检索评测集
# --------------------------------------------------------------------------
def build_retrieval_eval(count: int, seed: int) -> list[dict]:
    """生成 ``count`` 条查询，每条标一张金标主卡。"""
    rng = random.Random(seed)
    settings = get_settings()
    index = build_default_index(settings.data_dir)
    reply_cards = index.by_collection("student_reply")

    # 把每张回应卡对应的禁止说法卡也放进来（真实召回池就是两张卡一起）
    ban_cards = index.by_collection("ban")

    queries: list[dict] = []
    per_card = max(1, count // len(reply_cards))
    for card in reply_cards:
        keywords_pool = SCENE_KEYWORDS.get(card.scene, []) + PARAPHRASES.get(card.scene, [])
        card_bans = next(
            (b for b in ban_cards if b.scene == card.scene), None
        )
        for i in range(per_card):
            paraphrase = keywords_pool[i % len(keywords_pool)] if keywords_pool else ""
            distortion = rng.choice(
                card.payload.get("applicable_distortions") or GENERAL_DISTORTIONS
            )
            fit = rng.choice(card.payload.get("applicable_fit") or ["轻量引导"])
            steps = list(card.payload.get("steps") or [])
            rng.shuffle(steps)
            # 一半查询带上"精确术语"，另一半只带同义改写 —— 这正是
            # 文档里"BM25 靠逐字命中、向量找不到那么准"的那类样本
            exact = i % 2 == 0
            query_keywords = [paraphrase]
            if exact:
                query_keywords.insert(0, distortion)
            queries.append(
                {
                    "id": f"rq_{card.card_id}_{i}",
                    "gold_card_id": card.card_id,
                    "scene": card.scene,
                    "collection": "student_reply",
                    "distortion": [distortion],
                    "fit": fit,
                    "steps": steps,
                    "keywords": [k for k in query_keywords if k],
                    "variant": "exact" if exact else "paraphrase",
                    "ban_card_id": card_bans.card_id if card_bans else None,
                }
            )
    # 补齐到 count
    idx = 0
    while len(queries) < count and reply_cards:
        card = reply_cards[idx % len(reply_cards)]
        distortion = rng.choice(GENERAL_DISTORTIONS)
        queries.append(
            {
                "id": f"rq_extra_{idx}",
                "gold_card_id": card.card_id,
                "scene": card.scene,
                "collection": "student_reply",
                "distortion": [distortion],
                "fit": rng.choice(["轻量引导", "情绪疏导"]),
                "steps": [],
                "keywords": [distortion],
                "variant": "exact",
                "ban_card_id": None,
            }
        )
        idx += 1
    return queries[:count]


# --------------------------------------------------------------------------
# 侧写召回评测集
# --------------------------------------------------------------------------
#: 每段对话 8 轮，三件早期事实**分散在不同轮次**埋下：
#: 第 1 轮埋「优势」（最早，会掉出 12 条原话的窗口），
#: 第 5 轮埋「年级」，第 6 轮埋「上一次情绪诱因」（近期，窗口内）。
#: 这样"只留 12 轮原话"时年级和诱因还答得出来、优势答不出来 ——
#: 正是文档描述的"早期优势经常已经不在窗口里"。
ROUND_TEMPLATE = [
    "我比较擅长画画，画了三年了",                       # 0: 优势（早期）
    "美术老师说过我有感觉，我在画室里其实挺自在的",       # 1
    "我每天都会画一会儿，就是最近一碰到考试就慌",         # 2
    "同学都说我在美术上有天赋，我自己也这么觉得",         # 3
    "对了，我{grade}",                                   # 4: 年级
    "这次月考失利我挺难受的，一直在想这件事",             # 5: 诱因（近期）
    "我本来想报美术专业的，就是成绩一直上不来",           # 6
    "每次想到月考失利我都难受，也不敢跟别人说",           # 7
]


def build_profile_eval(dialogs: int, seed: int) -> list[dict]:
    """生成 ``dialogs`` 段对话，每段 8 轮，埋三件早期事实。"""
    rng = random.Random(seed + 7)
    grades = ["高二", "高一", "初三", "高二", "初一"]
    variations = [
        "我比较擅长画画，画了三年了",
        "我挺擅长画画的，都画了三年了",
        "我擅长画画，已经画了三年了",
    ]

    out: list[dict] = []
    for d in range(dialogs):
        grade = grades[d % len(grades)]
        turns: list[dict] = []
        for r in range(8):
            template = ROUND_TEMPLATE[r]
            if r == 0:
                student = variations[d % len(variations)]
            else:
                student = template.format(grade=grade)
            turns.append({"role": "student", "content": student})
            turns.append(
                {
                    "role": "assistant",
                    "content": "我在听你说，你愿意多讲一点吗？",
                }
            )
        out.append(
            {
                "id": f"dlg_{d}",
                "turns": turns,
                "gold": {
                    "grade": grade,
                    "strength": "画画",
                    "cause": "月考失利",
                },
            }
        )
    return out


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成合成评测集")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dialogs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.eval_seed

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    retrieval = build_retrieval_eval(args.queries, seed)
    path = EVAL_DIR / "retrieval_eval.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for item in retrieval:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(retrieval)} 条查询")

    dialogs = build_profile_eval(args.dialogs, seed)
    path2 = EVAL_DIR / "profile_eval.jsonl"
    with path2.open("w", encoding="utf-8") as fh:
        for item in dialogs:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{path2.relative_to(ROOT)}: {len(dialogs)} 段对话 × 8 轮")

    exact = sum(1 for q in retrieval if q["variant"] == "exact")
    print(f"  其中精确术语样本 {exact} 条 / 同义改写样本 {len(retrieval) - exact} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
