"""检索评测：金标主卡是否进入最终 Top5。

文档口径：

    20%命中率提升看的是金标卡片进没进最终 Top 5。200 条按场景写的查询，
    每条人工标了一张主卡。只走向量大约 62%，加上 BM25、RRF 和重排大约 82%，
    绝对高 20 个百分点。

本脚本在**合成评测集**上跑四个配置，输出每一环的命中率：

    A  只走向量            向量召回 top20 → 取 top5
    B  向量 + BM25（RRF）  两路各 top20 → RRF 融合 → 取 top5
    C  B + 重排            读卡片正文重排 → 取 top5
    D  完整管线（含截断）  再按断崖 + 动态 K + token 预算截断

用法::

    ./.venv/Scripts/python.exe scripts/gen_eval_data.py
    ./.venv/Scripts/python.exe scripts/eval_retrieval.py
    ./.venv/Scripts/python.exe scripts/eval_retrieval.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.index import build_default_index  # noqa: E402
from app.rag.retriever import Retriever, RetrieverConfig  # noqa: E402
from app.schemas import StructuredQuery  # noqa: E402

EVAL_FILE = ROOT / "eval" / "retrieval_eval.jsonl"


def _load() -> list[dict]:
    if not EVAL_FILE.exists():
        print(f"找不到评测集 {EVAL_FILE}，请先运行 scripts/gen_eval_data.py")
        return []
    with EVAL_FILE.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _hit(ranked_ids: list[str], gold: str) -> bool:
    return gold in ranked_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索 Top5 命中率评测")
    parser.add_argument("--json", default="", help="同时把报告写到该 JSON 文件")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    items = _load()
    if not items:
        return 1

    settings = get_settings()
    index = build_default_index(settings.data_dir)
    retriever = Retriever(
        index,
        RetrieverConfig(
            recall_top_k=settings.recall_top_k,
            vector_min_score=settings.vector_min_score,
            rrf_k=settings.rrf_k,
            rerank_min_score=settings.rerank_min_score,
            rerank_top_k=settings.rerank_top_k,
            dynamic_k_by_fit=dict(settings.dynamic_k_by_fit),
            score_gap_threshold=settings.score_gap_threshold,
            card_char_budget=settings.card_char_budget,
        ),
    )

    counters = {"vector": 0, "rrf": 0, "rerank": 0, "final": 0}
    by_variant: dict[str, dict[str, int]] = {}
    details: list[dict] = []

    for item in items:
        query = StructuredQuery(
            collection=item["collection"],
            scene=item["scene"],
            distortion=item["distortion"],
            fit=item["fit"],
            steps=item["steps"],
            keywords=item["keywords"],
        )
        gold = item["gold_card_id"]
        variant = item["variant"]

        result = retriever.search(query, collections=["student_reply", "ban"])

        # A：只走向量
        vector_ids = [c.card_id for c in result.recalled_vector][: args.top_k]
        # B：RRF 融合后
        rrf_ids = [c.card_id for c in result.fused][: args.top_k]
        # C：重排后
        rerank_ids = [c.card_id for c in result.reranked][: args.top_k]
        # D：完整管线（断崖 + 动态 K + token 预算）
        final_ids = [c.card_id for c in result.selected]

        hits = {
            "vector": _hit(vector_ids, gold),
            "rrf": _hit(rrf_ids, gold),
            "rerank": _hit(rerank_ids, gold),
            "final": _hit(final_ids, gold),
        }
        for key, ok in hits.items():
            if ok:
                counters[key] += 1
        bucket = by_variant.setdefault(variant, {k: 0 for k in counters})
        bucket["total"] = bucket.get("total", 0) + 1
        for key, ok in hits.items():
            if ok:
                bucket[key] += 1
        details.append({"id": item["id"], "gold": gold, **hits})

    total = len(items)
    print("=" * 66)
    print(f"检索 Top{args.top_k} 命中率评测（合成评测集，{total} 条查询）")
    print("=" * 66)
    labels = {
        "vector": "A 只走向量",
        "rrf": "B 向量+BM25(RRF)",
        "rerank": "C  +重排",
        "final": "D  +断崖/动态K截断",
    }
    baseline = counters["vector"] / total * 100
    for key in ("vector", "rrf", "rerank", "final"):
        rate = counters[key] / total * 100
        delta = f"  ({rate - baseline:+.1f} pp)" if key != "vector" else ""
        print(f"  {labels[key]:22s} {rate:5.1f}%  ({counters[key]}/{total}){delta}")

    print("-" * 66)
    print("按样本类型：")
    for variant, bucket in sorted(by_variant.items()):
        n = bucket.get("total", 0)
        if not n:
            continue
        print(
            f"  {variant:12s} n={n:3d}  向量 {bucket['vector'] / n * 100:5.1f}%"
            f"   完整管线 {bucket['final'] / n * 100:5.1f}%"
        )

    # ---------- 阈值敏感度：说明 0.35 是按什么标定的 ----------
    print("-" * 66)
    print("向量召回阈值敏感度（只走向量这一路）：")
    original = retriever.config.vector_min_score
    embedder = retriever.index.embedder
    original_calibrated = getattr(embedder, "calibrated_min_score", None)
    effective_now = retriever.effective_vector_min_score()
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        # 临时关掉 embedder 自带标定，强制用文档口径的阈值来测
        embedder.calibrated_min_score = None
        retriever.config.vector_min_score = threshold
        hit = 0
        for item in items:
            query = StructuredQuery(
                collection=item["collection"],
                scene=item["scene"],
                distortion=item["distortion"],
                fit=item["fit"],
                steps=item["steps"],
                keywords=item["keywords"],
            )
            res = retriever.search(query, collections=["student_reply", "ban"])
            ids = [c.card_id for c in res.recalled_vector][: args.top_k]
            if item["gold_card_id"] in ids:
                hit += 1
        note = "  (文档口径)" if abs(threshold - 0.35) < 1e-9 else ""
        mark = ""
        if abs(threshold - effective_now) < 1e-9:
            mark = "  ← 本地默认生效值"
        print(f"    阈值 {threshold:.2f} → 命中率 {hit / total * 100:5.1f}%{note}{mark}")
    retriever.config.vector_min_score = original
    embedder.calibrated_min_score = original_calibrated

    print("-" * 66)
    print("注：本地向量器是稀疏哈希占位实现（余弦分布整体偏低），文档的 0.35 阈值是")
    print("    按稠密 embedding 模型标定的，直接套用会砍掉近九成召回；本地实现因此")
    print("    自带标定阈值。绝对值不等价于原系统（62% → 82%），可参考的是各配置")
    print("    之间的相对关系与阈值敏感度。")

    if args.json:
        report = {
            "total": total,
            "hit_rate": {k: counters[k] / total for k in counters},
            "by_variant": by_variant,
            "baseline_vector": counters["vector"] / total,
            "final_pipeline": counters["final"] / total,
            "improvement_pp": (counters["final"] - counters["vector"]) / total * 100,
        }
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
