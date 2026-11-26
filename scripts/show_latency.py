"""打印最近几轮对话的耗时瀑布，回答"这次慢在哪一步"。

用法：
    ./.venv/Scripts/python.exe scripts/show_latency.py            # 最近 5 轮
    ./.venv/Scripts/python.exe scripts/show_latency.py -n 10      # 最近 10 轮
    ./.venv/Scripts/python.exe scripts/show_latency.py --trace trace_abc123

耗时来自 ``tool_call_log.duration_ms``（编排器逐步计时）。
注意：``emotion_recognition`` 与 ``risk_scorer`` 是**并行**执行的，
两者相加会高估总耗时；实际那一段等于两者较大值。
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "psych_companion.db"

#: 并行执行的步骤：同组内取最大值，而不是相加
PARALLEL_GROUPS = [{"emotion_recognition", "risk_scorer"}]


def load_rows(db: Path, trace: str | None, limit: int) -> list[tuple[str, int, str, float, str]]:
    if not db.exists():
        raise SystemExit(f"找不到数据库：{db}")
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    if trace:
        cur.execute(
            "SELECT trace_id, seq, tool_name, duration_ms, detail, ts FROM tool_call_log "
            "WHERE trace_id LIKE ? ORDER BY seq",
            (trace + "%",),
        )
        return cur.fetchall()
    cur.execute(
        "SELECT trace_id FROM tool_call_log GROUP BY trace_id ORDER BY MAX(ts) DESC LIMIT ?",
        (limit,),
    )
    trace_ids = [row[0] for row in cur.fetchall()]
    rows: list[tuple] = []
    for tid in reversed(trace_ids):
        cur.execute(
            "SELECT trace_id, seq, tool_name, duration_ms, detail, ts FROM tool_call_log "
            "WHERE trace_id = ? ORDER BY seq",
            (tid,),
        )
        rows.extend(cur.fetchall())
    return rows


def effective_total(items: list[tuple[str, float]]) -> float:
    """把并行组的耗时折算成"实际墙钟"：组内取最大，组间相加。"""
    total = 0.0
    consumed: set[str] = set()
    for name, ms in items:
        if name in consumed:
            continue
        group = next((g for g in PARALLEL_GROUPS if name in g), None)
        if group is None:
            total += ms
            consumed.add(name)
            continue
        ids = {n for n, _ in items if n in group}
        total += max(ms for n, ms in items if n in group)
        consumed |= ids
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="对话耗时瀑布")
    parser.add_argument("-n", "--limit", type=int, default=5, help="最近几轮（默认 5）")
    parser.add_argument("--trace", help="只看某一轮（可只给前缀）")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 路径")
    parser.add_argument("--min-ms", type=int, default=1, help="低于该值的步骤不打印（默认 1ms）")
    args = parser.parse_args()

    rows = load_rows(Path(args.db), args.trace, args.limit)
    if not rows:
        print("没有记录。先发一轮对话，或检查 --trace。")
        return

    current = None
    items: list[tuple[str, float]] = []
    details: dict[str, str] = {}
    started = 0.0

    def flush() -> None:
        if not items:
            return
        wall = effective_total(items)
        print(f"\n{'=' * 78}")
        print(f"{current}   {time.strftime('%H:%M:%S', time.localtime(started))}   实际约 {wall / 1000:.2f}s")
        print(f"{'=' * 78}")
        widest = max((ms for _, ms in items), default=1)
        for name, ms in sorted(items, key=lambda x: x[1], reverse=True):
            if ms < args.min_ms:
                continue
            bar = "█" * max(1, int(ms / widest * 28))
            share = ms / wall * 100 if wall else 0
            note = ""
            if name in ("emotion_recognition", "risk_scorer"):
                note = "  (与对方并行)"
            elif name.endswith("_retrieval"):
                tim = details.get(name, "")
                if "embed" in tim:
                    note = "  " + tim
            print(f"  {name:24s} {ms:8.0f} ms  {share:5.1f}%  {bar}{note}")
        print(f"  {'合计 (并行折算)':24s} {wall:8.0f} ms")
        biggest = max(items, key=lambda x: x[1])
        print(f"  → 最慢一步：{biggest[0]}（{biggest[1] / 1000:.2f}s）")

    for trace_id, _seq, name, duration_ms, detail, ts in rows:
        if trace_id != current:
            flush()
            current = trace_id
            items = []
            details = {}
            started = ts or 0.0
        ms = float(duration_ms or 0)
        items.append((name, ms))
        if detail:
            details[name] = str(detail)[:70]
    flush()


if __name__ == "__main__":
    main()
