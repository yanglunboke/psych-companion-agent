"""查询已落盘的 AI 调用 Prompt 日志。

用法::

    python scripts/show_prompt.py                      # 列出今天所有记录（概览）
    python scripts/show_prompt.py --trace trace_xxx    # 看某次请求的完整 prompt
    python scripts/show_prompt.py --worker guidance    # 只看某个 Worker
    python scripts/show_prompt.py --date 2026-11-26    # 看指定日期

    python scripts/show_prompt.py --trace trace_xxx --full   # 含完整 system+user 全文
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "data" / "prompts"


def _load(date: str) -> list[dict]:
    path = PROMPT_DIR / f"{date}.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt_ts(ts: float) -> str:
    import time

    return time.strftime("%H:%M:%S", time.localtime(ts))


def main() -> None:
    parser = argparse.ArgumentParser(description="查询 AI 调用 Prompt 日志")
    parser.add_argument("--trace", help="按 trace_id 过滤（支持前缀）")
    parser.add_argument("--worker", help="按 worker 名过滤，如 emotion/risk/needs/guidance")
    parser.add_argument("--date", help="日期，如 2026-11-26（默认今天）")
    parser.add_argument("--full", action="store_true", help="输出完整 system/user 全文")
    args = parser.parse_args()

    if args.date:
        date = args.date
    else:
        import time

        date = time.strftime("%Y-%m-%d")

    rows = _load(date)
    if not rows:
        print(f"无记录：{PROMPT_DIR / (date + '.jsonl')}")
        return

    if args.trace:
        rows = [r for r in rows if r["trace_id"].startswith(args.trace)]
    if args.worker:
        rows = [r for r in rows if r["worker"] == args.worker]

    if not rows:
        print("没有匹配的记录。")
        return

    if args.trace and args.full:
        for i, r in enumerate(rows, 1):
            print("=" * 72)
            print(f"[{i}] worker={r['worker']}  model={r['model']}  backend={r['backend']}  ts={_fmt_ts(r['ts'])}")
            print("=" * 72)
            print("----- SYSTEM PROMPT -----")
            print(r["system_prompt"])
            print("\n----- USER MESSAGE -----")
            print(r["user_message"])
            print()
        return

    # 概览模式
    print(f"日期 {date}，共 {len(rows)} 条：")
    print(f"{'时间':<10} {'trace_id':<26} {'worker':<10} {'model':<14} {'sys长度':<8} {'user长度'}")
    print("-" * 80)
    for r in rows:
        print(
            f"{_fmt_ts(r['ts']):<10} {r['trace_id']:<26} {r['worker']:<10} "
            f"{r['model']:<14} {len(r['system_prompt']):<8} {len(r['user_message'])}"
        )


if __name__ == "__main__":
    main()
