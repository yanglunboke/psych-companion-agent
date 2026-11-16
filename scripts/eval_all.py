"""一次性跑完两个评测并把报告写到 ``eval/report.json``。

用法::

    ./.venv/Scripts/python.exe scripts/eval_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_profile_recall  # noqa: E402
import eval_retrieval  # noqa: E402
import gen_eval_data  # noqa: E402

EVAL_DIR = ROOT / "eval"


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    retrieval_report = EVAL_DIR / "report_retrieval.json"
    profile_report = EVAL_DIR / "report_profile.json"

    print("\n>>> scripts/gen_eval_data.py\n")
    gen_eval_data.main([])

    print("\n>>> scripts/eval_retrieval.py\n")
    eval_retrieval.main(["--json", str(retrieval_report)])

    print("\n>>> scripts/eval_profile_recall.py\n")
    eval_profile_recall.main(["--json", str(profile_report)])

    combined = {
        "retrieval": json.loads(retrieval_report.read_text(encoding="utf-8")),
        "profile_recall": json.loads(profile_report.read_text(encoding="utf-8")),
    }
    out = EVAL_DIR / "report.json"
    out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总报告已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
