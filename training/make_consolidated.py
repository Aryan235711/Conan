"""One consolidated training set: every family and micro-skill, for training from a base model.

The 0.5B model reached its results through a chain of fifteen short runs
(v5 ... v11e), each adding a skill and replaying the earlier ones. A larger
model on a rented or free GPU should not repeat that chain: it trains once,
from the base model, on everything. This script assembles that set from the
trace files the chain already uses (all in the final trace format: reading
step, text-order key holders, state tracking, written-out comparisons,
per-person record lookup, line-by-line scenario scan).

    python3 training/make_consolidated.py                 # data/kaggle/sft_all.jsonl (about 6,900 rows)
    python3 training/make_consolidated.py --scale 0.4 --out data/generated/sft_all_small.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "generated"

# (file, case-id prefix or None, rows at scale 1.0, label)
PARTS = [
    # The one-stage dry run on a 40% sample (0.5B) was weakest on timeline cases (66%) and on
    # "no record" lookups (1 of 18), so both get more weight than the other parts.
    ("sft_interval.jsonl", None, 800, "curriculum: interval coverage"),
    ("sft_lookup.jsonl", None, 300, "curriculum: record lookup"),
    ("sft_lookup_v94.jsonl", None, 200, "curriculum: record lookup (balanced)"),
    ("sft_lookup_v9.jsonl", None, 200, "curriculum: record lookup (many no-record items)"),
    ("sft_reading_v93.jsonl", None, 800, "curriculum: paragraph reading"),
    ("sft_read_to_train.jsonl", None, 900, "timeline, templated"),
    ("sft_read_to_liar_train.jsonl", None, 400, "lying witness, templated"),
    ("sft_read_to_combo_train.jsonl", None, 500, "combined, templated"),
    ("sft_mliar_v11c.jsonl", None, 500, "one or two liars, templated"),
    ("sft_mixed.jsonl", None, 500, "mixed alibis, templated"),
    ("sft_prose_wide3.jsonl", "PYT", 400, "timeline, prose"),
    ("sft_prose_wide3.jsonl", "PYL", 400, "lying witness, prose"),
    ("sft_prose_wide3.jsonl", "PYC", 500, "combined, prose"),
    ("sft_prose_mixed.jsonl", None, 500, "mixed alibis, prose"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data" / "kaggle" / "sft_all.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows, counts = [], Counter()
    for name, prefix, n, label in PARTS:
        pool = [json.loads(l) for l in open(G / name, encoding="utf-8")]
        if prefix:
            pool = [r for r in pool if r.get("case_id", "").startswith(prefix)]
        take = min(len(pool), max(1, round(n * args.scale)))
        for r in rng.sample(pool, take):
            rows.append({"prompt": r["prompt"], "completion": r["completion"]})
        counts[label] = take
    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for label, n in counts.items():
        print(f"{n:5d}  {label}")
    print(f"{len(rows):5d}  total -> {out}")


if __name__ == "__main__":
    main()
