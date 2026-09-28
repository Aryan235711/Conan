"""Comparison curriculum: interval-coverage questions with worked answers.

Teaches the one skill the probe showed every model lacks.  Prompts use
three phrasings that the probe (benchmarks/interval_probe.py) never uses,
and items are drawn with a different seed, so the probe measures whether
the skill generalizes rather than whether a template was memorized.

    python3 training/make_interval_data.py --n 3000 --out data/generated/sft_interval.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from coverage_text import clock, covers, explain_coverage, explain_coverage_compact  # noqa: E402
from interval_probe import make_items  # noqa: E402

TEMPLATES = [
    "An alibi places someone elsewhere from {A} to {B}. The death happened between {L} and {H}. "
    "Does the alibi cover that whole period? Work it out step by step and end with 'Answer: yes' or 'Answer: no'.",
    "Verified whereabouts: {A} to {B}. Death window: {L} to {H}, on the same night, and times may pass midnight. "
    "Is the person accounted for during the entire death window? Reason step by step, then finish with "
    "'Answer: yes' or 'Answer: no'.",
    "Alibi: {A}-{B}. Window: {L}-{H}. Does the alibi cover the whole window? "
    "Check it step by step, then write 'Answer: yes' or 'Answer: no'.",
]


def build(n: int, seed: int, compact: bool = False) -> list[dict]:
    rng = random.Random(seed)
    probe_keys = {(x["a"], x["b"], x["lo"], x["hi"]) for x in make_items(200, seed=0)}
    rows, attempt = [], 0
    while len(rows) < n:
        attempt += 1
        for it in make_items(200, seed=seed * 1000 + attempt):
            if len(rows) >= n:
                break
            key = (it["a"], it["b"], it["lo"], it["hi"])
            if key in probe_keys:
                continue
            q = rng.choice(TEMPLATES).format(A=clock(it["a"]), B=clock(it["b"]), L=clock(it["lo"]), H=clock(it["hi"]))
            ans = "yes" if covers(*key) else "no"
            steps = explain_coverage_compact(*key) if compact else explain_coverage(*key)
            work = "\n".join(steps) + f"\nAnswer: {ans}"
            rows.append({"prompt": [{"role": "user", "content": q}],
                         "completion": [{"role": "assistant", "content": work}],
                         "answer": ans, "kind": it["kind"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--compact", action="store_true", help="shorter worked answers matching compact case traces")
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "sft_interval.jsonl"))
    args = ap.parse_args()
    rows = build(args.n, args.seed, args.compact)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    yes = sum(r["answer"] == "yes" for r in rows)
    print(f"wrote {len(rows)} items ({yes} yes, {len(rows) - yes} no) to {args.out}")


if __name__ == "__main__":
    main()
