"""Reading curriculum: state what each evidence paragraph says.

SFT v9.2 solves 94% of prose cases whose decisive paragraphs it reads right,
so reading is the bottleneck. Its most common misreading is an alibi claim
("W says W and S were together at P from a to b") read as W's own claim, with
the companion dropped, or with W and S swapped. In full-case traces the
reading step is about a tenth of the tokens; this curriculum gives reading
its own short items, the same recipe that fixed interval comparison (v5) and
record lookup (v7.1).

Each item shows one to four consecutive paragraphs of a prose case (the "wide3"
phrasing bank, training names only) and asks for the reading lines, in exactly
the format of the reading step in full traces (key holders in text order,
motives and background grouped at the end). Paragraphs whose roles are easy
to confuse (alibi claims, sightings, statements, multi-fact paragraphs) are
drawn more often.

    python3 training/make_reading_data.py --n 4000 --out data/generated/sft_reading_v93.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import make_sft_data as m  # noqa: E402
from detective_engine import prose  # noqa: E402
from detective_engine.generator import Fact  # noqa: E402

WEIGHT = {"witness_alibi": 4.0, "statement": 3.0, "testimony": 2.5, "record": 2.0, "point": 2.0, "alibi": 2.0,
          "keyholders": 2.0, "last_alive": 1.5}


def _victim(case: dict) -> str | None:
    mm = re.match(r"(.+?) was found dead", case["summary"])
    return mm.group(1) if mm else None


def reading_lines(case: dict, idx: list[int]) -> list[str]:
    """Reading lines for paragraphs idx (1-based), formatted as the reading step does."""
    facts = case["generator"]["facts"]
    fam = case["generator"]["family"]
    per: dict[int, list[str]] = {}
    for f in facts:
        if f["para"] in idx:
            per.setdefault(f["para"], []).append(
                m._read_fact(Fact(f["kind"], "", f["data"], f["role"]), case["evidence"][f["para"] - 1]))
    lines, motives, background = [], [], []
    for i in idx:
        e = f"E{i}"
        got = per.get(i, [])
        if got and all(x.startswith("a motive for ") for x in got):
            motives.append(f"{e} " + ", ".join(x[len("a motive for "):] for x in got))
        elif got and not all(x.startswith("background only") for x in got):
            lines.append(f"{e}: " + "; ".join(x for x in got if not x.startswith("background only")) + ".")
        elif i == 1 and fam in ("liar", "composite"):
            lines.append(f"{e}: the rule: one witness lies, records are reliable.")
        else:
            background.append(e)
    if motives:
        lines.append("Motives only: " + "; ".join(motives) + ".")
    if background:
        lines.append("No fact about who or when: " + ", ".join(background) + ".")
    return lines


def _weight(case: dict, i: int) -> float:
    kinds = [f["kind"] for f in case["generator"]["facts"] if f["para"] == i]
    if not kinds:
        return 0.3
    w = max(WEIGHT.get(k, 0.6) for k in kinds)
    return w * (1.5 if len(kinds) > 1 else 1.0)


def build(cases: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        case = rng.choice(cases)
        k = len(case["evidence"])
        start = rng.choices(range(1, k + 1), weights=[_weight(case, i) for i in range(1, k + 1)])[0]
        size = rng.choice([1, 1, 2, 3, 4])
        idx = list(range(start, min(k, start + size - 1) + 1))
        victim = _victim(case)
        head = (f"Paragraphs from a case file. The victim is {victim}."
                if victim else "Paragraphs from a case file about witness statements.")
        body = "\n".join(f"E{i}. {case['evidence'][i - 1]}" for i in idx)
        prompt = (f"{head}\n\n{body}\n\nState what each paragraph says, in the standard reading form "
                  "(key holders in the order named; motives and background grouped at the end).")
        rows.append({"prompt": [{"role": "user", "content": prompt}],
                     "completion": [{"role": "assistant", "content": "\n".join(reading_lines(case, idx))}],
                     "kinds": sorted({f["kind"] for f in case["generator"]["facts"] if f["para"] in idx})})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--cases", type=int, default=300, help="fresh prose cases per family to draw paragraphs from")
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "sft_reading_v93.jsonl"))
    args = ap.parse_args()
    m.TEXT_ORDER = True
    cases = []
    for j, (fam, levels) in enumerate((("timeline", (2, 3)), ("liar", (2, 3)), ("composite", (2, 3)))):
        cases += prose.generate(args.cases, args.seed + j, fam, levels, "A", "wide3", f"RD{fam[0].upper()}")
    rows = build(cases, args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter(k for r in rows for k in r["kinds"])
    print(f"wrote {len(rows)} reading items to {args.out}; paragraphs by fact kind: {dict(c.most_common())}")


if __name__ == "__main__":
    main()
