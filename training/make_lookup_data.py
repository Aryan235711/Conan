"""Record-lookup curriculum: find the record about a person at a time.

SFT v7 fails lying-witness cases mostly at lookup: for a claim about person X
at time T it either finds no record or pairs the claim with a record about a
different person or time. This curriculum teaches that one step with worked
answers, the same approach that took interval coverage from 42% to 99%.

Each item lists 5-10 reliable records, including decoys about the same person
at other times and other people at the same time, and asks whether X was
recorded at T and where. The worked answer lists X's records, then picks the
one at T or says there is none, and (when a claimed place is given) compares.

    python3 training/make_lookup_data.py --n 3000 --out data/generated/sft_lookup.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine.generator import POOLS, fmt  # noqa: E402
from detective_engine.liar import PLACES, SOURCES  # noqa: E402

TEMPLATES = [
    "Records:\n{records}\n\nSomeone claims {person} was at {claim} at {time}. Does a record confirm or contradict it? "
    "Work it out step by step and end with 'Answer: confirmed', 'Answer: contradicted' or 'Answer: no record'.",
    "Here are reliable records:\n{records}\n\nIs there a record of {person} at {time}? If so, where, and does it match "
    "the claim that they were at {claim}? End with 'Answer: confirmed', 'Answer: contradicted' or 'Answer: no record'.",
]


def build(n: int, seed: int, p_none: float = 0.2) -> list[dict]:
    rng = random.Random(seed)
    pool = POOLS["A"]
    places = PLACES["A"]
    rows = []
    while len(rows) < n:
        people = [f"{a} {b}" for a, b in zip(rng.sample(pool["first"], 4), rng.sample(pool["last"], 4))]
        start = rng.randint(6, 8) * 60
        times = [start + 60 * i for i in range(5)]
        target, t_target = rng.choice(people), rng.choice(times)
        records = set()
        # decoys: same person other times, other people same time
        for t in rng.sample([x for x in times if x != t_target], rng.randint(1, 3)):
            records.add((target, t, rng.choice(places)))
        for p in rng.sample([x for x in people if x != target], rng.randint(1, 3)):
            records.add((p, t_target, rng.choice(places)))
        for _ in range(rng.randint(1, 4)):
            records.add((rng.choice(people), rng.choice(times), rng.choice(places)))
        # outcome: p_none no record (default 20%), the rest split between confirmed and contradicted
        roll = rng.random()
        true_place = rng.choice(places)
        records = {r for r in records if not (r[0] == target and r[1] == t_target)}
        half = (1 - p_none) / 2
        if roll < 1 - p_none:
            records.add((target, t_target, true_place))
            claim = true_place if roll < half else rng.choice([p for p in places if p != true_place])
        else:
            claim = rng.choice(places)
        recs = list(records)
        # one record per (person, time): keep the first seen
        seen, uniq = set(), []
        for r in recs:
            if (r[0], r[1]) not in seen:
                seen.add((r[0], r[1]))
                uniq.append(r)
        rng.shuffle(uniq)
        lines = [f"E{i}. " + rng.choice(SOURCES["A"]).format(p=p, l=l, t=fmt(t)) for i, (p, t, l) in enumerate(uniq, 1)]
        eid = {(p, t): f"E{i}" for i, (p, t, _) in enumerate(uniq, 1)}
        place_of = {(p, t): l for p, t, l in uniq}
        mine = sorted([(t, eid[(target, t)], place_of[(target, t)]) for p, t, _ in uniq if p == target])
        work = []
        if mine:
            work.append(f"Records for {target}: " + "; ".join(f"{e} {l} at {fmt(t)}" for t, e, l in mine) + ".")
        else:
            work.append(f"Records for {target}: none.")
        if (target, t_target) in eid:
            e, l = eid[(target, t_target)], place_of[(target, t_target)]
            work.append(f"At {fmt(t_target)}: {e} places {target} at {l}.")
            if l == claim:
                work.append(f"The claim says {claim}; same place, so the record confirms it.")
                ans = "confirmed"
            else:
                work.append(f"The claim says {claim}; different place, so the record contradicts it.")
                ans = "contradicted"
        else:
            work.append(f"None of {target}'s records is at {fmt(t_target)}, so nothing confirms or contradicts the claim.")
            ans = "no record"
        q = rng.choice(TEMPLATES).format(records="\n".join(lines), person=target, claim=claim, time=fmt(t_target))
        rows.append({"prompt": [{"role": "user", "content": q}],
                     "completion": [{"role": "assistant", "content": "\n".join(work) + f"\nAnswer: {ans}"}],
                     "answer": ans})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--p-none", type=float, default=0.2, help="share of items with no matching record")
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "sft_lookup.jsonl"))
    args = ap.parse_args()
    rows = build(args.n, args.seed, args.p_none)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} items to {args.out}: {dict(Counter(r['answer'] for r in rows))}")


if __name__ == "__main__":
    main()
