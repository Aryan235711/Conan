"""Scorer ranking benchmark: does a scorer rank answers the way it should?

For each gold case (C001-C006) the same answer types are scored by the v1
keyword engine and the v2 verifiable scorer:

    expert  a careful, mostly-correct answer (see benchmarks/answers.py)
    echo    restates the prompt with no commitment or reasoning
    wrong   confidently endorses the false narrative
    soup    (v2 only) raw keyword dump with no final answer

A trustworthy scorer ranks expert > echo > wrong on every case and passes
the expert answer.  Exit code is 0 only if v2 meets both bars.

Run:  python3 benchmarks/scorer_ranking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answers import EXPERT_V1, EXPERT_V2, WRONG_V1, WRONG_V2, echo_v1, echo_v2, keyword_soup  # noqa: E402
from detective_engine.engine.reward_interface import RewardScorer  # noqa: E402
from detective_engine.engine.verifiable import VerifiableScorer  # noqa: E402

CASES = ["C001", "C002", "C003", "C004", "C005", "C006"]


def summarize(rows: list[tuple[str, float, float, float, bool]]) -> tuple[int, int, int]:
    ok = sum((e > c) + (c > w) + (e > w) for _, e, c, w, _ in rows)
    return ok, 3 * len(rows), sum(p for *_, p in rows)


def main() -> int:
    v1, v2 = RewardScorer(), VerifiableScorer()
    rows1, rows2 = [], []
    print(f"{'':5} | {'v1 keyword engine':^26} | {'v2 verifiable scorer':^34}")
    print(f"{'case':5} | {'expert':>7} {'echo':>7} {'wrong':>7}  | {'expert':>7} {'echo':>7} {'wrong':>7} {'soup':>7}")
    for cid in CASES:
        case = v2.case(cid)
        a = v1.score(cid, EXPERT_V1[cid]); b = v1.score(cid, echo_v1(case)); c = v1.score(cid, WRONG_V1[cid])
        rows1.append((cid, a["weighted"], b["weighted"], c["weighted"], a["passed"]))
        x = v2.score(cid, EXPERT_V2[cid]); y = v2.score(cid, echo_v2(case)); z = v2.score(cid, WRONG_V2[cid])
        s = v2.score(cid, keyword_soup(case))
        rows2.append((cid, x.reward, y.reward, z.reward, x.passed))
        print(f"{cid:5} | {a['weighted']:7.3f} {b['weighted']:7.3f} {c['weighted']:7.3f}  | "
              f"{x.reward:7.3f} {y.reward:7.3f} {z.reward:7.3f} {s.reward:7.3f}")

    ok1, tot1, p1 = summarize(rows1)
    ok2, tot2, p2 = summarize(rows2)
    print(f"\n{'':24} {'v1':>6} {'v2':>6}")
    print(f"{'pairwise ranking':24} {f'{ok1}/{tot1}':>6} {f'{ok2}/{tot2}':>6}")
    print(f"{'expert answers passed':24} {f'{p1}/{len(CASES)}':>6} {f'{p2}/{len(CASES)}':>6}")
    return 0 if ok2 == tot2 and p2 == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
