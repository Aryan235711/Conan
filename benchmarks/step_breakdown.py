"""Step-by-step error analysis for timeline-family evaluations.

A model can fail a case at several steps. This script checks each step of
every answer in a run against the case's structured facts:

    window_ok        time window overlaps the true window (IoU > 0.8)
    accident_out     the accident scenario is ruled out
    nokey_all_out    every suspect without a key is ruled out
    alibi_all_out    every suspect whose verified alibi covers the window is ruled out
    culprit_kept     the true culprit is NOT ruled out
    correct|steps    final pick is right, given every elimination step was right

It also splits "culprit kept" by the culprit's whereabouts line. A model that
treats any verified alibi as clearing, without comparing it with the window,
drops culprits whose alibi is verified but does not cover the window.

    python3 benchmarks/step_breakdown.py runs/hf_runs_sft_qwen0.5b [more run dirs] --split test_ood
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine.engine.models import TimeWindow  # noqa: E402
from detective_engine.engine.verifiable import _norm_id, parse_final_answer, window_iou  # noqa: E402
from detective_engine.generator import Fact, death_window, fmt  # noqa: E402


def analyse(run_dir: Path, raw: dict[str, dict], split: str) -> dict:
    c = collections.Counter()
    kind_kept = collections.defaultdict(lambda: [0, 0])
    win_kept = collections.defaultdict(lambda: [0, 0])
    n = n_steps = 0
    for line in open(run_dir / f"{split}.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        ans = parse_final_answer(rec["raw_output"])
        case = raw.get(rec["case_id"])
        if not isinstance(ans, dict) or case is None or case["generator"].get("family", "timeline") != "timeline":
            continue
        n += 1
        facts = [Fact(f["kind"], f["text"], f["data"], f["role"]) for f in case["generator"]["facts"]]
        labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in case["scenarios"]]
        sid = {lab: f"S{i}" for i, lab in enumerate(labels, 1)}
        ruled = {_norm_id(x, "S") for x in (ans.get("ruled_out") or [])}
        lo, hi = death_window(facts)
        keyholders = set(next(f.data["names"] for f in facts if f.kind == "keyholders"))
        culprit = labels[int(case["answer_key"]["true_scenario"][1:]) - 1]
        covered = [f.data["name"] for f in facts if f.kind == "alibi" and f.data["start"] <= lo and f.data["end"] >= hi]
        nokey = [lab for lab in labels if lab != "accident" and lab not in keyholders]

        ok = {
            "window_ok": window_iou(ans.get("time_window"), TimeWindow("t", fmt(lo), fmt(hi))) > 0.8,
            "accident_out": "accident" not in sid or sid["accident"] in ruled,
            "nokey_all_out": all(sid[x] in ruled for x in nokey),
            "alibi_all_out": all(sid[x] in ruled for x in covered),
            "culprit_kept": sid[culprit] not in ruled,
        }
        ok["all_steps_ok"] = all(ok[k] for k in ("accident_out", "nokey_all_out", "alibi_all_out", "culprit_kept"))
        ok["correct"] = bool(rec["correct"])
        for k, v in ok.items():
            c[k] += v
        if ok["all_steps_ok"]:
            n_steps += 1
            c["correct_given_steps"] += ok["correct"]

        where = next(f for f in facts if f.data.get("name") == culprit and f.kind in ("alibi", "testimony"))
        if where.kind == "testimony":
            kind = "unverified testimony"
        elif where.data["end"] < lo or where.data["start"] > hi:
            kind = "verified alibi outside the window"
        else:
            kind = "verified alibi covering part of the window"
        kind_kept[kind][0] += ok["culprit_kept"]
        kind_kept[kind][1] += 1
        w = "window right" if ok["window_ok"] else "window wrong"
        win_kept[w][0] += ok["culprit_kept"]
        win_kept[w][1] += 1
    return {"n": n, "n_steps": n_steps, "counts": c, "kind_kept": kind_kept, "win_kept": win_kept}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories under runs/")
    ap.add_argument("--split", default="test_ood")
    args = ap.parse_args()
    raw = {json.loads(l)["id"]: json.loads(l)
           for l in open(ROOT / "data" / "generated" / f"{args.split}.jsonl", encoding="utf-8") if l.strip()}
    cols = ["window_ok", "accident_out", "nokey_all_out", "alibi_all_out", "culprit_kept", "all_steps_ok", "correct"]
    print(f"{'run':44} {'n':>4} " + " ".join(f"{c[:13]:>13}" for c in cols) + f" {'correct|steps':>13}")
    results = []
    for r in args.runs:
        res = analyse(Path(r), raw, args.split)
        results.append((r, res))
        n = max(1, res["n"])
        vals = [res["counts"][c] / n for c in cols]
        cg = res["counts"]["correct_given_steps"] / max(1, res["n_steps"])
        print(f"{Path(r).name[-44:]:44} {res['n']:>4} " + " ".join(f"{v:13.2f}" for v in vals) + f" {cg:13.2f}")
    print("\nCulprit kept (not ruled out), by the culprit's whereabouts line and by window correctness:")
    for r, res in results:
        print(f"  {Path(r).name[-60:]}")
        for k, (a, b) in sorted(res["kind_kept"].items()):
            print(f"    {k:45} {a:>3}/{b:<3} {a / max(1, b):.0%}")
        for k, (a, b) in sorted(res["win_kept"].items()):
            print(f"    {k:45} {a:>3}/{b:<3} {a / max(1, b):.0%}")


if __name__ == "__main__":
    main()
