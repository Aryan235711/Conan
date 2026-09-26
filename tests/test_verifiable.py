"""Tests for the verifiable scorer (v2).

Covers parsing, each scoring component, and the gaming strategies that fool
the v1 keyword engine: hedging, citing every clue, self-contradiction,
confident wrong answers, and garbage output.

Run:  python3 tests/test_verifiable.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detective_engine.engine.verifiable import (  # noqa: E402
    VerifiableScorer, brier_skill, parse_final_answer, window_iou,
)
from detective_engine.engine.models import TimeWindow  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


scorer = VerifiableScorer()
GOLD = ["C001", "C002", "C003", "C004", "C005", "C006"]


def perfect(cid: str) -> dict:
    case = scorer.case(cid)
    key = case.answer_key
    n = len(case.scenarios)
    probs = {f"S{i}": 0.0 for i in range(1, n + 1)}
    alive = [s for s in probs if s not in key.ruled_out]
    for s in alive:
        probs[s] = 0.1 / max(1, len(alive) - 1) if s != key.true_scenario else 0.0
    probs[key.true_scenario] = 1.0 - sum(v for s, v in probs.items() if s != key.true_scenario)
    ans = {
        "most_likely": key.true_scenario,
        "probabilities": probs,
        "ruled_out": list(key.ruled_out),
        "key_evidence": list(key.key_evidence),
        "red_herrings": list(key.red_herrings),
    }
    if key.time_window:
        ans["time_window"] = {"earliest": key.time_window.earliest, "latest": key.time_window.latest}
    return ans


def uniform(cid: str) -> dict:
    n = len(scorer.case(cid).scenarios)
    return {"most_likely": "S1", "probabilities": {f"S{i}": 1 / n for i in range(1, n + 1)},
            "ruled_out": [], "key_evidence": [], "red_herrings": []}


def shotgun(cid: str) -> dict:
    case = scorer.case(cid)
    ans = perfect(cid)
    ans["key_evidence"] = [f"E{i}" for i in range(1, len(case.evidence) + 1)]
    ans["red_herrings"] = []
    return ans


def confident_wrong(cid: str) -> dict:
    case = scorer.case(cid)
    key = case.answer_key
    wrong = next(f"S{i}" for i in range(1, len(case.scenarios) + 1) if f"S{i}" != key.true_scenario)
    probs = {f"S{i}": 0.0 for i in range(1, len(case.scenarios) + 1)}
    probs[wrong] = 1.0
    return {"most_likely": wrong, "probabilities": probs, "ruled_out": [key.true_scenario],
            "key_evidence": [], "red_herrings": []}


print("\n=== VS-01: Every gold case is keyed and prompt-safe ===")
check("all six gold cases have answer keys", all(c in scorer.case_ids for c in GOLD), str(scorer.case_ids[:10]))
for cid in GOLD:
    case = scorer.case(cid)
    prompt = scorer.prompt(cid)
    check(f"{cid} prompt omits hidden truth", case.hidden_truth not in prompt)
    check(f"{cid} prompt numbers evidence and scenarios",
          f"E{len(case.evidence)}." in prompt and f"S{len(case.scenarios)}." in prompt)

print("\n=== VS-02: Perfect answers pass; baselines do not ===")
for cid in GOLD:
    p = scorer.score(cid, perfect(cid))
    u = scorer.score(cid, uniform(cid))
    s = scorer.score(cid, shotgun(cid))
    w = scorer.score(cid, confident_wrong(cid))
    check(f"{cid} perfect passes (reward {p.reward:.2f})", p.passed and p.reward > 0.9, str(p.feedback))
    check(f"{cid} uniform hedger fails (reward {u.reward:.2f})", not u.passed and u.reward < 0.35)
    check(f"{cid} shotgun evidence scores below perfect", s.components["evidence"] < 0.8 and s.reward < p.reward)
    check(f"{cid} confident wrong scores near zero (reward {w.reward:.2f})", w.reward < 0.2 and not w.passed)

print("\n=== VS-03: Consistency violations are penalized ===")
a = perfect("C004")
a["probabilities"] = {"S1": 0.6, "S2": 0.3, "S3": 0.1}
r = scorer.score("C004", a)
check("top pick not most probable -> violation", any("highest probability" in v for v in r.violations))
a = perfect("C004"); a["ruled_out"] = ["S1", "S2", "S3"]
r = scorer.score("C004", a)
check("ruling out own top pick -> violation", any("also listed as ruled out" in v for v in r.violations))
check("violations block a pass", not r.passed)

print("\n=== VS-04: Format handling ===")
text = "Reasoning...\n```json\n" + json.dumps(perfect("C001")) + "\n```"
check("parses fenced JSON from raw text", scorer.score("C001", text).passed)
check("think block ignored", parse_final_answer('<think>{"most_likely":"S1"}</think> {"most_likely":"S2"}')["most_likely"] == "S2")
check("last final answer wins", parse_final_answer('{"most_likely":"S1"} then {"most_likely":"S3"}')["most_likely"] == "S3")
g = scorer.score("C001", "I think the butler did it.")
check("garbage scores 0 and is not skipped", g.reward == 0.0 and not g.format_ok)
g = scorer.score("C001", {"most_likely": "S9"})
check("out-of-range scenario scores 0", g.reward == 0.0 and not g.format_ok)
alt = perfect("C001"); alt["most_likely"] = "scenario 2"; alt["key_evidence"] = [7, "e8", "E9", "E10"]
check("tolerant ID formats", scorer.score("C001", alt).passed)
pct = perfect("C006"); pct["probabilities"] = {k: v * 100 for k, v in pct["probabilities"].items()}
check("percentages are normalized", abs(scorer.score("C006", pct).reward - scorer.score("C006", perfect("C006")).reward) < 1e-9)

print("\n=== VS-05: Metric primitives ===")
check("brier skill: uniform = 0", abs(brier_skill({"S1": 1/3, "S2": 1/3, "S3": 1/3}, "S1")) < 1e-9)
check("brier skill: certain and right = 1", brier_skill({"S1": 1, "S2": 0, "S3": 0}, "S1") == 1.0)
check("brier skill: certain and wrong = 0", brier_skill({"S1": 0, "S2": 1, "S3": 0}, "S1") == 0.0)
tw = TimeWindow("tod", "23:14", "01:00")
check("window IoU exact overnight = 1", window_iou({"earliest": "23:14", "latest": "01:00"}, tw) == 1.0)
check("window IoU partial overlap in (0,1)", 0 < window_iou({"earliest": "00:00", "latest": "02:00"}, tw) < 1)
check("window IoU disjoint = 0", window_iou({"earliest": "02:00", "latest": "03:00"}, tw) == 0.0)
check("window IoU malformed = 0", window_iou({"earliest": "late", "latest": "01:00"}, tw) == 0.0)

print("\n" + "=" * 50)
print(f"  VERIFIABLE SCORER: {passed} passed, {failed} failed")
print("=" * 50)
if failed:
    sys.exit(1)
