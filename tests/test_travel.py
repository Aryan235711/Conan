"""Tests for the travel-alibi family (detective_engine/travel.py).

Each case is re-solved independently, minute by minute: a ticket holder is away
from the scene from (departure - walk to the station) to (arrival + trip back),
with the arrival read from the timetable row for that station and departure.
The trace's written-out arithmetic is checked against the same numbers.

Run:  python3 tests/test_travel.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.evaluate import agent_solver  # noqa: E402
from detective_engine.generator import fmt  # noqa: E402
from detective_engine.travel import generate  # noqa: E402
import make_sft_data  # noqa: E402

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def away(case: dict) -> dict[str, list[tuple[int, int]]]:
    facts = case["generator"]["facts"]
    table = {}
    for f in facts:
        if f["kind"] == "timetable":
            table[(f["data"]["station"], f["data"]["dep"])] = (f["data"]["dest"], f["data"]["arr"])
    walk = {f["data"]["station"]: f["data"]["minutes"] for f in facts if f["kind"] == "travel"}
    out: dict[str, list] = {}
    for f in facts:
        d = f["data"]
        if f["kind"] == "alibi":
            out.setdefault(d["name"], []).append((d["start"], d["end"]))
        if f["kind"] == "ticket":
            dest, arr = table[(d["station"], d["dep"])]
            out.setdefault(d["name"], []).append((d["dep"] - walk[d["station"]], arr + walk[dest]))
    return out


def independent(case: dict) -> set[str]:
    facts = case["generator"]["facts"]

    def possible(t: int) -> bool:
        for f in facts:
            d = f["data"]
            if f["kind"] == "last_alive" and t < d["time"]:
                return False
            if f["kind"] == "discovery" and t > d["time"]:
                return False
            if f["kind"] == "body_temp" and abs(t - (d["discovery"] - (37.0 - d["temp"]) * 60)) > 60:
                return False
        return True

    minutes = [t for t in range(24 * 60) if possible(t)]
    holders = next(set(f["data"]["names"]) for f in facts if f["kind"] == "keyholders")
    spans = away(case)
    return {s for s in holders if not any(all(a <= t <= b for t in minutes) for a, b in spans.get(s, []))}


print("Travel cases")
cases = generate(100, 701, (2, 3), "A", "T")
check("100 cases build", len(cases) == 100)
kinds = Counter(f["kind"] for c in cases for f in c["generator"]["facts"])
check("every case has tickets and timetable rows", kinds["ticket"] >= 100 and kinds["timetable"] >= 200, str(kinds))
bad_indep = bad_score = needs_travel = 0
for c in cases:
    labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in c["scenarios"]]
    truth = labels[int(c["answer_key"]["true_scenario"][1:]) - 1]
    bad_indep += independent(c) != {truth}
    case = CaseDefinition.from_dict(c)
    bad_score += score_answer(case, agent_solver(case, build_prompt(case), c)).reward < 0.999
    lo = int(c["answer_key"]["time_window"]["earliest"].replace(":", ""))
    for f in c["generator"]["facts"]:
        if f["kind"] == "ticket":
            needs_travel += 1
check("independent minute-by-minute solve agrees on every case", bad_indep == 0, str(bad_indep))
check("the verified answer scores 1.0 on every case", bad_score == 0, str(bad_score))

print("\nTraces")
make_sft_data.COMPACT = make_sft_data.TRACK = make_sft_data.VERBOSE_COMPARE = True
make_sft_data.INDEX_RECORDS = make_sft_data.READ = make_sft_data.TEXT_ORDER = True
bad_ans = bad_concl = bad_math = 0
for c in cases:
    tr = make_sft_data.write_trace(c)
    ans = json.loads(tr.split("```json")[1].split("```")[0])
    bad_ans += ans["most_likely"] != c["answer_key"]["true_scenario"] or ans["key_evidence"] != c["answer_key"]["key_evidence"]
    labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in c["scenarios"]]
    truth = labels[int(c["answer_key"]["true_scenario"][1:]) - 1]
    m = re.findall(r"Still possible: (.+)\.", tr)
    bad_concl += not m or m[-1] != f"{c['answer_key']['true_scenario']} {truth}"
    spans = away(c)
    for name, a, b in re.findall(r"So (.+?) could not have been at the scene from (\d\d:\d\d) to (\d\d:\d\d)\.", tr):
        bad_math += not any(fmt(x) == a and fmt(y) == b for x, y in spans.get(name, []))
check("trace answer equals the key", bad_ans == 0, str(bad_ans))
check("trace ends with only the culprit still possible", bad_concl == 0, str(bad_concl))
check("every written-out travel span is arithmetically right", bad_math == 0, str(bad_math))

check("one timetable row per station and departure",
      all(len({(f["data"]["station"], f["data"]["dep"]) for f in c["generator"]["facts"] if f["kind"] == "timetable"})
          == sum(f["kind"] == "timetable" for f in c["generator"]["facts"]) for c in cases))

print("\nProse")
from detective_engine import prose  # noqa: E402
pc = prose.generate(20, 702, "travel", (2, 3), "A", "wide3", "P") + prose.generate(20, 703, "travel", (2, 3), "B", "test", "Q")
check("40 prose travel cases build and re-verify", len(pc) == 40)


def _grams(words, n=4):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _runs(tpl):
    return [re.findall(r"[a-z0-9°'\-]+", c.lower()) for c in re.split(r"\{[a-z_]+\}", tpl)]


test_g = {g for v in prose.BANK.values() for tpl in v["test"] for w in _runs(tpl) for g in _grams(w)}
train_g = {g for v in prose.BANK.values() for tpl in v["train"] for w in _runs(tpl) for g in _grams(w)}
hits = {g for c in pc[:20] for e in c["evidence"] for g in _grams(re.findall(r"[a-z0-9°'\-]+", e.lower())) & (test_g - train_g)}
check("no four-word run of a test template appears in training-bank travel prose", not hits, str(sorted(hits)[:3]))
bad = 0
for c in pc:
    tr = make_sft_data.write_trace(c)
    bad += json.loads(tr.split("```json")[1].split("```")[0])["most_likely"] != c["answer_key"]["true_scenario"]
check("prose traces reach the answer key", bad == 0, str(bad))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
