"""Tests for the k-liar family (detective_engine/multi_liar.py).

Each case is re-solved independently: a set of k witnesses is consistent if
the cells fixed by records and everyone else's statements never clash, and
each hypothesised liar has a statement whose cell is not already fixed to the
same place. The answer must be the only consistent set.

Run:  python3 tests/test_multi_liar.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.evaluate import agent_solver  # noqa: E402
from detective_engine.multi_liar import generate  # noqa: E402
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


def independent(case: dict) -> set[frozenset]:
    g = case["generator"]
    facts = g["facts"]
    out = set()
    for hyp in combinations(g["witnesses"], g["k"]):
        fixed: dict[tuple, str] = {}
        clash = False
        for f in facts:
            if f["kind"] == "record" or (f["kind"] == "statement" and f["data"]["speaker"] not in hyp):
                for p, t, l, _ in f["data"]["atoms"]:
                    if fixed.setdefault((p, t), l) != l:
                        clash = True
        if clash:
            continue
        if all(any(fixed.get((p, t)) != l for f in facts if f["kind"] == "statement" and f["data"]["speaker"] == h
                   for p, t, l, _ in f["data"]["atoms"]) for h in hyp):
            out.add(frozenset(hyp))
    return out


print("k-liar cases")
cases = generate(120, 501, (2, 3), "A", "T")
check("120 cases build", len(cases) == 120)
ks = Counter(c["generator"]["k"] for c in cases)
check("one-liar and two-liar cases are mixed evenly", ks[1] == ks[2] == 60, str(ks))
check("the rule states the number of liars",
      all(("Exactly two witnesses" in c["evidence"][0]) == (c["generator"]["k"] == 2) for c in cases))
bad_indep = bad_score = 0
for c in cases:
    truth = frozenset(c["generator"]["scenario_sets"][int(c["answer_key"]["true_scenario"][1:]) - 1])
    bad_indep += independent(c) != {truth}
    case = CaseDefinition.from_dict(c)
    bad_score += score_answer(case, agent_solver(case, build_prompt(case), c)).reward < 0.999
check("independent solve finds exactly the answer on every case", bad_indep == 0, str(bad_indep))
check("the verified answer scores 1.0 on every case", bad_score == 0, str(bad_score))
check("two-liar cases list every pair as a scenario",
      all(len(c["scenarios"]) == len(c["generator"]["witnesses"]) * (len(c["generator"]["witnesses"]) - 1) // 2
          for c in cases if c["generator"]["k"] == 2))

print("\nTraces")
make_sft_data.COMPACT = make_sft_data.TRACK = make_sft_data.VERBOSE_COMPARE = True
make_sft_data.INDEX_RECORDS = make_sft_data.READ = make_sft_data.TEXT_ORDER = True
bad_ans = bad_concl = bad_rule = 0
for c in cases:
    tr = make_sft_data.write_trace(c)
    ans = json.loads(tr.split("```json")[1].split("```")[0])
    bad_ans += ans["most_likely"] != c["answer_key"]["true_scenario"]
    g = c["generator"]
    truth = g["scenario_sets"][int(c["answer_key"]["true_scenario"][1:]) - 1]
    m = re.search(r"Conclusion: (.+?) \(S\d+\) (is|are) lying", tr)
    bad_concl += not m or set(m.group(1).split(" and ")) != set(truth)
    word = "two witnesses lie" if g["k"] == 2 else "one witness lies"
    bad_rule += f"E1: the rule: {word}" not in tr
check("trace answer equals the key", bad_ans == 0, str(bad_ans))
check("trace conclusion names exactly the liars", bad_concl == 0, str(bad_concl))
check("reading step states how many lie", bad_rule == 0, str(bad_rule))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
