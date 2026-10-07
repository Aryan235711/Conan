"""Tests for the mixed-alibi family (detective_engine/mixed.py), templated and prose.

Each case is re-solved independently, minute by minute: the death window from
every timing fact, the liar from claims a record contradicts, and the culprit
as the key holder whom no truthful witness and no verified record covers for
the whole window. The answer must be the only one.

Run:  python3 tests/test_mixed.py
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

from detective_engine import prose  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.evaluate import agent_solver  # noqa: E402
from detective_engine.mixed import generate  # noqa: E402
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


def independent(case: dict) -> set[str]:
    facts = [f for f in case["generator"]["facts"]]
    witnesses = case["generator"]["witnesses"]

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
    records = [f["data"] for f in facts if f["kind"] == "record"]
    claims = [f["data"] for f in facts if f["kind"] == "witness_alibi"]
    verified = [f["data"] for f in facts if f["kind"] == "alibi"]
    holders = next(set(f["data"]["names"]) for f in facts if f["kind"] == "keyholders")

    def false_claim(c: dict) -> bool:
        return any(r["name"] in (c["witness"], c["suspect"]) and c["start"] <= r["time"] <= c["end"]
                   and r["place"] != c["place"] for r in records)

    liars = [w for w in witnesses if all(not false_claim(c) for c in claims if c["witness"] != w)]
    out = set()
    for liar in liars:
        for s in holders:
            spans = [(c["start"], c["end"]) for c in claims if c["suspect"] == s and c["witness"] != liar]
            spans += [(a["start"], a["end"]) for a in verified if a["name"] == s]
            if not any(all(a <= t <= b for t in minutes) for a, b in spans):
                out.add(s)
    return out


print("Mixed-alibi cases (templated)")
cases = generate(100, 601, (2, 3), "A", "T")
check("100 cases build", len(cases) == 100)
with_verified = sum(any(f["kind"] == "alibi" for f in c["generator"]["facts"]) for c in cases)
check("most cases mix verified and witness alibis", with_verified >= 60, str(with_verified))
lengths = Counter(len(c["evidence"]) for c in cases)
check("case length varies widely", max(lengths) - min(lengths) >= 8, str(sorted(lengths)))
bad_indep = bad_score = 0
for c in cases:
    labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in c["scenarios"]]
    truth = labels[int(c["answer_key"]["true_scenario"][1:]) - 1]
    bad_indep += independent(c) != {truth}
    case = CaseDefinition.from_dict(c)
    bad_score += score_answer(case, agent_solver(case, build_prompt(case), c)).reward < 0.999
check("independent minute-by-minute solve agrees on every case", bad_indep == 0, str(bad_indep))
check("the verified answer scores 1.0 on every case", bad_score == 0, str(bad_score))

print("\nMixed-alibi cases (prose)")
pcases = prose.generate(20, 602, "mixed", (2, 3), "A", "wide3", "P") + prose.generate(20, 603, "mixed", (2, 3), "B", "test", "Q")
check("40 prose cases build and re-verify", len(pcases) == 40)
check("prose cases carry verified alibis", sum(any(f["kind"] == "alibi" for f in c["generator"]["facts"]) for c in pcases) >= 20)

print("\nTraces")
make_sft_data.COMPACT = make_sft_data.TRACK = make_sft_data.VERBOSE_COMPARE = True
make_sft_data.INDEX_RECORDS = make_sft_data.READ = make_sft_data.TEXT_ORDER = True
bad_ans = bad_concl = bad_ver = 0
for c in cases + pcases:
    tr = make_sft_data.write_trace(c)
    ans = json.loads(tr.split("```json")[1].split("```")[0])
    bad_ans += ans["most_likely"] != c["answer_key"]["true_scenario"] or ans["key_evidence"] != c["answer_key"]["key_evidence"]
    labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in c["scenarios"]]
    truth = labels[int(c["answer_key"]["true_scenario"][1:]) - 1]
    m = re.findall(r"Still possible: (.+)\.", tr)
    bad_concl += not m or m[-1] != f"{c['answer_key']['true_scenario']} {truth}"
    holders = next(set(f["data"]["names"]) for f in c["generator"]["facts"] if f["kind"] == "keyholders")
    for f in c["generator"]["facts"]:
        if f["kind"] == "alibi" and f["data"]["name"] in holders:
            bad_ver += f"{f['data']['name']} is verified from" not in tr
check("trace answer equals the key", bad_ans == 0, str(bad_ans))
check("trace ends with only the culprit still possible", bad_concl == 0, str(bad_concl))
check("every key holder's verified alibi is checked in the trace", bad_ver == 0, str(bad_ver))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
