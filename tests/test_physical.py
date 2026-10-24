"""Tests for the physical-evidence family (detective_engine/physical.py), templated and prose.

Each case is re-solved independently, minute by minute: the death window from
cooling, signs of life and the stomach rule; access applies only if the body was
not moved; a suspect is cleared only by a verified alibi covering every minute.

Run:  python3 tests/test_physical.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from detective_engine import prose  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.evaluate import agent_solver  # noqa: E402
from detective_engine.physical import generate  # noqa: E402
import make_sft_data  # noqa: E402

passed = failed = 0
RULE = {"undigested": (0, 60), "partly digested": (60, 180), "empty": (240, 10_000)}


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def independent(case: dict) -> set[str]:
    facts = case["generator"]["facts"]
    meal = next((f["data"]["time"] for f in facts if f["kind"] == "meal"), None)
    state = next((f["data"]["state"] for f in facts if f["kind"] == "stomach"), None)

    def possible(t: int) -> bool:
        for f in facts:
            d = f["data"]
            if f["kind"] == "last_alive" and t < d["time"]:
                return False
            if f["kind"] == "discovery" and t > d["time"]:
                return False
            if f["kind"] == "body_temp" and abs(t - (d["discovery"] - (37.0 - d["temp"]) * 60)) > 60:
                return False
        if meal is not None and state:
            a, b = RULE[state]
            if not meal + a <= t <= meal + b:
                return False
        return True

    minutes = [t for t in range(24 * 60) if possible(t)]
    moved = any(f["kind"] == "lividity" and f["data"]["moved"] for f in facts)
    holders = next(set(f["data"]["names"]) for f in facts if f["kind"] == "keyholders")
    labels = [s.split(",")[0] for s in case["scenarios"] if not s.endswith("was an accident.")]
    out = set()
    for s in labels:
        if not moved and s not in holders:
            continue
        spans = [(f["data"]["start"], f["data"]["end"]) for f in facts if f["kind"] == "alibi" and f["data"]["name"] == s]
        if not any(all(a <= t <= b for t in minutes) for a, b in spans):
            out.add(s)
    return out


def truth(c: dict) -> str:
    labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in c["scenarios"]]
    return labels[int(c["answer_key"]["true_scenario"][1:]) - 1]


print("Physical-evidence cases (templated)")
cases = generate(100, 801, (2, 3), "A", "T")
check("100 cases build", len(cases) == 100)
kinds = lambda c: {f["kind"] for f in c["generator"]["facts"]}  # noqa: E731
check("most cases use the stomach", sum("stomach" in kinds(c) for c in cases) >= 50)
moved = [c for c in cases if any(f["kind"] == "lividity" and f["data"]["moved"] for f in c["generator"]["facts"])]
no_key = sum(truth(c) not in next(f["data"]["names"] for f in c["generator"]["facts"] if f["kind"] == "keyholders")
             for c in moved)
check("some culprits have no key because the body was moved", no_key >= 10, str(no_key))
bad_indep = bad_score = 0
for c in cases:
    bad_indep += independent(c) != {truth(c)}
    case = CaseDefinition.from_dict(c)
    bad_score += score_answer(case, agent_solver(case, build_prompt(case), c)).reward < 0.999
check("independent minute-by-minute solve agrees on every case", bad_indep == 0, str(bad_indep))
check("the verified answer scores 1.0 on every case", bad_score == 0, str(bad_score))

print("\nProse")
pc = prose.generate(20, 802, "physical", (2, 3), "A", "wide3", "P") + prose.generate(20, 803, "physical", (2, 3), "B", "test", "Q")
check("40 prose cases build and re-verify", len(pc) == 40)
check("independent solve agrees on prose cases", all(independent(c) == {truth(c)} for c in pc))


def _grams(words, n=4):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _runs(tpl):
    return [re.findall(r"[a-z0-9°'\-]+", c.lower()) for c in re.split(r"\{[a-z_]+\}", tpl)]


test_g = {g for v in prose.BANK.values() for tpl in v["test"] for w in _runs(tpl) for g in _grams(w)}
train_g = {g for v in prose.BANK.values() for tpl in v["train"] for w in _runs(tpl) for g in _grams(w)}
hits = {g for c in pc[:20] for e in c["evidence"] for g in _grams(re.findall(r"[a-z0-9°'\-]+", e.lower())) & (test_g - train_g)}
check("no four-word run of a test template appears in training-bank prose", not hits, str(sorted(hits)[:3]))

print("\nTraces")
make_sft_data.COMPACT = make_sft_data.TRACK = make_sft_data.VERBOSE_COMPARE = True
make_sft_data.INDEX_RECORDS = make_sft_data.READ = make_sft_data.TEXT_ORDER = True
bad_ans = bad_concl = 0
for c in cases + pc:
    tr = make_sft_data.write_trace(c)
    ans = json.loads(tr.split("```json")[1].split("```")[0])
    bad_ans += ans["most_likely"] != c["answer_key"]["true_scenario"] or ans["key_evidence"] != c["answer_key"]["key_evidence"]
    m = re.findall(r"Still possible: (.+)\.", tr)
    bad_concl += not m or m[-1] != f"{c['answer_key']['true_scenario']} {truth(c)}"
check("trace answer equals the key", bad_ans == 0, str(bad_ans))
check("trace ends with only the culprit still possible", bad_concl == 0, str(bad_concl))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
