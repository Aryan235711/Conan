"""Tests for case generator v2.

Checks every generated case against an independent, minute-by-minute
solver (not the generator's own), plus determinism, answer-key integrity,
prompt safety, and train/OOD vocabulary separation.

Run:  python3 tests/test_generator.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detective_engine.engine.case_validator import CaseQualityValidator  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.generator import POOLS, generate  # noqa: E402

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


def independent_solve(case: dict) -> set[str]:
    """Brute force over every minute: who was free, had access, and could it be an accident?"""
    facts = case["generator"]["facts"]
    lo, hi = 0, 24 * 60 - 1
    for f in facts:
        d = f["data"]
        if f["kind"] == "discovery":
            hi = min(hi, d["time"])
        elif f["kind"] == "last_alive":
            lo = max(lo, d["time"])
        elif f["kind"] == "body_temp":
            est = d["discovery"] - (37.0 - d["temp"]) * 60
            lo, hi = max(lo, est - 60), min(hi, est + 60)
    keyholders = next((set(f["data"]["names"]) for f in facts if f["kind"] == "keyholders"), None)
    locked = any(f["kind"] == "no_forced_entry" for f in facts)
    alibis = collections.defaultdict(list)
    for f in facts:
        if f["kind"] == "alibi":
            alibis[f["data"]["name"]].append((f["data"]["start"], f["data"]["end"]))
    suspects = {s.split(", the ")[0] for s in case["scenarios"] if ", the " in s}
    possible = set()
    t = int(lo)
    while t <= hi:
        for s in suspects:
            if locked and keyholders is not None and s not in keyholders:
                continue
            if not any(a <= t <= b for a, b in alibis[s]):
                possible.add(s)
        t += 1
    if any("accident" in s for s in case["scenarios"]) and not any(f["kind"] == "homicide" for f in facts):
        possible.add("accident")
    return possible


print("\n=== GEN-01: Every case has exactly one solution (independent solver) ===")
cases = generate(150, seed=11, levels=(1, 2, 3), pool="A") + generate(60, seed=12, levels=(3,), pool="B")
bad_unique, bad_truth, bad_key = [], [], []
validator = CaseQualityValidator()
for c in cases:
    sol = independent_solve(c)
    true_idx = int(c["answer_key"]["true_scenario"][1:]) - 1
    true_text = c["scenarios"][true_idx]
    if len(sol) != 1:
        bad_unique.append((c["id"], sol))
    elif next(iter(sol)) not in true_text:
        bad_truth.append(c["id"])
    problems = validator._answer_key_problems(CaseDefinition.from_dict(c))
    if problems:
        bad_key.append((c["id"], problems))
check(f"all {len(cases)} cases have a unique solution", not bad_unique, str(bad_unique[:3]))
check("the unique solution matches the answer key", not bad_truth, str(bad_truth[:3]))
check("all answer keys pass the validator", not bad_key, str(bad_key[:3]))

print("\n=== GEN-02: Answer keys are informative and unbiased ===")
check("every case has key evidence", all(c["answer_key"]["key_evidence"] for c in cases))
check("every case has red herrings", all(c["answer_key"]["red_herrings"] for c in cases))
pos = collections.Counter(c["answer_key"]["true_scenario"] for c in cases if len(c["scenarios"]) == 6)
check("true scenario position is spread (no slot > 35% for 6-scenario cases)",
      max(pos.values()) / sum(pos.values()) < 0.35, str(dict(pos)))
levels = collections.Counter(c["generator"]["level"] for c in cases)
check("all three difficulty levels generated", set(levels) == {1, 2, 3}, str(dict(levels)))

print("\n=== GEN-03: Determinism ===")
a = generate(20, seed=5)
b = generate(20, seed=5)
check("same seed gives identical cases", a == b)
check("different seeds differ", generate(20, seed=6) != a)

print("\n=== GEN-04: Prompt safety and scoring ===")
leaks = [c["id"] for c in cases if c["hidden_truth"] in build_prompt(CaseDefinition.from_dict(c))]
check("no prompt contains the hidden truth", not leaks, str(leaks[:3]))
fact_leak = [c["id"] for c in cases if '"kind"' in build_prompt(CaseDefinition.from_dict(c))]
check("no prompt contains generator facts", not fact_leak)
c0 = CaseDefinition.from_dict(cases[0])
k = c0.answer_key
perfect = {"most_likely": k.true_scenario,
           "probabilities": {f"S{i}": (1.0 if f"S{i}" == k.true_scenario else 0.0) for i in range(1, len(c0.scenarios) + 1)},
           "ruled_out": list(k.ruled_out), "key_evidence": list(k.key_evidence), "red_herrings": list(k.red_herrings),
           "time_window": {"earliest": k.time_window.earliest, "latest": k.time_window.latest}}
check("a perfect answer to a generated case scores ~1", score_answer(c0, perfect).reward > 0.99)

print("\n=== GEN-05: OOD split uses held-out vocabulary ===")
ood_names = set(POOLS["B"]["first"]) | set(POOLS["B"]["last"])
id_names = set(POOLS["A"]["first"]) | set(POOLS["A"]["last"])
check("pools A and B share no names", not (ood_names & id_names), str(ood_names & id_names))
check("pools A and B share no places", not (set(POOLS["A"]["place"]) & set(POOLS["B"]["place"])))

print("\n" + "=" * 50)
print(f"  GENERATOR: {passed} passed, {failed} failed")
print("=" * 50)
if failed:
    sys.exit(1)
