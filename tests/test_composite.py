"""Tests for the combined family (timeline + lying witness).

Every case is re-solved independently: the death window minute by minute,
the liar by checking each claim against every record, and the culprit by
brute force over (liar, suspect, minute).

Run:  python3 tests/test_composite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detective_engine.composite import generate  # noqa: E402
from detective_engine.engine.case_validator import CaseQualityValidator  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402

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


def independent(case: dict) -> tuple[set[str], set[str]]:
    facts = case["generator"]["facts"]
    witnesses = case["generator"]["witnesses"]
    # window, minute by minute
    def alive_possible(t: int) -> bool:
        for f in facts:
            d = f["data"]
            if f["kind"] == "last_alive" and t < d["time"]:
                return False
            if f["kind"] == "discovery" and t > d["time"]:
                return False
            if f["kind"] == "body_temp":
                est = d["discovery"] - (37.0 - d["temp"]) * 60
                if abs(t - est) > 60:
                    return False
        return True
    minutes = [t for t in range(0, 24 * 60) if alive_possible(t)]
    records = [f["data"] for f in facts if f["kind"] == "record"]
    claims = [f["data"] for f in facts if f["kind"] == "witness_alibi"]

    def contradicted(c: dict) -> bool:
        return any(r["name"] in (c["witness"], c["suspect"]) and c["start"] <= r["time"] <= c["end"]
                   and r["place"] != c["place"] for r in records)

    liars = {w for w in witnesses if all(not contradicted(c) for c in claims if c["witness"] != w)}
    keyholders = next(set(f["data"]["names"]) for f in facts if f["kind"] == "keyholders")
    suspects = [s.split(",")[0] for s in case["scenarios"] if "accident" not in s]
    culprits = set()
    for liar in liars:
        for s in suspects:
            if s not in keyholders:
                continue
            for t in minutes:
                if not any(c["suspect"] == s and c["witness"] != liar and c["start"] <= t <= c["end"] for c in claims):
                    culprits.add(s)
                    break
    return culprits, liars


print("\n=== CO-01: Unique culprit and liar, confirmed independently ===")
cases = generate(120, seed=51, pool="A") + generate(40, seed=52, pool="B")
bad, key_bad = [], []
validator = CaseQualityValidator()
for c in cases:
    culprits, liars = independent(c)
    truth = c["scenarios"][int(c["answer_key"]["true_scenario"][1:]) - 1].split(",")[0]
    if culprits != {truth} or len(liars) != 1:
        bad.append((c["id"], culprits, liars, truth))
    if validator._answer_key_problems(CaseDefinition.from_dict(c)):
        key_bad.append(c["id"])
check(f"all {len(cases)} cases: one culprit, one liar, matching the key", not bad, str(bad[:2]))
check("all answer keys pass the validator", not key_bad, str(key_bad[:3]))

print("\n=== CO-02: No surface shortcut ===")
unbalanced = 0
hits = 0
for c in cases:
    sus = [s.split(",")[0] for s in c["scenarios"] if "accident" not in s]
    counts = [sum(n in e for e in c["evidence"]) for n in sus]
    wcounts = [sum(n in e for e in c["evidence"]) for n in c["generator"]["witnesses"]]
    unbalanced += len(set(counts)) != 1 or len(set(wcounts)) != 1
check("every suspect and every witness is named equally often", unbalanced == 0, f"{unbalanced} unbalanced")
every_keyholder_vouched = all(
    sum(1 for f in c["generator"]["facts"] if f["kind"] == "witness_alibi")
    == len(next(f["data"]["names"] for f in c["generator"]["facts"] if f["kind"] == "keyholders"))
    for c in cases)
check("every key holder has a witness alibi, so alibi form reveals nothing", every_keyholder_vouched)

print("\n=== CO-03: Harder than the single families ===")
mean_len = sum(len(c["evidence"]) for c in cases) / len(cases)
mean_key = sum(len(c["answer_key"]["key_evidence"]) for c in cases) / len(cases)
check(f"cases are long (mean {mean_len:.1f} evidence lines)", mean_len >= 25)
check(f"answers need many decisive clues (mean {mean_key:.1f})", mean_key >= 7)

print("\n=== CO-04: Prompts and scoring ===")
check("no prompt contains the hidden truth",
      not any(c["hidden_truth"] in build_prompt(CaseDefinition.from_dict(c)) for c in cases))
c0 = CaseDefinition.from_dict(cases[0]); k = c0.answer_key
perfect = {"most_likely": k.true_scenario,
           "probabilities": {f"S{i}": float(f"S{i}" == k.true_scenario) for i in range(1, len(c0.scenarios) + 1)},
           "ruled_out": list(k.ruled_out), "key_evidence": list(k.key_evidence), "red_herrings": list(k.red_herrings),
           "time_window": {"earliest": k.time_window.earliest, "latest": k.time_window.latest}}
check("a perfect answer scores ~1", score_answer(c0, perfect).reward > 0.99)
check("deterministic for a fixed seed", generate(10, seed=3) == generate(10, seed=3))

print("\n" + "=" * 50)
print(f"  COMPOSITE FAMILY: {passed} passed, {failed} failed")
print("=" * 50)
if failed:
    sys.exit(1)
