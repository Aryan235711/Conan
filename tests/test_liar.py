"""Tests for the witness-consistency ("who is lying?") family.

Every case is re-solved by an independent backtracking search over concrete
worlds, which shares no code with the generator's constraint check.

Run:  python3 tests/test_liar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detective_engine.engine.case_validator import CaseQualityValidator  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.liar import generate  # noqa: E402

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


def world_exists(required, must_break, n_places) -> bool:
    """Backtracking search for a concrete world: every required atom holds and,
    if must_break is non-empty, at least one of those atoms fails."""
    cells = sorted({(a[0], a[1]) for a in required + must_break})
    named = sorted({a[2] for a in required + must_break})
    domain = named + [f"__other{i}" for i in range(max(0, n_places - len(named)))]
    assign: dict = {}

    def holds(atom) -> bool | None:
        v = assign.get((atom[0], atom[1]))
        if v is None:
            return None
        return (v == atom[2]) == atom[3]

    def ok_partial() -> bool:
        if any(holds(a) is False for a in required):
            return False
        if must_break and all(holds(a) is True for a in must_break):
            return False
        return True

    def bt(i: int) -> bool:
        if not ok_partial():
            return False
        if i == len(cells):
            return True
        for v in domain:
            assign[cells[i]] = v
            if bt(i + 1):
                return True
        del assign[cells[i]]
        return False

    return bt(0)


def independent_liars(case: dict) -> set[str]:
    facts = case["generator"]["facts"]
    n_places = case["generator"]["n_places"]
    records = [tuple(a) for f in facts if f["kind"] == "record" for a in f["data"]["atoms"]]
    witnesses = [s[: -len(" is lying.")] for s in case["scenarios"]]
    out = set()
    for h in witnesses:
        others = [tuple(a) for f in facts if f["kind"] == "statement" and f["data"]["speaker"] != h
                  for a in f["data"]["atoms"]]
        own = [tuple(a) for f in facts if f["kind"] == "statement" and f["data"]["speaker"] == h
               for a in f["data"]["atoms"]]
        if world_exists(records + others, own, n_places):
            out.add(h)
    return out


print("\n=== LI-01: Unique liar, confirmed independently ===")
cases = generate(150, seed=41, pool="A") + generate(45, seed=42, pool="B")
bad = []
validator = CaseQualityValidator()
key_bad = []
for c in cases:
    sol = independent_liars(c)
    truth = c["scenarios"][int(c["answer_key"]["true_scenario"][1:]) - 1][: -len(" is lying.")]
    if sol != {truth}:
        bad.append((c["id"], sol, truth))
    if validator._answer_key_problems(CaseDefinition.from_dict(c)):
        key_bad.append(c["id"])
check(f"all {len(cases)} cases have exactly the keyed liar", not bad, str(bad[:2]))
check("all answer keys pass the validator", not key_bad, str(key_bad[:3]))

print("\n=== LI-02: No surface shortcut ===")
unbalanced = 0
hits, chance = 0, 0.0
for c in cases:
    names = [s[: -len(" is lying.")] for s in c["scenarios"]]
    counts = [sum(n in e for e in c["evidence"]) for n in names]
    unbalanced += len(set(counts)) != 1
    chance += 1 / len(names)
    hits += counts.index(max(counts)) == int(c["answer_key"]["true_scenario"][1:]) - 1
check("every witness is named equally often", unbalanced == 0, f"{unbalanced} unbalanced")
pos = [int(c["answer_key"]["true_scenario"][1:]) for c in cases if len(c["scenarios"]) == 4]
check("liar position is spread across slots", max(pos.count(i) for i in range(1, 5)) / len(pos) < 0.4)

print("\n=== LI-03: Keys, prompts and scoring ===")
check("every case has key evidence and red herrings",
      all(c["answer_key"]["key_evidence"] and c["answer_key"]["red_herrings"] for c in cases))
check("no prompt contains the hidden truth or facts",
      not any(c["hidden_truth"] in build_prompt(CaseDefinition.from_dict(c)) or '"atoms"' in build_prompt(CaseDefinition.from_dict(c))
              for c in cases))
c0 = CaseDefinition.from_dict(cases[0]); k = c0.answer_key
perfect = {"most_likely": k.true_scenario,
           "probabilities": {f"S{i}": float(f"S{i}" == k.true_scenario) for i in range(1, len(c0.scenarios) + 1)},
           "ruled_out": list(k.ruled_out), "key_evidence": list(k.key_evidence), "red_herrings": list(k.red_herrings)}
check("a perfect answer scores ~1", score_answer(c0, perfect).reward > 0.99)
check("deterministic for a fixed seed", generate(10, seed=3) == generate(10, seed=3))

print("\n" + "=" * 50)
print(f"  LIAR FAMILY: {passed} passed, {failed} failed")
print("=" * 50)
if failed:
    sys.exit(1)
