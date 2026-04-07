"""Test causality validator V10 — anti-gaming, concept-origin, diversity, traps."""
from detective_engine.engine.causality_validator import (
    CausalityValidator, _significant_concepts,
)
from detective_engine.engine.models import AnalysisRecord
from detective_engine.engine.case_loader import CaseLoader

loader = CaseLoader()
cases, insights = loader.load_all()
case_map = {c.id: c for c in cases}
case = case_map["C001"]
print(f"✅ Loaded {case.id}: {case.title}")
print(f"   Evidence: {len(case.evidence)} items")

cv = CausalityValidator()

# ══════════════════════════════════════════════════════════════════
# TEST 1: STRONG analysis — honest chain through all phases
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("TEST 1: STRONG — honest derivation chain")
print("═" * 60)

strong = AnalysisRecord(
    observations=[
        "chair is slightly pulled back from the table",
        "notebook lies open on the table",
        "glass of water is half full",
        "window is slightly open",
        "curtains are moving inward",
        "room appears otherwise undisturbed",
    ],
    anomalies=[
        "chair pulled back yet room called undisturbed — contradiction",
        "notebook open suggests recent activity despite calm appearance",
    ],
    hypotheses={
        "chair is slightly pulled back from the table": [
            "someone was sitting and left recently — chair pushed back on exit",
            "chair was deliberately placed back to appear undisturbed",
        ],
        "notebook lies open on the table": [
            "someone was reading or writing — activity trace",
            "notebook was staged open to create the appearance of normal use",
        ],
        "glass of water is half full": [
            "glass was being used — someone was present recently",
            "glass was placed as part of staging the scene",
        ],
    },
    elimination_target="reject calm narrative — scene is staged",
    reasons=[
        "chair pulled back contradicts undisturbed room — activity happened",
        "notebook open and glass half full confirm recent presence",
        "curtains moving from open window means someone adjusted environment",
        "the calm appearance was deliberately constructed — staged scene",
    ],
    false_narrative_rejection=(
        "The room is calm so nothing happened is wrong — the chair, notebook, "
        "and glass all show someone was here, contradicting undisturbed"
    ),
    contradiction_notes=[
        "chair pulled and notebook open vs undisturbed appearance",
    ],
)

result_strong = cv.validate(case, strong)
print(f"\n  Total: {result_strong.total_score}/{result_strong.max_score}")
for d in result_strong.dimensions:
    icon = "✅" if d.score >= d.max_score else ("🔶" if d.score > 0 else "❌")
    print(f"  {icon} [{d.severity:10}] {d.name}: {d.score}/{d.max_score} — {d.details}")
print(f"  Full chains: {len(result_strong.full_chains)}")
print(f"  Phantoms:    {len(result_strong.phantom_concepts)}")
print(f"  Leaps:       {len(result_strong.inference_leaps)}")
if result_strong.chain_trace_lines:
    print(f"  Chain trace:")
    for line in result_strong.chain_trace_lines[:10]:
        print(line)

# ══════════════════════════════════════════════════════════════════
# TEST 2: WEAK analysis — phantom derivation
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("TEST 2: WEAK — phantom derivation (correct answer, broken chain)")
print("═" * 60)

weak = AnalysisRecord(
    observations=[
        "the room is quiet",
        "everything looks normal",
        "there is a table in the room",
        "there is light coming through the window",
        "the space feels empty",
    ],
    anomalies=[
        "something feels off about the scene",
    ],
    hypotheses={
        "the room is quiet": [
            "nobody is here right now",
            "the person left a while ago",
        ],
        "everything looks normal": [
            "nothing unusual happened",
            "maybe things are exactly as they should be",
        ],
        "there is a table in the room": [
            "the table is just furniture",
            "someone might have used the table",
        ],
    },
    elimination_target="scene is staged — reject the calm narrative",
    reasons=[
        "the chair being pulled back shows someone was recently sitting there",
        "notebook was left open proving activity contradicts undisturbed claim",
        "glass half full confirms presence",
        "curtains moving inward from open window means environment was altered",
    ],
    false_narrative_rejection=(
        "The room is calm so nothing happened is false — "
        "chair pulled, notebook open, glass half full"
    ),
    contradiction_notes=[
        "chair and notebook vs undisturbed",
    ],
)

result_weak = cv.validate(case, weak)
print(f"\n  Total: {result_weak.total_score}/{result_weak.max_score}")
for d in result_weak.dimensions:
    icon = "✅" if d.score >= d.max_score else ("🔶" if d.score > 0 else "❌")
    print(f"  {icon} [{d.severity:10}] {d.name}: {d.score}/{d.max_score} — {d.details}")
print(f"  Full chains: {len(result_weak.full_chains)}")
print(f"  Phantoms:    {len(result_weak.phantom_concepts)} → {result_weak.phantom_concepts[:8]}")
print(f"  Leaps:       {len(result_weak.inference_leaps)} → {result_weak.inference_leaps[:8]}")
if result_weak.chain_trace_lines:
    print(f"  Chain trace:")
    for line in result_weak.chain_trace_lines[:10]:
        print(line)

# ══════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("COMPARISON")
print("═" * 60)
print(f"  Strong: {result_strong.total_score}/{result_strong.max_score}  (honest chain)")
print(f"  Weak:   {result_weak.total_score}/{result_weak.max_score}  (phantom derivation)")
delta = result_strong.total_score - result_weak.total_score
print(f"  Delta:  +{delta} points advantage for honest reasoning")

assert result_strong.total_score > result_weak.total_score, (
    f"FAIL: Strong ({result_strong.total_score}) should beat "
    f"Weak ({result_weak.total_score})"
)
assert result_weak.phantom_concepts, "FAIL: Weak should have phantom derivations"
assert result_strong.max_score == 11, f"FAIL: max_score should be 11, got {result_strong.max_score}"

# Check severity labels present
severities = {d.severity for d in result_strong.dimensions}
assert "fatal" in severities, "FAIL: missing fatal severity dimension"
assert "bonus" in severities, "FAIL: missing bonus severity dimension"
assert "structural" in severities, "FAIL: missing structural severity dimension"

dim_names = {d.name for d in result_strong.dimensions}
assert "hypothesis_diversity" in dim_names, "FAIL: missing V10 hypothesis_diversity dimension"
assert "root_validity" in dim_names, "FAIL: missing V10 root_validity dimension"

# Check chain trace produced
assert result_strong.chain_trace_lines, "FAIL: strong should have chain trace lines"
assert result_weak.chain_trace_lines, "FAIL: weak should have chain trace lines"

print("\n✅ CORE ASSERTIONS PASSED — V10 causality validator works correctly")

# ══════════════════════════════════════════════════════════════════
# TEST 3: SYNONYM NORMALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("TEST 3: SYNONYM NORMALIZATION")
print("═" * 60)

# Test that different phrasings map to same canonical concept
pairs = [
    ("cleaned the surface", "wiped the surface", "clean"),
    ("fingerprints on glass", "prints on tumbler", "print"),
    ("broken window pane", "shattered windowsill", "break"),
    ("someone moved the chair", "seat was shifted", "move"),
]
all_passed = True
for phrase_a, phrase_b, expected_canonical in pairs:
    concepts_a = _significant_concepts(phrase_a)
    concepts_b = _significant_concepts(phrase_b)
    if expected_canonical in concepts_a and expected_canonical in concepts_b:
        print(f"  ✅ '{phrase_a}' ∩ '{phrase_b}' → {{{expected_canonical}}}")
    else:
        print(f"  ❌ FAIL: '{phrase_a}' → {concepts_a}, '{phrase_b}' → {concepts_b}")
        all_passed = False
assert all_passed, "FAIL: synonym normalization failed"

# ══════════════════════════════════════════════════════════════════
# TEST 4: CONFIDENCE QUALIFIER (hedging language)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("TEST 4: CONFIDENCE QUALIFIER")
print("═" * 60)

hedged = AnalysisRecord(
    observations=[
        "chair is slightly pulled back, likely indicating recent use",
        "notebook open suggests someone was reading or writing",
        "glass half full is consistent with recent presence",
    ],
    anomalies=[
        "chair position is more probable to be from someone leaving quickly",
        "the calm appearance probably masks recent activity",
    ],
    hypotheses={
        "chair is slightly pulled back from the table": [
            "likely someone was sitting and left — chair pushed back on exit",
            "possibly the chair was staged to appear undisturbed",
        ],
        "notebook lies open on the table": [
            "suggests recent reading activity",
        ],
    },
    elimination_target="on balance, the scene is staged — reject calm narrative",
    reasons=[
        "the weight of evidence points to recent activity",
        "curtains moving indicates environmental alteration",
    ],
    false_narrative_rejection="The calm narrative is unlikely given the physical evidence",
    contradiction_notes=[
        "chair pulled and notebook open vs undisturbed appearance",
    ],
)
result_hedged = cv.validate(case, hedged)
conf_dim = [d for d in result_hedged.dimensions if d.name == "confidence_qualifier"][0]
print(f"  Confidence qualifier: {conf_dim.score}/{conf_dim.max_score} — {conf_dim.details}")
assert conf_dim.score == 1, f"FAIL: hedged analysis should get confidence bonus, got {conf_dim.score}"
print("  ✅ Hedged language correctly detected → bonus point awarded")

# ══════════════════════════════════════════════════════════════════
# TEST 5: ANTI-GAMING — ungrounded hedging should NOT score
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("TEST 5: ANTI-GAMING CONFIDENCE")
print("═" * 60)

gamed = AnalysisRecord(
    observations=[
        "chair near table",
        "window open",
        "notebook on desk",
    ],
    anomalies=[
        "probably perhaps likely maybe on balance more probable",
        "diagnostic posterior prior confidence certainty uncertain",
    ],
    hypotheses={
        "chair near table": [
            "probably likely maybe something happened",
            "possibly perhaps some event occurred",
        ],
    },
    elimination_target="likely probably perhaps staged",
    reasons=[
        "more likely than not probably perhaps",
        "confidence posterior prior likely",
        "on balance possibly maybe",
    ],
)
result_gamed = cv.validate(case, gamed)
gamed_conf = [d for d in result_gamed.dimensions if d.name == "confidence_qualifier"][0]
print(f"  Anti-gaming qualifier: {gamed_conf.score}/{gamed_conf.max_score} — {gamed_conf.details}")
assert gamed_conf.score == 0, "FAIL: ungrounded hedging spam should not earn confidence bonus"
print("  ✅ Ungrounded hedging spam correctly rejected")

# ══════════════════════════════════════════════════════════════════
# TEST 6: ROOT VALIDITY / INFERENCE TRAPS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("TEST 6: INFERENCE TRAP PENALTY")
print("═" * 60)

trap_record = AnalysisRecord(
    observations=[
        "chair is slightly pulled back from the table",
        "notebook lies open on the table",
        "glass of water is half full",
        "window is slightly open",
        "curtains are moving inward",
    ],
    anomalies=[
        "window is open",
        "room appears calm",
    ],
    hypotheses={
        "window is slightly open": [
            "someone entered through the window",
            "someone exited through the window",
        ],
    },
    elimination_target="someone exited through the window",
    reasons=[
        "the window proves exit",
        "room is calm so nothing happened after that",
        "therefore the window is the answer",
    ],
)
trap_result = cv.validate(case, trap_record)
root_dim = [d for d in trap_result.dimensions if d.name == "root_validity"][0]
print(f"  Root validity: {root_dim.score}/{root_dim.max_score} — {root_dim.details}")
print(f"  Trap penalties: {trap_result.inference_trap_penalties}")
assert trap_result.inference_trap_penalties >= 1, "FAIL: trap reasoning should trigger a penalty"
assert trap_result.inference_traps_triggered, "FAIL: triggered traps should be reported"
print("  ✅ Inference trap penalty correctly applied")

print("\n" + "═" * 60)
print("ALL V10 TESTS PASSED")
print("═" * 60)
