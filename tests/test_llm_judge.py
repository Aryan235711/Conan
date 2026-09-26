"""Tests for LLMJudge._safe_json — verdict extraction from raw model output.

Regression suite for the fallback parser, which previously read negative
verdicts as positive ones ("INCOHERENT" -> COHERENT, "INVALID" -> VALID)
and crashed on non-object JSON.

Run:  python3 tests/test_llm_judge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detective_engine.engine.llm_judge import LLMJudge  # noqa: E402

passed = 0
failed = 0


def check(name: str, raw: str, expected: str) -> None:
    global passed, failed
    try:
        got = LLMJudge._safe_json(raw).get("score")
    except Exception as exc:  # a crash is a failure, not an abort
        got = f"CRASH {type(exc).__name__}: {exc}"
    if got == expected:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — expected {expected!r}, got {got!r}")


print("\n=== LJ-01: Negative verdicts are never flipped ===")
check("INCOHERENT in prose", "The chain is INCOHERENT because of gaps", "INCOHERENT")
check("INVALID in prose", "Verdict: INVALID", "INVALID")
check("WEAK with negated positive", "I think it is WEAK, not strong", "WEAK")
check("FRAGILE in prose", "This conclusion is fragile.", "FRAGILE")

print("\n=== LJ-02: Positive verdicts still parse ===")
check("COHERENT in prose", "Overall the chain is coherent.", "COHERENT")
check("SURVIVES in prose", "The elimination SURVIVES scrutiny.", "SURVIVES")
check("STRONG in prose", "STRONG: found an implicit contradiction", "STRONG")

print("\n=== LJ-03: JSON extraction ===")
check("plain JSON", '{"score": "COHERENT", "chain_gaps": []}', "COHERENT")
check("JSON wrapped in prose", 'Here you go: {"score": "WEAK", "missed": ["x"]} done', "WEAK")
check("lowercase score normalized", '{"score": "strong"}', "STRONG")
check("non-string score does not crash", '{"score": 1}', "1")
check("last scored object wins", '{"score": "STRONG"} revised: {"score": "WEAK"}', "WEAK")
check("object without score is ignored", '{"note": "x"} verdict WEAK', "WEAK")

print("\n=== LJ-04: Reasoning-model <think> blocks ===")
check(
    "think block with braces and both verdicts",
    '<think>should I say {STRONG} or WEAK?</think> {"score": "WEAK"}',
    "WEAK",
)
check(
    "think block mentioning positive, answer negative",
    "<think>It might be COHERENT...</think>\nFinal: INCOHERENT",
    "INCOHERENT",
)
check("unterminated think block", "<think>STRONG STRONG STRONG", "SKIP")

print("\n=== LJ-05: Non-object and empty output ===")
check("JSON list does not crash", '["WEAK"]', "WEAK")
check("empty string", "", "SKIP")
check("no verdict at all", "I cannot evaluate this.", "SKIP")

print("\n" + "=" * 50)
print(f"  LLM JUDGE PARSER: {passed} passed, {failed} failed")
print("=" * 50)

if failed:
    sys.exit(1)
