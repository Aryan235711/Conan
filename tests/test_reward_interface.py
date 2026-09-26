"""Tests for the reward interface (Path A — reward model for LLM training).

Verifies:
  1. Basic API — score(), case_prompt(), case_evidence(), score_batch()
  2. Good reasoning gets high scores, bad reasoning gets low scores
  3. Perception trace scoring when provided
  4. Feedback and penalties are populated correctly
  5. Generated cases (no evidence_meta) degrade gracefully

Usage:
    cd Conan && PYTHONPATH=. python tests/test_reward_interface.py
"""
from __future__ import annotations

import sys
import time

from detective_engine.engine.reward_interface import RewardScorer

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

scorer = RewardScorer()
results: list[dict] = []
passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        results.append({"name": name, "status": "PASS"})
    else:
        failed += 1
        results.append({"name": name, "status": "FAIL", "detail": detail})
        print(f"  FAIL: {name} — {detail}")


# ---------------------------------------------------------------------------
# RW-01: API surface — case_ids, case_prompt, case_evidence
# ---------------------------------------------------------------------------
print("\n=== RW-01: API Surface ===")

ids = scorer.case_ids
check("RW-01a: case_ids non-empty", len(ids) > 0, f"got {len(ids)}")
check("RW-01b: C001 in case_ids", "C001" in ids)

prompt = scorer.case_prompt("C001")
check("RW-01c: prompt has evidence", "evidence" in prompt and len(prompt["evidence"]) > 0)
check("RW-01d: prompt has no solution", "solution" not in prompt and "hidden_truth" not in prompt)
check("RW-01e: prompt has scenarios", "scenarios" in prompt)
check("RW-01f: prompt has must_reject", "must_reject_false_narrative" in prompt)

evidence = scorer.case_evidence("C001")
check("RW-01g: evidence is list", isinstance(evidence, list) and len(evidence) == 6)

# ---------------------------------------------------------------------------
# RW-02: Good reasoning on C001 — should score well
# ---------------------------------------------------------------------------
print("\n=== RW-02: Good Reasoning (C001) ===")

good_analysis = {
    "observations": [
        "chair pulled back from table",
        "notebook open on table",
        "glass half full",
        "window slightly open",
        "curtains moving inward",
    ],
    "anomalies": [
        "the room appears calm but chair position contradicts undisturbed scene",
        "window open with curtains moving inward suggests recent activity",
    ],
    "hypotheses": {
        "chair pulled back from table": [
            "someone left in a hurry — staged to look normal",
            "natural position from daily use",
        ],
        "notebook open on table": [
            "left mid-task — person was interrupted",
            "placed deliberately to suggest normalcy",
        ],
        "window slightly open": [
            "escape route — someone exited through the window",
            "ventilation — natural airflow",
        ],
    },
    "elimination_target": "the false narrative of calm is hiding inside — scene was staged",
    "reasons": [
        "chair pulled back contradicts an undisturbed room",
        "window open with inward curtains shows recent opening",
        "the calm appearance is deliberately curated to hide activity",
    ],
    "false_narrative_rejection": (
        "The claim that the room is calm and nothing happened is wrong because "
        "the chair position, open window, and curtain movement all contradict "
        "an undisturbed scene. The calm is staged and misleading."
    ),
    "contradiction_notes": [
        "room described as undisturbed but chair is pulled back and window is open — "
        "these physical details contradict the surface calm"
    ],
}

r = scorer.score("C001", good_analysis)

check("RW-02a: returns dict", isinstance(r, dict))
check("RW-02b: has pillars", "pillars" in r)
check("RW-02c: has weighted", "weighted" in r and 0.0 <= r["weighted"] <= 1.0)
check("RW-02d: has grade", "grade" in r and r["grade"] in "ABCDF")
check("RW-02e: content pillar > 0", r["pillars"]["content"] > 0, f"got {r['pillars']['content']}")
check("RW-02f: integrity pillar > 0", r["pillars"]["integrity"] > 0, f"got {r['pillars']['integrity']}")
check("RW-02g: weighted > 0.2", r["weighted"] > 0.2, f"got {r['weighted']}")
check("RW-02h: has feedback list", isinstance(r["feedback"], list))
check("RW-02i: has penalties list", isinstance(r["penalties"], list))
check("RW-02j: has details", isinstance(r["details"], dict))

# ---------------------------------------------------------------------------
# RW-03: Bad reasoning on C001 — should score poorly
# ---------------------------------------------------------------------------
print("\n=== RW-03: Bad Reasoning (C001) ===")

bad_analysis = {
    "observations": [
        "the room is clearly calm and undisturbed",
        "everything looks normal and natural",
    ],
    "anomalies": ["nothing unusual"],
    "hypotheses": {
        "the room is clearly calm and undisturbed": [
            "nobody was here — it was a natural quiet evening",
        ],
    },
    "elimination_target": "nobody was here",
    "reasons": ["the room is obviously calm so nothing happened"],
    "false_narrative_rejection": "",
    "contradiction_notes": [],
}

r_bad = scorer.score("C001", bad_analysis)

check("RW-03a: bad weighted < good weighted",
      r_bad["weighted"] < r["weighted"],
      f"bad={r_bad['weighted']} vs good={r['weighted']}")
check("RW-03b: bad grade worse", r_bad["grade"] >= r["grade"],  # F > A in ord
      f"bad={r_bad['grade']} vs good={r['grade']}")
check("RW-03c: bad has feedback", len(r_bad["feedback"]) > 0,
      "expected feedback about missed concepts")
check("RW-03d: bad has penalties", len(r_bad["penalties"]) > 0,
      "expected inference leak / forbidden reasoning penalties")

# ---------------------------------------------------------------------------
# RW-04: Perception trace scoring
# ---------------------------------------------------------------------------
print("\n=== RW-04: Perception Trace ===")

r_perc = scorer.score("C001", good_analysis, perception_trace={
    "initial": ["chair pulled back", "window open", "curtains moving"],
    "later": [
        "notebook open on table",
        "glass half full",
        "room appears undisturbed",
    ],
})

check("RW-04a: perception details present",
      "perception" in r_perc.get("details", {}),
      "expected perception sub-dict in details")
if "perception" in r_perc.get("details", {}):
    pd = r_perc["details"]["perception"]
    check("RW-04b: coverage computed", pd.get("coverage") is not None,
          f"got {pd.get('coverage')}")
    check("RW-04c: id_match_rate > 0", pd.get("id_match_rate", 0) > 0,
          f"got {pd.get('id_match_rate')}")

# ---------------------------------------------------------------------------
# RW-05: score_batch
# ---------------------------------------------------------------------------
print("\n=== RW-05: Batch Scoring ===")

batch = scorer.score_batch([
    {"case_id": "C001", "analysis": good_analysis},
    {"case_id": "C001", "analysis": bad_analysis},
])

check("RW-05a: batch returns 2 results", len(batch) == 2)
check("RW-05b: first > second weighted",
      batch[0]["weighted"] > batch[1]["weighted"],
      f"{batch[0]['weighted']} vs {batch[1]['weighted']}")

# ---------------------------------------------------------------------------
# RW-06: Generated case (no evidence_meta) — graceful degradation
# ---------------------------------------------------------------------------
print("\n=== RW-06: Generated Case Degradation ===")

# Find a generated case (C007+)
gen_ids = [cid for cid in scorer.case_ids if cid >= "C007"]
if gen_ids:
    gen_id = gen_ids[0]
    gen_prompt = scorer.case_prompt(gen_id)
    gen_analysis = {
        "observations": [e[:40] for e in gen_prompt["evidence"][:3]],
        "anomalies": ["something seems off"],
        "hypotheses": {gen_prompt["evidence"][0][:30]: ["one explanation", "another"]},
        "elimination_target": "the false narrative",
        "reasons": ["evidence contradicts the surface story"],
        "contradiction_notes": [],
    }
    r_gen = scorer.score(gen_id, gen_analysis)
    check("RW-06a: gen case scores without error", isinstance(r_gen, dict))
    check("RW-06b: gen perception is 0 (no meta)",
          r_gen["pillars"]["perception"] == 0.0,
          f"got {r_gen['pillars']['perception']}")
else:
    check("RW-06a: no generated cases to test", True)
    check("RW-06b: skipped", True)

# ---------------------------------------------------------------------------
# RW-07: Unknown case raises KeyError
# ---------------------------------------------------------------------------
print("\n=== RW-07: Error Handling ===")

try:
    scorer.score("ZZZZ", {})
    check("RW-07a: unknown case raises", False, "no exception raised")
except KeyError:
    check("RW-07a: unknown case raises KeyError", True)

# ---------------------------------------------------------------------------
# RW-08: Performance — single score call
# ---------------------------------------------------------------------------
print("\n=== RW-08: Performance ===")

t0 = time.perf_counter()
for _ in range(100):
    scorer.score("C001", good_analysis)
elapsed = time.perf_counter() - t0
avg_ms = elapsed / 100 * 1000

check("RW-08a: avg score < 10ms", avg_ms < 10, f"avg={avg_ms:.2f}ms")
print(f"  100 calls in {elapsed*1000:.1f}ms (avg {avg_ms:.2f}ms/call)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  REWARD INTERFACE: {passed} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    print("\nFailed tests:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  FAIL: {r['name']} — {r.get('detail', '')}")
    sys.exit(1)
