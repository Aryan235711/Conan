"""Tests for the reasoning parser + phase generator + score_raw integration.

Covers:
  1. Layer A — structure extraction from various LLM formatting styles
  2. Layer B — structural validity rejection
  3. Layer C — semantic grounding rejection
  4. PhasePromptGenerator — phase-separated generation flow
  5. score_raw — end-to-end free-form text → reward signal
  6. Trace provenance — only system-generated traces accepted

Usage:
    cd Conan && PYTHONPATH=. python tests/test_reasoning_parser.py
"""
from __future__ import annotations

import sys

from detective_engine.engine.reasoning_parser import (
    parse_reasoning_output,
    ParserConfig,
)
from detective_engine.engine.phase_generator import PhasePromptGenerator
from detective_engine.engine.reward_interface import RewardScorer

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

scorer = RewardScorer()
passed = 0
failed = 0
results: list[dict] = []


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
# PA-01: Layer A — Markdown heading format
# ---------------------------------------------------------------------------
print("\n=== PA-01: Layer A — Markdown Headings ===")

MARKDOWN_TEXT = """
## Observations
- A chair is slightly pulled back from the table
- A notebook lies open on the table
- A glass of water is half full
- A window is slightly open
- Curtains are moving inward

## Anomalies
- Chair position contradicts an undisturbed room
- Open window with inward curtain movement suggests recent activity

## Hypotheses
"chair pulled back":
- Someone left in a hurry and staged the scene
- Natural position from daily use

"window open":
- Escape route — someone exited through the window
- Ventilation — natural airflow

## Contradiction Detection
- Room described as undisturbed but chair is pulled back — physical contradiction

## False Narrative
The obvious but wrong explanation is that the room is calm and nothing happened. This fails because the chair position and open window contradict surface calm.

## Elimination
The false narrative of calm is wrong — the scene was staged to hide activity

## Reasoning
- Chair pulled back contradicts an undisturbed room
- Window open with inward curtains shows recent opening
- The calm appearance is deliberately curated to hide activity
"""

r = parse_reasoning_output(MARKDOWN_TEXT)
check("PA-01a: parses successfully", r is not None)
if r:
    a = r.analysis
    check("PA-01b: 5 observations", len(a["observations"]) == 5,
          f"got {len(a['observations'])}")
    check("PA-01c: 2 anomalies", len(a["anomalies"]) == 2,
          f"got {len(a['anomalies'])}")
    check("PA-01d: hypotheses parsed", len(a["hypotheses"]) >= 2,
          f"got {len(a['hypotheses'])}")
    check("PA-01e: elimination non-empty", len(a["elimination_target"]) > 10)
    check("PA-01f: 3 reasons", len(a["reasons"]) == 3,
          f"got {len(a['reasons'])}")
    check("PA-01g: false narrative present", len(a["false_narrative_rejection"]) > 10)
    check("PA-01h: contradiction notes", len(a["contradiction_notes"]) >= 1)
    check("PA-01i: raw_text preserved", r.raw_text == MARKDOWN_TEXT)

# ---------------------------------------------------------------------------
# PA-02: Layer A — Bold label format
# ---------------------------------------------------------------------------
print("\n=== PA-02: Layer A — Bold Labels ===")

BOLD_TEXT = """
**Observations:**
- The chair is pulled back slightly
- Notebook open on the table
- Glass half full of water
- Window slightly ajar
- Curtains moving inward from breeze

**Anomalies:**
- Chair out of place in an otherwise undisturbed room
- Window open suggests someone passed through

**Hypotheses:**
"chair pulled back":
- Staged scene to look normal
- Someone sat and left recently

"notebook open":
- Interrupted mid-task
- Deliberately placed

**Elimination:**
The room was staged — the calm is an illusion

**Reasoning:**
- Physical evidence contradicts the surface calm
- Chair and window positions reveal activity
- The undisturbed appearance is curated
"""

r2 = parse_reasoning_output(BOLD_TEXT)
check("PA-02a: bold format parses", r2 is not None)
if r2:
    check("PA-02b: observations found", len(r2.analysis["observations"]) >= 4)
    check("PA-02c: hypotheses found", len(r2.analysis["hypotheses"]) >= 2)

# ---------------------------------------------------------------------------
# PA-03: Layer A — Numbered format
# ---------------------------------------------------------------------------
print("\n=== PA-03: Layer A — Colon Labels ===")

COLON_TEXT = """
Observations:
1. Chair pulled back from the table slightly
2. Notebook is open on the table surface
3. A glass of water is half full
4. Window is slightly open and curtains blowing inward
5. Room otherwise appears undisturbed and calm

Anomalies:
1. Chair position contradicts the undisturbed room
2. Window open with moving curtains suggests recent passage

Hypotheses:
"chair pulled back":
- The chair was moved during a staged scene
- Someone sat there and left normally

"window open":
- Used as an escape route
- Just for ventilation purposes

Elimination:
The scene was clearly staged and the calm is fabricated

Reasoning:
1. Chair position is inconsistent with undisturbed claim
2. Window opening pattern suggests deliberate manipulation
3. Surface calm is a deception tactic
"""

r3 = parse_reasoning_output(COLON_TEXT)
check("PA-03a: colon format parses", r3 is not None)
if r3:
    check("PA-03b: 5 observations", len(r3.analysis["observations"]) == 5,
          f"got {len(r3.analysis['observations'])}")

# ---------------------------------------------------------------------------
# PA-04: Layer B — Missing core sections → reject
# ---------------------------------------------------------------------------
print("\n=== PA-04: Layer B — Structural Rejection ===")

# Missing observations AND hypotheses (2 missing > threshold of 1)
MISSING_SECTIONS = """
## Elimination
Something happened here

## Reasoning
- Because I said so
- The evidence shows it
"""

r4 = parse_reasoning_output(MISSING_SECTIONS)
check("PA-04a: missing 2+ sections rejected", r4 is None)

# Too few observations
FEW_OBS = """
## Observations
- chair

## Hypotheses
"chair":
- staged

## Elimination
Scene was staged for some reason here

## Reasoning
- Chair was moved to create the appearance of calm
"""

r5 = parse_reasoning_output(FEW_OBS)
check("PA-04b: 1-word observation rejected", r5 is None)

# ---------------------------------------------------------------------------
# PA-05: Layer C — Scaffolding gaming (perfect format, zero content)
# ---------------------------------------------------------------------------
print("\n=== PA-05: Layer C — Scaffolding Gaming ===")

# Scaffolding: hypothesis keys and explanations share NO tokens with observations.
# Reasoning shares no content tokens with observations either.
SCAFFOLD = """
## Observations
- alpha bravo charlie delta exists here
- echo foxtrot golf hotel noticed here
- india juliet kilo lima present there
- mike november oscar papa located nearby
- quebec romeo sierra tango found somewhere

## Anomalies
- something seems inconsistent about the situation

## Hypotheses
"zulu yankee xray whiskey":
- maybe the situation is unusual because reasons
- or perhaps the arrangement is natural overall

"venus mercury pluto neptune":
- perhaps someone created this arrangement recently
- or it was always configured in that fashion

## Elimination
The explanation is that circumstances are anomalous in this scenario

## Reasoning
- Because the anomaly shows inconsistency in the arrangement
- The pattern of configuration reveals deliberate setup clearly
- Surface conditions mask the underlying actual reality completely
"""

r6 = parse_reasoning_output(SCAFFOLD)
check("PA-05a: scaffolding with no cross-ref rejected", r6 is None)

# ---------------------------------------------------------------------------
# PA-06: Layer C — Hypothesis monoculture
# ---------------------------------------------------------------------------
print("\n=== PA-06: Layer C — Hypothesis Monoculture ===")

MONO = """
## Observations
- A chair is slightly pulled back from the table
- A notebook lies open on the table surface
- Window is slightly open with curtains moving

## Anomalies
- Chair position seems unusual for undisturbed room

## Hypotheses
"chair pulled back":
- The staged scene was arranged by someone here
- The staged scene was arranged by a person here

## Elimination
The scene was staged to create an illusion of calm

## Reasoning
- Chair pulled back reveals staging in this room
- Window position confirms deliberate manipulation here
- Scene calm is artificially constructed by staging
"""

r7 = parse_reasoning_output(MONO)
check("PA-06a: monoculture rejected", r7 is None)

# ---------------------------------------------------------------------------
# PA-07: Empty / garbage input
# ---------------------------------------------------------------------------
print("\n=== PA-07: Garbage Input ===")

check("PA-07a: empty string", parse_reasoning_output("") is None)
check("PA-07b: whitespace only", parse_reasoning_output("   \n  \n  ") is None)
check("PA-07c: random text", parse_reasoning_output("hello world foo bar baz") is None)
check("PA-07d: None-ish", parse_reasoning_output("I don't know what to say") is None)

# ---------------------------------------------------------------------------
# PA-08: score_raw end-to-end
# ---------------------------------------------------------------------------
print("\n=== PA-08: score_raw End-to-End ===")

raw_result = scorer.score_raw("C001", MARKDOWN_TEXT)
check("PA-08a: score_raw returns dict", raw_result is not None)
if raw_result:
    check("PA-08b: has weighted", "weighted" in raw_result)
    check("PA-08c: has parse_warnings", "parse_warnings" in raw_result)
    check("PA-08d: has raw_text", "raw_text" in raw_result)
    check("PA-08e: weighted > 0", raw_result["weighted"] > 0,
          f"got {raw_result['weighted']}")
    check("PA-08f: sections_found", len(raw_result.get("sections_found", [])) >= 4)

# score_raw on garbage → None
garbage_result = scorer.score_raw("C001", "lol what")
check("PA-08g: garbage → None", garbage_result is None)

# ---------------------------------------------------------------------------
# PA-09: Trace provenance — only system-generated accepted
# ---------------------------------------------------------------------------
print("\n=== PA-09: Trace Provenance ===")

# System trace (trusted)
sys_trace = {
    "initial": ["chair pulled back", "window open"],
    "later": ["notebook open", "glass half full"],
    "_system_generated": True,
}

# Self-reported trace (untrusted — no _system_generated flag)
fake_trace = {
    "initial": ["chair pulled back", "window open"],
    "later": ["notebook open", "glass half full"],
}

r_sys = scorer.score_raw("C001", MARKDOWN_TEXT, perception_trace=sys_trace)
r_fake = scorer.score_raw("C001", MARKDOWN_TEXT, perception_trace=fake_trace)

check("PA-09a: system trace accepted", r_sys is not None)
check("PA-09b: fake trace silently dropped", r_fake is not None)
if r_sys and r_fake:
    # With system trace, perception details should be richer
    sys_perc = r_sys.get("details", {}).get("perception")
    fake_perc = r_fake.get("details", {}).get("perception")
    check("PA-09c: system trace has perception details",
          sys_perc is not None and sys_perc.get("id_match_rate", 0) > 0,
          f"got {sys_perc}")

# ---------------------------------------------------------------------------
# PA-10: PhasePromptGenerator
# ---------------------------------------------------------------------------
print("\n=== PA-10: PhasePromptGenerator ===")

case_prompt = scorer.case_prompt("C001")
gen = PhasePromptGenerator(case_prompt)

# Generate prompts for all phases
p0 = gen.phase_prompt(0)
check("PA-10a: phase 0 prompt has evidence", "evidence" in p0.lower())
check("PA-10b: phase 0 prompt has micro-capture", "micro-capture" in p0.lower() or "MICRO" in p0)

# Record mock responses
gen.record_response(0, """
- Chair pulled back from table
- Window slightly open
- Notebook on table
""")

p1 = gen.phase_prompt(1)
check("PA-10c: phase 1 includes phase 0 context", "Phase 0" in p1)

gen.record_response(1, """
- Chair slightly pulled back from table
- Notebook lies open on the surface
- Glass of water half full on table
- Window is slightly open letting air in
- Curtains moving inward from the breeze
""")

gen.phase_prompt(2)
gen.record_response(2, """
- Chair position contradicts undisturbed room narrative
- Open window with inward curtains suggests activity
""")

gen.phase_prompt(3)
gen.record_response(3, """
"chair pulled back":
- Staged to look like normal activity
- Someone left in a hurry recently

"window open":
- Used as escape route by someone
- Simply opened for ventilation purposes
""")

gen.phase_prompt(4)
gen.record_response(4, "- Room undisturbed vs chair pulled back")

gen.phase_prompt(5)
gen.record_response(5, "The calm room narrative is false because physical evidence contradicts it")

gen.phase_prompt(6)
gen.record_response(6, """
- The calm scene is staged — this is the false narrative
- Chair position contradicts undisturbed claims strongly
- Window opening reveals deliberate recent activity here
- Surface calm was curated to hide something important
""")

# Build outputs
analysis = gen.build_analysis()
check("PA-10d: analysis has observations", len(analysis["observations"]) >= 4,
      f"got {len(analysis['observations'])}")
check("PA-10e: analysis has hypotheses", len(analysis["hypotheses"]) >= 1)
check("PA-10f: analysis has elimination", len(analysis["elimination_target"]) > 5)

trace = gen.build_trace()
check("PA-10g: trace is system-generated", trace.get("_system_generated") is True)
check("PA-10h: trace has initial", len(trace.get("initial", [])) >= 2)
check("PA-10i: trace has timestamps",
      "_phase_timestamps" in trace and len(trace["_phase_timestamps"]) > 0)

# Score with system trace
result = scorer.score("C001", analysis, perception_trace=trace)
check("PA-10j: phase gen score > 0", result["weighted"] > 0,
      f"got {result['weighted']}")

check("PA-10k: completed phases", len(gen.completed_phases) >= 5)

# ---------------------------------------------------------------------------
# PA-11: score_raw_batch
# ---------------------------------------------------------------------------
print("\n=== PA-11: score_raw_batch ===")

batch = scorer.score_raw_batch([
    {"case_id": "C001", "raw_text": MARKDOWN_TEXT},
    {"case_id": "C001", "raw_text": "garbage nonsense"},
])
check("PA-11a: batch returns 2", len(batch) == 2)
check("PA-11b: first parsed", batch[0] is not None)
check("PA-11c: second rejected", batch[1] is None)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  PARSER + PHASE GEN: {passed} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    print("\nFailed tests:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  FAIL: {r['name']} — {r.get('detail', '')}")
    sys.exit(1)
