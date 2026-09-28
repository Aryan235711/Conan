"""Integration, adversarial gaming, and performance tests.

Covers three gaps:
  1. Full pipeline integration — feed synthetic AnalysisRecords through
     the entire validator stack (reasoning graph, causality, Bayesian,
     perception, final scoring) without interactive I/O.
  2. Adversarial gaming — attempts to cheat or exploit the scoring
     system and verifies that defenses hold.
  3. Performance benchmarks — measures wall-clock time for key operations
     across all cases.

Usage:
    python test_integration.py
"""
from __future__ import annotations

import time
from dataclasses import field

from detective_engine.engine.case_loader import CaseLoader
from detective_engine.engine.models import (
    AnalysisRecord,
    ObservationEvent,
    PerceptionConfig,
    PerceptionTrace,
)
from detective_engine.engine.validator import Validator, PILLAR_WEIGHTS
from detective_engine.engine.reasoning_graph import ReasoningGraphValidator
from detective_engine.engine.causality_validator import CausalityValidator
from detective_engine.engine.bayesian_validator import BayesianValidator
from detective_engine.engine.perception_integrity import (
    PerceptionIntegrityValidator,
    match_evidence_ids,
    perception_adjustment,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

loader = CaseLoader()
all_cases, _ = loader.load_all()
cases = {c.id: c for c in all_cases}
cfg = PerceptionConfig()

validator = Validator()
reasoning = ReasoningGraphValidator()
causality = CausalityValidator()
bayesian = BayesianValidator()
perception = PerceptionIntegrityValidator(cfg)

results: list[dict] = []


def ts(offset: float = 0.0) -> float:
    return 1_000_000.0 + offset


def make_ev(text, phase, idx, t, ev_map):
    return ObservationEvent(
        text=text, timestamp=ts(t), phase=phase,
        order_index=idx, evidence_ids=match_evidence_ids(text, ev_map, cfg),
    )


def full_pipeline(
    case_id: str,
    record: AnalysisRecord,
    trace: PerceptionTrace | None = None,
    used_ids: set[str] | None = None,
):
    """Run the complete validator stack, return EvaluationResult + pillar scores."""
    case = cases[case_id]
    ev_map = getattr(case, "evidence_map", {})

    rg = reasoning.validate(case, record)
    caus = causality.validate(case, record)
    bay = bayesian.validate(case, record)

    perc_result = perception.validate(
        trace=trace, evidence_map=ev_map,
        used_in_reasoning_ids=used_ids or set(),
    )

    ev = validator.evaluate(
        case, record,
        reasoning_graph_score=rg.total_score,
        reasoning_graph_results=[
            {"name": l.name, "score": l.score, "details": l.details}
            for l in rg.links
        ],
        causality_score=caus.total_score,
        causality_max=caus.max_score,
        causality_results=[
            {"name": d.name, "score": d.score, "max": d.max_score,
             "details": d.details, "severity": d.severity}
            for d in caus.dimensions
        ],
        causality_phantom=caus.phantom_concepts,
        causality_leaps=caus.inference_leaps,
        causality_temporal=caus.temporal_violations,
        causality_chain_trace=caus.chain_trace_lines,
        causality_trap_penalties=caus.inference_trap_penalties,
        causality_traps_triggered=caus.inference_traps_triggered,
        bayesian_score=bay.total_score,
        bayesian_max=bay.max_score,
        bayesian_results=[
            {"name": d.name, "score": d.score, "max": d.max_score, "details": d.details}
            for d in bay.dimensions
        ],
        bayesian_traps=bay.cognitive_traps_triggered,
    )

    # Set perception pillar
    if perc_result is not None and perc_result.evaluation_confidence >= 0.60:
        adj = perception_adjustment(perc_result, cfg)
        ev.pillar_perception = max(0.0, min(1.0, 0.5 + adj / 4.0))
    ev.weighted_score = (
        PILLAR_WEIGHTS["content"] * ev.pillar_content
        + PILLAR_WEIGHTS["structure"] * ev.pillar_structure
        + PILLAR_WEIGHTS["integrity"] * ev.pillar_integrity
        + PILLAR_WEIGHTS["perception"] * ev.pillar_perception
    )

    return ev, perc_result


def check(name, condition, detail=""):
    status = "✅" if condition else "❌"
    msg = f"  {status} {name}"
    if detail and not condition:
        msg += f" — {detail}"
    print(msg)
    results.append({"name": name, "pass": condition})
    return condition


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: FULL PIPELINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("SECTION 1: FULL PIPELINE INTEGRATION")
print("═" * 60)

# --- INT-01: C001 — strong honest run end-to-end ---
print("\n── INT-01: C001 strong honest run ──")

ev_map_c1 = getattr(cases["C001"], "evidence_map", {})
record_c1 = AnalysisRecord(
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
            "chair was deliberately placed to appear undisturbed",
        ],
        "notebook lies open on the table": [
            "someone was reading or writing — activity trace",
            "notebook was staged open to simulate normal use",
        ],
        "glass of water is half full": [
            "glass was being used — someone present recently",
            "glass placed as part of staging the scene",
        ],
    },
    contradiction_notes=[
        "chair pulled back means someone was here yet room is undisturbed",
    ],
    false_narrative_rejection=(
        "the story that the room was calm and nobody was here is wrong — "
        "the chair and notebook are activity traces that contradict the "
        "undisturbed appearance. The scene was staged to look normal."
    ),
    elimination_target="the false narrative of calm is hiding inside — scene was staged",
    reasons=[
        "chair displaced from table proves recent physical presence",
        "notebook open shows interrupted activity",
        "undisturbed appearance contradicts activity traces — scene was staged",
    ],
)

trace_c1 = PerceptionTrace(
    initial=[
        make_ev("chair slightly pulled back", "initial", 0, 0, ev_map_c1),
        make_ev("room appears undisturbed", "initial", 1, 1, ev_map_c1),
        make_ev("notebook open on table", "initial", 2, 2, ev_map_c1),
    ],
    later=[
        make_ev("window slightly open", "later", 0, 10, ev_map_c1),
        make_ev("curtains moving inward", "later", 1, 11, ev_map_c1),
        make_ev("glass of water half full", "later", 2, 12, ev_map_c1),
        make_ev("window safety stop opens only 10 cm", "later", 3, 13, ev_map_c1),
        make_ev("no wardrobe and under the bed is empty", "later", 4, 14, ev_map_c1),
        make_ev("glass has no fingerprints and no lip marks", "later", 5, 15, ev_map_c1),
        make_ev("notebook page dated three weeks ago and blank", "later", 6, 16, ev_map_c1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
record_c1.perception_trace = trace_c1

ev, perc = full_pipeline("C001", record_c1, trace_c1, {"ev_01", "ev_02", "ev_06"})

check("INT-01 passes", ev.passed)
check("INT-01 grade A or B", ev.grade in ("A", "B"), f"got {ev.grade}")
check("INT-01 elimination correct", ev.elimination_correct)
check("INT-01 false narrative rejected", ev.false_narrative_rejected)
check("INT-01 causality > 0", ev.causality_score > 0, f"got {ev.causality_score}")
check("INT-01 reasoning graph > 0", ev.reasoning_graph_score > 0)
check("INT-01 pillar_content > 0", ev.pillar_content > 0, f"got {ev.pillar_content:.2f}")
check("INT-01 pillar_integrity > 0", ev.pillar_integrity > 0, f"got {ev.pillar_integrity:.2f}")
check("INT-01 weighted_score > 0", ev.weighted_score > 0, f"got {ev.weighted_score:.2f}")
check("INT-01 perception coverage", perc.coverage is not None and perc.coverage >= 0.75)
print(f"  Score: {ev.earned}/{ev.max_possible} ({ev.confidence_score:.0%}) "
      f"weighted={ev.weighted_score:.2f}")

# --- INT-02: C004 — timeline case with correct physics-based reasoning ---
print("\n── INT-02: C004 timeline case ──")

ev_map_c4 = getattr(cases["C004"], "evidence_map", {})
record_c4 = AnalysisRecord(
    observations=[
        "body found at 08:00 in locked apartment",
        "rigor mortis in jaw and neck but not limbs",
        "half-eaten meal dry and cold on table",
        "thermostat turned off at 23:14",
        "apartment temperature 16 degrees",
        "witness a claims phone call at 01:00",
        "witness b saw victim at shop 22:30",
        "receipt timestamped 22:28",
        "phone shows no outgoing calls after 23:00",
    ],
    anomalies=[
        "witness a claims call at 01:00 but phone shows no calls after 23:00 — contradiction",
        "partial rigor at 08:00 suggests death 8-10 hours prior, around 22:00-00:00",
    ],
    hypotheses={
        "rigor mortis in jaw and neck but not limbs": [
            "death occurred 8-10 hours before discovery, around 22:00-00:00",
            "cold apartment slowed rigor — death may have been slightly later",
        ],
        "thermostat turned off at 23:14": [
            "victim turned off heating as last deliberate action before death",
            "another person turned off heating after the victim died",
        ],
        "phone shows no outgoing calls after 23:00": [
            "victim was dead or incapacitated by 23:00 — witness a lied about the 01:00 call",
            "victim's phone was taken or disabled after 23:00",
        ],
    },
    contradiction_notes=[
        "witness a says spoke to victim at 01:00 but phone records show no calls after 23:00",
        "partial rigor mortis timeline contradicts witness a claim of victim alive at 01:00",
    ],
    false_narrative_rejection=(
        "the story that witness a spoke to the victim at 01:00 is false — "
        "phone records contradict this. Witness a fabricated the call to "
        "move the timeline past the actual time of death."
    ),
    elimination_target="witness a is truthful",
    reasons=[
        "phone records show no outgoing calls after 23:00 — witness a lied about 01:00 call",
        "partial rigor at 08:00 places death between 22:00 and 00:00",
        "thermostat shutoff at 23:14 was likely the victim's last action",
    ],
)

trace_c4 = PerceptionTrace(
    initial=[
        make_ev("rigor mortis jaw and neck not limbs", "initial", 0, 0, ev_map_c4),
        make_ev("phone no outgoing calls after 23:00", "initial", 1, 1, ev_map_c4),
    ],
    later=[
        make_ev("thermostat turned off 23:14", "later", 0, 10, ev_map_c4),
        make_ev("witness a claims call at 01:00", "later", 1, 11, ev_map_c4),
        make_ev("receipt timestamped 22:28", "later", 2, 12, ev_map_c4),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
record_c4.perception_trace = trace_c4

ev4, perc4 = full_pipeline("C004", record_c4, trace_c4, {"ev_02", "ev_04", "ev_09"})

check("INT-02 passes", ev4.passed)
check("INT-02 grade A or B", ev4.grade in ("A", "B"), f"got {ev4.grade}")
check("INT-02 elimination correct", ev4.elimination_correct)
check("INT-02 pillars all set", ev4.pillar_content > 0 and ev4.pillar_integrity > 0)

# --- INT-03: C001 — weak run should fail ---
print("\n── INT-03: C001 weak run (should fail) ──")

record_weak = AnalysisRecord(
    observations=["there is a room", "the room has stuff", "things look normal",
                   "nothing special", "everything is fine"],
    anomalies=["nothing anomalous"],
    hypotheses={"there is a room": ["maybe someone was here", "maybe not"]},
    contradiction_notes=[],
    false_narrative_rejection="it seems fine",
    elimination_target="everything is normal",
    reasons=["the room looks fine"],
)

ev_weak, _ = full_pipeline("C001", record_weak)

check("INT-03 does NOT pass", not ev_weak.passed)
check("INT-03 grade D or F", ev_weak.grade in ("D", "F"), f"got {ev_weak.grade}")
check("INT-03 low confidence", ev_weak.confidence_score < 0.40, f"got {ev_weak.confidence_score:.2f}")

# --- INT-04: Pillar weights sum to 1.0 ---
print("\n── INT-04: Pillar weight invariants ──")
weight_sum = sum(PILLAR_WEIGHTS.values())
check("Pillar weights sum to 1.0", abs(weight_sum - 1.0) < 1e-9, f"sum={weight_sum}")
check("All weights positive", all(w > 0 for w in PILLAR_WEIGHTS.values()))
check("Integrity has highest weight", PILLAR_WEIGHTS["integrity"] >= max(
    PILLAR_WEIGHTS["content"], PILLAR_WEIGHTS["structure"], PILLAR_WEIGHTS["perception"]))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: ADVERSARIAL GAMING TESTS
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 60)
print("SECTION 2: ADVERSARIAL GAMING DEFENSES")
print("═" * 60)

# --- ADV-01: Keyword stuffing — dump all solution keywords without reasoning ---
print("\n── ADV-01: Keyword stuffing ──")

case_c1 = cases["C001"]
stuffed_keywords = " ".join(case_c1.solution.direct_answer_keywords)
all_concept_terms = []
for rule in case_c1.solution.required_concept_rules:
    for group in rule.concept_groups:
        all_concept_terms.extend(group.terms)
keyword_soup = " ".join(all_concept_terms) + " " + stuffed_keywords

record_stuff = AnalysisRecord(
    observations=[keyword_soup, keyword_soup, keyword_soup, keyword_soup, keyword_soup],
    anomalies=[keyword_soup],
    hypotheses={keyword_soup: [keyword_soup]},
    contradiction_notes=[keyword_soup],
    false_narrative_rejection=keyword_soup + " wrong false reject staged",
    elimination_target=case_c1.solution.elimination_target or "",
    reasons=[keyword_soup],
)

ev_stuff, _ = full_pipeline("C001", record_stuff)

check("ADV-01 keyword stuff still limited by causality",
      ev_stuff.causality_score < ev_stuff.causality_max,
      f"causality={ev_stuff.causality_score}/{ev_stuff.causality_max}")
check("ADV-01 causality score is low (phantom concepts)",
      ev_stuff.causality_score <= 6,
      f"causality={ev_stuff.causality_score}/{ev_stuff.causality_max}")
check("ADV-01 confidence < 85% (shouldn't get A easily)",
      ev_stuff.confidence_score < 0.85,
      f"confidence={ev_stuff.confidence_score:.0%}")

# --- ADV-02: Phantom derivation — correct answer with no observation chain ---
print("\n── ADV-02: Phantom derivation (right answer, no chain) ──")

record_phantom = AnalysisRecord(
    observations=["room is quiet", "door is closed", "lights are on",
                   "floor is clean", "air feels still"],
    anomalies=["nothing stands out"],
    hypotheses={"room is quiet": [
        "nobody has been here",
        "someone left quietly",
    ]},
    contradiction_notes=[
        "chair pulled back yet room undisturbed — staged scene"
    ],
    false_narrative_rejection=(
        "the calm room narrative is wrong — it was staged and fabricated"
    ),
    elimination_target="scene was staged",
    reasons=[
        "the scene was staged to appear undisturbed",
        "chair displacement proves recent activity — deliberately hidden",
        "notebook open confirms someone was interrupted",
    ],
)

ev_phantom, _ = full_pipeline("C001", record_phantom)

# Causality should catch phantoms: "chair" and "notebook" appear in reasons
# but were NEVER observed (observations mention "room", "door", "lights" etc.)
check("ADV-02 has phantom concepts",
      len(ev_phantom.causality_phantom) > 0,
      f"phantoms={ev_phantom.causality_phantom}")
check("ADV-02 causality penalized",
      ev_phantom.causality_score < 8,
      f"causality={ev_phantom.causality_score}/{ev_phantom.causality_max}")

# --- ADV-03: Hypothesis monoculture (only one explanation per observation) ---
print("\n── ADV-03: Hypothesis monoculture ──")

record_mono = AnalysisRecord(
    observations=[
        "chair is slightly pulled back from the table",
        "notebook lies open on the table",
        "window is slightly open",
        "curtains are moving inward",
        "room appears otherwise undisturbed",
    ],
    anomalies=["chair and undisturbed room contradict"],
    hypotheses={
        "chair is slightly pulled back from the table": [
            "the scene was staged",
            "someone staged the scene",  # near-duplicate
        ],
        "notebook lies open on the table": [
            "staged notebook",
            "notebook was staged",       # near-duplicate
        ],
    },
    contradiction_notes=["chair pulled back yet room undisturbed"],
    false_narrative_rejection="the calm narrative is wrong — staged fabricated",
    elimination_target="nobody was here",
    reasons=["scene was staged", "chair proves someone was here"],
)

ev_mono, _ = full_pipeline("C001", record_mono)

# Hypothesis diversity should be penalized — all hypotheses say the same thing
check("ADV-03 diversity score < max",
      ev_mono.causality_score < ev_mono.causality_max,
      f"causality={ev_mono.causality_score}/{ev_mono.causality_max}")

# --- ADV-04: Observation inference leak spam ---
print("\n── ADV-04: Inference leak in observations ──")

record_leak = AnalysisRecord(
    observations=[
        "chair is pulled back which clearly means someone sat there",
        "notebook is open which obviously suggests interrupted work",
        "room is undisturbed which implies staging",
        "window probably lets someone escape",
        "curtains must mean wind or exit",
    ],
    anomalies=["chair vs undisturbed"],
    hypotheses={"chair": ["staged scene", "recent departure"]},
    contradiction_notes=["chair vs undisturbed — contradiction"],
    false_narrative_rejection="calm room is wrong — staged fabricated",
    elimination_target="nobody was here",
    reasons=["scene staged", "chair proves presence"],
)

ev_leak, _ = full_pipeline("C001", record_leak)

check("ADV-04 all 5 observations flagged as inference leaks",
      ev_leak.observation_purity_penalty >= 5,
      f"penalty={ev_leak.observation_purity_penalty}")

# --- ADV-05: Forbidden pattern trigger — accepting calm/normal narrative ---
print("\n── ADV-05: Forbidden reasoning pattern ──")

record_forbidden = AnalysisRecord(
    observations=[
        "chair is slightly pulled back", "notebook open", "window open",
        "curtains moving inward", "room undisturbed",
    ],
    anomalies=["nothing anomalous really"],
    hypotheses={"room undisturbed": [
        "the room is calm and peaceful — nothing happened",
        "everything is natural and normal here",
    ]},
    contradiction_notes=[],
    false_narrative_rejection="",
    elimination_target="the scene is calm and nothing unusual happened — natural normal state",
    reasons=[
        "the quiet undisturbed room is natural and innocent",
        "everything appears normal — no deception occurred",
    ],
)

ev_forbidden, _ = full_pipeline("C001", record_forbidden)

check("ADV-05 forbidden penalty triggered",
      ev_forbidden.forbidden_penalty > 0,
      f"penalty={ev_forbidden.forbidden_penalty}")
check("ADV-05 does NOT pass",
      not ev_forbidden.passed)

# --- ADV-06: Perception gaming — claim to notice everything instantly ---
print("\n── ADV-06: Perception gaming (perfect micro-capture, nothing used) ──")

ev_map_c1 = getattr(cases["C001"], "evidence_map", {})
trace_game = PerceptionTrace(
    initial=[
        make_ev("chair slightly pulled back", "initial", 0, 0, ev_map_c1),
        make_ev("notebook open on table", "initial", 1, 0.1, ev_map_c1),
        make_ev("room appears undisturbed", "initial", 2, 0.2, ev_map_c1),
    ],
    later=[
        make_ev("window slightly open", "later", 0, 0.3, ev_map_c1),
        make_ev("curtains moving inward", "later", 1, 0.4, ev_map_c1),
        make_ev("glass of water half full", "later", 2, 0.5, ev_map_c1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)

ev_game, perc_game = full_pipeline(
    "C001", record_weak, trace_game, set()  # perfect perception, zero reasoning
)

check("ADV-06 decorative_observation flag",
      "decorative_observation" in perc_game.adversarial_flags)
check("ADV-06 diagnostic_dropout flag",
      "diagnostic_dropout" in perc_game.adversarial_flags)
check("ADV-06 still fails overall",
      not ev_game.passed)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 60)
print("SECTION 3: PERFORMANCE BENCHMARKS")
print("═" * 60)

N_ITERS = 50

# Benchmark: full pipeline per case
print(f"\n── Pipeline latency ({N_ITERS} iterations each) ──")
for cid in sorted(cases):
    case = cases[cid]
    ev_map = getattr(case, "evidence_map", {})

    # Build a minimal valid record
    rec = AnalysisRecord(
        observations=[e[:60] for e in case.evidence[:5]] + (["filler observation"] * max(0, 5 - len(case.evidence))),
        anomalies=["tension between clues noted"],
        hypotheses={case.evidence[0][:60]: ["explanation one", "explanation two"]} if case.evidence else {},
        contradiction_notes=["fact a vs fact b"],
        false_narrative_rejection="the false narrative is wrong — reject it staged",
        elimination_target=case.solution.elimination_target or "unknown",
        reasons=["reason one", "reason two", "reason three"],
    )

    trace = PerceptionTrace(
        initial=[make_ev(case.evidence[0][:60] if case.evidence else "clue", "initial", 0, 0, ev_map)],
        later=[make_ev(case.evidence[1][:60] if len(case.evidence) > 1 else "clue2", "later", 0, 10, ev_map)],
        first_hypothesis_timestamp=ts(20),
        first_elimination_timestamp=ts(80),
    )

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        full_pipeline(cid, rec, trace, set())
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / N_ITERS) * 1000
    print(f"  {cid}: {avg_ms:6.1f} ms/run  (total {elapsed:.2f}s for {N_ITERS} runs)")

    check(f"PERF {cid} under 100ms", avg_ms < 100, f"{avg_ms:.1f}ms")

# Benchmark: evidence ID matching
print(f"\n── Evidence ID matching latency ({N_ITERS * 10} calls) ──")
ev_map_c5 = getattr(cases["C005"], "evidence_map", {})
test_texts = [
    "partner is visibly crying and pacing",
    "single plate single cutlery one glass used",
    "phone calls stopped at 23:00 and neighbor heard arguing",
    "suspicious stranger near building confirmed as delivery driver",
]

t0 = time.perf_counter()
for _ in range(N_ITERS * 10):
    for txt in test_texts:
        match_evidence_ids(txt, ev_map_c5, cfg)
elapsed = time.perf_counter() - t0

total_calls = N_ITERS * 10 * len(test_texts)
avg_us = (elapsed / total_calls) * 1_000_000
print(f"  {total_calls} calls in {elapsed:.3f}s — avg {avg_us:.0f} us/call")
check("PERF evidence matching under 500us", avg_us < 500, f"{avg_us:.0f}us")

# Benchmark: case loading
print(f"\n── Case loading latency ({N_ITERS} iterations) ──")
t0 = time.perf_counter()
for _ in range(N_ITERS):
    loader2 = CaseLoader()
    loader2.load_all()
elapsed = time.perf_counter() - t0

avg_ms = (elapsed / N_ITERS) * 1000
print(f"  {N_ITERS} loads in {elapsed:.2f}s — avg {avg_ms:.1f} ms/load")
# 250 ms, not 50: loading ~500 local cases takes 30-60 ms and exceeded 50 ms whenever
# training or evaluation shared the machine. 250 ms still catches a real regression.
check("PERF case loading under 250ms", avg_ms < 250, f"{avg_ms:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 60)
print("SUMMARY")
print("═" * 60)

total = len(results)
passed = sum(1 for r in results if r["pass"])
print(f"\nTotal checks: {total}")
print(f"Passed:       {passed} / {total}")

if passed < total:
    print("\nFailed:")
    for r in results:
        if not r["pass"]:
            print(f"  ✗ {r['name']}")
else:
    print("\nAll checks passed.")

# Exit non-zero on any failed check so runners and CI can detect it.
if passed < total:
    import sys
    sys.exit(1)
