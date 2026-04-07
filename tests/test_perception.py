"""Perception integrity system validation.

Runs 20 synthetic sessions — honest, adversarial, and edge-case —
against both C001 and C002 evidence maps.  Reports per-scenario metric
readings and flag accuracy.

Usage:
    python test_perception.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from detective_engine.engine.case_loader import CaseLoader
from detective_engine.engine.models import (
    EvidenceMeta,
    ObservationEvent,
    PerceptionConfig,
    PerceptionTrace,
)
from detective_engine.engine.perception_integrity import (
    PerceptionIntegrityValidator,
    match_evidence_ids,
    should_probe,
    should_probe_stage1,
    select_probes,
    perception_adjustment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cfg = PerceptionConfig()
validator = PerceptionIntegrityValidator(cfg)

loader = CaseLoader()
all_cases, _ = loader.load_all()
cases_by_id = {c.id: c for c in all_cases}


def emap(case_id: str) -> dict[str, EvidenceMeta]:
    return getattr(cases_by_id[case_id], "evidence_map", {})


def ts(offset: float = 0.0) -> float:
    return 1_000_000.0 + offset  # deterministic base timestamp


def make_event(
    text: str,
    phase: str,
    idx: int,
    t_offset: float,
    ev_map: dict[str, EvidenceMeta],
) -> ObservationEvent:
    return ObservationEvent(
        text=text,
        timestamp=ts(t_offset),
        phase=phase,  # type: ignore[arg-type]
        order_index=idx,
        evidence_ids=match_evidence_ids(text, ev_map, cfg),
    )


def run(
    scenario_name: str,
    trace: PerceptionTrace,
    ev_map: dict[str, EvidenceMeta],
    used_ids: set[str],
    expect_flags: list[str] | None = None,
    expect_no_flags: list[str] | None = None,
    expect_probe: bool | None = None,
    note: str = "",
) -> dict[str, Any]:
    result = validator.validate(trace, ev_map, used_ids)
    adj = perception_adjustment(result, cfg)
    probe_fires = should_probe(result, cfg)

    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else " N/A"

    # Pass/fail checks
    checks: list[str] = []
    if expect_flags:
        for f in expect_flags:
            if f in result.adversarial_flags:
                checks.append(f"  ✅ flag '{f}' correctly detected")
            else:
                checks.append(f"  ❌ MISS: flag '{f}' should have fired")
    if expect_no_flags:
        for f in expect_no_flags:
            if f not in result.adversarial_flags:
                checks.append(f"  ✅ flag '{f}' correctly absent")
            else:
                checks.append(f"  ❌ FALSE POSITIVE: flag '{f}' should NOT fire")
    if expect_probe is not None:
        if probe_fires == expect_probe:
            checks.append(f"  ✅ probe trigger = {probe_fires} (expected {expect_probe})")
        else:
            checks.append(f"  ❌ probe trigger = {probe_fires} (expected {expect_probe})")

    all_pass = all("✅" in c for c in checks) if checks else True

    row = {
        "name": scenario_name,
        "cov": result.coverage,
        "ret": result.retention_score,
        "late": result.late_injection_score,
        "uptake": result.causal_uptake_score,
        "sal": result.salience_distortion,
        "lock": result.narrative_lock_in,
        "neutrality": result.strategic_neutrality_score,
        "match": result.id_match_rate,
        "conf": result.evaluation_confidence,
        "adj": adj,
        "flags": result.adversarial_flags,
        "probe": probe_fires,
        "checks": checks,
        "pass": all_pass,
        "note": note,
    }

    sep = "─" * 60
    status = "✅ PASS" if all_pass else "❌ FAIL"
    print(f"\n{sep}")
    print(f"SCENARIO: {scenario_name}  [{status}]")
    if note:
        print(f"  Note: {note}")
    print(f"  coverage:      {_fmt(result.coverage)}")
    print(f"  retention:     {_fmt(result.retention_score)}")
    print(f"  late_inject:   {_fmt(result.late_injection_score)}")
    print(f"  causal_uptake: {result.causal_uptake_score:.2f}")
    print(f"  salience_dist: {_fmt(result.salience_distortion)}")
    print(f"  narrative_lock:{_fmt(result.narrative_lock_in)}")
    print(f"  neutrality:    {result.strategic_neutrality_score:.2f}")
    print(f"  id_match_rate: {result.id_match_rate:.0%}")
    print(f"  eval_conf:     {result.evaluation_confidence:.0%}")
    print(f"  adj (Phase B): {adj:+.2f}")
    print(f"  flags:         {result.adversarial_flags or '—'}")
    print(f"  probe fires:   {probe_fires}")
    for c in checks:
        print(c)

    return row


# ---------------------------------------------------------------------------
# C001 evidence map
# ---------------------------------------------------------------------------

C1 = emap("C001")
C2 = emap("C002")

# Critical IDs for C001: ev_01 (chair), ev_02 (notebook), ev_04 (window), ev_06 (undisturbed)
# Critical IDs for C002: would be defined once we annotate it

results: list[dict] = []


# ===========================================================================
# HONEST RUNS (C001)
# ===========================================================================

# 1. Honest — strong all-round
trace = PerceptionTrace(
    initial=[
        make_event("chair slightly pulled back", "initial", 0, 0, C1),
        make_event("room appears undisturbed", "initial", 1, 1, C1),
        make_event("window slightly open", "initial", 2, 2, C1),
    ],
    later=[
        make_event("notebook open on table", "later", 0, 10, C1),
        make_event("curtains moving inward", "later", 1, 11, C1),
        make_event("glass of water half full", "later", 2, 12, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "HONEST-01: Strong honest run",
    trace, C1,
    used_ids={"ev_01", "ev_02", "ev_06"},
    expect_flags=[],
    expect_no_flags=["decorative_observation", "late_fairness_injection"],
    expect_probe=False,
    note="All critical clues noticed early and used in reasoning",
))

# 2. Honest — average run (misses one critical clue in micro-capture)
trace = PerceptionTrace(
    initial=[
        make_event("window slightly open", "initial", 0, 0, C1),
        make_event("curtains moving inward", "initial", 1, 1, C1),
    ],
    later=[
        make_event("chair pulled from table", "later", 0, 10, C1),
        make_event("notebook open on table", "later", 1, 11, C1),
        make_event("room appears undisturbed", "later", 2, 12, C1),
        make_event("glass of water half full", "later", 3, 13, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "HONEST-02: Average run (window-biased micro-capture)",
    trace, C1,
    used_ids={"ev_01", "ev_06"},
    expect_flags=["salience_capture", "late_fairness_injection", "decorative_observation"],
    expect_probe=True,
    note="Noticed high-salience window/curtains first; causal_uptake=0.33 (2/6 used) — decorative is correct",
))

# 3. Honest — good reasoning but forgot one observation
trace = PerceptionTrace(
    initial=[
        make_event("chair slightly pulled back", "initial", 0, 0, C1),
        make_event("notebook lying open", "initial", 1, 1, C1),
    ],
    later=[
        make_event("window slightly open", "later", 0, 10, C1),
        make_event("curtains moving inward", "later", 1, 11, C1),
        make_event("room appears otherwise undisturbed", "later", 2, 12, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "HONEST-03: Good micro-capture, used all in reasoning",
    trace, C1,
    used_ids={"ev_01", "ev_02", "ev_06"},
    expect_no_flags=["late_fairness_injection"],
    note="Diagnostic clues noticed first; all used downstream",
))


# ===========================================================================
# ADVERSARIAL RUNS — intentionally bad patterns (C001)
# ===========================================================================

# 4. Adversarial — all critical clues noticed ONLY after hypothesis
trace = PerceptionTrace(
    initial=[
        make_event("curtains moving inward", "initial", 0, 0, C1),  # low diagnostic
        make_event("glass of water half full", "initial", 1, 1, C1),  # low diagnostic
    ],
    later=[
        make_event("chair slightly pulled back", "later", 0, 10, C1),   # critical
        make_event("notebook open on table", "later", 1, 11, C1),        # critical
        make_event("room appears undisturbed", "later", 2, 12, C1),      # critical
        make_event("window slightly open", "later", 3, 13, C1),          # critical
    ],
    first_hypothesis_timestamp=ts(5),   # hypothesis formed very early
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "ADVERSARIAL-01: All critical clues injected AFTER hypothesis",
    trace, C1,
    used_ids={"ev_01", "ev_02", "ev_04", "ev_06"},
    expect_flags=["late_fairness_injection", "premature_commitment"],
    expect_no_flags=["decorative_observation"],  # uptake is high (0.67) — not decorative
    expect_probe=True,
    note="Classic late injection: hypothesis formed before critical clues noticed",
))

# 5. Adversarial — decorative observation (noticed but never used)
trace = PerceptionTrace(
    initial=[
        make_event("chair slightly pulled back", "initial", 0, 0, C1),
        make_event("notebook open on table", "initial", 1, 1, C1),
        make_event("window slightly open", "initial", 2, 2, C1),
    ],
    later=[
        make_event("curtains moving inward", "later", 0, 10, C1),
        make_event("room appears undisturbed", "later", 1, 11, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "ADVERSARIAL-02: Observations noticed but none used in reasoning",
    trace, C1,
    used_ids=set(),  # nothing fed into reasoning
    expect_flags=["decorative_observation", "checklist_symmetry", "diagnostic_dropout"],
    expect_probe=True,
    note="User listed observations as checklist but never integrated any",
))

# 6. Adversarial — strategic neutrality pattern
# Looks balanced on paper: 100% coverage, but all arrived late, nothing used
trace = PerceptionTrace(
    initial=[
        make_event("curtains moving inward", "initial", 0, 0, C1),
        make_event("glass of water half full", "initial", 1, 1, C1),
    ],
    later=[
        make_event("chair slightly pulled back", "later", 0, 10, C1),
        make_event("notebook open on table", "later", 1, 11, C1),
        make_event("room appears undisturbed", "later", 2, 12, C1),
        make_event("window slightly open", "later", 3, 13, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "ADVERSARIAL-03: Strategic neutrality (balanced but late, unused)",
    trace, C1,
    used_ids={"ev_03"},  # only glass used — irrelevant
    expect_flags=["decorative_observation", "checklist_symmetry"],
    expect_probe=True,
    note="100% eventual coverage, all critical IDs arrived late, low uptake",
))

# 7. Adversarial — revision laundering (adds critical clues in revision phase)
trace = PerceptionTrace(
    initial=[
        make_event("curtains moving inward", "initial", 0, 0, C1),
        make_event("glass of water half full", "initial", 1, 1, C1),
    ],
    later=[
        make_event("window slightly open", "later", 0, 5, C1),
    ],
    revisions=[
        ObservationEvent(
            text="room appears undisturbed contradicting chair position",
            timestamp=ts(90),
            phase="revision",
            order_index=0,
            evidence_ids=["ev_06", "ev_01"],
            source="probe",
        )
    ],
    first_hypothesis_timestamp=ts(10),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "ADVERSARIAL-04: Revision laundering (critical insight added after elimination)",
    trace, C1,
    used_ids={"ev_06"},
    expect_flags=["premature_commitment"],
    expect_probe=True,
    note="Contradiction anchor ev_06 only appears in revision after lock-in",
))

# 8. Adversarial — salience capture (only vivid, misleading clues noticed)
trace = PerceptionTrace(
    initial=[
        make_event("window slightly open", "initial", 0, 0, C1),    # high salience + misdirection
        make_event("curtains moving inward", "initial", 1, 1, C1),  # high salience + misdirection
    ],
    later=[
        make_event("window slightly open again noted", "later", 0, 10, C1),
        make_event("curtains moving strongly", "later", 1, 11, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "ADVERSARIAL-05: Salience capture (only vivid misdirection clues noticed)",
    trace, C1,
    used_ids={"ev_04"},
    expect_flags=["salience_capture"],
    expect_no_flags=["late_fairness_injection"],  # late_inject=0.00 — nothing arrived late
    expect_probe=True,
    note="User fixated entirely on window/curtains — diagnostic clues never noticed",
))

# 9. Adversarial — premature commitment (hypothesis before seeing critical clues)
trace = PerceptionTrace(
    initial=[
        make_event("window slightly open", "initial", 0, 0, C1),
    ],
    later=[
        make_event("chair pulled back", "later", 0, 20, C1),
        make_event("room undisturbed", "later", 1, 21, C1),
        make_event("notebook open", "later", 2, 22, C1),
    ],
    first_hypothesis_timestamp=ts(2),   # instantly after seeing just 1 clue
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "ADVERSARIAL-06: Premature commitment (hypothesis after 1 micro-capture clue)",
    trace, C1,
    used_ids={"ev_01"},
    expect_flags=["premature_commitment"],
    expect_probe=True,
    note="Committed to hypothesis after single clue — critical coverage was ~25%",
))


# ===========================================================================
# EDGE CASES
# ===========================================================================

# 10. Edge — no micro-capture (Phase 0 skipped or empty)
trace = PerceptionTrace(
    initial=[],  # nothing captured in Phase 0
    later=[
        make_event("chair slightly pulled back", "later", 0, 10, C1),
        make_event("notebook open on table", "later", 1, 11, C1),
        make_event("room appears undisturbed", "later", 2, 12, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "EDGE-01: No micro-capture (Phase 0 empty)",
    trace, C1,
    used_ids={"ev_01", "ev_06"},
    expect_no_flags=["late_fairness_injection"],  # no micro-capture → no reference point for 'late'
    note="retention_score=None; late_fairness_injection must not fire without micro-capture anchor",
))

# 11. Edge — no evidence_map (unannotated case)
trace = PerceptionTrace(
    initial=[
        make_event("chair slightly pulled back", "initial", 0, 0, {}),
    ],
    later=[
        make_event("notebook open", "later", 0, 10, {}),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "EDGE-02: No evidence_meta (unannotated case)",
    trace, {},  # empty evidence map
    used_ids=set(),
    note="Salience-dependent metrics should all be None; confidence ~0.30",
))

# 12. Edge — very low id_match_rate
trace = PerceptionTrace(
    initial=[
        make_event("something vague happened here", "initial", 0, 0, C1),
        make_event("there was stuff on the floor", "initial", 1, 1, C1),
    ],
    later=[
        make_event("general messy area visible", "later", 0, 10, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "EDGE-03: Very low id_match_rate (vague observations)",
    trace, C1,
    used_ids=set(),
    note="ID-dependent metrics should be None; confidence capped at 35%",
))

# 13. Edge — single observation in micro-capture matches multiple IDs
trace = PerceptionTrace(
    initial=[
        make_event("chair pulled back and window open and notebook on table", "initial", 0, 0, C1),
    ],
    later=[
        make_event("room appears undisturbed", "later", 0, 10, C1),
        make_event("glass half full and curtains moving", "later", 1, 11, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "EDGE-04: Multi-clue sentence in micro-capture",
    trace, C1,
    used_ids={"ev_01", "ev_02", "ev_04"},
    note="Should match ev_01, ev_02, ev_04 from one sentence (capped at 3)",
))

# 14. Edge — perfect honest run with all 6 clues
trace = PerceptionTrace(
    initial=[
        make_event("chair slightly pulled back from table", "initial", 0, 0, C1),
        make_event("room appears undisturbed which is suspicious", "initial", 1, 1, C1),
        make_event("notebook lies open on table", "initial", 2, 2, C1),
    ],
    later=[
        make_event("window slightly open", "later", 0, 10, C1),
        make_event("curtains moving inward", "later", 1, 11, C1),
        make_event("glass of water half full", "later", 2, 12, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "EDGE-05: Perfect run — all 6 clues, critical ones early, all used",
    trace, C1,
    used_ids={"ev_01", "ev_02", "ev_04", "ev_05", "ev_06"},
    expect_no_flags=["late_fairness_injection", "premature_commitment", "salience_capture"],
    expect_probe=False,
    note="Baseline: what a perfect run should look like",
))


# ===========================================================================
# COGNITIVE DISCONNECT TESTS (key insight: noticed early but never used)
# ===========================================================================

# 15. Disconnect — noticed contradiction anchor early, never used it
trace = PerceptionTrace(
    initial=[
        make_event("room appears undisturbed", "initial", 0, 0, C1),   # ev_06 — critical contradiction anchor
        make_event("chair pulled back from table", "initial", 1, 1, C1),  # ev_01 — critical
    ],
    later=[
        make_event("notebook open on table", "later", 0, 10, C1),
        make_event("window slightly open", "later", 1, 11, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "DISCONNECT-01: Noticed contradiction anchor early, ignored in reasoning",
    trace, C1,
    used_ids={"ev_04"},  # only used window — never used ev_06 (the tension anchor)
    expect_flags=["decorative_observation"],
    expect_probe=True,
    note="COGNITIVE DISCONNECT: ev_06 registered in micro-capture but dropped from reasoning",
))

# 16. Disconnect — noticed ALL clues, used none
trace = PerceptionTrace(
    initial=[
        make_event("chair slightly pulled back", "initial", 0, 0, C1),
        make_event("notebook open", "initial", 1, 1, C1),
        make_event("room undisturbed", "initial", 2, 2, C1),
    ],
    later=[
        make_event("window open", "later", 0, 10, C1),
        make_event("curtains moving inward", "later", 1, 11, C1),
        make_event("glass half full", "later", 2, 12, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "DISCONNECT-02: Noticed everything, used nothing (decorative listing)",
    trace, C1,
    used_ids=set(),
    expect_flags=["decorative_observation", "checklist_symmetry", "diagnostic_dropout"],
    expect_probe=True,
    note="COGNITIVE DISCONNECT: maximum observation, zero integration",
))

# 17. Disconnect — noticed diagnostic clue, pivoted to misdirection in reasoning
trace = PerceptionTrace(
    initial=[
        make_event("room appears undisturbed", "initial", 0, 0, C1),  # ev_06 — highest diag weight
        make_event("chair pulled back", "initial", 1, 1, C1),          # ev_01
    ],
    later=[
        make_event("window slightly open", "later", 0, 10, C1),
        make_event("curtains moving inward", "later", 1, 11, C1),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "DISCONNECT-03: Noticed high-diagnostic clue, reasoned from low-diagnostic clues",
    trace, C1,
    used_ids={"ev_04", "ev_05"},  # only window/curtains — the misdirection pair
    expect_flags=["diagnostic_dropout"],
    expect_probe=True,
    note="COGNITIVE DISCONNECT: diagnostic ev_06 noticed early, reasoning anchored on misdirection",
))


# ===========================================================================
# C002 RUNS (no evidence_meta yet — graceful degradation test)
# ===========================================================================

# 18. C002 — honest run without metadata
c2_texts_initial = ["no fingerprints on the metal object", "object sits in high contact area"]
c2_texts_later = ["there are no signs the object was replaced", "surrounding area appears normal"]

trace = PerceptionTrace(
    initial=[make_event(t, "initial", i, i, C2) for i, t in enumerate(c2_texts_initial)],
    later=[make_event(t, "later", i, i + 10, C2) for i, t in enumerate(c2_texts_later)],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C002-01: Honest run on unannotated C002",
    trace, C2,  # empty map — no evidence_meta yet
    used_ids=set(),
    note="C002 has no evidence_meta yet; salience metrics should all be N/A",
))

# 19. C002 — adversarial: conclusion stated before negative-evidence insight
# (now annotated — reload fresh map)
C2 = emap("C002")

trace = PerceptionTrace(
    initial=[make_event("area appears normal", "initial", 0, 0, C2)],  # ev_04 — false calm
    later=[
        make_event("no fingerprints on frequently used object", "later", 0, 30, C2),
        make_event("high contact location", "later", 1, 31, C2),
        make_event("no signs that the object was recently replaced", "later", 2, 32, C2),
    ],
    first_hypothesis_timestamp=ts(5),   # hypothesis before critical clues
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C002-02: Adversarial — low-diagnostic clue in micro-capture, critical clues injected late",
    trace, C2,
    used_ids=set(),
    expect_flags=["late_fairness_injection", "premature_commitment", "decorative_observation"],
    expect_probe=True,
    note="salience_capture should NOT fire — ev_04 has salience=0.50, not high enough",
))

# C002-03: Honest run — abstract quiet clues (ev_02 and ev_03) noticed first
# These have very LOW salience — noticing them early is the ideal (anti-salience-bias) pattern
trace = PerceptionTrace(
    initial=[
        make_event("the object sits in high-contact location", "initial", 0, 0, C2),   # ev_02 quiet anchor
        make_event("no signs the object was recently replaced", "initial", 1, 1, C2),  # ev_03 negative evidence
    ],
    later=[
        make_event("no fingerprints on frequently used object", "later", 0, 10, C2),
        make_event("surrounding area appears normal", "later", 1, 11, C2),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C002-03: Ideal run — quiet diagnostic clues noticed first (anti-salience bias)",
    trace, C2,
    used_ids={"ev_01", "ev_02", "ev_03"},
    expect_no_flags=["salience_capture", "diagnostic_dropout"],
    expect_probe=False,
    note="timing_consistency should be low (quiet clues first) — this is HEALTHY, not suspicious",
))

# C002-04: Salience bias — noticed only the obvious clue (missing fingerprints) first
# ev_01 has high salience AND high misdirection risk — fixating on it looks like salience capture
trace = PerceptionTrace(
    initial=[
        make_event("no fingerprints visible on the object", "initial", 0, 0, C2),  # ev_01 high salience
    ],
    later=[
        make_event("object is in high-contact location", "later", 0, 10, C2),
        make_event("no signs the object was recently replaced", "later", 1, 11, C2),
        make_event("area appears normal", "later", 2, 12, C2),
    ],
    first_hypothesis_timestamp=ts(5),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C002-04: Salience bias — only the vivid absence clue noticed first",
    trace, C2,
    used_ids={"ev_01"},
    expect_flags=["salience_capture", "late_fairness_injection", "decorative_observation"],
    expect_probe=True,
    note="ev_01 has salience=0.80 — noticing ONLY it shows salience bias; critical clues ev_02+ev_03 arrive late, low uptake",
))

# C002-05: Diagnostic dropout — noticed ev_02 and ev_03 early, reasoned only from ev_01
trace = PerceptionTrace(
    initial=[
        make_event("object in high contact location", "initial", 0, 0, C2),  # ev_02
        make_event("no replacement signs visible", "initial", 1, 1, C2),    # ev_03
    ],
    later=[
        make_event("no fingerprints on object", "later", 0, 10, C2),
        make_event("area appears normal", "later", 1, 11, C2),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C002-05: Cognitive disconnect — quiet anchors noticed early, ignored in reasoning",
    trace, C2,
    used_ids={"ev_01"},  # only the obvious clue used — ev_02, ev_03 dropped
    expect_flags=["diagnostic_dropout"],
    expect_probe=True,
    note="ev_02 and ev_03 are critical and high-diagnostic; noticing them but not using them is a clear disconnect",
))

# ===========================================================================
# C005 RUNS — psychological-adversarial, emotional salience bias
# ===========================================================================

C5 = emap("C005")

# C005-01: Honest run — physical evidence noticed first, emotional clue noted but not used to prove innocence
trace = PerceptionTrace(
    initial=[
        make_event("single plate single cutlery one glass used", "initial", 0, 0, C5),    # ev_04 critical
        make_event("phone shows zero calls after 23:00", "initial", 1, 1, C5),           # ev_06 critical
    ],
    later=[
        make_event("partner visibly crying and pacing", "later", 0, 10, C5),              # ev_02 trap
        make_event("neighbor heard arguing around 22:30", "later", 1, 11, C5),            # ev_08 critical
        make_event("partner says cooked dinner for both of us", "later", 2, 12, C5),      # ev_03
        make_event("suspicious stranger confirmed as delivery driver", "later", 3, 13, C5), # ev_10 false clue
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C005-01: Honest run — physical evidence first, emotional trap noticed but filtered",
    trace, C5,
    used_ids={"ev_04", "ev_06", "ev_08"},
    expect_no_flags=["salience_capture", "diagnostic_dropout"],
    expect_probe=False,
    note="Physical contradiction anchors in micro-capture; timing_consistency should be LOW (good)",
))

# C005-02: Emotional salience capture — crying/pacing noticed first, physical evidence later
trace = PerceptionTrace(
    initial=[
        make_event("partner is visibly crying and pacing", "initial", 0, 0, C5),   # ev_02 high-salience trap
        make_event("suspicious stranger near building", "initial", 1, 1, C5),      # ev_10 false clue
        make_event("car found 3km away keys in ignition", "initial", 2, 2, C5),    # ev_09 misdirection
    ],
    later=[
        make_event("single plate and cutlery in kitchen", "later", 0, 10, C5),
        make_event("phone calls stopped at 23:00", "later", 1, 11, C5),
        make_event("neighbor heard arguing 22:30", "later", 2, 12, C5),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C005-02: Emotional salience capture — crying + false clues dominate micro-capture",
    trace, C5,
    used_ids={"ev_02", "ev_09", "ev_10"},  # only emotional/misdirection clues used
    expect_flags=["salience_capture", "late_fairness_injection"],
    expect_probe=True,
    note="ev_02 (salience=0.92) + ev_10 (salience=0.88) in micro-capture = classic emotional salience bias; no diagnostic_dropout because no critical+high-diag clues in micro-capture",
))

# C005-03: Adversarial — noticed physical evidence early, used false clues in reasoning
trace = PerceptionTrace(
    initial=[
        make_event("single plate in kitchen contradicts dinner claim", "initial", 0, 0, C5), # ev_04
        make_event("phone gap zero calls after 23:00", "initial", 1, 1, C5),                # ev_06
    ],
    later=[
        make_event("suspicious stranger near building", "later", 0, 10, C5),
        make_event("broken window in stairwell", "later", 1, 11, C5),
        make_event("wallet found on car seat", "later", 2, 12, C5),
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C005-03: Cognitive disconnect — physical anchors noticed, false clues used in reasoning",
    trace, C5,
    used_ids={"ev_10", "ev_11", "ev_12"},  # only false clues used
    expect_flags=["diagnostic_dropout"],
    expect_probe=True,
    note="ev_04+ev_06 noticed early (high diagnostic) but dropped; false clues drove conclusion",
))

# C005-04: Strategic neutrality — noticed all 12 items eventually, none of the critical ones early, low uptake
trace = PerceptionTrace(
    initial=[
        make_event("partner crying and pacing", "initial", 0, 0, C5),    # ev_02 trap
        make_event("car keys left in ignition", "initial", 1, 1, C5),    # ev_09 misdirection
    ],
    later=[
        make_event("single plate single cutlery", "later", 0, 10, C5),       # ev_04
        make_event("phone calls stopped at 23:00", "later", 1, 11, C5),      # ev_06
        make_event("partner called police 07:15", "later", 2, 12, C5),       # ev_01
        make_event("dinner claim but single plate", "later", 3, 13, C5),     # ev_03
        make_event("tried calling all night", "later", 4, 14, C5),           # ev_05
        make_event("neighbor heard arguing 22:30", "later", 5, 15, C5),      # ev_08
        make_event("suspicious stranger confirmed delivery driver", "later", 6, 16, C5),  # ev_10
        make_event("broken window pre-existing damage", "later", 7, 17, C5),  # ev_11
        make_event("wallet on car seat spouse always left it there", "later", 8, 18, C5), # ev_12
    ],
    first_hypothesis_timestamp=ts(20),
    first_elimination_timestamp=ts(80),
)
results.append(run(
    "C005-04: Strategic neutrality — high coverage, all critical clues arrived late, low uptake",
    trace, C5,
    used_ids={"ev_02"},  # only emotional trap used — didn't integrate the physical evidence
    expect_flags=["late_fairness_injection", "salience_capture", "checklist_symmetry"],
    expect_probe=True,
    note="Looks balanced on paper; emotional trap ev_02 drives conclusion despite physical evidence available",
))


# ===========================================================================
# PROBE ROTATION TEST
# ===========================================================================

# 20. Test probe rotation — same scenario, probes_seen accumulates
print("\n" + "═" * 60)
print("PROBE ROTATION TEST")
print("═" * 60)

trace = PerceptionTrace(
    initial=[make_event("curtains moving inward", "initial", 0, 0, C1)],
    later=[
        make_event("chair pulled back", "later", 0, 10, C1),
        make_event("room undisturbed", "later", 1, 11, C1),
    ],
    first_hypothesis_timestamp=ts(5),
    first_elimination_timestamp=ts(80),
)
result = validator.validate(trace, C1, used_in_reasoning_ids=set())

probes_seen: set[str] = set()
for attempt in range(1, 4):
    probes = select_probes(result, cfg, probes_seen)
    print(f"\n  Attempt {attempt} — probes_seen={len(probes_seen)}")
    for p in probes:
        print(f"    • {p}")
    probes_seen.update(probes)

print("\n  (After exhausting all variants, rotation resets — attempt 4:)")
probes = select_probes(result, cfg, probes_seen)
for p in probes:
    print(f"    • {p}")


# ===========================================================================
# SUMMARY
# ===========================================================================

print("\n\n" + "═" * 60)
print("SUMMARY")
print("═" * 60)

total = len(results)
passed = sum(1 for r in results if r["pass"])
print(f"\nTotal scenarios: {total}")
print(f"Passed checks:   {passed} / {total}")

if passed < total:
    print("\nFailed:")
    for r in results:
        if not r["pass"]:
            print(f"  ✗ {r['name']}")
            for c in r["checks"]:
                if "❌" in c:
                    print(f"      {c.strip()}")

# Metric range summary
print("\nMetric ranges across all scenarios:")
for key, label in [
    ("cov", "coverage"),
    ("ret", "retention"),
    ("late", "late_injection"),
    ("uptake", "causal_uptake"),
    ("sal", "salience_distort"),
    ("neutrality", "strategic_neutral"),
]:
    vals = [r[key] for r in results if r[key] is not None]
    if vals:
        print(f"  {label:22s}  min={min(vals):.2f}  max={max(vals):.2f}  avg={sum(vals)/len(vals):.2f}")

print("\nFlag frequency:")
all_flags: dict[str, int] = {}
for r in results:
    for f in r["flags"]:
        all_flags[f] = all_flags.get(f, 0) + 1
for flag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
    print(f"  {flag:30s}  {count} times")
