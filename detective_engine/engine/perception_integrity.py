"""Perception Integrity Validator — V11/V12 layer.

Evaluates *what was noticed*, *when*, and *how stably* observations persisted
into reasoning.  Separate from the causality validator (which checks derivation
integrity) and the reasoning graph validator (which checks chain structure).

Three failure types this module distinguishes:
    1. Perception miss      — clue never registered early
    2. Memory loss          — clue was initially observed but later dropped
    3. Strategic repair     — clue appears late in a way inconsistent with
                              genuine early perception

Design principles:
    • All ID-dependent metrics return None when id_match_rate is too low.
    • Salience-dependent metrics return None when evidence_meta is absent.
    • strategic_neutrality_score is a soft 0.0–1.0, not a binary flag.
    • Probe selection is rotation-aware via probes_seen set from user profile.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from .models import (
    EvidenceMeta,
    ObservationEvent,
    PerceptionConfig,
    PerceptionIntegrityResult,
    PerceptionTrace,
    normalize_key,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Short evidence word whitelist
# Words under 4 chars that carry real evidence meaning and must NOT be
# filtered by the standard len > 3 token filter.
# ---------------------------------------------------------------------------

SHORT_EVIDENCE_WORDS: frozenset[str] = frozenset({
    "ink", "key", "gun", "bag", "cup", "cut", "tie", "tag",
    "lid", "pen", "cap", "mud", "ash", "oil", "ice", "arm",
    "leg", "eye", "ear", "jaw", "rib", "toe", "lip", "hat",
    "map", "log", "rod", "pin", "saw", "axe", "bow", "net",
})


# ---------------------------------------------------------------------------
# Probe bank — two variants per failure mode for rotation
# ---------------------------------------------------------------------------

PROBE_BANK: dict[str, list[str]] = {
    "late_injection": [
        "Which clue did you notice late, and why?",
        "Name a clue that only appeared in your reasoning after Phase 3.",
    ],
    "salience_distortion": [
        "List 2 low-salience clues that may matter.",
        "Which detail is easy to overlook but changes the story?",
    ],
    "narrative_lock_in": [
        "List 2 clues that weaken your current hypothesis.",
        "What would have to be true for your hypothesis to be wrong?",
    ],
    "low_causal_uptake": [
        "Name 1 observed detail you have not used in your reasoning yet.",
        "Which observation contributed least to your conclusion?",
    ],
    "diagnostic_dropout": [
        "You noticed a key clue early but didn't use it. Which one, and why?",
        "Name the most important clue you observed but left out of your reasoning.",
    ],
    "general": [
        "List 1 detail you almost ignored.",
        "Which clue felt least important to you at first?",
    ],
}


# ---------------------------------------------------------------------------
# Evidence ID matching
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    """Normalise text, strip punctuation, and return a filtered token set.

    Keeps tokens that are either > 3 chars or in the short-word whitelist.
    Punctuation is stripped so that 'suspicious' and "'suspicious'" match.
    """
    cleaned = re.sub(r"[^\w\s]", " ", normalize_key(text))
    return {
        w for w in cleaned.split()
        if len(w) > 3 or w in SHORT_EVIDENCE_WORDS
    }


def _match_single(
    phrase: str,
    evidence_map: dict[str, EvidenceMeta],
    config: PerceptionConfig,
) -> str | None:
    """Find the best-matching evidence ID for a single phrase.

    Uses token overlap between the phrase and each evidence_meta.text.
    Returns None if best score < config.id_match_threshold.
    """
    tokens = _token_set(phrase)
    if not tokens:
        return None

    best_id, best_score = None, 0.0
    for eid, meta in evidence_map.items():
        meta_tokens = _token_set(meta.text)
        if not meta_tokens:
            continue
        overlap = len(tokens & meta_tokens) / len(meta_tokens)
        if overlap > best_score:
            best_score, best_id = overlap, eid

    return best_id if best_score >= config.id_match_threshold else None


def match_evidence_ids(
    text: str,
    evidence_map: dict[str, EvidenceMeta],
    config: PerceptionConfig,
) -> list[str]:
    """Return a list of evidence IDs matched from a free-text observation.

    Splits input on conjunctions and punctuation so that multi-clue sentences
    like "chair pulled back and window open" can match multiple IDs.
    Deduplicates results; each ID appears at most once.

    Note: Capped at the first 3 IDs to prevent overcounting from overly
    long sentences.  (TODO: make cap configurable via PerceptionConfig.)
    """
    phrases = re.split(r"\band\b|\bas well as\b|&|,|;", normalize_key(text))
    matched: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        eid = _match_single(phrase.strip(), evidence_map, config)
        if eid and eid not in seen:
            matched.append(eid)
            seen.add(eid)
        if len(matched) >= 3:  # cap to avoid overcounting
            break
    return matched


# ---------------------------------------------------------------------------
# Core metric functions
# (each is a pure function — easy to test in isolation)
# ---------------------------------------------------------------------------

def retention_score(initial_ids: set[str], later_ids: set[str]) -> float:
    """Fraction of initial observations that persist into later phases.

    High = stable perception.
    Low  = memory drop or weak encoding.
    """
    if not initial_ids:
        return 0.0
    retained = initial_ids & later_ids
    return len(retained) / len(initial_ids)


def late_injection_score(
    initial_ids: set[str],
    later_ids: set[str],
    critical_ids: set[str],
) -> float:
    """Fraction of critical clues that appear late without early registration.

    High = possible strategic repair or memory contamination.
    """
    if not critical_ids:
        return 0.0
    injected = (later_ids - initial_ids) & critical_ids
    return len(injected) / len(critical_ids)


def causal_uptake_score(
    observed_ids: set[str],
    used_in_reasoning_ids: set[str],
) -> float:
    """Fraction of observed clues that actually influenced later reasoning.

    High = observations were cognitively integrated.
    Low  = decorative or checklist-style observation.
    """
    if not observed_ids:
        return 0.0
    used = observed_ids & used_in_reasoning_ids
    return len(used) / len(observed_ids)


def coverage_score(
    observed_ids: set[str],
    critical_ids: set[str],
) -> float:
    """Fraction of critical clues that were noticed at all.

    Low = likely perceptual omission.
    """
    if not critical_ids:
        return 1.0
    return len(observed_ids & critical_ids) / len(critical_ids)


def salience_distortion(
    observed_ids: set[str],
    evidence_map: dict[str, EvidenceMeta],
) -> float:
    """Measures whether the user over-selects vivid / misleading clues.

    High = likely attraction to vivid but low-value cues.
    Returns a value in [0, 1].
    """
    if not observed_ids:
        return 0.0

    salient = 0.0
    diagnostic = 0.0
    for eid in observed_ids:
        meta = evidence_map.get(eid)
        if not meta:
            continue
        salient += meta.salience_level * max(meta.misdirection_risk, 0.1)
        diagnostic += meta.diagnostic_weight

    if diagnostic <= 0:
        return min(1.0, salient)
    return max(0.0, min(1.0, (salient - diagnostic) / max(diagnostic, 1e-6)))


def narrative_lock_in(
    trace: PerceptionTrace,
    initial_critical_coverage: float,
) -> float:
    """Measures hypothesis commitment before sufficient critical observation coverage.

    High = early top-down story formation before enough evidence registration.
    Uses first_hypothesis_timestamp (set at Phase 3 onset) as the anchor.
    Note: this is an approximation — CLI timestamps include typing time.
    """
    if trace.first_hypothesis_timestamp is None:
        return 0.0
    if initial_critical_coverage >= 0.6:
        return 0.0
    return 1.0 - initial_critical_coverage


def timing_consistency(
    trace: PerceptionTrace,
    evidence_map: dict[str, EvidenceMeta],
) -> float:
    """Average salience of the initial (micro-capture) observations.

    Interpretation:
        extremely low  = suspiciously 'too perfect' capture of only quiet clues
        middle range   = typical human attention distribution (healthy)
        extremely high = likely salience-driven bias
    """
    if not trace.initial:
        return 0.0

    score = 0.0
    counted = 0
    for event in trace.initial:
        for eid in event.evidence_ids:
            meta = evidence_map.get(eid)
            if meta:
                score += meta.salience_level
                counted += 1
                break  # one meta score per event is enough

    return score / len(trace.initial) if trace.initial else 0.0


# ---------------------------------------------------------------------------
# Adversarial detection
# ---------------------------------------------------------------------------

def _compute_strategic_neutrality_score(
    result: PerceptionIntegrityResult,
    config: PerceptionConfig,
) -> float:
    """Soft 0.0–1.0 score for strategic neutrality pattern.

    Only meaningful when evaluation_confidence >= 0.60.
    Pattern: looks balanced on paper, but balance arrived late and did not
    affect reasoning — a strong signal of performative fairness.
    """
    if result.evaluation_confidence < 0.60:
        return 0.0
    if (
        result.coverage is None
        or result.late_injection_score is None
        or result.causal_uptake_score is None
    ):
        return 0.0

    triggered = (
        result.coverage > config.neutrality_coverage_threshold
        and result.late_injection_score > config.neutrality_late_injection_threshold
        and result.causal_uptake_score < config.neutrality_causal_uptake_threshold
    )
    if not triggered:
        return 0.0

    coverage_excess = (
        (result.coverage - config.neutrality_coverage_threshold)
        / max(1.0 - config.neutrality_coverage_threshold, 1e-6)
    )
    injection_excess = (
        (result.late_injection_score - config.neutrality_late_injection_threshold)
        / max(1.0 - config.neutrality_late_injection_threshold, 1e-6)
    )
    uptake_deficit = (
        (config.neutrality_causal_uptake_threshold - result.causal_uptake_score)
        / max(config.neutrality_causal_uptake_threshold, 1e-6)
    )
    return min(1.0, max(0.0, (coverage_excess + injection_excess + uptake_deficit) / 3.0))


def _collect_adversarial_flags(
    result: PerceptionIntegrityResult,
    config: PerceptionConfig,
    initial_ids: set[str] | None = None,
    used_ids: set[str] | None = None,
    evidence_map: dict[str, EvidenceMeta] | None = None,
) -> list[str]:
    """Generate a list of adversarial flag strings based on metric thresholds.

    Parameters
    ----------
    result       : PerceptionIntegrityResult (partially filled — neutrality already set)
    config       : PerceptionConfig
    initial_ids  : evidence IDs from micro-capture only (for disconnect detection)
    used_ids     : evidence IDs that appeared in reasoning (for disconnect detection)
    evidence_map : full evidence metadata (for diagnostic_dropout check)
    """
    flags: list[str] = []

    # late_fairness_injection — fires independently when late injection is high.
    # Requires retention_score to be available (i.e., micro-capture was present),
    # otherwise "late" has no reference point and the flag is meaningless.
    if (
        result.late_injection_score is not None
        and result.late_injection_score > 0.50
        and result.retention_score is not None  # micro-capture must exist
    ):
        flags.append("late_fairness_injection")

    if result.causal_uptake_score < config.probe_causal_uptake_threshold:
        flags.append("decorative_observation")

    if (
        result.late_injection_score is not None
        and result.late_injection_score > config.probe_late_injection_threshold
        and result.narrative_lock_in is not None
        and result.narrative_lock_in > config.probe_narrative_lock_in_threshold
    ):
        flags.append("revision_laundering")

    if (
        result.coverage is not None
        and result.coverage > 0.80
        and result.causal_uptake_score < 0.35
    ):
        flags.append("checklist_symmetry")

    # salience_capture — fires when micro-capture is dominated by high-salience clues.
    # Uses timing_consistency (avg salience of initial events) rather than
    # salience_distortion (which normalises over the full observed set and loses
    # the signal when misdirection and diagnostic weights are close).
    if (
        result.timing_consistency is not None
        and result.timing_consistency > 0.70
    ):
        flags.append("salience_capture")

    if (
        result.narrative_lock_in is not None
        and result.narrative_lock_in > config.probe_narrative_lock_in_threshold
    ):
        flags.append("premature_commitment")

    # diagnostic_dropout — cognitive disconnect: a high-diagnostic clue was noticed
    # in micro-capture but NONE of those clues appeared in downstream reasoning.
    # This is the clearest signal of perception–reasoning mismatch.
    if (
        initial_ids
        and used_ids is not None
        and evidence_map
    ):
        early_high_diag = {
            eid for eid in initial_ids
            if evidence_map.get(eid)
            and evidence_map[eid].diagnostic_weight >= 0.65
            and evidence_map[eid].critical
        }
        if early_high_diag and not (early_high_diag & used_ids):
            flags.append("diagnostic_dropout")

    return flags


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------

def should_probe(
    result: PerceptionIntegrityResult,
    config: PerceptionConfig,
) -> bool:
    """Return True if Stage 2 probes should be shown after Phase 6."""
    return (
        (result.salience_distortion is not None
         and result.salience_distortion > config.probe_salience_distortion_threshold)
        or (result.timing_consistency is not None
            and result.timing_consistency > 0.70)
        or (result.late_injection_score is not None
            and result.late_injection_score > config.probe_late_injection_threshold)
        or (result.narrative_lock_in is not None
            and result.narrative_lock_in > config.probe_narrative_lock_in_threshold)
        or result.strategic_neutrality_score > 0.0
        or result.causal_uptake_score < config.probe_causal_uptake_threshold
        or "diagnostic_dropout" in result.adversarial_flags
    )


def should_probe_stage1(
    initial_crit_coverage: float,
    trace: PerceptionTrace,
    config: PerceptionConfig,
) -> bool:
    """Return True if a light Stage 1 probe should fire after Phase 3."""
    lock_in = narrative_lock_in(trace, initial_crit_coverage)
    return (
        initial_crit_coverage < config.stage1_coverage_threshold
        or lock_in > config.stage1_lock_in_threshold
    )


def select_probes(
    result: PerceptionIntegrityResult,
    config: PerceptionConfig,
    probes_seen: set[str],
) -> list[str]:
    """Return 1–2 targeted probe prompts, avoiding already-shown ones.

    Parameters
    ----------
    result      : PerceptionIntegrityResult
    config      : PerceptionConfig
    probes_seen : set of probe strings already shown on previous attempts
                  for this case (from user_profile.probes_seen_per_case)
    """
    candidates: list[str] = []

    if (result.late_injection_score is not None
            and result.late_injection_score > config.probe_late_injection_threshold):
        candidates.extend(PROBE_BANK["late_injection"])

    if (result.salience_distortion is not None
            and result.salience_distortion > config.probe_salience_distortion_threshold):
        candidates.extend(PROBE_BANK["salience_distortion"])

    if (result.narrative_lock_in is not None
            and result.narrative_lock_in > config.probe_narrative_lock_in_threshold):
        candidates.extend(PROBE_BANK["narrative_lock_in"])

    if result.causal_uptake_score < config.probe_causal_uptake_threshold:
        candidates.extend(PROBE_BANK["low_causal_uptake"])

    if "diagnostic_dropout" in result.adversarial_flags:
        candidates.extend(PROBE_BANK["diagnostic_dropout"])

    if not candidates:
        candidates = list(PROBE_BANK["general"])

    # Rotation: prefer probes not yet seen; fall back to full list if exhausted
    fresh = [p for p in candidates if p not in probes_seen]
    if not fresh:
        fresh = candidates  # reset rotation

    return fresh[:2]


# ---------------------------------------------------------------------------
# Evaluation confidence helper
# ---------------------------------------------------------------------------

def _compute_evaluation_confidence(
    has_evidence_meta: bool,
    has_micro_capture: bool,
    id_match_rate: float,
    config: PerceptionConfig,
) -> float:
    """Estimate how much to trust the perception metrics.

    Returns a value in [0.30, 0.90].
    Capped at 0.35 when id_match_rate is below min_reliable_match_rate.
    """
    if id_match_rate < config.min_reliable_match_rate:
        return 0.35

    if has_evidence_meta and has_micro_capture:
        return 0.90
    if has_evidence_meta:
        return 0.55
    if has_micro_capture:
        return 0.65
    return 0.30


# ---------------------------------------------------------------------------
# Timing data quality helper
# ---------------------------------------------------------------------------

def _timing_data_quality(trace: PerceptionTrace) -> str:
    """Assess the reliability of timestamp-based ordering.

    CLI timestamps include typing latency, so they are always 'coarse' at best.
    Returns 'reliable' only when all three anchor timestamps are present and
    in the expected order (Phase 0 < Phase 3 < Phase 6).
    """
    t0 = trace.initial[0].timestamp if trace.initial else None
    t3 = trace.first_hypothesis_timestamp
    t6 = trace.first_elimination_timestamp

    if t0 is None or t3 is None or t6 is None:
        return "unavailable"

    if t0 <= t3 <= t6:
        return "coarse"  # ordering is valid but gaps are not meaningful

    return "unavailable"  # timestamps out of order — don't use them


# ---------------------------------------------------------------------------
# Main extraction helpers
# ---------------------------------------------------------------------------

def _all_observed_ids(trace: PerceptionTrace) -> set[str]:
    """All unique evidence IDs across initial + later observations."""
    ids: set[str] = set()
    for event in trace.initial + trace.later:
        ids.update(event.evidence_ids)
    return ids


def _initial_ids(trace: PerceptionTrace) -> set[str]:
    """Evidence IDs from micro-capture (Phase 0) only."""
    ids: set[str] = set()
    for event in trace.initial:
        ids.update(event.evidence_ids)
    return ids


def _later_ids(trace: PerceptionTrace) -> set[str]:
    """Evidence IDs from Phase 1+ observations (after micro-capture)."""
    ids: set[str] = set()
    for event in trace.later:
        ids.update(event.evidence_ids)
    return ids


# ---------------------------------------------------------------------------
# Phase B score adjustment
# ---------------------------------------------------------------------------

def perception_adjustment(
    result: PerceptionIntegrityResult,
    config: PerceptionConfig,
) -> float:
    """Compute a bounded score adjustment to add to the V10 score.

    Only applied when evaluation_confidence >= config.adjustment_min_confidence.
    Returns 0.0 if confidence is too low to trust the metrics.
    Clamped to [config.adjustment_min, config.adjustment_max].
    """
    if result.evaluation_confidence < config.adjustment_min_confidence:
        return 0.0

    coverage = result.coverage or 0.0
    retention = result.retention_score or 0.0
    late_inj = result.late_injection_score or 0.0
    sal_dist = result.salience_distortion or 0.0
    lock_in = result.narrative_lock_in or 0.0

    raw = (
        0.8 * coverage
        + 0.5 * retention
        + 0.6 * result.causal_uptake_score
        - 0.7 * late_inj
        - 0.6 * sal_dist
        - 0.6 * lock_in
    )
    return max(config.adjustment_min, min(config.adjustment_max, 2.0 * (raw - 0.5)))


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------

class PerceptionIntegrityValidator:
    """Validates perception integrity from a PerceptionTrace.

    Usage::

        validator = PerceptionIntegrityValidator()
        result = validator.validate(
            trace=record.perception_trace,
            evidence_map=case_evidence_map,        # dict[str, EvidenceMeta] or {}
            used_in_reasoning_ids=reasoning_ids,   # from causality validator output
        )
    """

    def __init__(self, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()

    def validate(
        self,
        trace: PerceptionTrace | None,
        evidence_map: dict[str, EvidenceMeta],
        used_in_reasoning_ids: set[str] | None = None,
    ) -> PerceptionIntegrityResult:
        """Run all perception integrity checks.

        Parameters
        ----------
        trace                   : PerceptionTrace or None (returns minimal result)
        evidence_map            : dict[str, EvidenceMeta] from case JSON; may be empty
        used_in_reasoning_ids   : set of evidence IDs that appeared in reasoning
                                  phases (passed in from causality validator or
                                  extracted from elimination+reasons text)
        """
        if trace is None:
            return self._empty_result()

        has_meta = bool(evidence_map)
        has_micro = bool(trace.initial)
        used_ids = used_in_reasoning_ids or set()

        # Evidence ID sets
        init_ids = _initial_ids(trace)
        later_ids_set = _later_ids(trace)
        all_ids = init_ids | later_ids_set
        critical_ids = {eid for eid, m in evidence_map.items() if m.critical}

        # id_match_rate: fraction of all observation events that matched ≥1 ID
        total_events = len(trace.initial) + len(trace.later)
        matched_events = sum(
            1 for ev in trace.initial + trace.later if ev.evidence_ids
        )
        id_match_rate = matched_events / total_events if total_events > 0 else 0.0

        eval_confidence = _compute_evaluation_confidence(
            has_evidence_meta=has_meta,
            has_micro_capture=has_micro,
            id_match_rate=id_match_rate,
            config=self.config,
        )

        # Suppress ID-dependent metrics when match rate too low
        reliable = id_match_rate >= self.config.min_reliable_match_rate

        cov = coverage_score(all_ids, critical_ids) if reliable else None
        ret = retention_score(init_ids, later_ids_set) if reliable and init_ids else None
        late_inj = (
            late_injection_score(init_ids, later_ids_set, critical_ids)
            if reliable
            else None
        )
        causal_uptake = causal_uptake_score(all_ids, used_ids)

        # Metadata-dependent metrics
        if has_meta:
            sal_dist = salience_distortion(all_ids, evidence_map)
            init_crit_cov = coverage_score(init_ids, critical_ids) if reliable else 0.0
            lock_in = narrative_lock_in(trace, init_crit_cov)
            t_cons = timing_consistency(trace, evidence_map)
        else:
            sal_dist = None
            lock_in = None
            t_cons = None

        timing_quality = _timing_data_quality(trace)

        # Build preliminary result (strategic neutrality needs the result object)
        result = PerceptionIntegrityResult(
            coverage=cov,
            retention_score=ret,
            late_injection_score=late_inj,
            causal_uptake_score=causal_uptake,
            salience_distortion=sal_dist,
            narrative_lock_in=lock_in,
            timing_consistency=t_cons,
            strategic_neutrality_score=0.0,  # computed below
            evaluation_confidence=eval_confidence,
            timing_data_quality=timing_quality,
            id_match_rate=id_match_rate,
        )

        # Adversarial detection (needs complete result)
        result.strategic_neutrality_score = _compute_strategic_neutrality_score(
            result, self.config
        )
        result.adversarial_flags = _collect_adversarial_flags(
            result, self.config,
            initial_ids=init_ids,
            used_ids=used_ids,
            evidence_map=evidence_map if has_meta else None,
        )

        return result

    def _empty_result(self) -> PerceptionIntegrityResult:
        """Return a minimal result when no trace is available."""
        return PerceptionIntegrityResult(
            coverage=None,
            retention_score=None,
            late_injection_score=None,
            causal_uptake_score=0.0,
            salience_distortion=None,
            narrative_lock_in=None,
            timing_consistency=None,
            strategic_neutrality_score=0.0,
            evaluation_confidence=0.30,
            timing_data_quality="unavailable",
            id_match_rate=0.0,
        )
