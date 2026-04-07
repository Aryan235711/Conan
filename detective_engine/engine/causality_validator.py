"""Causality validator — evidence derivation integrity checker (V10).

The existing ReasoningGraphValidator checks structural *shape*:
    "Do observations and hypotheses share keywords?"

This module checks structural *flow*:
    "Does each reasoning phase actually DERIVE from the previous one?"

Seven validation dimensions (weighted — not all failures are equal):

    1. Evidence Provenance       (0-3)  FATAL      — phantom derivation is fatal
    2. Logical Continuity        (0-2)  SERIOUS    — inference leaps are serious
    3. Counter-Evidence Handling  (0-2)  MINOR      — 3-tier: ignored/mentioned/refuted
    4. Temporal Integrity        (0-1)  STRUCTURAL — concept-origin tracking
    5. Confidence Qualifier      (0-1)  BONUS      — grounded hedging only
    6. Hypothesis Diversity      (0-2)  SERIOUS    — competing explanations must differ
    7. Root Validity             (0-N)  PENALTY    — inference trap detection

Total budget: 0-11 points + trap penalties (always active).

V10 changes from V9.1:
    Fix #1: Confidence anti-gaming — hedging phrases must co-occur with
            evidence concepts in the same sentence.
    Fix #2: Temporal concept-origin — replace brittle 70% heuristic with
            concept-origin tracking + live phase timestamps.
    Fix #4: Counter-evidence 3-tier — expand from binary (0/1) to
            ignored (0) / mentioned (1) / refuted (2).
    Fix #5: Hypothesis diversity — new dimension checking that Phase 3
            competing explanations are semantically distinct.
    Fix #3: Root validity — inference traps defined per-case in JSON;
            penalty for triggering known wrong inferences.

The key insight:

    A hypothesis can be CORRECT but NOT DERIVED.

    "The scene is staged" is right — but if the user never observed the
    contradiction between activity traces and "undisturbed", the hypothesis
    appeared from nowhere.  That's a *phantom derivation*.

Failure modes (ordered by severity):

    PHANTOM  — Conclusion references evidence never processed       [FATAL]
    LEAP     — Conclusion skips the hypothesis bridge               [SERIOUS]
    BACKFILL — Elimination-specific concepts absent from observations [STRUCTURAL]
    BLIND    — Counter-evidence was completely ignored               [MINOR]
    TRAP     — User fell into a known wrong inference               [PENALTY]

Concept normalization:

    Synonym clusters map related words to canonical concepts:
    "cleaned", "wiped", "scrubbed", "washed" -> {clean}
    This ensures "cleaned surface" and "wiped prints" are recognized as the
    same evidence concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    AnalysisRecord,
    CaseDefinition,
    normalize_key,
)


# ---------------------------------------------------------------------------
# Synonym clusters — semantic equivalence classes for evidence concepts
# ---------------------------------------------------------------------------

_SYNONYM_CLUSTERS: dict[str, list[str]] = {
    # Physical actions
    "clean": ["cleaned", "wiped", "scrubbed", "washed", "polished", "sanitized"],
    "remove": ["removed", "erased", "deleted", "eliminated", "cleared", "stripped"],
    "break": ["broken", "shattered", "cracked", "smashed", "fractured", "snapped"],
    "move": ["moved", "shifted", "displaced", "repositioned", "relocated"],
    "open": ["opened", "ajar", "unlocked", "unsealed", "uncovered"],
    "close": ["closed", "shut", "sealed", "locked", "latched"],
    "hide": ["hidden", "concealed", "obscured", "covered", "masked", "buried"],
    "stage": ["staged", "arranged", "planted", "fabricated", "constructed"],
    # Evidence concepts
    "print": ["fingerprint", "fingerprints", "prints", "thumbprint"],
    "trace": ["traces", "residue", "remnants", "marks", "markings"],
    "blood": ["bloodstain", "bloodstains", "bleeding", "bloody"],
    "poison": ["poisoned", "toxin", "toxic", "venom", "lethal"],
    # States
    "disturb": ["disturbed", "undisturbed", "disrupted", "tampered", "touched"],
    "calm": ["peaceful", "quiet", "serene", "tranquil"],
    "recent": ["recently", "fresh", "freshly", "newly"],
    # Positions
    "pull": ["pulled", "pulling", "dragged", "tugged", "drawn"],
    "push": ["pushed", "pushing", "shoved", "pressed"],
    # Objects (forensic)
    "knife": ["blade", "dagger", "cutting"],
    "glass": ["glasses", "tumbler", "goblet"],
    "chair": ["seat", "stool"],
    "window": ["windows", "pane", "windowsill"],
    "door": ["doorway", "entrance"],
    "note": ["notebook", "notepad", "journal", "diary", "notes", "writing"],
    # Temporal
    "timeline": ["timestamp", "timing", "timed", "schedule"],
}

# Build reverse lookup: word -> canonical label
_WORD_TO_CANONICAL: dict[str, str] = {}
for _canonical, _synonyms in _SYNONYM_CLUSTERS.items():
    for _word in _synonyms:
        _WORD_TO_CANONICAL[_word.lower()] = _canonical
    _WORD_TO_CANONICAL[_canonical.lower()] = _canonical


# ---------------------------------------------------------------------------
# Stop words — common 4+ letter words that carry no evidence meaning
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "were", "they",
    "their", "them", "than", "then", "also", "just", "only", "very",
    "when", "what", "which", "where", "would", "could", "should",
    "about", "after", "before", "between", "during", "into", "more",
    "most", "some", "such", "other", "because", "since", "both",
    "each", "does", "like", "make", "made", "much", "many", "well",
    "being", "will", "come", "came", "gone", "went", "said", "says",
    "same", "still", "even", "over", "down", "here", "there", "these",
    "those", "your", "don't", "can't", "won't", "isn't",
    "aren't", "didn't", "doesn't", "wasn't", "weren't", "hasn't",
    "look", "looks", "seem", "seems", "show", "shows", "tell", "tells",
    "used", "using", "need", "needs", "want", "wants", "take", "takes",
    "give", "gives", "keep", "keeps", "left", "right", "back", "next",
    "might", "maybe", "must", "shall", "quite", "rather", "already",
    "often", "never", "always", "really", "actually", "certainly",
    "something", "everything", "nothing", "anything", "someone",
    "everyone", "appears", "otherwise", "slightly",
})


def _normalize_word(word: str) -> str:
    """Map a word to its canonical form via synonym clusters."""
    lower = word.lower()
    return _WORD_TO_CANONICAL.get(lower, lower)


def _significant_concepts(text: str) -> set[str]:
    """Extract meaningful concepts from text (>3 chars, not stop, normalized).

    Returns canonical concept labels, not raw words.  This means
    "cleaned surface" and "wiped prints" both yield the concept {clean}.
    """
    raw_words = normalize_key(text).split()
    result: set[str] = set()
    for w in raw_words:
        w = w.strip(".,;:!?\"'()[]{}-—")   # strip trailing punctuation
        if len(w) > 3 and w not in _STOP_WORDS:
            result.add(_normalize_word(w))
    return result


# ---------------------------------------------------------------------------
# Confidence / hedging detection
# ---------------------------------------------------------------------------

_HEDGING_PHRASES = [
    "likely", "unlikely", "probably", "possibly", "perhaps",
    "suggests", "indicates", "consistent with", "points to",
    "more probable", "less probable", "most probable", "least probable",
    "high probability", "low probability", "moderate probability",
    "strong evidence", "weak evidence", "diagnostic",
    "weighing", "weight of evidence", "balance of evidence",
    "base rate", "prior", "posterior",
    "confidence", "certainty", "uncertain",
    "rules out", "does not rule out", "cannot exclude",
    "more likely than not", "on balance",
]


def _count_hedging(texts: list[str]) -> int:
    """Count how many hedging/probabilistic phrases appear (legacy, ungrounded)."""
    combined = normalize_key(" ".join(texts))
    return sum(1 for phrase in _HEDGING_PHRASES if phrase in combined)


def _count_grounded_hedging(texts: list[str], evidence_concepts: set[str]) -> int:
    """Count hedging phrases that co-occur with evidence concepts in the same sentence.

    V10 anti-gaming: A hedging phrase only counts if the SAME sentence also
    contains at least one evidence concept from the case.  This prevents
    spam-stuffing generic hedges like "probably something happened" that
    carry no case-specific content.

    Returns the number of *sentences* that are both hedged AND grounded.
    """
    grounded = 0
    for text in texts:
        # Split into sentences (rough but effective for this use case)
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in ".!?":
                sentences.append("".join(current).strip())
                current = []
        if current:
            remainder = "".join(current).strip()
            if remainder:
                sentences.append(remainder)

        # Also treat each comma-separated item as a potential sentence
        # (user inputs are often comma-separated, not period-terminated)
        if not sentences or (len(sentences) == 1 and "." not in text):
            sentences = [s.strip() for s in text.split(",") if s.strip()]

        for sentence in sentences:
            norm = normalize_key(sentence)
            has_hedge = any(phrase in norm for phrase in _HEDGING_PHRASES)
            if not has_hedge:
                continue
            # Check if sentence contains at least one evidence concept
            sentence_concepts = _significant_concepts(sentence)
            if sentence_concepts & evidence_concepts:
                grounded += 1
    return grounded


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CausalityDimension:
    """Score for a single causality dimension."""
    name: str
    score: int
    max_score: int
    details: str
    severity: str = "medium"    # "fatal" | "serious" | "minor" | "structural" | "bonus"


@dataclass
class EvidenceThread:
    """Tracks one evidence concept through the reasoning chain.

    A concept is any meaningful word extracted from the case's evidence list,
    normalized through synonym clusters.  For each concept, we record whether
    it appeared in each reasoning phase.
    """
    concept: str
    in_observations: bool = False
    in_hypotheses: bool = False
    in_elimination: bool = False

    @property
    def full_chain(self) -> bool:
        return self.in_observations and self.in_hypotheses and self.in_elimination

    @property
    def is_phantom(self) -> bool:
        return self.in_elimination and not self.in_observations

    @property
    def is_leap(self) -> bool:
        return self.in_observations and self.in_elimination and not self.in_hypotheses

    @property
    def is_dormant(self) -> bool:
        return self.in_observations and not self.in_elimination

    @property
    def status_icon(self) -> str:
        if self.full_chain:
            return "\u26d3"      # chain
        if self.is_phantom:
            return "\U0001f47b"  # ghost
        if self.is_leap:
            return "\u26a1"      # lightning
        if self.is_dormant:
            return "\U0001f4a4"  # zzz
        return "\u00b7"

    @property
    def phase_trace(self) -> str:
        """Show which phases the concept appeared in: [O H E] format."""
        o = "O" if self.in_observations else "\u00b7"
        h = "H" if self.in_hypotheses else "\u00b7"
        e = "E" if self.in_elimination else "\u00b7"
        return f"[{o} {h} {e}]"


@dataclass
class CausalityResult:
    """Aggregated causality analysis."""
    dimensions: list[CausalityDimension] = field(default_factory=list)
    total_score: int = 0
    max_score: int = 11
    evidence_threads: list[EvidenceThread] = field(default_factory=list)
    phantom_concepts: list[str] = field(default_factory=list)
    inference_leaps: list[str] = field(default_factory=list)
    full_chains: list[str] = field(default_factory=list)
    ignored_counter_evidence: list[str] = field(default_factory=list)
    temporal_violations: list[str] = field(default_factory=list)
    chain_trace_lines: list[str] = field(default_factory=list)
    inference_trap_penalties: int = 0
    inference_traps_triggered: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CausalityValidator:
    """Validates logical derivation integrity across reasoning phases.

    Seven dimensions with weighted severity (V10):
        1. Evidence Provenance       (0-3) FATAL      — phantom = hardest penalty
        2. Logical Continuity        (0-2) SERIOUS    — inference leaps
        3. Counter-Evidence Handling  (0-2) MINOR      — ignored/mentioned/refuted
        4. Temporal Integrity        (0-1) STRUCTURAL — concept-origin tracking
        5. Confidence Qualifier      (0-1) BONUS      — grounded hedging only
        6. Hypothesis Diversity      (0-2) SERIOUS    — competing explanations
        7. Root Validity             (0-N) PENALTY    — inference trap detection

    Uses synonym-normalized concepts (not raw string matching).
    """

    MIN_EVIDENCE_CONCEPTS = 3

    def validate(
        self,
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> CausalityResult:
        """Run all seven causality checks."""

        obs_concepts = self._phase_concepts(
            record.observations + record.anomalies
        )
        hyp_concepts = self._phase_concepts(
            [exp for exps in record.hypotheses.values() for exp in exps]
        )
        elim_concepts = self._phase_concepts(
            [record.elimination_target] + record.reasons
        )

        evidence_vocab: set[str] = set()
        for ev in case.evidence:
            evidence_vocab |= _significant_concepts(ev)

        threads = self._build_threads(
            evidence_vocab, obs_concepts, hyp_concepts, elim_concepts,
        )

        d1 = self._check_provenance(threads)
        d2 = self._check_continuity(threads)
        d3 = self._check_counter_evidence(case, record)
        d4 = self._check_temporal_integrity(record, obs_concepts, elim_concepts)
        d5 = self._check_confidence_qualifier(record, evidence_vocab)
        d6 = self._check_hypothesis_diversity(record)
        d7, trap_penalty, traps_triggered = self._check_root_validity(case, record)

        dims = [d1, d2, d3, d4, d5, d6]
        if d7 is not None:
            dims.append(d7)
        total = max(0, sum(d.score for d in dims) - trap_penalty)

        phantoms = [t.concept for t in threads if t.is_phantom]
        leaps = [t.concept for t in threads if t.is_leap]
        full = [t.concept for t in threads if t.full_chain]
        ignored = self._find_ignored_counter(case, record)

        temporal_viol = []
        if d4.score == 0 and "violation" in d4.details.lower():
            temporal_viol = [d4.details]

        chain_lines = self._build_chain_trace(threads)

        return CausalityResult(
            dimensions=dims,
            total_score=total,
            max_score=11,
            evidence_threads=threads,
            phantom_concepts=phantoms,
            inference_leaps=leaps,
            full_chains=full,
            ignored_counter_evidence=ignored,
            temporal_violations=temporal_viol,
            chain_trace_lines=chain_lines,
            inference_trap_penalties=trap_penalty,
            inference_traps_triggered=traps_triggered,
        )

    # ------------------------------------------------------------------
    # Phase concept extraction (synonym-normalized)
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_concepts(texts: list[str]) -> set[str]:
        result: set[str] = set()
        for t in texts:
            result |= _significant_concepts(t)
        return result

    # ------------------------------------------------------------------
    # Thread builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_threads(
        evidence_vocab: set[str],
        obs_concepts: set[str],
        hyp_concepts: set[str],
        elim_concepts: set[str],
    ) -> list[EvidenceThread]:
        threads: list[EvidenceThread] = []
        for concept in sorted(evidence_vocab):
            threads.append(EvidenceThread(
                concept=concept,
                in_observations=concept in obs_concepts,
                in_hypotheses=concept in hyp_concepts,
                in_elimination=concept in elim_concepts,
            ))
        return threads

    # ------------------------------------------------------------------
    # Per-concept chain trace (for feedback — Fix #4)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chain_trace(threads: list[EvidenceThread]) -> list[str]:
        """Build human-readable per-concept chain trace.

        Shows EXACTLY which link broke for each evidence concept:
            chain  chair [O H E] -- full chain
            ghost  glass [. . E] -- PHANTOM: in elimination but never observed
            bolt   knife [O . E] -- LEAP: skipped hypothesis bridge
        """
        lines: list[str] = []
        for t in threads:
            icon = t.status_icon
            trace = t.phase_trace
            if t.full_chain:
                desc = "full chain"
            elif t.is_phantom:
                desc = "PHANTOM: in elimination but never observed"
            elif t.is_leap:
                desc = "LEAP: observed -> elimination, skipped hypothesis"
            elif t.is_dormant:
                desc = "dormant: observed but unused in elimination"
            elif t.in_hypotheses and not t.in_elimination:
                desc = "hypothesized but not used in elimination"
            else:
                desc = "not referenced in reasoning"
            lines.append(f"    {icon} {t.concept:<18} {trace}  -- {desc}")
        return lines

    # ------------------------------------------------------------------
    # Dimension 1: Evidence Provenance (0-3) -- FATAL severity
    # ------------------------------------------------------------------

    def _check_provenance(
        self,
        threads: list[EvidenceThread],
    ) -> CausalityDimension:
        """Are hypotheses built on evidence that was actually observed?

        WEIGHTED HEAVILY (0-3) because phantom derivation is the most
        serious reasoning failure.

        Scoring:
            3 -- >=80% hypothesis evidence concepts grounded in observations
            2 -- >=60% grounded
            1 -- >=40% grounded
            0 -- <40% or insufficient concepts
        """
        hyp_evidence = [t for t in threads if t.in_hypotheses]

        if len(hyp_evidence) < self.MIN_EVIDENCE_CONCEPTS:
            return CausalityDimension(
                name="evidence_provenance",
                score=0,
                max_score=3,
                details=(
                    f"Only {len(hyp_evidence)} evidence concepts in hypotheses "
                    f"(need >={self.MIN_EVIDENCE_CONCEPTS}). "
                    f"Hypotheses must reference specific case evidence."
                ),
                severity="fatal",
            )

        grounded = sum(1 for t in hyp_evidence if t.in_observations)
        ratio = grounded / len(hyp_evidence)

        if ratio >= 0.80:
            score = 3
        elif ratio >= 0.60:
            score = 2
        elif ratio >= 0.40:
            score = 1
        else:
            score = 0

        phantoms = [t.concept for t in hyp_evidence if not t.in_observations]
        phantom_note = ""
        if phantoms:
            phantom_note = f" Unobserved: {', '.join(phantoms[:5])}."

        return CausalityDimension(
            name="evidence_provenance",
            score=score,
            max_score=3,
            details=(
                f"{grounded}/{len(hyp_evidence)} hypothesis evidence concepts "
                f"traced to observations ({ratio:.0%}).{phantom_note}"
            ),
            severity="fatal",
        )

    # ------------------------------------------------------------------
    # Dimension 2: Logical Continuity (0-2) -- SERIOUS severity
    # ------------------------------------------------------------------

    def _check_continuity(
        self,
        threads: list[EvidenceThread],
    ) -> CausalityDimension:
        """Does the elimination derive from hypotheses, not from raw evidence?"""
        elim_evidence = [t for t in threads if t.in_elimination]

        if len(elim_evidence) < self.MIN_EVIDENCE_CONCEPTS:
            return CausalityDimension(
                name="logical_continuity",
                score=0,
                max_score=2,
                details=(
                    f"Only {len(elim_evidence)} evidence concepts in elimination "
                    f"(need >={self.MIN_EVIDENCE_CONCEPTS}). "
                    f"Elimination must reference specific case evidence."
                ),
                severity="serious",
            )

        bridged = sum(1 for t in elim_evidence if t.in_hypotheses)
        ratio = bridged / len(elim_evidence)

        if ratio >= 0.50:
            score = 2
        elif ratio >= 0.25:
            score = 1
        else:
            score = 0

        leaps = [t.concept for t in elim_evidence if not t.in_hypotheses]
        leap_note = ""
        if leaps:
            leap_note = f" Unbridged: {', '.join(leaps[:5])}."

        return CausalityDimension(
            name="logical_continuity",
            score=score,
            max_score=2,
            details=(
                f"{bridged}/{len(elim_evidence)} elimination concepts "
                f"bridged through hypotheses ({ratio:.0%}).{leap_note}"
            ),
            severity="serious",
        )

    # ------------------------------------------------------------------
    # Dimension 3: Counter-Evidence Handling (0-2) -- MINOR severity
    # ------------------------------------------------------------------

    # Reasoning indicators that show the user actively REFUTED counter-evidence
    _REFUTATION_MARKERS = (
        "because", "contradicts", "inconsistent", "rules out", "however",
        "although", "despite", "nevertheless", "even though", "but",
        "doesn't explain", "does not explain", "fails because",
        "undermined by", "refuted by", "weakened by", "cannot account",
        "disproved", "incompatible", "conflicts with", "overridden",
    )

    @staticmethod
    def _check_counter_evidence(
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> CausalityDimension:
        """Was evidence challenging the conclusion acknowledged AND refuted?

        V10 3-tier scoring (expanded from binary 0/1):
            0 — counter-evidence completely ignored
            1 — counter-evidence mentioned / acknowledged
            2 — counter-evidence actively refuted with reasoning
        """
        if not case.contradictions:
            return CausalityDimension(
                name="counter_evidence",
                score=2,
                max_score=2,
                details="No contradictions defined -- passes by default.",
                severity="minor",
            )

        elim_text = normalize_key(
            record.elimination_target + " " + " ".join(record.reasons)
        )
        contra_text = normalize_key(
            " ".join(record.contradiction_notes)
            + " " + " ".join(record.anomalies)
        )
        # Also check reasons for refutation reasoning
        reasons_text = normalize_key(" ".join(record.reasons))

        acknowledged = 0
        refuted = 0
        total_counter = 0

        for contradiction in case.contradictions:
            a_in_elim = any(
                normalize_key(k) in elim_text
                for k in contradiction.fact_a_keywords
            )
            b_in_elim = any(
                normalize_key(k) in elim_text
                for k in contradiction.fact_b_keywords
            )

            if a_in_elim and not b_in_elim:
                counter_keywords = contradiction.fact_b_keywords
            elif b_in_elim and not a_in_elim:
                counter_keywords = contradiction.fact_a_keywords
            else:
                continue

            total_counter += 1
            mentioned = any(
                normalize_key(k) in contra_text for k in counter_keywords
            )
            if mentioned:
                acknowledged += 1
                # Check if the user also provided reasoning about the counter-evidence
                # Look for refutation markers near the counter-evidence keywords
                combined_reasoning = contra_text + " " + reasons_text
                has_refutation = any(
                    marker in combined_reasoning
                    for marker in CausalityValidator._REFUTATION_MARKERS
                )
                if has_refutation:
                    refuted += 1

        if total_counter == 0:
            return CausalityDimension(
                name="counter_evidence",
                score=2,
                max_score=2,
                details="Could not determine counter-evidence alignment -- passes by default.",
                severity="minor",
            )

        # 3-tier scoring
        if refuted > 0:
            score = 2
            tier = "refuted"
        elif acknowledged > 0:
            score = 1
            tier = "mentioned"
        else:
            score = 0
            tier = "ignored"

        return CausalityDimension(
            name="counter_evidence",
            score=score,
            max_score=2,
            details=(
                f"{acknowledged}/{total_counter} counter-evidence acknowledged, "
                f"{refuted} actively refuted — tier: {tier}."
            ),
            severity="minor",
        )

    # ------------------------------------------------------------------
    # Dimension 4: Temporal Integrity (0-1) -- STRUCTURAL severity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_temporal_integrity(
        record: AnalysisRecord,
        obs_concepts: set[str],
        elim_concepts: set[str],
    ) -> CausalityDimension:
        """Did reasoning phases happen in the correct order?

        V10: Replaces the brittle 70% heuristic with concept-origin tracking.

        Two checks (either can trigger violation):

        1. Timestamp-based (if available from case_runner):
           Phases must be chronologically ordered.

        2. Concept-origin tracking (always available):
           For each concept in elimination, check if it first appeared in
           observations or hypotheses.  If >50% of elimination concepts
           have NO upstream origin, the user likely wrote the conclusion
           first and backfilled weak observations.
        """
        violations: list[str] = []

        # Timestamp-based check (if available)
        if hasattr(record, "phase_timestamps") and record.phase_timestamps:
            ts = record.phase_timestamps
            obs_t = ts.get("observations")
            hyp_t = ts.get("hypotheses")
            elim_t = ts.get("elimination")

            if obs_t and hyp_t and obs_t > hyp_t:
                violations.append("hypotheses timestamped BEFORE observations")
            if hyp_t and elim_t and hyp_t > elim_t:
                violations.append("elimination timestamped BEFORE hypotheses")
            if obs_t and elim_t and obs_t > elim_t:
                violations.append("elimination timestamped BEFORE observations")

        # Concept-origin tracking (V10 — replaces 70% heuristic)
        # Check what fraction of elimination concepts have NO upstream origin
        if len(elim_concepts) >= 3:
            orphan_concepts = elim_concepts - obs_concepts
            # Also check hypothesis concepts as valid upstream
            hyp_concepts = set()
            for exps in record.hypotheses.values():
                for exp in exps:
                    hyp_concepts |= _significant_concepts(exp)
            orphan_concepts = orphan_concepts - hyp_concepts

            orphan_ratio = len(orphan_concepts) / len(elim_concepts)
            if orphan_ratio > 0.50:
                violations.append(
                    f"{len(orphan_concepts)}/{len(elim_concepts)} elimination "
                    f"concepts ({orphan_ratio:.0%}) have no upstream origin "
                    f"in observations or hypotheses — possible backfill"
                )

        if violations:
            return CausalityDimension(
                name="temporal_integrity",
                score=0,
                max_score=1,
                details=f"Temporal violation: {'; '.join(violations)}.",
                severity="structural",
            )

        return CausalityDimension(
            name="temporal_integrity",
            score=1,
            max_score=1,
            details="Phase ordering consistent — concept origins verified.",
            severity="structural",
        )

    # ------------------------------------------------------------------
    # Dimension 5: Confidence Qualifier (0-1) -- BONUS
    # ------------------------------------------------------------------

    @staticmethod
    def _check_confidence_qualifier(
        record: AnalysisRecord,
        evidence_concepts: set[str],
    ) -> CausalityDimension:
        """Does the user show calibrated confidence with grounded hedging?

        V10 anti-gaming: hedging language ONLY counts if it co-occurs
        with case-specific evidence concepts in the same sentence.

        "likely" by itself = gaming.
        "chair likely indicates recent presence" = grounded hedging.

        BONUS point — you don't lose for being deterministic, but you
        gain for showing case-grounded probabilistic calibration.
        """
        all_text = (
            record.observations
            + record.anomalies
            + [exp for exps in record.hypotheses.values() for exp in exps]
            + record.reasons
            + [record.elimination_target]
            + [record.false_narrative_rejection]
            + record.contradiction_notes
        )

        grounded_count = _count_grounded_hedging(all_text, evidence_concepts)
        raw_count = _count_hedging(all_text)

        if grounded_count >= 3:
            return CausalityDimension(
                name="confidence_qualifier",
                score=1,
                max_score=1,
                details=(
                    f"{grounded_count} grounded hedging phrases detected "
                    f"(co-occur with evidence concepts) — calibrated reasoning. "
                    f"({raw_count} total hedging phrases.)"
                ),
                severity="bonus",
            )

        return CausalityDimension(
            name="confidence_qualifier",
            score=0,
            max_score=1,
            details=(
                f"Only {grounded_count} grounded hedging phrases ({raw_count} total, "
                f"but {raw_count - grounded_count} lack evidence concept context). "
                f"Hedging must co-occur with case evidence: "
                f"'chair likely indicates...' not just 'likely something...'."
            ),
            severity="bonus",
        )

    # ------------------------------------------------------------------
    # Dimension 6: Hypothesis Diversity (0-2) -- SERIOUS severity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_hypothesis_diversity(
        record: AnalysisRecord,
    ) -> CausalityDimension:
        """Are Phase 3 competing explanations semantically distinct?

        V10 Fix #5: For standard (non-Bayesian) cases, Phase 3 collects
        2-3 explanations per observation but V9.1 never validated them.

        Checks:
        1. Each observation's explanations must be semantically distinct
           (concept overlap < 60% between any pair).
        2. At least one observation must have genuinely competing hypotheses.

        Scoring:
            2 — All observations have diverse explanations, ≥2 obs checked
            1 — Some diversity detected but ≥1 observation has near-duplicate explanations
            0 — All explanations are near-duplicates or too few hypotheses
        """
        if not record.hypotheses:
            return CausalityDimension(
                name="hypothesis_diversity",
                score=0,
                max_score=2,
                details="No hypotheses provided — cannot assess diversity.",
                severity="serious",
            )

        obs_checked = 0
        obs_diverse = 0

        for obs, explanations in record.hypotheses.items():
            if len(explanations) < 2:
                continue
            obs_checked += 1

            # Check pairwise concept overlap between explanations
            concept_sets = [_significant_concepts(exp) for exp in explanations]
            all_pairs_distinct = True

            for i in range(len(concept_sets)):
                for j in range(i + 1, len(concept_sets)):
                    set_a = concept_sets[i]
                    set_b = concept_sets[j]
                    if not set_a or not set_b:
                        continue
                    overlap = len(set_a & set_b)
                    union = len(set_a | set_b)
                    if union > 0 and overlap / union >= 0.60:
                        all_pairs_distinct = False
                        break
                if not all_pairs_distinct:
                    break

            if all_pairs_distinct:
                obs_diverse += 1

        if obs_checked == 0:
            return CausalityDimension(
                name="hypothesis_diversity",
                score=0,
                max_score=2,
                details="No observations with multiple explanations to compare.",
                severity="serious",
            )

        if obs_diverse == obs_checked:
            score = 2
        elif obs_diverse > 0:
            score = 1
        else:
            score = 0

        return CausalityDimension(
            name="hypothesis_diversity",
            score=score,
            max_score=2,
            details=(
                f"{obs_diverse}/{obs_checked} observations have semantically "
                f"distinct competing explanations (< 60% concept overlap)."
            ),
            severity="serious",
        )

    # ------------------------------------------------------------------
    # Dimension 7: Root Validity (penalty) -- inference trap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_root_validity(
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> tuple[CausalityDimension | None, int, list[str]]:
        """Check if user fell into known inference traps.

        V10 Fix #3: Cases can define inference_traps — common wrong
        inferences that seem logical but are factually wrong.

        This is PENALTY-BASED: you don't earn points for avoiding traps,
        but you LOSE points for triggering them.

        Returns:
            (dimension_or_None, total_penalty, list_of_triggered_trap_descriptions)
        """
        # Check if case has inference traps defined
        inference_traps = getattr(case.solution, "inference_traps", None)
        if not inference_traps:
            return None, 0, []

        # Build user's full reasoning corpus
        all_reasoning = normalize_key(" ".join(
            [record.elimination_target]
            + record.reasons
            + [exp for exps in record.hypotheses.values() for exp in exps]
            + record.anomalies
        ))

        triggered: list[str] = []
        total_penalty = 0

        for trap in inference_traps:
            # A trap is triggered if the user's reasoning contains the trap keywords
            trap_keywords = getattr(trap, "trap_keywords", [])
            penalty = getattr(trap, "penalty", 1)
            description = getattr(trap, "description", "unknown trap")

            if trap_keywords and any(
                normalize_key(kw) in all_reasoning for kw in trap_keywords
            ):
                triggered.append(description)
                total_penalty += penalty

        if not triggered:
            dim = CausalityDimension(
                name="root_validity",
                score=0,
                max_score=0,
                details=f"No inference traps triggered ({len(inference_traps)} traps checked).",
                severity="bonus",
            )
            return dim, 0, []

        dim = CausalityDimension(
            name="root_validity",
            score=0,
            max_score=0,
            details=(
                f"⚠ {len(triggered)} inference trap(s) triggered "
                f"(−{total_penalty} pts): {'; '.join(triggered)}"
            ),
            severity="fatal",
        )
        return dim, total_penalty, triggered

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ignored_counter(
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> list[str]:
        """Return descriptions of counter-evidence completely ignored."""
        if not case.contradictions:
            return []

        elim_text = normalize_key(
            record.elimination_target + " " + " ".join(record.reasons)
        )
        contra_text = normalize_key(
            " ".join(record.contradiction_notes)
            + " " + " ".join(record.anomalies)
        )

        ignored: list[str] = []
        for contradiction in case.contradictions:
            a_in_elim = any(
                normalize_key(k) in elim_text
                for k in contradiction.fact_a_keywords
            )
            b_in_elim = any(
                normalize_key(k) in elim_text
                for k in contradiction.fact_b_keywords
            )

            if a_in_elim and not b_in_elim:
                counter_keywords = contradiction.fact_b_keywords
            elif b_in_elim and not a_in_elim:
                counter_keywords = contradiction.fact_a_keywords
            else:
                continue

            if not any(normalize_key(k) in contra_text for k in counter_keywords):
                ignored.append(contradiction.description)

        return ignored
