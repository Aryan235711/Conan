"""Reasoning graph validator — structural analysis of the user's logic chain.

Validates the three-step chain:
    Observation → Hypothesis → Elimination

Each link is scored independently:
    1. Grounding   — do observations support hypotheses?
    2. Justification — do hypotheses justify the elimination?
    3. Completeness — was all key evidence addressed?

Returns a ReasoningGraphResult with per-link scores and a total (0–3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AnalysisRecord, CaseDefinition, normalize_key, contains_any


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LinkScore:
    """Score for a single link in the reasoning chain."""
    name: str          # "grounding" | "justification" | "completeness"
    score: int         # 0 or 1
    max_score: int = 1
    details: str = ""


@dataclass
class ReasoningGraphResult:
    """Aggregated structural analysis of the reasoning chain."""
    links: list[LinkScore] = field(default_factory=list)
    total_score: int = 0
    max_score: int = 3
    chain_valid: bool = False  # True if all 3 links score 1
    ungrounded_hypotheses: list[str] = field(default_factory=list)
    unjustified_elimination: bool = False
    missed_evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ReasoningGraphValidator:
    """Validates the observation→hypothesis→elimination chain.

    This is a deterministic, rule-based checker (no LLM).  It verifies
    structural integrity of the reasoning chain independent of content
    correctness (that's the Validator's job).
    """

    # Minimum overlap ratio to consider a hypothesis "grounded"
    GROUNDING_THRESHOLD = 0.30

    def validate(
        self,
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> ReasoningGraphResult:
        """Run all three structural checks."""
        grounding = self._check_grounding(record)
        justification = self._check_justification(record)
        completeness = self._check_completeness(case, record)

        links = [grounding, justification, completeness]
        total = sum(l.score for l in links)

        return ReasoningGraphResult(
            links=links,
            total_score=total,
            chain_valid=(total == 3),
            ungrounded_hypotheses=self._find_ungrounded(record),
            unjustified_elimination=(justification.score == 0),
            missed_evidence=self._find_missed_evidence(case, record),
        )

    # ------------------------------------------------------------------
    # Link 1: Grounding — observations support hypotheses?
    # ------------------------------------------------------------------

    def _check_grounding(self, record: AnalysisRecord) -> LinkScore:
        """Check that hypotheses reference terms from observations."""
        obs_corpus = normalize_key(" ".join(record.observations))
        obs_words = set(obs_corpus.split())

        grounded_count = 0
        total_hyps = 0

        for _obs_key, explanations in record.hypotheses.items():
            for exp in explanations:
                total_hyps += 1
                exp_words = set(normalize_key(exp).split())
                # Count meaningful words (>3 chars) shared between observation corpus and hypothesis
                shared = obs_words & exp_words
                meaningful_shared = {w for w in shared if len(w) > 3}
                if meaningful_shared:
                    grounded_count += 1

        if total_hyps == 0:
            return LinkScore(
                name="grounding", score=0,
                details="No hypotheses provided.",
            )

        ratio = grounded_count / total_hyps
        passed = ratio >= self.GROUNDING_THRESHOLD

        return LinkScore(
            name="grounding",
            score=1 if passed else 0,
            details=f"{grounded_count}/{total_hyps} hypotheses reference observation terms ({ratio:.0%}).",
        )

    # ------------------------------------------------------------------
    # Link 2: Justification — hypotheses justify elimination?
    # ------------------------------------------------------------------

    def _check_justification(self, record: AnalysisRecord) -> LinkScore:
        """Check that the elimination target and reasons reference hypothesis content."""
        hyp_corpus = normalize_key(" ".join(
            exp for exps in record.hypotheses.values() for exp in exps
        ))
        hyp_words = {w for w in hyp_corpus.split() if len(w) > 3}

        reason_corpus = normalize_key(
            " ".join(record.reasons) + " " + record.elimination_target
        )
        reason_words = {w for w in reason_corpus.split() if len(w) > 3}

        if not hyp_words or not reason_words:
            return LinkScore(
                name="justification", score=0,
                details="No hypotheses or reasons provided.",
            )

        shared = hyp_words & reason_words
        # Need at least some concept overlap
        ratio = len(shared) / min(len(hyp_words), len(reason_words)) if min(len(hyp_words), len(reason_words)) > 0 else 0
        passed = len(shared) >= 2 and ratio >= 0.15

        return LinkScore(
            name="justification",
            score=1 if passed else 0,
            details=f"{len(shared)} shared concepts between hypotheses and elimination ({ratio:.0%} overlap).",
        )

    # ------------------------------------------------------------------
    # Link 3: Completeness — was key evidence addressed?
    # ------------------------------------------------------------------

    def _check_completeness(
        self, case: CaseDefinition, record: AnalysisRecord,
    ) -> LinkScore:
        """Check that the user addressed most of the evidence."""
        all_user_text = normalize_key(
            " ".join(record.observations)
            + " " + " ".join(record.anomalies)
            + " " + " ".join(record.reasons)
            + " " + " ".join(record.contradiction_notes)
        )

        addressed = 0
        for ev in case.evidence:
            # Extract significant words from each evidence item
            ev_words = [w for w in normalize_key(ev).split() if len(w) > 3]
            if any(w in all_user_text for w in ev_words):
                addressed += 1

        total = len(case.evidence)
        if total == 0:
            return LinkScore(name="completeness", score=1, details="No evidence to check.")

        ratio = addressed / total
        passed = ratio >= 0.50

        return LinkScore(
            name="completeness",
            score=1 if passed else 0,
            details=f"{addressed}/{total} evidence items addressed ({ratio:.0%}).",
        )

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _find_ungrounded(self, record: AnalysisRecord) -> list[str]:
        """Return hypotheses that don't reference any observation terms."""
        obs_words = set(normalize_key(" ".join(record.observations)).split())

        ungrounded = []
        for _obs_key, explanations in record.hypotheses.items():
            for exp in explanations:
                exp_words = set(normalize_key(exp).split())
                shared = {w for w in (obs_words & exp_words) if len(w) > 3}
                if not shared:
                    ungrounded.append(exp)
        return ungrounded

    def _find_missed_evidence(
        self, case: CaseDefinition, record: AnalysisRecord,
    ) -> list[str]:
        """Return evidence items not referenced anywhere in the user's analysis."""
        all_text = normalize_key(
            " ".join(record.observations)
            + " " + " ".join(record.anomalies)
            + " " + " ".join(record.reasons)
            + " " + " ".join(record.contradiction_notes)
        )

        missed = []
        for ev in case.evidence:
            ev_words = [w for w in normalize_key(ev).split() if len(w) > 3]
            if not any(w in all_text for w in ev_words):
                missed.append(ev)
        return missed
