"""Bayesian reasoning validator — scores probabilistic thinking quality.

Evaluates whether the user demonstrates proper Bayesian reasoning when a
case has multiple logically valid conclusions.  The five dimensions:

    1. Multi-hypothesis identification  (0–1)  Did you see ALL valid paths?
    2. Correct probability ranking       (0–2)  Did you rank the most probable first?
    3. Evidence weighting quality         (0–2)  Diagnostic vs neutral evidence?
    4. Prior / base-rate awareness        (0–1)  Statistics, behavioral patterns?
    5. Convergence reasoning              (0–1)  Independent lines converging?

Total budget: 0–7 points.

Probability can come from FIVE channels (the user asked this exact question):
    • Statistical base rates   — "82% of poisoning cases involve sustained access"
    • Behavioral patterns      — "Elena lied about the chemical purchase"
    • Physical evidence weight  — "sustained access is more diagnostic than brief proximity"
    • Temporal / contextual    — "2-4h window matches Elena's continuous access"
    • Convergence              — "access + purchase + lie + base rate all point to Elena"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    AnalysisRecord,
    BayesianSolution,
    CaseDefinition,
    normalize_key,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BayesianDimension:
    """Score for a single Bayesian reasoning dimension."""
    name: str
    score: int
    max_score: int
    details: str


@dataclass
class BayesianResult:
    """Aggregated Bayesian reasoning evaluation."""
    dimensions: list[BayesianDimension] = field(default_factory=list)
    total_score: int = 0
    max_score: int = 7
    cognitive_traps_triggered: list[str] = field(default_factory=list)
    most_probable_identified: bool = False


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class BayesianValidator:
    """Scores probabilistic reasoning quality for multi-hypothesis cases.

    Only activates when the case's solution contains a ``bayesian`` field.
    For standard (single-answer) cases, returns a zero-score result with
    ``max_score=0`` so the overall budget is unaffected.
    """

    def validate(
        self,
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> BayesianResult:
        bayesian = case.solution.bayesian
        if not bayesian:
            return BayesianResult(total_score=0, max_score=0)

        dims: list[BayesianDimension] = []

        # 1. Multi-hypothesis identification (0-1)
        d1 = self._score_multi_hypothesis(bayesian, record)
        dims.append(d1)

        # 2. Correct probability ranking (0-2)
        d2 = self._score_ranking(bayesian, record)
        dims.append(d2)

        # 3. Evidence weighting (0-2)
        d3 = self._score_evidence_weighting(bayesian, record)
        dims.append(d3)

        # 4. Prior awareness (0-1)
        d4 = self._score_prior_awareness(bayesian, record)
        dims.append(d4)

        # 5. Convergence reasoning (0-1)
        d5 = self._score_convergence(bayesian, record)
        dims.append(d5)

        # Cognitive traps
        traps = self._check_cognitive_traps(bayesian, record)

        total = sum(d.score for d in dims)

        return BayesianResult(
            dimensions=dims,
            total_score=total,
            max_score=7,
            cognitive_traps_triggered=traps,
            most_probable_identified=d2.score >= 1,
        )

    # ------------------------------------------------------------------
    # Dimension 1: Multi-hypothesis identification (0-1)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_multi_hypothesis(
        bayesian: BayesianSolution, record: AnalysisRecord,
    ) -> BayesianDimension:
        """Did the user identify multiple valid hypotheses?"""
        # Primary: check explicit probability_ranking
        ranking = record.probability_ranking
        if len(ranking) >= 2:
            user_labels = normalize_key(" ".join(
                str(r.get("hypothesis", "")) for r in ranking
            ))
            case_labels = [normalize_key(h.label) for h in bayesian.hypotheses]
            matches = sum(
                1 for cl in case_labels
                if any(w in user_labels for w in cl.split() if len(w) > 3)
            )
            if matches >= 2:
                return BayesianDimension(
                    name="multi_hypothesis", score=1, max_score=1,
                    details=f"Identified {matches}/{len(bayesian.hypotheses)} hypotheses.",
                )

        # Fallback: check all reasoning text for multiple suspect references
        corpus = normalize_key(" ".join(
            record.reasons + record.anomalies
            + record.evidence_weight_notes
            + record.prior_reasoning
            + [exp for exps in record.hypotheses.values() for exp in exps]
        ))
        mentioned = sum(
            1 for h in bayesian.hypotheses
            if any(
                normalize_key(word) in corpus
                for word in h.label.split() if len(word) > 3
            )
        )
        if mentioned >= 2:
            return BayesianDimension(
                name="multi_hypothesis", score=1, max_score=1,
                details=f"Referenced {mentioned} hypotheses in reasoning.",
            )

        return BayesianDimension(
            name="multi_hypothesis", score=0, max_score=1,
            details="Only one hypothesis considered — missed alternative explanations.",
        )

    # ------------------------------------------------------------------
    # Dimension 2: Correct probability ranking (0-2)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_ranking(
        bayesian: BayesianSolution, record: AnalysisRecord,
    ) -> BayesianDimension:
        """Did the user correctly rank the most probable hypothesis?"""
        most_probable = next(
            (h for h in bayesian.hypotheses if h.id == bayesian.most_probable),
            None,
        )
        if not most_probable:
            return BayesianDimension(
                name="correct_ranking", score=0, max_score=2,
                details="Case configuration error — most_probable not found.",
            )

        mp_label = normalize_key(most_probable.label)
        mp_words = [w for w in mp_label.split() if len(w) > 3]

        # Primary: check explicit probability_ranking
        if record.probability_ranking:
            top = record.probability_ranking[0]
            top_label = normalize_key(str(top.get("hypothesis", "")))
            if any(w in top_label for w in mp_words):
                return BayesianDimension(
                    name="correct_ranking", score=2, max_score=2,
                    details=f"Correctly ranked '{most_probable.label}' as most probable.",
                )
            # Partial credit: mentioned but not top-ranked
            all_labels = normalize_key(" ".join(
                str(r.get("hypothesis", "")) for r in record.probability_ranking
            ))
            if any(w in all_labels for w in mp_words):
                return BayesianDimension(
                    name="correct_ranking", score=1, max_score=2,
                    details=f"Identified '{most_probable.label}' but not ranked first.",
                )

        # Fallback: check reasoning text for reference to the correct suspect
        corpus = normalize_key(
            record.elimination_target + " "
            + " ".join(record.reasons)
            + " ".join(record.evidence_weight_notes)
        )
        if any(w in corpus for w in mp_words):
            return BayesianDimension(
                name="correct_ranking", score=1, max_score=2,
                details=f"Referenced '{most_probable.label}' in reasoning (no explicit ranking).",
            )

        return BayesianDimension(
            name="correct_ranking", score=0, max_score=2,
            details=f"Did not identify '{most_probable.label}' as most probable.",
        )

    # ------------------------------------------------------------------
    # Dimension 3: Evidence weighting (0-2)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_evidence_weighting(
        bayesian: BayesianSolution, record: AnalysisRecord,
    ) -> BayesianDimension:
        """Did the user distinguish diagnostic from neutral evidence?"""
        high_diagnostic = [
            w for w in bayesian.evidence_weights
            if w.diagnostic_power in ("high", "moderate")
        ]
        if not high_diagnostic:
            return BayesianDimension(
                name="evidence_weighting", score=0, max_score=2,
                details="No diagnostic evidence defined in case.",
            )

        # Build corpus from all reasoning + explicit weight notes
        reasoning_corpus = normalize_key(" ".join(
            record.reasons + record.anomalies
            + record.evidence_weight_notes
            + record.prior_reasoning
            + [exp for exps in record.hypotheses.values() for exp in exps]
        ))

        # Weighting language
        weight_keywords = [
            "diagnostic", "strongest", "most important", "key evidence",
            "decisive", "critical", "most telling", "strongest indicator",
            "weighs heavily", "points strongly", "most significant",
            "more likely", "less likely", "probability", "probable",
            "convincing", "compelling", "definitive", "incriminating",
        ]
        has_weighting_language = any(
            normalize_key(kw) in reasoning_corpus for kw in weight_keywords
        )

        # Did user reference specific diagnostic evidence explanations?
        # Require 3+ matching words from the explanation (not just one common word)
        diagnostic_refs = 0
        for ew in high_diagnostic:
            explanation_words = [
                w for w in normalize_key(ew.explanation).split() if len(w) > 4
            ]
            matches = sum(1 for w in explanation_words[:8] if w in reasoning_corpus)
            if matches >= 3:
                diagnostic_refs += 1

        # Gate: without explicit evidence_weight_notes, cap at 1/2
        has_explicit_notes = len(record.evidence_weight_notes) >= 1

        score = 0
        if has_explicit_notes and (diagnostic_refs >= 2 or (diagnostic_refs >= 1 and has_weighting_language)):
            score = 2
        elif diagnostic_refs >= 2 or (diagnostic_refs >= 1 and has_weighting_language):
            score = 1
        elif diagnostic_refs >= 1 or has_weighting_language:
            score = 1

        return BayesianDimension(
            name="evidence_weighting", score=score, max_score=2,
            details=(
                f"Referenced {diagnostic_refs}/{len(high_diagnostic)} diagnostic "
                f"evidence items. Weighting language: "
                f"{'yes' if has_weighting_language else 'no'}."
            ),
        )

    # ------------------------------------------------------------------
    # Dimension 4: Prior awareness (0-1)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_prior_awareness(
        bayesian: BayesianSolution, record: AnalysisRecord,
    ) -> BayesianDimension:
        """Did the user reference base rates, behavioral patterns, or statistics?"""
        corpus = normalize_key(" ".join(
            record.prior_reasoning
            + record.reasons
            + record.evidence_weight_notes
            + record.anomalies
        ))

        prior_keywords = [
            "base rate", "statistic", "typically", "most cases",
            "usually", "pattern", "behavioral", "historically",
            "common", "rare", "likelihood", "prior", "baseline",
            "frequency", "percentage", "majority", "often",
            "probability", "bayesian", "update", "percent",
        ]

        # Also check if user references the probability_channels text
        channel_words: list[str] = []
        for channel_desc in bayesian.probability_channels.values():
            channel_words.extend(
                w for w in normalize_key(channel_desc).split() if len(w) > 4
            )

        has_prior_language = any(
            normalize_key(kw) in corpus for kw in prior_keywords
        )
        has_channel_reference = any(w in corpus for w in channel_words[:12])

        if has_prior_language or has_channel_reference:
            return BayesianDimension(
                name="prior_awareness", score=1, max_score=1,
                details="Demonstrated awareness of base rates or probabilistic priors.",
            )

        return BayesianDimension(
            name="prior_awareness", score=0, max_score=1,
            details="No reference to base rates, behavioral patterns, or statistics.",
        )

    # ------------------------------------------------------------------
    # Dimension 5: Convergence reasoning (0-1)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_convergence(
        bayesian: BayesianSolution, record: AnalysisRecord,
    ) -> BayesianDimension:
        """Did the user recognize multiple independent evidence lines converging?"""
        corpus = normalize_key(" ".join(
            record.reasons
            + record.evidence_weight_notes
            + record.prior_reasoning
            + record.anomalies
        ))

        convergence_keywords = [
            "converge", "multiple lines", "independent", "together",
            "combined", "all point", "several factors", "taken together",
            "collectively", "reinforc", "corroborat", "consistent with",
            "accumulation", "weight of evidence", "multiple indicators",
            "all suggest", "each supports", "multiple pieces",
            "independently", "all lead", "points the same",
        ]

        if any(normalize_key(kw) in corpus for kw in convergence_keywords):
            return BayesianDimension(
                name="convergence", score=1, max_score=1,
                details="Recognized convergent evidence from independent lines.",
            )

        return BayesianDimension(
            name="convergence", score=0, max_score=1,
            details="Treated evidence in isolation — did not recognize convergence.",
        )

    # ------------------------------------------------------------------
    # Cognitive traps (informational — not scored, but flagged)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_cognitive_traps(
        bayesian: BayesianSolution, record: AnalysisRecord,
    ) -> list[str]:
        """Check if user fell into designed cognitive traps."""
        corpus = normalize_key(" ".join(
            record.reasons
            + [exp for exps in record.hypotheses.values() for exp in exps]
            + [record.elimination_target]
            + record.evidence_weight_notes
        ))

        triggered: list[str] = []
        for trap in bayesian.cognitive_traps:
            if any(normalize_key(kw) in corpus for kw in trap.penalty_keywords):
                triggered.append(f"{trap.name}: {trap.description}")

        return triggered
