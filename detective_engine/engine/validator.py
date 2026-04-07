"""Deterministic evaluation engine — the source of truth.

The Validator owns every scoring decision.  The optional LLM judge adds
supplementary scores but NEVER overrides the rule engine.

Scoring is organized into four weighted pillars:

    CONTENT (30%)      What you found — concept rules, contradictions, insight usage,
                       false narrative rejection.
    STRUCTURE (25%)    How you reasoned — reasoning graph chain, LLM judge, Bayesian
                       reasoning (when applicable).
    INTEGRITY (35%)    How you derived — causality/derivation integrity (evidence
                       provenance, logical continuity, counter-evidence, temporal,
                       confidence qualifier, hypothesis diversity, inference traps).
    PERCEPTION (10%)   How you noticed — perception integrity (coverage, retention,
                       late injection, causal uptake, salience, lock-in).

Each pillar is scored 0.0–1.0 independently, then combined via weights.
The raw additive budget is preserved for backward compatibility.
"""
from __future__ import annotations

# Pillar weights — sum to 1.0
PILLAR_WEIGHTS: dict[str, float] = {
    "content": 0.30,
    "structure": 0.25,
    "integrity": 0.35,
    "perception": 0.10,
}

from .models import (
    AnalysisRecord,
    CaseDefinition,
    EvaluationResult,
    contains_any,
    normalize_key,
)

INFERENCE_MARKERS = (
    "suggests", "means", "implies", "must", "probably",
    "clearly", "obviously", "someone", "missing", "strong wind",
)


class Validator:
    """Pure-logic evaluation — no LLM, no side effects."""

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        case: CaseDefinition,
        record: AnalysisRecord,
        *,
        llm_verdict: str = "",
        llm_reasoning: str = "",
        llm_score: int = 0,
        llm_role_results: list[dict] | None = None,
        llm_critique: str = "",
        llm_counterarguments: list[str] | None = None,
        reasoning_graph_results: list[dict] | None = None,
        reasoning_graph_score: int = 0,
        bayesian_score: int = 0,
        bayesian_max: int = 0,
        bayesian_results: list[dict] | None = None,
        bayesian_traps: list[str] | None = None,
        causality_score: int = 0,
        causality_max: int = 11,
        causality_results: list[dict] | None = None,
        causality_phantom: list[str] | None = None,
        causality_leaps: list[str] | None = None,
        causality_temporal: list[str] | None = None,
        causality_chain_trace: list[str] | None = None,
        causality_trap_penalties: int = 0,
        causality_traps_triggered: list[str] | None = None,
    ) -> EvaluationResult:
        corpus = self._build_corpus(record)

        concept_results, concept_score = self._score_concept_rules(case, corpus)
        contradiction_results, contradiction_score = self._score_contradictions(case, record)
        insight_results, insight_score = self._score_insight_usage(case, corpus)
        false_narrative_rejected = self._check_false_narrative(case, record)
        forbidden_results, forbidden_penalty = self._score_forbidden(case, corpus)
        purity_penalty = self._purity_penalty(record)

        elimination_correct = contains_any(
            record.elimination_target,
            [case.solution.elimination_target] if case.solution.elimination_target else [],
        )
        direct_answer_correct = (
            contains_any(record.elimination_target, case.solution.direct_answer_keywords)
            or contains_any(corpus, case.solution.direct_answer_keywords)
        )

        # aggregates
        earned = (
            concept_score
            + contradiction_score
            + insight_score
            + reasoning_graph_score     # 0–3
            + llm_score                 # 0–3
            + bayesian_score            # 0–7 (only for Bayesian cases)
            + causality_score           # 0–11 (always active, already has trap penalty applied)
        )
        if case.solution.must_reject_false_narrative:
            earned += 1 if false_narrative_rejected else 0

        max_possible = (
            sum(r.points for r in case.solution.required_concept_rules)
            + sum(c.points for c in case.contradictions)
            + sum(r.points for r in case.solution.insight_usage_rules)
            + (1 if case.solution.must_reject_false_narrative else 0)
            + 3  # reasoning graph slots (always counted)
            + 3  # LLM slots (always counted so confidence doesn't inflate when LLM is off)
            + bayesian_max  # 7 for Bayesian cases, 0 otherwise
            + causality_max  # 11 for derivation integrity (always active)
        )

        net = max(0, earned - forbidden_penalty - purity_penalty)
        confidence = net / max_possible if max_possible > 0 else 0.0

        answer_ok = elimination_correct or direct_answer_correct

        # ── Pillar scores (0.0–1.0 each) ──
        content_max = (
            sum(r.points for r in case.solution.required_concept_rules)
            + sum(c.points for c in case.contradictions)
            + sum(r.points for r in case.solution.insight_usage_rules)
            + (1 if case.solution.must_reject_false_narrative else 0)
        )
        content_earned = (
            concept_score + contradiction_score + insight_score
            + (1 if false_narrative_rejected and case.solution.must_reject_false_narrative else 0)
        )
        # Apply penalties proportionally to content pillar
        content_net = max(0, content_earned - forbidden_penalty - purity_penalty)
        pillar_content = content_net / content_max if content_max > 0 else 0.0

        structure_max = 3 + 3 + bayesian_max  # reasoning graph + LLM + Bayesian
        structure_earned = reasoning_graph_score + llm_score + bayesian_score
        pillar_structure = structure_earned / structure_max if structure_max > 0 else 0.0

        pillar_integrity = causality_score / causality_max if causality_max > 0 else 0.0

        # Perception pillar is set externally after perception validation
        pillar_perception = 0.0

        weighted_score = (
            PILLAR_WEIGHTS["content"] * pillar_content
            + PILLAR_WEIGHTS["structure"] * pillar_structure
            + PILLAR_WEIGHTS["integrity"] * pillar_integrity
            + PILLAR_WEIGHTS["perception"] * pillar_perception
        )

        grade = self._grade(confidence, answer_ok)
        passed = grade in ("A", "B") and answer_ok

        return EvaluationResult(
            elimination_correct=elimination_correct,
            direct_answer_correct=direct_answer_correct,
            false_narrative_rejected=false_narrative_rejected,
            concept_results=concept_results,
            concept_score=concept_score,
            contradiction_results=contradiction_results,
            contradiction_score=contradiction_score,
            insight_usage_results=insight_results,
            insight_usage_score=insight_score,
            forbidden_results=forbidden_results,
            forbidden_penalty=forbidden_penalty,
            observation_purity_penalty=purity_penalty,
            reasoning_graph_results=reasoning_graph_results or [],
            reasoning_graph_score=reasoning_graph_score,
            llm_verdict=llm_verdict,
            llm_reasoning=llm_reasoning,
            llm_score=llm_score,
            llm_role_results=llm_role_results or [],
            llm_critique=llm_critique,
            llm_counterarguments=llm_counterarguments or [],
            bayesian_score=bayesian_score,
            bayesian_max=bayesian_max,
            bayesian_results=bayesian_results or [],
            bayesian_traps=bayesian_traps or [],
            causality_score=causality_score,
            causality_max=causality_max,
            causality_results=causality_results or [],
            causality_phantom=causality_phantom or [],
            causality_leaps=causality_leaps or [],
            causality_temporal=causality_temporal or [],
            causality_chain_trace=causality_chain_trace or [],
            causality_trap_penalties=causality_trap_penalties,
            causality_traps_triggered=causality_traps_triggered or [],
            earned=earned,
            max_possible=max_possible,
            confidence_score=confidence,
            grade=grade,
            passed=passed,
            pillar_content=pillar_content,
            pillar_structure=pillar_structure,
            pillar_integrity=pillar_integrity,
            pillar_perception=pillar_perception,
            weighted_score=weighted_score,
        )

    # ------------------------------------------------------------------
    # inference-leak detection (called externally by the runner)
    # ------------------------------------------------------------------

    @staticmethod
    def flag_inference_leaks(observations: list[str]) -> list[str]:
        return [o for o in observations if contains_any(o, INFERENCE_MARKERS)]

    # ------------------------------------------------------------------
    # internal scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _build_corpus(record: AnalysisRecord) -> str:
        parts = (
            record.anomalies
            + record.reasons
            + record.contradiction_notes
            + [record.false_narrative_rejection]
            + [record.elimination_target]
            + [exp for exps in record.hypotheses.values() for exp in exps]
        )
        return normalize_key(" ".join(parts))

    @staticmethod
    def _score_concept_rules(case: CaseDefinition, corpus: str):
        results: list[dict] = []
        score = 0
        for rule in case.solution.required_concept_rules:
            matched = rule.evaluate(corpus)
            if matched:
                score += rule.points
            results.append({
                "rule": rule.name, "matched": matched,
                "description": rule.description, "points": rule.points,
            })
        return results, score

    @staticmethod
    def _score_contradictions(case: CaseDefinition, record: AnalysisRecord):
        corpus = normalize_key(" ".join(
            record.contradiction_notes + record.anomalies + record.reasons
        ))
        results: list[dict] = []
        score = 0
        for c in case.contradictions:
            detected = c.detected_in(corpus)
            if detected:
                score += c.points
            results.append({
                "description": c.description, "detected": detected, "points": c.points,
            })
        return results, score

    @staticmethod
    def _score_insight_usage(case: CaseDefinition, corpus: str):
        results: list[dict] = []
        score = 0
        for rule in case.solution.insight_usage_rules:
            verified = rule.verified(corpus)
            if verified:
                score += rule.points
            results.append({
                "insight": rule.insight_key, "description": rule.description,
                "verified": verified, "points": rule.points,
            })
        return results, score

    @staticmethod
    def _check_false_narrative(case: CaseDefinition, record: AnalysisRecord) -> bool:
        if not case.solution.must_reject_false_narrative or not case.false_narrative:
            return False
        fn = normalize_key(record.false_narrative_rejection)
        narrative_words = [w for w in case.false_narrative.lower().split() if len(w) > 3]
        rejection_words = [
            "wrong", "false", "incorrect", "fail", "reject", "unlikely",
            "staged", "planted", "fabricat", "doesn't explain",
            "not true", "deceptive", "mislead",
        ]
        return (
            any(w in fn for w in narrative_words)
            and any(w in fn for w in rejection_words)
        )

    @staticmethod
    def _score_forbidden(case: CaseDefinition, corpus: str):
        results: list[dict] = []
        penalty = 0
        for fp in case.solution.forbidden_patterns:
            triggered = fp.triggered(corpus)
            if triggered:
                penalty += fp.penalty
            results.append({
                "description": fp.description, "triggered": triggered, "penalty": fp.penalty,
            })
        return results, penalty

    @staticmethod
    def _purity_penalty(record: AnalysisRecord) -> int:
        return sum(
            1 for o in record.observations if contains_any(o, INFERENCE_MARKERS)
        )

    @staticmethod
    def _grade(confidence: float, answer_ok: bool) -> str:
        if confidence >= 0.85 and answer_ok:
            return "A"
        if confidence >= 0.65 and answer_ok:
            return "B"
        if confidence >= 0.45:
            return "C"
        if confidence >= 0.25:
            return "D"
        return "F"
