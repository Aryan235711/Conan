"""Reward interface — structured scoring for LLM training pipelines.

Thin wrapper over the full validator stack.  Takes a case ID + reasoning
dict, returns pillar scores + granular feedback suitable for use as an
RL reward signal or evaluation metric.

Two entry points:

    score(case_id, analysis_dict)
        For pre-structured input (already parsed or from phase generator).

    score_raw(case_id, raw_text)
        For free-form LLM text.  Runs the 3-layer parser (structure →
        validity → semantic grounding) and rejects garbage before scoring.

Usage:

    from detective_engine.engine.reward_interface import RewardScorer

    scorer = RewardScorer()                 # loads all cases once

    # Structured input
    result = scorer.score("C001", {
        "observations": ["overturned chair", "open window", ...],
        "anomalies": ["room looks staged", ...],
        "hypotheses": {"overturned chair": ["struggle", "staged"]},
        "elimination_target": "the scene was calm — hiding inside staged",
        "reasons": ["chair contradicts calm", ...],
    })

    # Free-form LLM text
    result = scorer.score_raw("C001", llm_output_text)
    if result is None:
        # unparseable or semantically empty — skip this sample
        ...

    print(result["weighted"])       # 0.0–1.0
    print(result["grade"])          # A/B/C/D/F
    print(result["feedback"])       # what was missed
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .case_loader import CaseLoader
from .reasoning_parser import parse_reasoning_output, ParserConfig, ParseResult
from .models import (
    AnalysisRecord,
    EvidenceMeta,
    ObservationEvent,
    PerceptionConfig,
    PerceptionTrace,
    normalize_key,
)
from .validator import Validator, PILLAR_WEIGHTS
from .reasoning_graph import ReasoningGraphValidator
from .causality_validator import CausalityValidator
from .bayesian_validator import BayesianValidator
from .perception_integrity import (
    PerceptionIntegrityValidator,
    match_evidence_ids,
    perception_adjustment,
)


class RewardScorer:
    """Stateless reward function over the detective engine validator stack.

    Instantiate once (loads all cases), then call ``score()`` per sample.
    Thread-safe — all validators are pure functions with no shared state.
    """

    def __init__(self, cases_dir: str | None = None):
        loader = CaseLoader(cases_dir)
        all_cases, _ = loader.load_all()
        self._cases = {c.id: c for c in all_cases}
        self._validator = Validator()
        self._reasoning = ReasoningGraphValidator()
        self._causality = CausalityValidator()
        self._bayesian = BayesianValidator()
        self._perc_config = PerceptionConfig()
        self._perception = PerceptionIntegrityValidator(self._perc_config)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def case_ids(self) -> list[str]:
        """All loaded case IDs."""
        return sorted(self._cases.keys())

    def case_evidence(self, case_id: str) -> list[str]:
        """Return the evidence list for a case (the LLM's input)."""
        return list(self._cases[case_id].evidence)

    def case_prompt(self, case_id: str) -> dict[str, Any]:
        """Return everything an LLM needs to attempt a case.

        Includes evidence, detective questions, scenarios, and protocol —
        but NOT the solution, hidden truth, or scoring rules.
        """
        case = self._cases[case_id]
        prompt: dict[str, Any] = {
            "id": case.id,
            "title": case.title,
            "category": case.category,
            "summary": case.summary,
            "evidence": case.evidence,
            "detective_questions": case.detective_questions,
            "analysis_protocol": case.analysis_protocol,
            "scenarios": case.scenarios,
            "has_contradictions": len(case.contradictions) > 0,
            "must_reject_false_narrative": case.solution.must_reject_false_narrative,
            "is_bayesian": case.solution.bayesian is not None,
        }
        if case.solution.bayesian:
            prompt["hypothesis_count"] = len(case.solution.bayesian.hypotheses)
        return prompt

    def score(
        self,
        case_id: str,
        analysis: dict[str, Any],
        *,
        perception_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a reasoning attempt and return structured reward signals.

        Parameters
        ----------
        case_id : str
            Case to evaluate against (e.g. "C001").
        analysis : dict
            LLM's reasoning output.  Required keys:
                observations, anomalies, hypotheses,
                elimination_target, reasons
            Optional keys:
                false_narrative_rejection, contradiction_notes,
                probability_ranking, evidence_weight_notes, prior_reasoning
        perception_trace : dict, optional
            If provided, enables perception pillar scoring.  Keys:
                initial: list[dict]   — micro-capture observations
                later: list[dict]     — later observations

        Returns
        -------
        dict with:
            pillars     — per-pillar 0.0–1.0 scores
            weighted    — weighted combination (the reward signal)
            grade       — A/B/C/D/F
            passed      — bool
            confidence  — raw confidence score
            earned/max  — raw point totals
            penalties   — list of triggered penalties
            feedback    — list of what was missed (for curriculum)
            details     — full breakdown for debugging
        """
        if case_id not in self._cases:
            raise KeyError(f"Unknown case: {case_id}. Available: {self.case_ids}")

        case = self._cases[case_id]
        evidence_map: dict[str, EvidenceMeta] = getattr(case, "evidence_map", {})

        # Build AnalysisRecord from dict
        record = self._build_record(analysis)

        # Build perception trace if provided
        trace = None
        used_reasoning_ids: set[str] = set()
        if perception_trace:
            trace = self._build_trace(perception_trace, evidence_map)
            record.perception_trace = trace
            used_reasoning_ids = self._derive_used_ids(record, evidence_map)

        # --- Run validator stack ---
        rg = self._reasoning.validate(case, record)
        caus = self._causality.validate(case, record)
        bay = self._bayesian.validate(case, record)
        perc = self._perception.validate(
            trace=trace,
            evidence_map=evidence_map,
            used_in_reasoning_ids=used_reasoning_ids,
        )

        ev = self._validator.evaluate(
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
                {"name": d.name, "score": d.score, "max": d.max_score,
                 "details": d.details}
                for d in bay.dimensions
            ],
            bayesian_traps=bay.cognitive_traps_triggered,
        )

        # Set perception pillar
        if perc is not None and perc.evaluation_confidence >= 0.60:
            adj = perception_adjustment(perc, self._perc_config)
            ev.pillar_perception = max(0.0, min(1.0, 0.5 + adj / 4.0))
        ev.weighted_score = (
            PILLAR_WEIGHTS["content"] * ev.pillar_content
            + PILLAR_WEIGHTS["structure"] * ev.pillar_structure
            + PILLAR_WEIGHTS["integrity"] * ev.pillar_integrity
            + PILLAR_WEIGHTS["perception"] * ev.pillar_perception
        )

        # --- Build response ---
        return self._format_result(ev, rg, caus, bay, perc, case)

    def score_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score multiple samples. Each dict must have 'case_id' and 'analysis' keys.

        Optional key: 'perception_trace'.
        """
        return [
            self.score(
                s["case_id"],
                s["analysis"],
                perception_trace=s.get("perception_trace"),
            )
            for s in samples
        ]

    def score_raw(
        self,
        case_id: str,
        raw_text: str,
        *,
        perception_trace: dict[str, Any] | None = None,
        parser_config: ParserConfig | None = None,
    ) -> dict[str, Any] | None:
        """Score free-form LLM text after parsing and validation.

        Runs the 3-layer parser (structure → validity → semantic grounding).
        Returns ``None`` if the text is unparseable or fails validation —
        the caller should skip this sample, NOT score it as 0.

        Parameters
        ----------
        case_id : str
            Case to evaluate against.
        raw_text : str
            Free-form LLM output text.
        perception_trace : dict, optional
            System-generated trace (from ``PhasePromptGenerator.build_trace()``).
            MUST have ``_system_generated: True`` to be trusted.
        parser_config : ParserConfig, optional
            Override parser thresholds.

        Returns
        -------
        dict with reward signals (same shape as ``score()``), plus:
            parse_warnings — soft issues from the parser
            raw_text       — original text preserved for analysis
        Or None if parsing/validation failed.
        """
        parsed = parse_reasoning_output(raw_text, config=parser_config)
        if parsed is None:
            return None

        # Validate perception trace provenance
        safe_trace = None
        if perception_trace is not None:
            if perception_trace.get("_system_generated"):
                safe_trace = perception_trace
            # else: silently drop untrusted trace — don't penalize, just ignore

        result = self.score(
            case_id,
            parsed.analysis,
            perception_trace=safe_trace,
        )

        # Attach parser metadata
        result["parse_warnings"] = parsed.warnings
        result["raw_text"] = raw_text
        result["sections_found"] = parsed.sections_found

        return result

    def score_raw_batch(
        self,
        samples: list[dict[str, Any]],
        *,
        parser_config: ParserConfig | None = None,
    ) -> list[dict[str, Any] | None]:
        """Score multiple free-form text samples.

        Each dict must have 'case_id' and 'raw_text' keys.
        Optional: 'perception_trace' (system-generated).
        Returns None for unparseable samples.
        """
        return [
            self.score_raw(
                s["case_id"],
                s["raw_text"],
                perception_trace=s.get("perception_trace"),
                parser_config=parser_config,
            )
            for s in samples
        ]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_record(analysis: dict[str, Any]) -> AnalysisRecord:
        return AnalysisRecord(
            observations=analysis.get("observations", []),
            anomalies=analysis.get("anomalies", []),
            hypotheses=analysis.get("hypotheses", {}),
            elimination_target=analysis.get("elimination_target", ""),
            reasons=analysis.get("reasons", []),
            false_narrative_rejection=analysis.get("false_narrative_rejection", ""),
            contradiction_notes=analysis.get("contradiction_notes", []),
            probability_ranking=analysis.get("probability_ranking", []),
            evidence_weight_notes=analysis.get("evidence_weight_notes", []),
            prior_reasoning=analysis.get("prior_reasoning", []),
        )

    def _build_trace(
        self,
        trace_dict: dict[str, Any],
        evidence_map: dict[str, EvidenceMeta],
    ) -> PerceptionTrace:
        cfg = self._perc_config

        def _make_events(
            items: list[dict], phase: str,
        ) -> list[ObservationEvent]:
            events = []
            for idx, item in enumerate(items):
                text = item if isinstance(item, str) else item.get("text", "")
                events.append(ObservationEvent(
                    text=text,
                    timestamp=1_000_000.0 + idx * 0.1,
                    phase=phase,
                    order_index=idx,
                    evidence_ids=match_evidence_ids(text, evidence_map, cfg),
                ))
            return events

        initial = _make_events(trace_dict.get("initial", []), "initial")
        later = _make_events(trace_dict.get("later", []), "later")

        return PerceptionTrace(
            initial=initial,
            later=later,
            first_hypothesis_timestamp=1_000_000.0 + 1.0,
            first_elimination_timestamp=1_000_000.0 + 2.0,
        )

    @staticmethod
    def _derive_used_ids(
        record: AnalysisRecord,
        evidence_map: dict[str, EvidenceMeta],
    ) -> set[str]:
        if not record.perception_trace or not evidence_map:
            return set()

        reasoning_text = normalize_key(
            " ".join(record.reasons)
            + " " + record.elimination_target
            + " " + " ".join(record.anomalies)
        )
        used = set()
        for ev_event in (
            record.perception_trace.initial + record.perception_trace.later
        ):
            for eid in ev_event.evidence_ids:
                meta = evidence_map.get(eid)
                if meta:
                    ev_tokens = {
                        w for w in normalize_key(meta.text).split() if len(w) > 3
                    }
                    if ev_tokens & set(reasoning_text.split()):
                        used.add(eid)
        return used

    @staticmethod
    def _format_result(ev, rg, caus, bay, perc, case) -> dict[str, Any]:
        # Penalties
        penalties = []
        for f in ev.forbidden_results:
            if f["triggered"]:
                penalties.append({
                    "type": "forbidden_reasoning",
                    "description": f["description"],
                    "points": -f["penalty"],
                })
        if ev.observation_purity_penalty > 0:
            penalties.append({
                "type": "inference_leak",
                "description": "Observations contained interpretive language",
                "points": -ev.observation_purity_penalty,
            })
        if caus.inference_traps_triggered:
            for trap in caus.inference_traps_triggered:
                penalties.append({
                    "type": "inference_trap",
                    "description": trap,
                    "points": -1,
                })

        # Feedback — what was missed
        feedback = []
        for r in ev.concept_results:
            if not r["matched"]:
                feedback.append(f"Missed concept: {r['rule']} — {r['description']}")
        for c in ev.contradiction_results:
            if not c["detected"]:
                feedback.append(f"Missed contradiction: {c['description']}")
        if case.solution.must_reject_false_narrative and not ev.false_narrative_rejected:
            feedback.append(f"Failed to reject false narrative: \"{case.false_narrative}\"")
        if not ev.elimination_correct and not ev.direct_answer_correct:
            feedback.append("Wrong elimination target / answer")
        if caus.phantom_concepts:
            feedback.append(f"Phantom derivations: {', '.join(caus.phantom_concepts[:3])}")
        if caus.inference_leaps:
            feedback.append(f"Inference leaps: {', '.join(caus.inference_leaps[:3])}")

        # Detailed sub-scores for debugging / analysis
        details = {
            "concept_score": ev.concept_score,
            "contradiction_score": ev.contradiction_score,
            "insight_usage_score": ev.insight_usage_score,
            "reasoning_graph_score": rg.total_score,
            "reasoning_graph_max": rg.max_score,
            "causality_score": caus.total_score,
            "causality_max": caus.max_score,
            "bayesian_score": bay.total_score,
            "bayesian_max": bay.max_score,
            "false_narrative_rejected": ev.false_narrative_rejected,
            "elimination_correct": ev.elimination_correct,
            "direct_answer_correct": ev.direct_answer_correct,
        }
        if perc is not None:
            details["perception"] = {
                "coverage": perc.coverage,
                "retention": perc.retention_score,
                "late_injection": perc.late_injection_score,
                "causal_uptake": perc.causal_uptake_score,
                "salience_distortion": perc.salience_distortion,
                "narrative_lock_in": perc.narrative_lock_in,
                "evaluation_confidence": perc.evaluation_confidence,
                "id_match_rate": perc.id_match_rate,
                "adversarial_flags": perc.adversarial_flags,
            }

        return {
            "pillars": {
                "content": round(ev.pillar_content, 4),
                "structure": round(ev.pillar_structure, 4),
                "integrity": round(ev.pillar_integrity, 4),
                "perception": round(ev.pillar_perception, 4),
            },
            "weighted": round(ev.weighted_score, 4),
            "grade": ev.grade,
            "passed": ev.passed,
            "confidence": round(ev.confidence_score, 4),
            "earned": ev.earned,
            "max": ev.max_possible,
            "penalties": penalties,
            "feedback": feedback,
            "details": details,
        }
