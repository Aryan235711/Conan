"""Interactive case runner — the 6-phase protocol + evaluation reporting.

This module owns ALL user interaction (input/output).  The Validator,
LLMJudge, ReasoningGraphValidator, and UserProfile are injected and have
no I/O side effects.
"""

from __future__ import annotations

from datetime import datetime
import json
import textwrap
import time
from dataclasses import asdict

from .models import (
    AnalysisRecord,
    CaseDefinition,
    EvaluationResult,
    EvidenceMeta,
    Insight,
    ObservationEvent,
    PerceptionConfig,
    PerceptionTrace,
    normalize_key,
    sanitize_text,
    split_csv,
)
from .insight_graph import InsightGraph
from .llm_judge import LLMJudge
from .reasoning_graph import ReasoningGraphValidator
from .bayesian_validator import BayesianValidator
from .causality_validator import CausalityValidator
from .perception_integrity import (
    PerceptionIntegrityValidator,
    match_evidence_ids,
    should_probe,
    should_probe_stage1,
    select_probes,
    coverage_score,
)
from .user_profile import UserProfile
from .validator import Validator


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _prompt_non_empty(prompt: str) -> str:
    while True:
        value = sanitize_text(input(prompt))
        if value:
            return value
        print("⚠  Please enter a non-empty response.")


def _prompt_csv(prompt: str, minimum: int = 1) -> list[str]:
    while True:
        values = split_csv(input(prompt))
        if len(values) >= minimum:
            return values
        print(f"⚠  Please provide at least {minimum} item(s).")


def _section(title: str) -> None:
    print(f"\n{'═' * 14} {title} {'═' * 14}")


def _wrap(text: str) -> str:
    return textwrap.fill(text, width=88)


# ---------------------------------------------------------------------------
# CaseRunner — orchestrates a single session
# ---------------------------------------------------------------------------

class CaseRunner:
    """Orchestrates the interactive detective training loop.

    Combines:
        InsightGraph             — dependency / unlock tracking (DAG)
        Validator                — deterministic scoring (source of truth)
        LLMJudge                 — multi-role reasoning quality supplement
        ReasoningGraphValidator  — observation→hypothesis→elimination chain checker
        UserProfile              — skill progression tracking across sessions
    """

    def __init__(
        self,
        graph: InsightGraph,
        validator: Validator | None = None,
        llm_judge: LLMJudge | None = None,
        reasoning_validator: ReasoningGraphValidator | None = None,
        bayesian_validator: BayesianValidator | None = None,
        causality_validator: CausalityValidator | None = None,
        perception_validator: PerceptionIntegrityValidator | None = None,
        perception_config: PerceptionConfig | None = None,
        user_profile: UserProfile | None = None,
    ):
        self.graph = graph
        self.validator = validator or Validator()
        self.llm = llm_judge or LLMJudge(enabled=False)
        self.reasoning = reasoning_validator or ReasoningGraphValidator()
        self.bayesian = bayesian_validator or BayesianValidator()
        self.causality = causality_validator or CausalityValidator()
        self.perception_config = perception_config or PerceptionConfig()
        self.perception = perception_validator or PerceptionIntegrityValidator(self.perception_config)
        self.profile = user_profile or UserProfile()
        self.history: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # case description
    # ------------------------------------------------------------------

    def describe_case(self, cid: str) -> str:
        c = self.graph.cases[cid]
        lines = [
            f"{c.id} — {c.title}",
            _wrap(c.summary), "",
            "Evidence:",
            *[f"  • {e}" for e in c.evidence], "",
            "Detective questions:",
            *[f"  • {q}" for q in c.detective_questions], "",
            "Three-layer protocol:",
            *[f"  {i}. {s}" for i, s in enumerate(c.analysis_protocol, 1)],
        ]
        if c.scenarios:
            lines += ["", "Candidate scenarios:", *[f"  • {s}" for s in c.scenarios]]
        if c.contradictions:
            lines += ["", f"⚠️  This scene contains {len(c.contradictions)} hidden contradiction(s) to detect."]
        if c.solution.must_reject_false_narrative:
            lines += ["⚠️  You must identify and reject the false narrative."]
        if c.solution.bayesian:
            lines += [
                "",
                "\ud83c\udfb2  BAYESIAN CASE \u2014 Multiple conclusions are logically valid.",
                "   You must RANK hypotheses by probability, not just pick one.",
                f"   {len(c.solution.bayesian.hypotheses)} competing hypotheses to evaluate.",
            ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 6-phase data collection
    # ------------------------------------------------------------------

    def collect_analysis(
        self,
        case: CaseDefinition,
        evidence_map: dict[str, EvidenceMeta] | None = None,
    ) -> AnalysisRecord:
        phase_timestamps: dict[str, str] = {}
        evidence_map = evidence_map or {}
        cfg = self.perception_config

        _section(f"CASE {case.id} — {case.title}")
        print(_wrap(case.summary))
        print("\nEvidence:")
        for e in case.evidence:
            print(f"  • {e}")
        print("\nProtocol:")
        for i, s in enumerate(case.analysis_protocol, 1):
            print(f"  {i}. {s}")

        # ── Phase 0: MICRO-CAPTURE (V11) ──────────────────────────────
        _section("PHASE 0: MICRO-CAPTURE")
        print("List the first 3 raw details you notice. Be brief — no explanations.")
        micro_items = _prompt_csv("> ", minimum=1)
        micro_items = micro_items[:3]  # cap at 3
        micro_events: list[ObservationEvent] = []
        for idx, item in enumerate(micro_items):
            micro_events.append(ObservationEvent(
                text=item,
                timestamp=time.time(),
                phase="initial",
                order_index=idx,
                evidence_ids=match_evidence_ids(item, evidence_map, cfg),
            ))

        # Build initial perception trace with micro-capture data
        trace = PerceptionTrace(initial=micro_events)

        # Phase 1
        _section("PHASE 1: RAW OBSERVATIONS")
        print("List ONLY what you see. No interpretations. No assumptions.")
        observations = _prompt_csv("> ", minimum=5)
        phase_timestamps["observations"] = datetime.now().isoformat()

        # Record Phase 1 observations as 'later' events in the trace
        for idx, obs in enumerate(observations):
            trace.later.append(ObservationEvent(
                text=obs,
                timestamp=time.time(),
                phase="later",
                order_index=idx,
                evidence_ids=match_evidence_ids(obs, evidence_map, cfg),
            ))

        flagged = self.validator.flag_inference_leaks(observations)
        if flagged:
            print("\n⚠️  INFERENCE LEAK DETECTED")
            for item in flagged:
                print(f"  ⚠ {item}")
            print("Penalty will apply.\n")

        # Phase 2
        _section("PHASE 2: ANOMALIES & TENSIONS")
        print("What tensions or oddities exist between the facts?")
        anomalies = _prompt_csv("> ", minimum=2)

        # Phase 3 — set hypothesis timestamp BEFORE prompts (earliest possible point)
        _section("PHASE 3: MULTIPLE EXPLANATIONS")
        trace.first_hypothesis_timestamp = time.time()
        print("For each of the first 3 observations, list 2–3 explanations.")
        hypotheses: dict[str, list[str]] = {}
        for obs in observations[:3]:
            exps = _prompt_csv(f"  Explanations for '{obs}' (min 2):\n  > ", minimum=2)
            hypotheses[obs] = exps
        phase_timestamps["hypotheses"] = datetime.now().isoformat()

        # ── Stage 1 light probe (after Phase 3, before Phase 4) ──────
        if evidence_map:
            critical_ids = {eid for eid, m in evidence_map.items() if m.critical}
            init_ids = {eid for ev in trace.initial for eid in ev.evidence_ids}
            init_crit_cov = coverage_score(init_ids, critical_ids)
            if should_probe_stage1(init_crit_cov, trace, cfg):
                trace.probe_stage_1_shown = True
                print("\n💡 PERCEPTION CHECK (optional — press Enter to skip)")
                print("Before you commit: list 1 detail you almost ignored.")
                raw = sanitize_text(input("> "))
                if raw:
                    trace.revisions.append(ObservationEvent(
                        text=raw,
                        timestamp=time.time(),
                        phase="revision",
                        order_index=0,
                        evidence_ids=match_evidence_ids(raw, evidence_map, cfg),
                        source="probe",
                    ))
                    print("Noted.")
        else:
            init_crit_cov = 0.0

        # Phase 4
        contradiction_notes: list[str] = []
        if case.contradictions:
            _section("PHASE 4: CONTRADICTION DETECTION")
            print(f"{len(case.contradictions)} hidden contradiction(s) to detect.")
            print("Identify pairs of facts that cannot coexist under the surface story.")
            contradiction_notes = _prompt_csv("> ", minimum=1)

        # Phase 5
        fn_rejection = ""
        if case.solution.must_reject_false_narrative and case.false_narrative:
            _section("PHASE 5: FALSE NARRATIVE REJECTION")
            print("What is the most obvious but WRONG explanation this scene sells?")
            print("State it and explain why it fails.")
            fn_rejection = _prompt_non_empty("> ")

        # Phase 6 — set elimination timestamp BEFORE prompts
        _section("PHASE 6: ELIMINATION & REASONING CHAIN")
        trace.first_elimination_timestamp = time.time()
        if case.scenarios:
            print("Scenarios:")
            for s in case.scenarios:
                print(f"  • {s}")
        elim = _prompt_non_empty("Which scenario do you ELIMINATE?\n> ")
        reasons = _prompt_csv("Reasoning chain (comma separated, min 3):\n> ", minimum=3)
        phase_timestamps["elimination"] = datetime.now().isoformat()

        # ── Bayesian phases (only for multi-hypothesis cases) ─────────
        probability_ranking: list[dict] = []
        evidence_weight_notes: list[str] = []
        prior_reasoning: list[str] = []

        if case.solution.bayesian:
            _section("PHASE 7: PROBABILITY RANKING")
            print("Multiple hypotheses are logically valid in this case.")
            print("Rank ALL valid suspects/conclusions from MOST to LEAST probable.")
            print("Format: suspect name (one per line, most probable first)")
            for idx in range(len(case.solution.bayesian.hypotheses)):
                label = f"  #{idx+1} (most probable)" if idx == 0 else f"  #{idx+1}"
                hyp = _prompt_non_empty(f"{label}: ")
                level = "high" if idx == 0 else ("medium" if idx == 1 else "low")
                probability_ranking.append({"hypothesis": hyp, "rank": idx + 1, "level": level})

            _section("PHASE 8: EVIDENCE WEIGHTING")
            print("Which evidence is MOST diagnostic (strongly shifts probability)?")
            print("Which evidence is misleading or neutral? Explain.")
            evidence_weight_notes = _prompt_csv(
                "Evidence weight notes (comma separated, min 2):\n> ", minimum=2,
            )

            _section("PHASE 9: PRIOR REASONING")
            print("What base rates, statistical patterns, behavioral signals,")
            print("or past-activity indicators influenced your probability ranking?")
            prior_reasoning = _prompt_csv(
                "Prior reasoning notes (comma separated, min 1):\n> ", minimum=1,
            )

        return AnalysisRecord(
            observations=observations,
            anomalies=anomalies,
            hypotheses=hypotheses,
            elimination_target=elim,
            reasons=reasons,
            false_narrative_rejection=fn_rejection,
            contradiction_notes=contradiction_notes,
            probability_ranking=probability_ranking,
            evidence_weight_notes=evidence_weight_notes,
            prior_reasoning=prior_reasoning,
            phase_timestamps=phase_timestamps,
            perception_trace=trace,
        )

    # ------------------------------------------------------------------
    # evaluation + report
    # ------------------------------------------------------------------

    def run_case(self, cid: str) -> None:
        if cid not in self.graph.cases:
            print("Invalid case ID.")
            return

        case = self.graph.cases[cid]

        if not self.graph.is_unlocked(cid):
            missing = self.graph.missing_for(cid)
            print(f"Case {cid} is locked. Missing insights: {', '.join(missing)}")
            return

        # Retrieve evidence_map if the case was loaded with perception metadata
        evidence_map: dict[str, EvidenceMeta] = getattr(case, "evidence_map", {})

        print(self.describe_case(cid))
        record = self.collect_analysis(case, evidence_map=evidence_map)

        # ── Reasoning graph analysis ──
        rg_result = self.reasoning.validate(case, record)
        rg_results_list = [
            {"name": l.name, "score": l.score, "details": l.details}
            for l in rg_result.links
        ]

        # ── Bayesian reasoning analysis (V8) ──
        bay_result = self.bayesian.validate(case, record)
        bay_results_list = [
            {"name": d.name, "score": d.score, "max": d.max_score, "details": d.details}
            for d in bay_result.dimensions
        ]

        # ── Causality validation (derivation integrity) ──
        caus_result = self.causality.validate(case, record)
        caus_results_list = [
            {"name": d.name, "score": d.score, "max": d.max_score,
             "details": d.details, "severity": d.severity}
            for d in caus_result.dimensions
        ]

        # ── Perception integrity validation (V11) ──
        # Derive used evidence IDs from the perception trace's later observations
        # that also appear in the reasons/elimination text (causal uptake proxy).
        used_reasoning_ids: set[str] = set()
        if record.perception_trace and evidence_map:
            reasoning_text = normalize_key(
                " ".join(record.reasons)
                + " " + record.elimination_target
                + " " + " ".join(record.anomalies)
            )
            for ev_event in record.perception_trace.initial + record.perception_trace.later:
                for eid in ev_event.evidence_ids:
                    meta = evidence_map.get(eid)
                    if meta:
                        # Check if any token from this evidence text appears in reasoning
                        ev_tokens = {
                            w for w in normalize_key(meta.text).split() if len(w) > 3
                        }
                        if ev_tokens & set(reasoning_text.split()):
                            used_reasoning_ids.add(eid)
        perc_result = self.perception.validate(
            trace=record.perception_trace,
            evidence_map=evidence_map,
            used_in_reasoning_ids=used_reasoning_ids,
        )

        # Stage 2 probes — after Phase 6, before final report
        if should_probe(perc_result, self.perception_config):
            probes_seen = self.profile.get_probes_seen(cid)
            probes = select_probes(perc_result, self.perception_config, probes_seen)
            if probes:
                _section("PERCEPTION PROBES")
                print("Your observation pattern triggered a few targeted questions.")
                print("Answer briefly (optional — press Enter to skip each).\n")
                for probe in probes:
                    print(f"  • {probe}")
                    raw = sanitize_text(input("    > "))
                    if raw and record.perception_trace:
                        record.perception_trace.revisions.append(ObservationEvent(
                            text=raw,
                            timestamp=time.time(),
                            phase="revision",
                            order_index=len(record.perception_trace.revisions),
                            evidence_ids=match_evidence_ids(raw, evidence_map, self.perception_config),
                            source="probe",
                        ))
                if record.perception_trace:
                    record.perception_trace.probe_stage_2_shown = True
                    record.perception_trace.probes_answered = probes
                self.profile.record_probes_seen(cid, probes)

        # ── LLM multi-role evaluation ──
        llm_result = self.llm.evaluate_full(case, record)

        # ── Deterministic evaluation (source of truth) ──
        ev = self.validator.evaluate(
            case, record,
            llm_verdict=llm_result.verdict,
            llm_reasoning=llm_result.reasoning,
            llm_score=llm_result.total_score,
            llm_role_results=[
                {"role": r.role, "score": r.score, "points": r.points}
                for r in llm_result.roles
            ],
            llm_critique=llm_result.critique,
            llm_counterarguments=llm_result.counterarguments,
            reasoning_graph_results=rg_results_list,
            reasoning_graph_score=rg_result.total_score,
            bayesian_score=bay_result.total_score,
            bayesian_max=bay_result.max_score,
            bayesian_results=bay_results_list,
            bayesian_traps=bay_result.cognitive_traps_triggered,
            causality_score=caus_result.total_score,
            causality_max=caus_result.max_score,
            causality_results=caus_results_list,
            causality_phantom=caus_result.phantom_concepts,
            causality_leaps=caus_result.inference_leaps,
            causality_temporal=caus_result.temporal_violations,
            causality_chain_trace=caus_result.chain_trace_lines,
            causality_trap_penalties=caus_result.inference_trap_penalties,
            causality_traps_triggered=caus_result.inference_traps_triggered,
        )

        # ── Set perception pillar + recompute weighted score ──
        if perc_result is not None and perc_result.evaluation_confidence >= 0.60:
            from detective_engine.engine.perception_integrity import perception_adjustment
            adj = perception_adjustment(perc_result, self.perception_config)
            # Normalize adjustment (-2..+2) into 0..1 pillar score
            ev.pillar_perception = max(0.0, min(1.0, 0.5 + adj / 4.0))
        from detective_engine.engine.validator import PILLAR_WEIGHTS
        ev.weighted_score = (
            PILLAR_WEIGHTS["content"] * ev.pillar_content
            + PILLAR_WEIGHTS["structure"] * ev.pillar_structure
            + PILLAR_WEIGHTS["integrity"] * ev.pillar_integrity
            + PILLAR_WEIGHTS["perception"] * ev.pillar_perception
        )

        self._print_report(case, ev, rg_result, bay_result, caus_result, perc_result)

        # ── LLM detailed feedback ──
        if llm_result.enabled:
            _section("LLM FEEDBACK")
            if llm_result.counterarguments:
                print("  Devil's advocate objections:")
                for i, ca in enumerate(llm_result.counterarguments, 1):
                    print(f"    {i}. {ca}")
            if llm_result.critique:
                print(f"\n  Trainer critique:\n{textwrap.indent(llm_result.critique, '    ')}")

        self.history[case.id] = {"record": asdict(record), "evaluation": asdict(ev)}

        # ── User profile tracking ──
        extra_errors = []
        if rg_result.ungrounded_hypotheses:
            from .user_profile import ERROR_UNGROUNDED_HYPOTHESIS
            extra_errors.append(ERROR_UNGROUNDED_HYPOTHESIS)
        if rg_result.missed_evidence:
            from .user_profile import ERROR_MISSED_EVIDENCE
            extra_errors.append(ERROR_MISSED_EVIDENCE)

        attempt = self.profile.build_attempt(
            case_id=case.id,
            evaluation=ev,
            reasoning_score=rg_result.total_score,
            llm_score=llm_result.total_score,
            extra_errors=extra_errors,
            perception=perc_result,
        )
        self.profile.record_attempt(attempt)

        # ── Unlock ──
        if ev.passed:
            key = case.solution.unlock_key
            if key and key in self.graph.insights:
                newly = self.graph.unlock(key)
                ins = self.graph.insights[key]
                print(f"\n🔓 Insight unlocked: {ins.key} — {ins.title}")
                print(f"   Transfer rule: {ins.transfer_rule}")
                for ncid in newly:
                    nc = self.graph.cases[ncid]
                    print(f"   🔓 Case {ncid}: {nc.title} is now UNLOCKED")
            else:
                print("\n✅ Case solved.")
        elif ev.grade == "C":
            print("\n⚠️  Partial understanding. Strengthen contradiction detection and insight application.")
        elif ev.grade == "D":
            print("\n❌ Weak reasoning. Return to Phase 1 and rebuild from raw observations.")
        else:
            print("\n❌ Case failed. observe → explain → contradict → reject → eliminate.")

    # ------------------------------------------------------------------
    # report printer
    # ------------------------------------------------------------------

    def _print_report(self, case: CaseDefinition, ev: EvaluationResult, rg_result=None, bay_result=None, caus_result=None, perc_result=None) -> None:
        _section("EVALUATION REPORT")

        icon = "✅" if ev.elimination_correct else "❌"
        print(f"{icon} Elimination target correct: {ev.elimination_correct}")
        icon = "✅" if ev.direct_answer_correct else "❌"
        print(f"{icon} Direct answer detected: {ev.direct_answer_correct}")

        # ── Pillar Summary ──
        def _bar(score: float, width: int = 20) -> str:
            filled = round(score * width)
            return f"[{'#' * filled}{'.' * (width - filled)}] {score:.0%}"

        print(f"\n{'━' * 44}")
        print(f"  CONTENT     (30%)  {_bar(ev.pillar_content)}")
        print(f"  STRUCTURE   (25%)  {_bar(ev.pillar_structure)}")
        print(f"  INTEGRITY   (35%)  {_bar(ev.pillar_integrity)}")
        print(f"  PERCEPTION  (10%)  {_bar(ev.pillar_perception)}")
        print(f"  {'─' * 40}")
        print(f"  WEIGHTED SCORE     {_bar(ev.weighted_score)}")
        print(f"{'━' * 44}")

        # Concept rules
        print(f"\n— Concept Rules ({ev.concept_score} pts) —")
        for r in ev.concept_results:
            icon = "✅" if r["matched"] else "❌"
            print(f"  {icon} {r['rule']}: {r['description']} [{r['points']}pt]")

        # Contradictions
        if ev.contradiction_results:
            print(f"\n— Contradiction Detection ({ev.contradiction_score} pts) —")
            for c in ev.contradiction_results:
                icon = "✅" if c["detected"] else "❌"
                print(f"  {icon} {c['description']} [{c['points']}pt]")

        # Insight usage
        if ev.insight_usage_results:
            print(f"\n— Prior Insight Verification ({ev.insight_usage_score} pts) —")
            for r in ev.insight_usage_results:
                icon = "✅" if r["verified"] else "❌"
                print(f"  {icon} [{r['insight']}] {r['description']} [{r['points']}pt]")

        # False narrative
        if case.solution.must_reject_false_narrative:
            icon = "✅" if ev.false_narrative_rejected else "❌"
            print(f"\n— False Narrative Rejection —")
            print(f"  {icon} Must reject: \"{case.false_narrative}\"")

        # Reasoning graph
        if rg_result:
            print(f"\n— Reasoning Chain Analysis ({rg_result.total_score}/{rg_result.max_score} pts) —")
            for link in rg_result.links:
                icon = "✅" if link.score > 0 else "❌"
                print(f"  {icon} {link.name}: {link.details}")
            if rg_result.ungrounded_hypotheses:
                print(f"  ⚠ Ungrounded hypotheses:")
                for h in rg_result.ungrounded_hypotheses[:3]:
                    print(f"    • {h}")
            if rg_result.missed_evidence:
                print(f"  ⚠ Missed evidence:")
                for e in rg_result.missed_evidence[:3]:
                    print(f"    • {e}")

        # LLM Judge (multi-role)
        if ev.llm_role_results:
            print(f"\n— LLM Judge ({ev.llm_score}/3 pts) —")
            role_icons = {
                "STRONG": "✅", "COHERENT": "✅", "SURVIVES": "✅",
                "WEAK": "❌", "INCOHERENT": "❌", "FRAGILE": "❌",
                "SKIP": "⏭️",
            }
            for r in ev.llm_role_results:
                icon = role_icons.get(r.get("score", "SKIP"), "❓")
                print(f"  {icon} {r['role']}: {r['score']} [{r.get('points', 0)}pt]")
        elif ev.llm_verdict == "SKIP":
            print(f"\n— LLM Judge (skipped) —")

        # Bayesian reasoning (V8)
        if bay_result and bay_result.max_score > 0:
            print(f"\n— Bayesian Reasoning ({bay_result.total_score}/{bay_result.max_score} pts) —")
            for d in bay_result.dimensions:
                icon = "✅" if d.score >= d.max_score else ("🔶" if d.score > 0 else "❌")
                print(f"  {icon} {d.name}: {d.score}/{d.max_score} — {d.details}")
            if bay_result.cognitive_traps_triggered:
                print(f"  ⚠ Cognitive traps triggered:")
                for trap in bay_result.cognitive_traps_triggered:
                    print(f"    • {trap}")

        # Causality validation (derivation integrity)
        if caus_result:
            print(f"\n— Derivation Integrity ({caus_result.total_score}/{caus_result.max_score} pts) —")
            severity_icons = {
                "fatal": "🔴", "serious": "🟠", "minor": "🟡",
                "structural": "🔵", "bonus": "🟢",
            }
            for d in caus_result.dimensions:
                s_icon = severity_icons.get(d.severity, "❓")
                score_icon = "✅" if d.score >= d.max_score else ("🔶" if d.score > 0 else "❌")
                print(f"  {score_icon} {s_icon} {d.name}: {d.score}/{d.max_score} [{d.severity}] — {d.details}")
            if caus_result.temporal_violations:
                print(f"  ⏰ Temporal violations:")
                for tv in caus_result.temporal_violations:
                    print(f"    • {tv}")
            if caus_result.chain_trace_lines:
                print(f"  ⛓  Evidence concept chain trace:")
                for line in caus_result.chain_trace_lines:
                    print(line)
            if caus_result.inference_traps_triggered:
                print(f"  ⚠ Inference traps triggered (−{caus_result.inference_trap_penalties} pts):")
                for trap in caus_result.inference_traps_triggered:
                    print(f"    • {trap}")

        # Perception integrity (V11)
        if perc_result is not None:
            print(f"\n— Perception Integrity (confidence: {perc_result.evaluation_confidence:.0%}) —")

            def _fmt(v: float | None) -> str:
                return f"{v:.2f}" if v is not None else "N/A"

            print(f"  Coverage:            {_fmt(perc_result.coverage)}")
            print(f"  Retention:           {_fmt(perc_result.retention_score)}")
            print(f"  Late injection:      {_fmt(perc_result.late_injection_score)}")
            print(f"  Causal uptake:       {_fmt(perc_result.causal_uptake_score):.2f}")
            print(f"  Salience distortion: {_fmt(perc_result.salience_distortion)}")
            print(f"  Narrative lock-in:   {_fmt(perc_result.narrative_lock_in)}")
            print(f"  Timing quality:      {perc_result.timing_data_quality}")
            print(f"  ID match rate:       {perc_result.id_match_rate:.0%}")

            if perc_result.id_match_rate < 0.40:
                print("  ⚠ Match rate too low for reliable perception scoring")

            if perc_result.strategic_neutrality_score > 0.0:
                print(f"  ⚠ Strategic neutrality: {perc_result.strategic_neutrality_score:.2f}")

            if perc_result.adversarial_flags:
                print("  Adversarial flags:")
                for flag in perc_result.adversarial_flags:
                    print(f"    • {flag}")

            if perc_result.recommended_probes:
                print("  Suggested probes:")
                for probe in perc_result.recommended_probes:
                    print(f"    • {probe}")

        # Forbidden
        triggered = [f for f in ev.forbidden_results if f["triggered"]]
        if triggered:
            print(f"\n— ⚠️  Forbidden Reasoning (−{ev.forbidden_penalty} pts) —")
            for f in triggered:
                print(f"  ⚠ {f['description']} [−{f['penalty']}pt]")

        # Purity
        if ev.observation_purity_penalty > 0:
            print(f"\n— ⚠️  Observation Impurity (−{ev.observation_purity_penalty} pts) —")

        # Score
        total_penalty = ev.forbidden_penalty + ev.observation_purity_penalty
        print(f"\n{'━' * 44}")
        print(f"  Earned:     {ev.earned} / {ev.max_possible} pts")
        print(f"  Penalties:  −{total_penalty} pts")
        print(f"  Confidence: {ev.confidence_score:.0%}")
        print(f"  Grade:      {ev.grade}")
        print(f"{'━' * 44}")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("DETECTIVE REASONING ENGINE V11 — ADVERSARIAL INTELLIGENCE FRAMEWORK")
        print("Concept matching • Contradiction detection • Reasoning chain analysis")
        print("Derivation integrity • Perception integrity • Hypothesis diversity • Trap detection • Multi-role LLM Judge • Skill tracking")
        print("Protocol: observe → explain → contradict → reject → eliminate\n")

        while True:
            print(self.graph.render_case_table())
            print("\nCommands: <case ID> | insights | graph | profile | state | exit")
            choice = normalize_key(input("> "))

            if choice == "exit":
                print("Session closed.")
                return
            if choice == "insights":
                self.graph_show_insights()
                continue
            if choice == "graph":
                print(self.graph.export_json())
                continue
            if choice == "profile":
                print(self.profile.render_profile())
                continue
            if choice == "state":
                print(json.dumps({
                    "unlocked": sorted(self.graph.unlocked),
                    "history": self.history,
                }, indent=2))
                continue
            upper = choice.upper()
            if upper in self.graph.cases:
                self.run_case(upper)
                continue
            print("Unknown command.")

    def graph_show_insights(self) -> None:
        _section("UNLOCKED INSIGHTS")
        if not self.graph.unlocked:
            print("  (none yet)")
            return
        for key in sorted(self.graph.unlocked):
            ins = self.graph.insights[key]
            print(f"  🔑 {ins.key}: {ins.title}")
            print(f"     type: {ins.reasoning_type}")
            print(f"     rule: {ins.transfer_rule}")
