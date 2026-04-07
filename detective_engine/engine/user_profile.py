"""User profile — tracks skill progression, weak areas, and repeated mistakes.

Persists to a JSON file between sessions.  Each case attempt is recorded with
scores, errors, and timestamps.  The profile computes rolling statistics to
show the user where they're strong and where they need work.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CaseAttempt:
    """A single attempt at a case."""
    case_id: str
    timestamp: float              # time.time()
    grade: str                    # A/B/C/D/F
    passed: bool
    confidence: float             # 0.0–1.0
    elimination_correct: bool
    contradiction_score: int
    contradiction_max: int
    concept_score: int
    concept_max: int
    insight_score: int
    insight_max: int
    forbidden_penalty: int
    purity_penalty: int
    reasoning_chain_score: int    # 0–3 from ReasoningGraphValidator
    llm_score: int                # 0–3 from LLMJudge multi-role
    errors: list[str] = field(default_factory=list)  # specific error tags
    # V11: perception integrity snapshot (None when module inactive / no trace)
    perception_coverage: float | None = None
    perception_late_injection: float | None = None
    perception_causal_uptake: float | None = None
    perception_salience_distortion: float | None = None
    perception_narrative_lock_in: float | None = None
    perception_strategic_neutrality: float | None = None
    perception_adversarial_flags: list[str] = field(default_factory=list)
    perception_eval_confidence: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CaseAttempt:
        return cls(**d)


@dataclass
class SkillProfile:
    """Rolling statistics across all attempts."""
    total_attempts: int = 0
    total_passed: int = 0
    pass_rate: float = 0.0

    # per-skill averages (0.0–1.0)
    observation_quality: float = 0.0   # 1 - (avg purity penalty / max observations)
    contradiction_detection: float = 0.0
    concept_matching: float = 0.0
    elimination_accuracy: float = 0.0
    reasoning_chain_quality: float = 0.0  # avg reasoning chain score / 3
    llm_reasoning_quality: float = 0.0    # avg llm score / 3

    # error frequency
    common_errors: dict[str, int] = field(default_factory=dict)

    # trend (last 5 attempts)
    recent_grades: list[str] = field(default_factory=list)
    improving: bool = False

    # V11: perception integrity trends (None when no perception data available)
    avg_perception_coverage: float | None = None
    avg_late_injection: float | None = None
    avg_causal_uptake: float | None = None
    avg_salience_distortion: float | None = None
    avg_narrative_lock_in: float | None = None
    common_perception_failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error tags
# ---------------------------------------------------------------------------

ERROR_INFERENCE_LEAK = "inference_leak"
ERROR_MISSED_CONTRADICTION = "missed_contradiction"
ERROR_FORBIDDEN_TRIGGERED = "forbidden_triggered"
ERROR_WRONG_ELIMINATION = "wrong_elimination"
ERROR_MISSED_FALSE_NARRATIVE = "missed_false_narrative"
ERROR_WEAK_REASONING_CHAIN = "weak_reasoning_chain"
ERROR_MISSED_EVIDENCE = "missed_evidence"
ERROR_UNGROUNDED_HYPOTHESIS = "ungrounded_hypothesis"


# ---------------------------------------------------------------------------
# Profile manager
# ---------------------------------------------------------------------------

class UserProfile:
    """Manages a single user's training history.

    Parameters
    ----------
    profile_path : Path | str | None
        Where to persist the JSON profile.  Defaults to ``detective_engine/data/profile.json``.
    """

    def __init__(self, profile_path: Path | str | None = None):
        default = Path(__file__).resolve().parent.parent / "data" / "profile.json"
        self.path = Path(profile_path) if profile_path else default
        self.attempts: list[CaseAttempt] = []
        self.probes_seen_per_case: dict[str, list[str]] = {}  # V11: probe rotation
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                import dataclasses
                raw_attempts = data.get("attempts", [])
                self.attempts = []
                known_fields = {f.name for f in dataclasses.fields(CaseAttempt)}
                for a in raw_attempts:
                    # Forward-compat: ignore unknown fields from older saves
                    filtered = {k: v for k, v in a.items() if k in known_fields}
                    self.attempts.append(CaseAttempt(**filtered))
                self.probes_seen_per_case = data.get("probes_seen_per_case", {})
            except (json.JSONDecodeError, TypeError, KeyError):
                self.attempts = []
                self.probes_seen_per_case = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "attempts": [a.to_dict() for a in self.attempts],
            "probes_seen_per_case": self.probes_seen_per_case,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    def record_attempt(self, attempt: CaseAttempt) -> None:
        """Record a case attempt and persist."""
        self.attempts.append(attempt)
        self.save()

    def get_probes_seen(self, case_id: str) -> set[str]:
        """Return the set of probe strings already shown for a case."""
        return set(self.probes_seen_per_case.get(case_id, []))

    def record_probes_seen(self, case_id: str, probes: list[str]) -> None:
        """Add probe strings to the seen set for a case and persist."""
        existing = self.probes_seen_per_case.get(case_id, [])
        updated = list(dict.fromkeys(existing + probes))  # deduplicate, preserve order
        self.probes_seen_per_case[case_id] = updated
        self.save()

    def build_attempt(
        self,
        case_id: str,
        evaluation: Any,        # EvaluationResult
        reasoning_score: int = 0,
        llm_score: int = 0,
        extra_errors: list[str] | None = None,
        perception: Any = None,  # PerceptionIntegrityResult | None
    ) -> CaseAttempt:
        """Build a CaseAttempt from an EvaluationResult + optional extras."""
        errors: list[str] = list(extra_errors or [])

        if evaluation.observation_purity_penalty > 0:
            errors.append(ERROR_INFERENCE_LEAK)
        if evaluation.forbidden_penalty > 0:
            errors.append(ERROR_FORBIDDEN_TRIGGERED)
        if not evaluation.elimination_correct:
            errors.append(ERROR_WRONG_ELIMINATION)
        if not evaluation.false_narrative_rejected and any(
            r.get("verified") is False for r in evaluation.insight_usage_results
        ):
            errors.append(ERROR_MISSED_FALSE_NARRATIVE)

        # Check for missed contradictions
        missed_c = sum(1 for c in evaluation.contradiction_results if not c["detected"])
        if missed_c > 0:
            errors.append(ERROR_MISSED_CONTRADICTION)

        if reasoning_score < 2:
            errors.append(ERROR_WEAK_REASONING_CHAIN)

        # V11: extract perception fields if result is available
        perc_coverage = None
        perc_late_inj = None
        perc_uptake = None
        perc_sal = None
        perc_lock = None
        perc_neutrality = None
        perc_flags: list[str] = []
        perc_conf = None
        if perception is not None:
            perc_coverage = perception.coverage
            perc_late_inj = perception.late_injection_score
            perc_uptake = perception.causal_uptake_score
            perc_sal = perception.salience_distortion
            perc_lock = perception.narrative_lock_in
            perc_neutrality = perception.strategic_neutrality_score
            perc_flags = list(perception.adversarial_flags)
            perc_conf = perception.evaluation_confidence

        return CaseAttempt(
            case_id=case_id,
            timestamp=time.time(),
            grade=evaluation.grade,
            passed=evaluation.passed,
            confidence=evaluation.confidence_score,
            elimination_correct=evaluation.elimination_correct,
            contradiction_score=evaluation.contradiction_score,
            contradiction_max=sum(c["points"] for c in evaluation.contradiction_results),
            concept_score=evaluation.concept_score,
            concept_max=sum(r["points"] for r in evaluation.concept_results),
            insight_score=evaluation.insight_usage_score,
            insight_max=sum(r["points"] for r in evaluation.insight_usage_results),
            forbidden_penalty=evaluation.forbidden_penalty,
            purity_penalty=evaluation.observation_purity_penalty,
            reasoning_chain_score=reasoning_score,
            llm_score=llm_score,
            errors=errors,
            perception_coverage=perc_coverage,
            perception_late_injection=perc_late_inj,
            perception_causal_uptake=perc_uptake,
            perception_salience_distortion=perc_sal,
            perception_narrative_lock_in=perc_lock,
            perception_strategic_neutrality=perc_neutrality,
            perception_adversarial_flags=perc_flags,
            perception_eval_confidence=perc_conf,
        )

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------

    def compute_profile(self) -> SkillProfile:
        """Compute rolling statistics from all attempts."""
        n = len(self.attempts)
        if n == 0:
            return SkillProfile()

        total_passed = sum(1 for a in self.attempts if a.passed)

        # Per-skill averages
        obs_quality_scores = []
        contra_scores = []
        concept_scores = []
        elim_correct_scores = []
        chain_scores = []
        llm_scores = []

        error_freq: dict[str, int] = {}

        for a in self.attempts:
            # Observation quality: 1.0 means no purity penalty
            max_obs = max(a.purity_penalty + 1, 1)  # avoid div by zero
            obs_quality_scores.append(max(0, 1.0 - a.purity_penalty / max_obs))

            # Contradiction detection ratio
            if a.contradiction_max > 0:
                contra_scores.append(a.contradiction_score / a.contradiction_max)
            else:
                contra_scores.append(1.0)

            # Concept matching ratio
            if a.concept_max > 0:
                concept_scores.append(a.concept_score / a.concept_max)
            else:
                concept_scores.append(1.0)

            elim_correct_scores.append(1.0 if a.elimination_correct else 0.0)
            chain_scores.append(a.reasoning_chain_score / 3.0)
            llm_scores.append(a.llm_score / 3.0)

            for err in a.errors:
                error_freq[err] = error_freq.get(err, 0) + 1

        def avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        recent = [a.grade for a in self.attempts[-5:]]
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

        improving = False
        if len(recent) >= 3:
            recent_vals = [grade_values.get(g, 0) for g in recent]
            improving = recent_vals[-1] > recent_vals[0]

        # V11: compute perception trends from attempts that have perception data
        perc_attempts = [a for a in self.attempts if a.perception_eval_confidence is not None]

        def avg_perc(values: list[float | None]) -> float | None:
            real = [v for v in values if v is not None]
            return sum(real) / len(real) if real else None

        avg_perc_coverage = avg_perc([a.perception_coverage for a in perc_attempts])
        avg_late_inj = avg_perc([a.perception_late_injection for a in perc_attempts])
        avg_causal_uptake = avg_perc([a.perception_causal_uptake for a in perc_attempts])
        avg_sal_dist = avg_perc([a.perception_salience_distortion for a in perc_attempts])
        avg_lock_in = avg_perc([a.perception_narrative_lock_in for a in perc_attempts])

        # Most common perception adversarial flags
        flag_freq: dict[str, int] = {}
        for a in perc_attempts:
            for flag in a.perception_adversarial_flags:
                flag_freq[flag] = flag_freq.get(flag, 0) + 1
        common_perc_failures = [
            f for f, _ in sorted(flag_freq.items(), key=lambda x: -x[1])[:5]
        ]

        return SkillProfile(
            total_attempts=n,
            total_passed=total_passed,
            pass_rate=total_passed / n,
            observation_quality=avg(obs_quality_scores),
            contradiction_detection=avg(contra_scores),
            concept_matching=avg(concept_scores),
            elimination_accuracy=avg(elim_correct_scores),
            reasoning_chain_quality=avg(chain_scores),
            llm_reasoning_quality=avg(llm_scores),
            common_errors=dict(sorted(error_freq.items(), key=lambda x: -x[1])),
            recent_grades=recent,
            improving=improving,
            avg_perception_coverage=avg_perc_coverage,
            avg_late_injection=avg_late_inj,
            avg_causal_uptake=avg_causal_uptake,
            avg_salience_distortion=avg_sal_dist,
            avg_narrative_lock_in=avg_lock_in,
            common_perception_failures=common_perc_failures,
        )

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def render_profile(self) -> str:
        """Render the user profile as a formatted string."""
        p = self.compute_profile()
        if p.total_attempts == 0:
            return "📊 No attempts recorded yet."

        lines = [
            "📊 USER SKILL PROFILE",
            f"{'━' * 44}",
            f"  Attempts:            {p.total_attempts}",
            f"  Passed:              {p.total_passed} ({p.pass_rate:.0%})",
            f"  Recent grades:       {' → '.join(p.recent_grades)}",
            f"  Trend:               {'📈 Improving' if p.improving else '📉 Needs work'}",
            f"",
            f"  SKILL BREAKDOWN:",
            f"  {'─' * 40}",
            f"  Observation quality:     {self._bar(p.observation_quality)}  {p.observation_quality:.0%}",
            f"  Contradiction detection: {self._bar(p.contradiction_detection)}  {p.contradiction_detection:.0%}",
            f"  Concept matching:        {self._bar(p.concept_matching)}  {p.concept_matching:.0%}",
            f"  Elimination accuracy:    {self._bar(p.elimination_accuracy)}  {p.elimination_accuracy:.0%}",
            f"  Reasoning chain:         {self._bar(p.reasoning_chain_quality)}  {p.reasoning_chain_quality:.0%}",
        ]

        if p.common_errors:
            lines += ["", "  COMMON MISTAKES:"]
            for err, count in list(p.common_errors.items())[:5]:
                lines.append(f"    ⚠ {err}: {count}× ")

        # V11: perception integrity section
        has_perc = any(
            v is not None for v in [
                p.avg_perception_coverage,
                p.avg_late_injection,
                p.avg_causal_uptake,
            ]
        )
        if has_perc:
            def _pf(v: float | None) -> str:
                return f"{v:.2f}" if v is not None else "N/A"

            lines += [
                "",
                "  PERCEPTION INTEGRITY TRENDS:",
                f"  {'─' * 40}",
                f"  Avg coverage:         {_pf(p.avg_perception_coverage)}",
                f"  Avg late injection:   {_pf(p.avg_late_injection)}",
                f"  Avg causal uptake:    {_pf(p.avg_causal_uptake)}",
                f"  Avg salience distort: {_pf(p.avg_salience_distortion)}",
                f"  Avg narrative lock-in:{_pf(p.avg_narrative_lock_in)}",
            ]
            if p.common_perception_failures:
                lines.append("  Recurring perception flags:")
                for flag in p.common_perception_failures:
                    lines.append(f"    • {flag}")
            # Adaptive perception recommendations
            if p.avg_late_injection is not None and p.avg_late_injection > 0.40:
                lines.append("    → You often register key clues late. Try the micro-capture phase more carefully.")
            if p.avg_salience_distortion is not None and p.avg_salience_distortion > 0.60:
                lines.append("    → You tend to fixate on dramatic clues. Practice noticing quiet details.")
            if p.avg_narrative_lock_in is not None and p.avg_narrative_lock_in > 0.70:
                lines.append("    → You commit to hypotheses early. Delay conclusions until more clues are mapped.")

        # Adaptive recommendations
        lines += ["", "  RECOMMENDATIONS:"]
        if p.observation_quality < 0.7:
            lines.append("    → Separate observations from interpretations. Only state what you SEE.")
        if p.contradiction_detection < 0.5:
            lines.append("    → Cross-reference every pair of facts. Look for tensions.")
        if p.concept_matching < 0.5:
            lines.append("    → Use more specific terminology from the evidence.")
        if p.elimination_accuracy < 0.5:
            lines.append("    → Build your elimination from evidence, not intuition.")
        if p.reasoning_chain_quality < 0.5:
            lines.append("    → Ensure each hypothesis references specific observations.")
        if p.pass_rate >= 0.8:
            lines.append("    → Excellent work. Try the harder cases or increase speed.")

        lines.append(f"{'━' * 44}")
        return "\n".join(lines)

    @staticmethod
    def _bar(value: float, width: int = 10) -> str:
        """Render a text progress bar."""
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    # ------------------------------------------------------------------
    # case-specific history
    # ------------------------------------------------------------------

    def case_history(self, case_id: str) -> list[CaseAttempt]:
        return [a for a in self.attempts if a.case_id == case_id]

    def best_grade(self, case_id: str) -> str | None:
        history = self.case_history(case_id)
        if not history:
            return None
        grade_order = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        return max(history, key=lambda a: grade_order.get(a.grade, 0)).grade

    def weakness_report(self) -> list[str]:
        """Return the top 3 weakest skills."""
        p = self.compute_profile()
        skills = [
            ("observation_quality", p.observation_quality),
            ("contradiction_detection", p.contradiction_detection),
            ("concept_matching", p.concept_matching),
            ("elimination_accuracy", p.elimination_accuracy),
            ("reasoning_chain", p.reasoning_chain_quality),
        ]
        skills.sort(key=lambda x: x[1])
        return [s[0] for s in skills[:3]]
