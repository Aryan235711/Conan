"""Core data models for the Detective Reasoning Engine.

Every dataclass here is JSON-serializable via ``dataclasses.asdict`` and can be
reconstructed from a plain dict via ``from_dict`` class methods (used by the
JSON case loader).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal


# ---------------------------------------------------------------------------
# Text helpers (used by matching primitives)
# ---------------------------------------------------------------------------

def sanitize_text(text: str) -> str:
    """Collapse whitespace and strip, preserving meaning."""
    return " ".join(text.strip().split())


def normalize_key(text: str) -> str:
    return sanitize_text(text).lower()


def split_csv(text: str) -> list[str]:
    parts = [sanitize_text(p) for p in text.split(",")]
    return [p for p in parts if p]


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_key(text)
    return any(normalize_key(kw) in normalized for kw in keywords)


# ---------------------------------------------------------------------------
# Concept matching primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConceptGroup:
    """A cluster of related terms.  Matching requires ANY one term."""
    terms: tuple[str, ...]

    def matches(self, text: str) -> bool:
        normalized = normalize_key(text)
        return any(normalize_key(t) in normalized for t in self.terms)


# convenient factory
def CG(*terms: str) -> ConceptGroup:
    """Shorthand constructor for ConceptGroup."""
    return ConceptGroup(terms=terms)


# alias used by JSON loader
Concept = ConceptGroup


def concept_match(text: str, groups: list[ConceptGroup]) -> bool:
    """True only when EVERY group has ≥1 hit (AND across groups, OR within)."""
    return all(g.matches(text) for g in groups)


# ---------------------------------------------------------------------------
# Knowledge / insight
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Insight:
    key: str
    title: str
    reasoning_type: str
    summary: str
    transfer_rule: str

    @classmethod
    def from_dict(cls, d: dict) -> Insight:
        return cls(**d)


# ---------------------------------------------------------------------------
# Evaluation sub-components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConceptRule:
    """Reasoning checkpoint — requires ALL groups to match."""
    name: str
    description: str
    concept_groups: list[ConceptGroup]
    points: int = 1

    def evaluate(self, text: str) -> bool:
        return concept_match(text, self.concept_groups)

    @classmethod
    def from_dict(cls, d: dict) -> ConceptRule:
        return cls(
            name=d["name"],
            description=d["description"],
            concept_groups=[ConceptGroup(terms=tuple(g)) for g in d["concept_groups"]],
            points=d.get("points", 1),
        )


@dataclass(frozen=True)
class ForbiddenPattern:
    """Wrong reasoning — if ALL groups match, a penalty is applied."""
    description: str
    concept_groups: list[ConceptGroup]
    penalty: int = 1

    def triggered(self, text: str) -> bool:
        return concept_match(text, self.concept_groups)

    @classmethod
    def from_dict(cls, d: dict) -> ForbiddenPattern:
        return cls(
            description=d["description"],
            concept_groups=[ConceptGroup(terms=tuple(g)) for g in d["concept_groups"]],
            penalty=d.get("penalty", 1),
        )


@dataclass(frozen=True)
class Contradiction:
    """Two facts in tension.  User must reference BOTH sides."""
    fact_a_keywords: tuple[str, ...]
    fact_b_keywords: tuple[str, ...]
    description: str
    points: int = 1

    def detected_in(self, text: str) -> bool:
        n = normalize_key(text)
        a = any(normalize_key(k) in n for k in self.fact_a_keywords)
        b = any(normalize_key(k) in n for k in self.fact_b_keywords)
        return a and b

    @classmethod
    def from_dict(cls, d: dict) -> Contradiction:
        return cls(
            fact_a_keywords=tuple(d["fact_a_keywords"]),
            fact_b_keywords=tuple(d["fact_b_keywords"]),
            description=d["description"],
            points=d.get("points", 1),
        )


@dataclass(frozen=True)
class InsightUsageRule:
    """Verifies a previously unlocked insight is actively APPLIED."""
    insight_key: str
    description: str
    concept_groups: list[ConceptGroup]
    points: int = 1

    def verified(self, text: str) -> bool:
        return concept_match(text, self.concept_groups)

    @classmethod
    def from_dict(cls, d: dict) -> InsightUsageRule:
        return cls(
            insight_key=d["insight_key"],
            description=d["description"],
            concept_groups=[ConceptGroup(terms=tuple(g)) for g in d["concept_groups"]],
            points=d.get("points", 1),
        )


# ---------------------------------------------------------------------------
# Bayesian reasoning components (V8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BayesianHypothesis:
    """A hypothesis with prior and posterior probabilities."""
    id: str
    label: str
    prior: float           # P(H) before evidence
    posterior: float       # P(H|E) after all evidence
    key_supporting: list[str] = field(default_factory=list)
    key_weakening: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> BayesianHypothesis:
        return cls(
            id=d["id"], label=d["label"],
            prior=d["prior"], posterior=d["posterior"],
            key_supporting=d.get("key_supporting", []),
            key_weakening=d.get("key_weakening", []),
        )


@dataclass(frozen=True)
class BayesianEvidenceWeight:
    """How diagnostic a single piece of evidence is for each hypothesis."""
    evidence_index: int
    diagnostic_power: str    # "high" | "moderate" | "low" | "neutral" | "misleading"
    favors: str              # hypothesis ID this evidence supports
    explanation: str

    @classmethod
    def from_dict(cls, d: dict) -> BayesianEvidenceWeight:
        return cls(**d)


@dataclass(frozen=True)
class CognitiveTrap:
    """A reasoning error the case is designed to trigger."""
    name: str
    description: str
    penalty_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> CognitiveTrap:
        return cls(
            name=d["name"], description=d["description"],
            penalty_keywords=d.get("penalty_keywords", []),
        )


@dataclass(frozen=True)
class InferenceTrap:
    """A known wrong inference that seems logical but is factually wrong.

    V10: Cases define these so the engine can penalize users who fall
    into common reasoning errors.  Unlike ForbiddenPattern (which checks
    for reasoning WORDS), InferenceTrap checks for specific wrong
    CONCLUSIONS or causal claims.

    Example: "window open → someone entered through the window"
    is a common but wrong inference for C001 (the window is for staging).
    """
    description: str
    trap_keywords: list[str] = field(default_factory=list)
    penalty: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> InferenceTrap:
        return cls(
            description=d["description"],
            trap_keywords=d.get("trap_keywords", []),
            penalty=d.get("penalty", 1),
        )


@dataclass(frozen=True)
class BayesianSolution:
    """Full probabilistic solution for a multi-hypothesis case.

    Contains the ground-truth probability distribution, diagnostic
    evidence weights, and cognitive traps the case is designed to test.
    """
    hypotheses: list[BayesianHypothesis]
    most_probable: str                   # hypothesis ID
    evidence_weights: list[BayesianEvidenceWeight] = field(default_factory=list)
    probability_channels: dict[str, str] = field(default_factory=dict)
    acceptable_posterior_range: tuple[float, float] = (0.6, 0.9)
    cognitive_traps: list[CognitiveTrap] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> BayesianSolution:
        return cls(
            hypotheses=[
                BayesianHypothesis.from_dict(h) for h in d["hypotheses"]
            ],
            most_probable=d["most_probable"],
            evidence_weights=[
                BayesianEvidenceWeight.from_dict(w)
                for w in d.get("evidence_weights", [])
            ],
            probability_channels=d.get("probability_channels", {}),
            acceptable_posterior_range=tuple(
                d.get("acceptable_posterior_range", [0.6, 0.9])
            ),
            cognitive_traps=[
                CognitiveTrap.from_dict(t)
                for t in d.get("cognitive_traps", [])
            ],
        )


# ---------------------------------------------------------------------------
# Case solution & definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseSolution:
    unlock_key: str
    elimination_target: str = ""
    direct_answer_keywords: list[str] = field(default_factory=list)
    required_concept_rules: list[ConceptRule] = field(default_factory=list)
    forbidden_patterns: list[ForbiddenPattern] = field(default_factory=list)
    insight_usage_rules: list[InsightUsageRule] = field(default_factory=list)
    must_reject_false_narrative: bool = False
    bayesian: BayesianSolution | None = None
    inference_traps: list[InferenceTrap] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> CaseSolution:
        return cls(
            unlock_key=d["unlock_key"],
            elimination_target=d.get("elimination_target", ""),
            direct_answer_keywords=d.get("direct_answer_keywords", []),
            required_concept_rules=[
                ConceptRule.from_dict(r) for r in d.get("required_concept_rules", [])
            ],
            forbidden_patterns=[
                ForbiddenPattern.from_dict(f) for f in d.get("forbidden_patterns", [])
            ],
            insight_usage_rules=[
                InsightUsageRule.from_dict(r) for r in d.get("insight_usage_rules", [])
            ],
            must_reject_false_narrative=d.get("must_reject_false_narrative", False),
            bayesian=(
                BayesianSolution.from_dict(d["bayesian"])
                if "bayesian" in d else None
            ),
            inference_traps=[
                InferenceTrap.from_dict(t) for t in d.get("inference_traps", [])
            ],
        )


@dataclass(frozen=True)
class CaseDefinition:
    id: str
    title: str
    category: str
    summary: str
    evidence: list[str]
    detective_questions: list[str]
    analysis_protocol: list[str]
    scenarios: list[str] = field(default_factory=list)
    requires_all: list[str] = field(default_factory=list)
    teaches: list[str] = field(default_factory=list)
    hidden_truth: str = ""
    false_narrative: str = ""
    contradictions: list[Contradiction] = field(default_factory=list)
    solution: CaseSolution = field(default_factory=lambda: CaseSolution(unlock_key=""))

    @classmethod
    def from_dict(cls, d: dict) -> CaseDefinition:
        return cls(
            id=d["id"],
            title=d["title"],
            category=d["category"],
            summary=d["summary"],
            evidence=d["evidence"],
            detective_questions=d["detective_questions"],
            analysis_protocol=d["analysis_protocol"],
            scenarios=d.get("scenarios", []),
            requires_all=d.get("requires_all", []),
            teaches=d.get("teaches", []),
            hidden_truth=d.get("hidden_truth", ""),
            false_narrative=d.get("false_narrative", ""),
            contradictions=[
                Contradiction.from_dict(c) for c in d.get("contradictions", [])
            ],
            solution=CaseSolution.from_dict(d["solution"]),
        )


# ---------------------------------------------------------------------------
# User input record
# ---------------------------------------------------------------------------

@dataclass
class AnalysisRecord:
    observations: list[str]
    anomalies: list[str]
    hypotheses: dict[str, list[str]]
    elimination_target: str
    reasons: list[str]
    false_narrative_rejection: str = ""
    contradiction_notes: list[str] = field(default_factory=list)
    # V8: Bayesian reasoning fields
    probability_ranking: list[dict] = field(default_factory=list)
    evidence_weight_notes: list[str] = field(default_factory=list)
    prior_reasoning: list[str] = field(default_factory=list)
    # V9.1: temporal integrity
    phase_timestamps: dict[str, str] = field(default_factory=dict)
    # V11: perception trace (micro-capture + later observations)
    perception_trace: PerceptionTrace | None = None


# ---------------------------------------------------------------------------
# Evaluation result — confidence scored
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    elimination_correct: bool
    direct_answer_correct: bool
    false_narrative_rejected: bool

    concept_results: list[dict]
    concept_score: int

    contradiction_results: list[dict]
    contradiction_score: int

    insight_usage_results: list[dict]
    insight_usage_score: int

    forbidden_results: list[dict]
    forbidden_penalty: int

    observation_purity_penalty: int

    # Reasoning graph (observation→hypothesis→elimination chain)
    reasoning_graph_results: list[dict] = field(default_factory=list)
    reasoning_graph_score: int = 0     # 0–3
    reasoning_graph_max: int = 3

    # LLM judge — multi-role (3 roles × 1pt each)
    llm_verdict: str = ""          # "VALID" / "INVALID" / "SKIP"
    llm_reasoning: str = ""
    llm_score: int = 0             # 0–3 (sum of 3 roles)
    llm_role_results: list[dict] = field(default_factory=list)
    llm_critique: str = ""
    llm_counterarguments: list[str] = field(default_factory=list)

    # Bayesian reasoning (V8)
    bayesian_score: int = 0
    bayesian_max: int = 0
    bayesian_results: list[dict] = field(default_factory=list)
    bayesian_traps: list[str] = field(default_factory=list)

    # Causality validation (derivation integrity — V10)
    causality_score: int = 0        # 0–11
    causality_max: int = 11
    causality_results: list[dict] = field(default_factory=list)
    causality_phantom: list[str] = field(default_factory=list)
    causality_leaps: list[str] = field(default_factory=list)
    causality_temporal: list[str] = field(default_factory=list)
    causality_chain_trace: list[str] = field(default_factory=list)
    causality_trap_penalties: int = 0
    causality_traps_triggered: list[str] = field(default_factory=list)

    # aggregates
    earned: int = 0
    max_possible: int = 0
    confidence_score: float = 0.0  # 0.0 – 1.0
    grade: str = "F"
    passed: bool = False

    # Weighted pillar scores (0.0–1.0 each) — computed by Validator
    pillar_content: float = 0.0       # What You Found (concepts, contradictions, insights, narrative)
    pillar_structure: float = 0.0     # How You Reasoned (reasoning graph, LLM, Bayesian)
    pillar_integrity: float = 0.0     # How You Derived (causality/derivation)
    pillar_perception: float = 0.0    # How You Noticed (perception integrity)
    weighted_score: float = 0.0       # Weighted combination of pillars

    # Perception integrity (V11) — None until perception module is active
    perception_score: float | None = None


# ---------------------------------------------------------------------------
# V11: Perception integrity types
# ---------------------------------------------------------------------------

@dataclass
class ObservationEvent:
    """A single observation captured at a specific phase and time."""
    text: str
    timestamp: float
    phase: Literal["initial", "later", "revision"]
    order_index: int
    evidence_ids: list[str] = field(default_factory=list)  # matched evidence IDs (plural)
    source: Literal["user", "probe", "system"] = "user"
    confidence: float = 1.0


@dataclass
class PerceptionTrace:
    """Full observation timeline across phases."""
    initial: list[ObservationEvent] = field(default_factory=list)   # micro-capture (Phase 0)
    later: list[ObservationEvent] = field(default_factory=list)     # Phase 1+ observations
    revisions: list[ObservationEvent] = field(default_factory=list) # probe-triggered additions
    first_hypothesis_timestamp: float | None = None   # set at Phase 3 onset
    first_elimination_timestamp: float | None = None  # set at Phase 6 onset
    probe_stage_1_shown: bool = False
    probe_stage_2_shown: bool = False
    probes_answered: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceMeta:
    """Per-evidence perception metadata (optional; author-annotated in case JSON)."""
    id: str
    text: str
    diagnostic_weight: float    # how much this clue discriminates among hypotheses
    salience_level: float       # how naturally human attention is drawn to it
    misdirection_risk: float    # how likely it is to capture attention unhelpfully
    salience_confidence: float  # author confidence in the above estimates
    bias_tags: list[str] = field(default_factory=list)
    critical: bool = False      # marks clues perception scoring cares most about

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceMeta:
        return cls(
            id=d["id"],
            text=d["text"],
            diagnostic_weight=d["diagnostic_weight"],
            salience_level=d["salience_level"],
            misdirection_risk=d["misdirection_risk"],
            salience_confidence=d["salience_confidence"],
            bias_tags=d.get("bias_tags", []),
            critical=d.get("critical", False),
        )


@dataclass
class PerceptionIntegrityResult:
    """Output of the perception integrity validator."""
    # Core metrics — None when id_match_rate < min_reliable_match_rate
    coverage: float | None
    retention_score: float | None
    late_injection_score: float | None
    causal_uptake_score: float

    # Metadata-dependent — None when evidence_meta absent from case JSON
    salience_distortion: float | None
    narrative_lock_in: float | None
    timing_consistency: float | None

    # Adversarial — soft 0.0–1.0 score, gated by evaluation_confidence
    strategic_neutrality_score: float
    adversarial_flags: list[str] = field(default_factory=list)

    # Probe recommendations (caller passes probes_seen for rotation)
    recommended_probes: list[str] = field(default_factory=list)

    # Evaluation quality signals
    evaluation_confidence: float = 0.30   # 0.30–0.90
    timing_data_quality: str = "unavailable"  # "reliable" / "coarse" / "unavailable"
    id_match_rate: float = 0.0


@dataclass
class PerceptionConfig:
    """All tunable thresholds for the perception integrity module.

    Defaults match the V11/V12 spec initial estimates.
    Pass a customised instance to ``PerceptionIntegrityValidator.validate()``
    to tune without touching module code.
    """
    # Strategic neutrality detection
    neutrality_coverage_threshold: float = 0.80
    neutrality_late_injection_threshold: float = 0.50
    neutrality_causal_uptake_threshold: float = 0.40

    # Stage 1 probe triggers (after Phase 3)
    stage1_coverage_threshold: float = 0.50
    stage1_lock_in_threshold: float = 0.70

    # Stage 2 probe triggers (after Phase 6)
    probe_salience_distortion_threshold: float = 0.60
    probe_late_injection_threshold: float = 0.40
    probe_narrative_lock_in_threshold: float = 0.70
    probe_causal_uptake_threshold: float = 0.40

    # Evidence ID matching
    id_match_threshold: float = 0.25
    min_reliable_match_rate: float = 0.40
    # TODO: max_ids_per_observation: int = 2  (future — prevents overcounting)

    # Phase B score adjustment bounds
    adjustment_min: float = -2.0
    adjustment_max: float = 2.0
    adjustment_min_confidence: float = 0.60  # don't adjust below this confidence
