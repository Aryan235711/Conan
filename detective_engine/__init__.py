"""Detective Reasoning Engine — Adversarial Intelligence Framework."""

from detective_engine.engine.models import (
    AnalysisRecord,
    AnswerKey,
    CaseDefinition,
    CaseSolution,
    Concept,
    ConceptGroup,
    ConceptRule,
    Contradiction,
    EvaluationResult,
    ForbiddenPattern,
    Insight,
    InsightUsageRule,
)
from detective_engine.engine.case_loader import CaseLoader
from detective_engine.engine.case_runner import CaseRunner
from detective_engine.engine.case_validator import CaseQualityValidator
from detective_engine.engine.insight_graph import InsightGraph
from detective_engine.engine.llm_judge import LLMJudge, LLMFullResult, LLMRoleResult
from detective_engine.engine.reasoning_graph import ReasoningGraphValidator, ReasoningGraphResult
from detective_engine.engine.bayesian_validator import BayesianValidator, BayesianResult, BayesianDimension
from detective_engine.engine.causality_validator import CausalityValidator, CausalityResult
from detective_engine.engine.user_profile import UserProfile, SkillProfile
from detective_engine.engine.validator import Validator
from detective_engine.engine.reward_interface import RewardScorer
from detective_engine.engine.verifiable import VerifiableScorer, VerifiableResult, build_prompt, parse_final_answer

__all__ = [
    "AnalysisRecord",
    "AnswerKey",
    "BayesianDimension",
    "BayesianResult",
    "BayesianValidator",
    "CaseDefinition",
    "CaseLoader",
    "CaseQualityValidator",
    "CaseRunner",
    "CaseSolution",
    "CausalityResult",
    "CausalityValidator",
    "Concept",
    "ConceptGroup",
    "ConceptRule",
    "Contradiction",
    "EvaluationResult",
    "ForbiddenPattern",
    "Insight",
    "InsightGraph",
    "InsightUsageRule",
    "LLMFullResult",
    "LLMJudge",
    "LLMRoleResult",
    "ReasoningGraphResult",
    "ReasoningGraphValidator",
    "SkillProfile",
    "UserProfile",
    "RewardScorer",
    "VerifiableResult",
    "VerifiableScorer",
    "build_prompt",
    "parse_final_answer",
    "Validator",
]
