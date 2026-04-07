"""LLM Judge — multi-role reasoning evaluator using a local model.

Design principles:
    🔒 Rule engine = source of truth.  LLM adds SUPPLEMENTARY points.
    ✅ LLM evaluates reasoning *quality*, not truth.
    ❌ LLM NEVER decides pass/fail alone.

Three scoring roles (each worth +1 point):
    1. Implicit contradiction detection — did the user spot non-obvious tensions?
    2. Reasoning chain coherence — does observation→hypothesis→elimination hold?
    3. Counterargument stress-test — can the user's conclusion survive objections?

Plus two non-scoring qualitative outputs:
    4. Critique — training feedback on logical gaps.
    5. Counterarguments — concrete objections for the user to consider.

Supports:
    • Ollama  (default — ``ollama run <model>``)
    • Any OpenAI-compatible local endpoint (llama.cpp, LM Studio, vLLM, etc.)

If no model is available the judge silently returns skip results.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from dataclasses import dataclass, field

from .models import AnalysisRecord, CaseDefinition


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_IMPLICIT_CONTRADICTION_PROMPT = textwrap.dedent("""\
    You are a strict logical evaluator for a detective reasoning exercise.

    CASE EVIDENCE:
    {evidence}

    HIDDEN TRUTH (for evaluation only — the user does not see this):
    {hidden_truth}

    USER'S CONTRADICTION NOTES:
    {contradictions}

    USER'S ANOMALY ANALYSIS:
    {anomalies}

    TASK: Evaluate whether the user identified implicit (non-obvious) contradictions
    between facts — tensions that are NOT stated explicitly in the evidence but emerge
    when facts are cross-referenced.

    Score the user:
    - "STRONG" = found at least one implicit contradiction (facts that shouldn't coexist).
    - "WEAK"   = only restated explicit facts or missed obvious tensions.

    Return EXACTLY this JSON (no markdown, no explanation outside the JSON):
    {{"score": "STRONG" or "WEAK", "implicit_found": ["<contradiction 1>", ...], "missed": ["<missed tension 1>", ...]}}
""")

_COHERENCE_PROMPT = textwrap.dedent("""\
    You are evaluating the logical chain of a detective reasoning exercise.

    CASE EVIDENCE:
    {evidence}

    USER'S OBSERVATIONS:
    {observations}

    USER'S HYPOTHESES:
    {hypotheses}

    USER'S ELIMINATION TARGET:
    {elimination_target}

    USER'S REASONING CHAIN:
    {reasoning}

    TASK: Evaluate whether the reasoning chain is COHERENT:
    1. Do the observations actually support the hypotheses?
    2. Do the hypotheses actually justify the elimination?
    3. Are there logical jumps (conclusions not supported by prior steps)?

    Score the user:
    - "COHERENT"   = observations→hypotheses→elimination forms a valid chain.
    - "INCOHERENT" = there are gaps, jumps, or unsupported conclusions.

    Return EXACTLY this JSON (no markdown, no explanation outside the JSON):
    {{"score": "COHERENT" or "INCOHERENT", "chain_gaps": ["<gap 1>", ...], "strongest_link": "<best supported step>"}}
""")

_COUNTERARGUMENT_PROMPT = textwrap.dedent("""\
    You are a devil's advocate reviewing a detective's conclusion.

    CASE EVIDENCE:
    {evidence}

    USER ELIMINATED: {elimination_target}
    REASONING: {reasoning}

    TASK: Generate 2-3 counterarguments that challenge the user's elimination.
    These should be plausible alternative explanations or weaknesses in the logic.

    Then evaluate: can the user's conclusion SURVIVE these objections?
    - "SURVIVES"  = the elimination is robust despite counterarguments.
    - "FRAGILE"   = the counterarguments expose fatal weaknesses.

    Return EXACTLY this JSON (no markdown, no explanation outside the JSON):
    {{"score": "SURVIVES" or "FRAGILE", "counterarguments": ["<objection 1>", "<objection 2>", ...], "verdict_reason": "<one sentence>"}}
""")

_CRITIQUE_PROMPT = textwrap.dedent("""\
    You are a senior detective trainer reviewing a student's reasoning.

    CASE EVIDENCE:
    {evidence}

    STUDENT'S OBSERVATIONS:
    {observations}

    STUDENT'S REASONING:
    {reasoning}

    Point out:
    1. Any observation that is actually an interpretation (inference leak).
    2. Any reasoning gap — evidence that was ignored.
    3. Any logical jump — conclusions not supported by stated observations.
    4. Specific improvements they should make next time.

    Be strict but constructive.  Keep response under 200 words.
""")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LLMRoleResult:
    """Result from a single LLM evaluation role."""
    role: str          # "contradiction" | "coherence" | "counterargument"
    score: str         # "STRONG"/"WEAK", "COHERENT"/"INCOHERENT", "SURVIVES"/"FRAGILE", "SKIP"
    points: int        # 0 or 1
    details: dict = field(default_factory=dict)
    raw: str = ""


@dataclass
class LLMFullResult:
    """Aggregated result from all LLM evaluation roles."""
    roles: list[LLMRoleResult] = field(default_factory=list)
    total_score: int = 0        # 0–3
    max_score: int = 3
    critique: str = ""
    counterarguments: list[str] = field(default_factory=list)
    enabled: bool = False

    @property
    def verdict(self) -> str:
        """Legacy compat: VALID if ≥2 of 3 roles pass, INVALID otherwise."""
        if not self.enabled:
            return "SKIP"
        return "VALID" if self.total_score >= 2 else "INVALID"

    @property
    def reasoning(self) -> str:
        parts = []
        for r in self.roles:
            if r.score != "SKIP":
                parts.append(f"{r.role}: {r.score}")
        return "; ".join(parts) if parts else "(LLM skipped)"


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Multi-role LLM reasoning evaluator.

    Parameters
    ----------
    model : str
        Model name for ``ollama run`` (e.g. ``"deepseek-r1"``, ``"qwen2.5-coder"``).
    endpoint : str | None
        If set, uses HTTP POST to an OpenAI-compatible endpoint instead of ollama CLI.
    enabled : bool
        Master switch.  When ``False``, all calls return skip results instantly.
    timeout : int
        Seconds before giving up on the model.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        endpoint: str | None = None,
        enabled: bool = False,
        timeout: int = 90,
    ):
        self.model = model
        self.endpoint = endpoint
        self.enabled = enabled
        self.timeout = timeout

    # ------------------------------------------------------------------
    # public API — full multi-role evaluation
    # ------------------------------------------------------------------

    def evaluate_full(
        self,
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> LLMFullResult:
        """Run all three scoring roles + critique.  Returns LLMFullResult."""
        if not self.enabled:
            return LLMFullResult(
                roles=[
                    LLMRoleResult(role="contradiction", score="SKIP", points=0),
                    LLMRoleResult(role="coherence", score="SKIP", points=0),
                    LLMRoleResult(role="counterargument", score="SKIP", points=0),
                ],
                total_score=0,
                critique="(LLM judge disabled — no critique available.)",
                enabled=False,
            )

        r1 = self._evaluate_implicit_contradictions(case, record)
        r2 = self._evaluate_coherence(case, record)
        r3 = self._evaluate_counterarguments(case, record)
        critique = self._generate_critique(case, record)

        roles = [r1, r2, r3]
        total = sum(r.points for r in roles)
        counterargs = r3.details.get("counterarguments", [])

        return LLMFullResult(
            roles=roles,
            total_score=total,
            critique=critique,
            counterarguments=counterargs,
            enabled=True,
        )

    # ------------------------------------------------------------------
    # legacy compat — single-verdict API
    # ------------------------------------------------------------------

    def evaluate_elimination(
        self,
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> LLMFullResult:
        """Legacy wrapper — returns the full multi-role result."""
        return self.evaluate_full(case, record)

    def critique_reasoning(
        self,
        case: CaseDefinition,
        record: AnalysisRecord,
    ) -> str:
        """Get qualitative critique only (for training feedback)."""
        if not self.enabled:
            return "(LLM judge disabled — no critique available.)"
        return self._generate_critique(case, record)

    # ------------------------------------------------------------------
    # Role 1: Implicit contradiction detection
    # ------------------------------------------------------------------

    def _evaluate_implicit_contradictions(
        self, case: CaseDefinition, record: AnalysisRecord,
    ) -> LLMRoleResult:
        prompt = _IMPLICIT_CONTRADICTION_PROMPT.format(
            evidence="\n".join(f"- {e}" for e in case.evidence),
            hidden_truth=case.hidden_truth or "(none)",
            contradictions=", ".join(record.contradiction_notes) or "(none)",
            anomalies=", ".join(record.anomalies) or "(none)",
        )
        raw = self._call(prompt)
        parsed = self._safe_json(raw)
        score_str = parsed.get("score", "WEAK").upper()
        points = 1 if score_str == "STRONG" else 0
        return LLMRoleResult(
            role="contradiction",
            score=score_str,
            points=points,
            details=parsed,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Role 2: Reasoning chain coherence
    # ------------------------------------------------------------------

    def _evaluate_coherence(
        self, case: CaseDefinition, record: AnalysisRecord,
    ) -> LLMRoleResult:
        hyp_text = "; ".join(
            f"{k}: {', '.join(v)}" for k, v in record.hypotheses.items()
        )
        prompt = _COHERENCE_PROMPT.format(
            evidence="\n".join(f"- {e}" for e in case.evidence),
            observations=", ".join(record.observations),
            hypotheses=hyp_text or "(none)",
            elimination_target=record.elimination_target,
            reasoning=", ".join(record.reasons),
        )
        raw = self._call(prompt)
        parsed = self._safe_json(raw)
        score_str = parsed.get("score", "INCOHERENT").upper()
        points = 1 if score_str == "COHERENT" else 0
        return LLMRoleResult(
            role="coherence",
            score=score_str,
            points=points,
            details=parsed,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Role 3: Counterargument stress-test
    # ------------------------------------------------------------------

    def _evaluate_counterarguments(
        self, case: CaseDefinition, record: AnalysisRecord,
    ) -> LLMRoleResult:
        prompt = _COUNTERARGUMENT_PROMPT.format(
            evidence="\n".join(f"- {e}" for e in case.evidence),
            elimination_target=record.elimination_target,
            reasoning=", ".join(record.reasons),
        )
        raw = self._call(prompt)
        parsed = self._safe_json(raw)
        score_str = parsed.get("score", "FRAGILE").upper()
        points = 1 if score_str == "SURVIVES" else 0
        return LLMRoleResult(
            role="counterargument",
            score=score_str,
            points=points,
            details=parsed,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Critique (non-scoring)
    # ------------------------------------------------------------------

    def _generate_critique(
        self, case: CaseDefinition, record: AnalysisRecord,
    ) -> str:
        prompt = _CRITIQUE_PROMPT.format(
            evidence="\n".join(f"- {e}" for e in case.evidence),
            observations=", ".join(record.observations),
            reasoning=", ".join(record.reasons),
        )
        return self._call(prompt)

    # ------------------------------------------------------------------
    # model communication
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> str:
        """Send prompt to local model and return raw text."""
        if self.endpoint:
            return self._call_http(prompt)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.stdout.strip()
        except FileNotFoundError:
            return '{"score": "SKIP", "reason": "ollama not found on PATH."}'
        except subprocess.TimeoutExpired:
            return '{"score": "SKIP", "reason": "Model timed out."}'
        except Exception as exc:
            return f'{{"score": "SKIP", "reason": "Error: {exc}"}}'

    def _call_http(self, prompt: str) -> str:
        """Call an OpenAI-compatible local endpoint."""
        try:
            import urllib.request
            body = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }).encode()
            req = urllib.request.Request(
                self.endpoint,  # type: ignore[arg-type]
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f'{{"score": "SKIP", "reason": "HTTP error: {exc}"}}'

    # ------------------------------------------------------------------
    # response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json(raw: str) -> dict:
        """Extract JSON from LLM response (tolerant of messy output)."""
        # Try to find JSON object in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

        # Try the whole string
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass

        # Keyword fallback
        upper = raw.upper()
        for positive in ("STRONG", "COHERENT", "SURVIVES", "VALID"):
            if positive in upper:
                return {"score": positive}
        for negative in ("WEAK", "INCOHERENT", "FRAGILE", "INVALID"):
            if negative in upper:
                return {"score": negative}

        return {"score": "SKIP", "reason": "Could not parse LLM response."}
